"""The ``dagent`` command-line entrypoint.

Importing this module registers the built-in agents, so a workflow file can name them.
Commands arrive with the phase that gives them something to do: ``resume`` with durable
state (Phase 5) and ``inspect`` with the run inspector (Phase 7).
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from collections.abc import Mapping
from typing import Annotated

import typer

import dagent.agents  # noqa: F401  — imported for the side effect of registering agents
from dagent import __version__
from dagent.errors import DagentError, PolicyError
from dagent.graph.validate import validate as validate_graph
from dagent.loader import load_workflow_file
from dagent.models.state import NodeState, NodeStateRecord, RunState
from dagent.models.workflow import Workflow
from dagent.policy.limits import Budget, Limits
from dagent.policy.run import FailureMode, RunPolicy
from dagent.runtime.executor import Executor
from dagent.runtime.model import ModelClient, NullModelClient, StubModelClient
from dagent.runtime.registry import default_registry
from dagent.store.base import StateStore
from dagent.store.memory import InMemoryStateStore

app = typer.Typer(
    name="dagent",
    help="A durable, async DAG execution engine for multi-agent AI workflows.",
    no_args_is_help=True,
)

WorkflowFile = Annotated[pathlib.Path, typer.Argument(help="Path to a workflow YAML file.")]


@app.callback()
def main() -> None:
    """Group the subcommands.

    Without this, Typer collapses a single-command app into a bare entrypoint, and
    ``dagent version`` would stop working the moment a second command is added.
    """


@app.command()
def version() -> None:
    """Print the installed dagent version."""
    typer.echo(__version__)


@app.command()
def validate(workflow_file: WorkflowFile) -> None:
    """Check a workflow file without running it.

    Every rule the executor applies at submit time, applied here instead, so a typo costs
    a second rather than a partly-executed run.
    """
    workflow = _load(workflow_file)
    typer.echo(f"{workflow.name}: {len(workflow.nodes)} node(s), valid")
    for node in workflow.nodes:
        upstream = ", ".join(node.depends_on) or "-"
        typer.echo(f"  {node.id} [{node.agent}] <- {upstream}")


@app.command()
def run(
    workflow_file: WorkflowFile,
    run_id: Annotated[str, typer.Option(help="Identifier for this run.")] = "cli",
    provider: Annotated[
        str, typer.Option(help="Model provider: 'gemini', 'stub' for a dry run, or 'none'.")
    ] = "gemini",
    show_outputs: Annotated[bool, typer.Option(help="Print each node's output.")] = True,
    max_concurrency: Annotated[
        int | None, typer.Option(help="Cap on nodes running at once. Default: unlimited.")
    ] = None,
    provider_concurrency: Annotated[
        int | None, typer.Option(help="Cap on nodes per provider. Default: unlimited.")
    ] = None,
    max_tokens: Annotated[
        int | None, typer.Option(help="Token ceiling for the whole run. Default: unlimited.")
    ] = None,
    on_failure: Annotated[
        str,
        typer.Option(help="run_to_completion | fail_fast | skip_downstream."),
    ] = FailureMode.RUN_TO_COMPLETION.value,
) -> None:
    """Run a workflow end to end and report what each node did.

    Credentials come from the environment and nowhere else, so run this under
    ``uv run --env-file .env dagent run ...`` or export the key yourself.

    Retries and timeouts are per node and come from the workflow file's ``policy`` block.
    The options here are per run: they are how much of the machine, and of the budget,
    this particular execution is allowed to use.
    """
    workflow = _load(workflow_file)
    try:
        policy = _run_policy(
            max_concurrency=max_concurrency,
            provider_concurrency=provider_concurrency,
            max_tokens=max_tokens,
            on_failure=on_failure,
        )
    except DagentError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc

    code = asyncio.run(
        _execute(
            workflow,
            run_id=run_id,
            provider=provider,
            show_outputs=show_outputs,
            policy=policy,
        )
    )
    raise typer.Exit(code)


def _run_policy(
    *,
    max_concurrency: int | None,
    provider_concurrency: int | None,
    max_tokens: int | None,
    on_failure: str,
) -> RunPolicy:
    """Assemble the run-level policy from the command line.

    Raises:
        PolicyError: If a cap is out of range or the failure mode is not one of the three.
    """
    try:
        failure_mode = FailureMode(on_failure)
    except ValueError as exc:
        modes = ", ".join(mode.value for mode in FailureMode)
        raise PolicyError(f"unknown failure mode {on_failure!r}; expected one of {modes}") from exc

    return RunPolicy(
        failure_mode=failure_mode,
        limits=Limits(
            max_concurrency=max_concurrency,
            default_per_provider=provider_concurrency,
        ),
        budget=Budget(max_tokens=max_tokens),
    )


def _load(workflow_file: pathlib.Path) -> Workflow:
    """Load and fully validate a workflow file, or exit with a readable message."""
    try:
        workflow = load_workflow_file(workflow_file)
        validate_graph(workflow, known_agents=default_registry.names())
    except DagentError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    return workflow


def _model_client(provider: str) -> ModelClient:
    """Build the one model client the whole run shares.

    Raises:
        DagentError: If the provider is unknown, or its credentials are missing.
    """
    if provider == "stub":
        return StubModelClient(lambda request: f"[stub reply to {request.prompt[:60]!r}]")
    if provider == "none":
        return NullModelClient()
    if provider == "gemini":
        from dagent.agents.gemini import GeminiClient

        return GeminiClient.from_env()
    raise DagentError(f"unknown provider {provider!r}; expected 'gemini', 'stub', or 'none'")


async def _execute(
    workflow: Workflow,
    *,
    run_id: str,
    provider: str,
    show_outputs: bool,
    policy: RunPolicy,
) -> int:
    """Execute the workflow, print a report, and return the process exit code."""
    store = InMemoryStateStore()

    try:
        client = _model_client(provider)
    except DagentError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        return 2

    try:
        result = await Executor(
            registry=default_registry, store=store, model=client, policy=policy
        ).run(workflow, run_id=run_id)
    except DagentError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        return 2
    finally:
        closer = getattr(client, "aclose", None)
        if closer is not None:
            await closer()

    typer.echo(f"run {result.run_id}: {result.state}")
    for node_id in sorted(result.nodes):
        record = result.nodes[node_id]
        detail = f"  {record.error}" if record.error else ""
        typer.echo(f"  {node_id}: {record.state}{detail}")

    calls = await store.load_model_calls(result.run_id)
    if calls:
        tokens = sum(call.response.total_tokens for call in calls)
        typer.echo(f"\n{len(calls)} model call(s), {tokens} token(s) recorded")

    if policy.budget.refused:
        typer.secho(
            f"budget: {policy.budget.describe()} — {policy.budget.refusals} model call(s) refused",
            fg=typer.colors.YELLOW,
        )

    if show_outputs:
        await _print_outputs(store, result.run_id, result.nodes)

    return 0 if result.state is RunState.SUCCEEDED else 1


async def _print_outputs(
    store: StateStore, run_id: str, nodes: Mapping[str, NodeStateRecord]
) -> None:
    """Print the output of every node that produced one."""
    for node_id in sorted(nodes):
        if nodes[node_id].state is not NodeState.SUCCESS:
            continue
        output = await store.load_output(run_id, node_id)
        typer.echo(f"\n--- {node_id} ---")
        typer.echo(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    app()
