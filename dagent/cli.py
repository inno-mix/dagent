"""The ``dagent`` command-line entrypoint.

Importing this module registers the built-in agents, so a workflow file can name them.
Commands arrive with the phase that gives them something to do: ``resume`` with durable
state (Phase 5) and ``inspect`` with the run inspector (Phase 7).
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated, TypeAlias

import typer
from pydantic import ValidationError as PydanticValidationError

import dagent.agents  # noqa: F401  — imported for the side effect of registering agents
from dagent import __version__
from dagent.errors import DagentError, PolicyError, StoreError
from dagent.graph.validate import validate as validate_graph
from dagent.loader import load_workflow_file
from dagent.models.state import NodeState, NodeStateRecord, RunState, RunStateRecord
from dagent.models.workflow import Policy, Workflow
from dagent.observability import logging as obs_logging
from dagent.observability.inspector import inspect_run
from dagent.policy.limits import Budget, Limits
from dagent.policy.run import FailureMode, RunPolicy
from dagent.runtime.executor import Executor
from dagent.runtime.model import ModelClient, NullModelClient, StubModelClient
from dagent.runtime.registry import default_registry
from dagent.store import POSTGRES_DSN_ENV
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
def inspect(
    run_id: Annotated[str, typer.Argument(help="Identifier of the run to dump.")],
    store: Annotated[
        str, typer.Option(help=f"State store: 'postgres' via ${POSTGRES_DSN_ENV}, or 'memory'.")
    ] = "postgres",
    outputs: Annotated[bool, typer.Option(help="Include each node's output.")] = True,
    indent: Annotated[int, typer.Option(help="JSON indent; 0 for one line.")] = 2,
) -> None:
    """Dump everything the store knows about a run, as JSON (FR-10).

    Node states, attempt counts, timings, errors, the definition as actually executed —
    expanded, if a planner grew it — every recorded model call, and each node's output.
    Enough to reconstruct what happened without the process that ran it.

    Defaults to the Postgres store, because a run worth inspecting is usually one that has
    already finished in some other process.
    """
    code = asyncio.run(_inspect(run_id, store=store, outputs=outputs, indent=indent))
    raise typer.Exit(code)


async def _inspect(run_id: str, *, store: str, outputs: bool, indent: int) -> int:
    """Assemble the report and print it, or explain why not."""
    try:
        state_store = await _open_store(store)
    except DagentError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        return 2

    try:
        report = await inspect_run(state_store, run_id, outputs=outputs)
    except DagentError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        return 2
    finally:
        await _aclose(state_store)

    typer.echo(json.dumps(report, indent=indent or None, ensure_ascii=False, default=str))
    return 0


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
    store: Annotated[
        str, typer.Option(help=f"State store: 'memory', or 'postgres' via ${POSTGRES_DSN_ENV}.")
    ] = "memory",
    max_attempts: Annotated[
        int, typer.Option(help="Attempts for nodes that declare no policy of their own.")
    ] = 1,
    node_timeout: Annotated[
        float | None,
        typer.Option(help="Timeout in seconds for nodes that declare no policy."),
    ] = None,
    log_level: Annotated[
        str, typer.Option(help="Structured log level: debug, info, warning, error, or 'off'.")
    ] = "off",
    json_logs: Annotated[
        bool, typer.Option(help="Render logs as JSON rather than for a human.")
    ] = False,
) -> None:
    """Run a workflow end to end and report what each node did.

    Credentials come from the environment and nowhere else, so run this under
    ``uv run --env-file .env dagent run ...`` or export the key yourself.

    Retries and timeouts are per node and come from the workflow file's ``policy`` block.
    The options here are per run: they are how much of the machine, and of the budget,
    this particular execution is allowed to use.
    """
    _logging(log_level, json_logs)
    workflow = _load(workflow_file)
    try:
        policy = _run_policy(
            max_concurrency=max_concurrency,
            provider_concurrency=provider_concurrency,
            max_tokens=max_tokens,
            on_failure=on_failure,
            max_attempts=max_attempts,
            node_timeout=node_timeout,
        )
    except DagentError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc

    code = asyncio.run(
        _drive(
            _start(workflow, run_id=run_id),
            run_id=run_id,
            provider=provider,
            store=store,
            show_outputs=show_outputs,
            policy=policy,
        )
    )
    raise typer.Exit(code)


@app.command()
def resume(
    run_id: Annotated[str, typer.Argument(help="Identifier of the run to continue.")],
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
    store: Annotated[
        str, typer.Option(help=f"State store: 'postgres' via ${POSTGRES_DSN_ENV}, or 'memory'.")
    ] = "postgres",
    max_attempts: Annotated[
        int, typer.Option(help="Attempts for nodes that declare no policy of their own.")
    ] = 1,
    node_timeout: Annotated[
        float | None,
        typer.Option(help="Timeout in seconds for nodes that declare no policy."),
    ] = None,
    log_level: Annotated[
        str, typer.Option(help="Structured log level: debug, info, warning, error, or 'off'.")
    ] = "off",
    json_logs: Annotated[
        bool, typer.Option(help="Render logs as JSON rather than for a human.")
    ] = False,
) -> None:
    """Continue a run that was interrupted, from wherever it got to.

    Takes no workflow file: the definition was stored with the run, so a resumed run
    cannot drift onto a different graph than the one that was interrupted.

    Defaults to the Postgres store, because that is the only one whose state outlives the
    process that crashed — resuming from memory in a fresh process has nothing to read.

    Nodes that already succeeded are skipped. Nodes that were mid-flight are re-dispatched
    under the same idempotency key they were already using, so an agent that deduplicates
    on ``ctx.idempotency_key`` commits its effect exactly once across the crash.
    """
    _logging(log_level, json_logs)
    try:
        policy = _run_policy(
            max_concurrency=max_concurrency,
            provider_concurrency=provider_concurrency,
            max_tokens=max_tokens,
            on_failure=on_failure,
            max_attempts=max_attempts,
            node_timeout=node_timeout,
        )
    except DagentError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc

    code = asyncio.run(
        _drive(
            _continue(run_id),
            run_id=run_id,
            provider=provider,
            store=store,
            show_outputs=show_outputs,
            policy=policy,
        )
    )
    raise typer.Exit(code)


def _logging(level: str, json_logs: bool) -> None:
    """Turn structured logging on, unless the caller asked for none.

    Off by default: the report this CLI already prints is what a person wants, and a run
    that also narrates every state transition is a run whose output nobody reads. Anyone
    who needs the narration asks for it.
    """
    if level.lower() == "off":
        return
    obs_logging.configure(level=level.upper(), json=json_logs)


def _run_policy(
    *,
    max_concurrency: int | None,
    provider_concurrency: int | None,
    max_tokens: int | None,
    on_failure: str,
    max_attempts: int = 1,
    node_timeout: float | None = None,
) -> RunPolicy:
    """Assemble the run-level policy from the command line.

    ``max_attempts`` and ``node_timeout`` become the defaults for nodes that declare no
    policy — which includes every node a planner generates at run time, since a generated
    node has no file to have written one in. Without this the dynamic workflow was
    strictly less robust than the hand-written one, which a live 503 demonstrated.

    Raises:
        PolicyError: If a cap is out of range or the failure mode is not one of the three.
    """
    try:
        failure_mode = FailureMode(on_failure)
    except ValueError as exc:
        modes = ", ".join(mode.value for mode in FailureMode)
        raise PolicyError(f"unknown failure mode {on_failure!r}; expected one of {modes}") from exc

    try:
        defaults = Policy(max_attempts=max_attempts, timeout_s=node_timeout)
    except PydanticValidationError as exc:
        raise PolicyError(f"invalid node defaults: {exc}") from exc

    return RunPolicy(
        failure_mode=failure_mode,
        node_defaults=defaults,
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


Drive: TypeAlias = Callable[[Executor], Awaitable[RunStateRecord]]
"""What to ask the executor to do — start a workflow, or continue a run."""


def _start(workflow: Workflow, *, run_id: str) -> Drive:
    """Begin a new run."""

    async def go(executor: Executor) -> RunStateRecord:
        return await executor.run(workflow, run_id=run_id)

    return go


def _continue(run_id: str) -> Drive:
    """Pick a run back up from the store."""

    async def go(executor: Executor) -> RunStateRecord:
        return await executor.resume(run_id)

    return go


async def _open_store(kind: str) -> StateStore:
    """Open the state store the caller asked for.

    Raises:
        DagentError: If the kind is unknown, or Postgres was asked for without a DSN.
    """
    if kind == "memory":
        return InMemoryStateStore()
    if kind == "postgres":
        dsn = os.environ.get(POSTGRES_DSN_ENV)
        if not dsn:
            raise StoreError(
                f"{POSTGRES_DSN_ENV} is not set; export a Postgres DSN, or pass --store memory "
                "for a run that does not need to outlive this process"
            )
        # Imported here so `dagent` stays usable with the `postgres` extra uninstalled.
        from dagent.store.postgres import PostgresStateStore

        return await PostgresStateStore.connect(dsn)
    raise StoreError(f"unknown store {kind!r}; expected 'memory' or 'postgres'")


async def _drive(
    drive: Drive,
    *,
    run_id: str,
    provider: str,
    store: str,
    show_outputs: bool,
    policy: RunPolicy,
) -> int:
    """Execute or resume, print a report, and return the process exit code."""
    try:
        state_store = await _open_store(store)
        client = _model_client(provider)
    except DagentError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        return 2

    try:
        result = await drive(
            Executor(registry=default_registry, store=state_store, model=client, policy=policy)
        )
    except DagentError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        return 2
    finally:
        await _aclose(client)

    try:
        return await _report(state_store, result, policy=policy, show_outputs=show_outputs)
    finally:
        await _aclose(state_store)


async def _aclose(resource: object) -> None:
    """Close a client or store that has something to close."""
    for name in ("aclose", "close"):
        closer = getattr(resource, name, None)
        if closer is not None:
            await closer()
            return


async def _report(
    store: StateStore, result: RunStateRecord, *, policy: RunPolicy, show_outputs: bool
) -> int:
    """Print what the run did and return the process exit code."""
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
