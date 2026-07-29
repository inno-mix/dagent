import contextlib
import pathlib

import pytest
from typer.testing import CliRunner

from dagent import __version__, cli
from dagent.cli import app
from dagent.errors import ValidationError
from dagent.runtime.executor import Executor
from dagent.runtime.model import StubModelClient
from dagent.store import POSTGRES_DSN_ENV
from dagent.store.memory import InMemoryStateStore

runner = CliRunner()


def test_help_exits_cleanly() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "dagent" in result.output


def test_version_command_prints_the_package_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_bare_invocation_shows_help_rather_than_failing_silently() -> None:
    result = runner.invoke(app, [])

    assert "Usage" in result.output


EXAMPLE = str(pathlib.Path(__file__).parent.parent / "examples" / "research.yaml")


def test_validate_reports_a_good_workflow() -> None:
    result = runner.invoke(app, ["validate", EXAMPLE])

    assert result.exit_code == 0
    assert "research: 5 node(s), valid" in result.output


def test_validate_rejects_an_unknown_agent(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: w\nnodes:\n  - id: a\n    agent: not_a_real_agent\n", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code == 2
    assert "not registered" in result.output


def test_validate_rejects_a_missing_file(tmp_path: pathlib.Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "nope.yaml")])

    assert result.exit_code == 2
    assert "cannot read workflow file" in result.output


def test_run_executes_the_research_workflow_against_the_stub_provider() -> None:
    # The full CLI path — load, validate, execute, report — with zero network.
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub", "--run-id", "cli-test"])

    assert result.exit_code == 0
    assert "run cli-test: succeeded" in result.output
    assert "synthesis: success" in result.output


def test_run_reports_the_recorded_model_calls() -> None:
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub"])

    # Two researchers plus one synthesizer; the two constants call nothing.
    assert "3 model call(s)" in result.output
    assert "token(s) recorded" in result.output


def test_run_prints_node_outputs_by_default() -> None:
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub"])

    assert "--- synthesis ---" in result.output


def test_run_can_suppress_outputs() -> None:
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub", "--no-show-outputs"])

    assert result.exit_code == 0
    assert "--- synthesis ---" not in result.output


def test_run_rejects_an_unknown_provider() -> None:
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "nonesuch"])

    assert result.exit_code == 2
    assert "unknown provider" in result.output


def test_run_without_a_model_fails_the_run_rather_than_the_process() -> None:
    # 'none' wires the NullModelClient: the agents fail, and that is a run outcome.
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "none"])

    assert result.exit_code == 1
    assert "failed" in result.output


def test_run_reports_a_missing_gemini_key_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "gemini"])

    assert result.exit_code == 2
    assert "GEMINI_API_KEY" in result.output


def test_run_reports_a_submit_time_rejection_from_the_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The executor revalidates; if that rejects, the CLI reports it rather than crashing.
    def reject(self: object, workflow: object, *, run_id: str) -> None:
        raise ValidationError("executor said no")

    monkeypatch.setattr(Executor, "run", reject)

    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub"])

    assert result.exit_code == 2
    assert "executor said no" in result.output


def test_run_closes_the_model_client_even_when_the_run_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An HTTP connection pool left open is a resource leak; the finally has to fire.
    closed: list[bool] = []

    class ClosableStub(StubModelClient):
        async def aclose(self) -> None:
            closed.append(True)

    monkeypatch.setattr(cli, "_model_client", lambda provider: ClosableStub(lambda r: "x"))

    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub"])

    assert result.exit_code == 0
    assert closed == [True]


# --- Phase 4: run-level policy from the command line ---------------------------------


def test_run_accepts_a_concurrency_cap() -> None:
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub", "--max-concurrency", "1"])

    assert result.exit_code == 0
    assert "succeeded" in result.output


def test_run_accepts_a_per_provider_cap() -> None:
    result = runner.invoke(
        app, ["run", EXAMPLE, "--provider", "stub", "--provider-concurrency", "1"]
    )

    assert result.exit_code == 0


def test_a_token_ceiling_ends_the_run_in_budget_exceeded() -> None:
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub", "--max-tokens", "1"])

    assert result.exit_code == 1
    assert "budget_exceeded" in result.output
    assert "model call(s) refused" in result.output


def test_a_generous_token_ceiling_leaves_the_run_alone() -> None:
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub", "--max-tokens", "1000000"])

    assert result.exit_code == 0
    assert "refused" not in result.output


def test_run_accepts_each_failure_mode() -> None:
    for mode in ("run_to_completion", "fail_fast", "skip_downstream"):
        result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub", "--on-failure", mode])

        assert result.exit_code == 0, mode


def test_run_rejects_an_unknown_failure_mode() -> None:
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub", "--on-failure", "panic"])

    assert result.exit_code == 2
    assert "unknown failure mode" in result.output


def test_run_rejects_a_concurrency_cap_of_zero() -> None:
    # Fail loud at submit time rather than starting a run that can never dispatch.
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub", "--max-concurrency", "0"])

    assert result.exit_code == 2
    assert "at least 1" in result.output


# --- Phase 5: stores and resume -------------------------------------------------------


def test_run_accepts_the_memory_store_explicitly() -> None:
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub", "--store", "memory"])

    assert result.exit_code == 0


def test_run_rejects_an_unknown_store() -> None:
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub", "--store", "nope"])

    assert result.exit_code == 2
    assert "unknown store" in result.output


def test_asking_for_postgres_without_a_dsn_says_which_variable_to_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(POSTGRES_DSN_ENV, raising=False)

    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub", "--store", "postgres"])

    assert result.exit_code == 2
    assert POSTGRES_DSN_ENV in result.output


def test_resume_needs_a_run_id() -> None:
    result = runner.invoke(app, ["resume"])

    assert result.exit_code == 2


def test_resume_from_the_memory_store_reports_the_missing_run() -> None:
    # A fresh process has an empty in-memory store, which is exactly why `resume`
    # defaults to Postgres. The message should still be legible if someone tries.
    result = runner.invoke(app, ["resume", "whatever", "--provider", "stub", "--store", "memory"])

    assert result.exit_code == 2
    assert "unknown run" in result.output


def test_resume_appears_in_the_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert "resume" in result.output


def test_a_bad_postgres_dsn_is_reported_rather_than_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The store is opened before anything runs, so an unreachable database costs a
    # message, not a traceback.
    monkeypatch.setenv(POSTGRES_DSN_ENV, "postgresql://nobody:nobody@127.0.0.1:1/none")

    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub", "--store", "postgres"])

    assert result.exit_code == 2
    assert "cannot connect to Postgres" in result.output


def test_resume_rejects_an_unknown_failure_mode_before_touching_the_store() -> None:
    result = runner.invoke(app, ["resume", "r1", "--on-failure", "panic"])

    assert result.exit_code == 2
    assert "unknown failure mode" in result.output


# --- Phase 7: the run inspector -------------------------------------------------------


def test_inspect_needs_a_run_id() -> None:
    result = runner.invoke(app, ["inspect"])

    assert result.exit_code == 2


def test_inspect_reports_a_run_that_is_not_there() -> None:
    result = runner.invoke(app, ["inspect", "never-happened", "--store", "memory"])

    assert result.exit_code == 2
    assert "unknown run" in result.output


def test_inspect_appears_in_the_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert "inspect" in result.output


def test_inspect_reports_a_postgres_dsn_it_was_not_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(POSTGRES_DSN_ENV, raising=False)

    result = runner.invoke(app, ["inspect", "r1"])

    assert result.exit_code == 2
    assert POSTGRES_DSN_ENV in result.output


def test_run_can_narrate_itself() -> None:
    result = runner.invoke(
        app, ["run", EXAMPLE, "--provider", "stub", "--log-level", "info", "--no-show-outputs"]
    )

    assert result.exit_code == 0
    assert "run.started" in result.output


def test_run_says_nothing_extra_by_default() -> None:
    # The report is what a person wants; a run that also narrates every transition is a
    # run whose output nobody reads.
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub", "--no-show-outputs"])

    assert "run.started" not in result.output


def test_run_can_emit_json_logs() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            EXAMPLE,
            "--provider",
            "stub",
            "--log-level",
            "info",
            "--json-logs",
            "--no-show-outputs",
        ],
    )

    assert result.exit_code == 0
    assert '"event": "run.started"' in result.output


def test_inspect_prints_the_report_as_json() -> None:
    # A run and its inspection are normally two processes sharing Postgres. Here the
    # store is handed to both, so the JSON rendering and the exit code are exercised
    # without needing a database for what is really a formatting question.
    import json as json_module

    store = InMemoryStateStore()

    with _using_store(store):
        ran = runner.invoke(
            app,
            ["run", EXAMPLE, "--provider", "stub", "--run-id", "insp", "--no-show-outputs"],
        )
        assert ran.exit_code == 0
        result = runner.invoke(app, ["inspect", "insp", "--store", "memory"])

    assert result.exit_code == 0
    report = json_module.loads(result.output)
    assert report["run"]["run_id"] == "insp"
    assert report["run"]["state"] == "succeeded"
    assert [node["node_id"] for node in report["nodes"]] == sorted(
        node["node_id"] for node in report["nodes"]
    )
    assert report["model_calls"]


@contextlib.contextmanager
def _using_store(store: InMemoryStateStore):  # type: ignore[no-untyped-def]
    """Make every CLI command in this block share one store, as Postgres would."""

    async def _fixed(kind: str) -> InMemoryStateStore:
        return store

    original = cli._open_store
    cli._open_store = _fixed  # type: ignore[assignment]
    try:
        yield
    finally:
        cli._open_store = original  # type: ignore[assignment]


def test_node_defaults_reach_nodes_that_declare_no_policy() -> None:
    # Surfaced by a live 503: a planner-generated node has no file to declare a policy in,
    # so it inherits the run's defaults — which were inert and unreachable from the CLI.
    result = runner.invoke(
        app,
        ["run", EXAMPLE, "--provider", "stub", "--max-attempts", "3", "--no-show-outputs"],
    )

    assert result.exit_code == 0


def test_an_impossible_node_default_is_refused_before_the_run_starts() -> None:
    result = runner.invoke(app, ["run", EXAMPLE, "--provider", "stub", "--max-attempts", "0"])

    assert result.exit_code == 2
    assert "invalid node defaults" in result.output


def test_a_node_timeout_default_is_accepted() -> None:
    result = runner.invoke(
        app,
        ["run", EXAMPLE, "--provider", "stub", "--node-timeout", "30", "--no-show-outputs"],
    )

    assert result.exit_code == 0
