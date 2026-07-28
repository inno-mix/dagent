import pathlib

import pytest
from typer.testing import CliRunner

from dagent import __version__, cli
from dagent.cli import app
from dagent.errors import ValidationError
from dagent.runtime.executor import Executor
from dagent.runtime.model import StubModelClient

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
