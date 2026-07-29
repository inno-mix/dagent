"""The load driver has to actually run, or the numbers in the README are folklore.

Deliberately tiny and untimed: a performance assertion in a unit suite measures the
machine that happens to be running it, and fails on a busy laptop for reasons that have
nothing to do with the code. What is checked here is that the driver still works — that
its shapes build valid graphs and the engine executes them — so a refactor that breaks it
is caught before someone reruns the benchmark and quotes a number from a stale script.
"""

import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest

DRIVER = pathlib.Path(__file__).parent.parent / "benchmarks" / "load.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dagent_benchmark_load", DRIVER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_the_driver_exists() -> None:
    assert DRIVER.is_file()


@pytest.mark.parametrize("shape", ["wide", "chain", "layered"])
def test_every_shape_builds_a_valid_graph_of_the_size_asked_for(shape: str) -> None:
    module = _load()

    workflow = module.BUILDERS[shape](12)

    assert len(workflow.nodes) == 12


@pytest.mark.parametrize("shape", ["wide", "chain", "layered"])
@pytest.mark.asyncio
async def test_every_shape_runs_to_completion(shape: str) -> None:
    module = _load()

    result = await module.measure(shape, 8, repeat=1, latency_ms=0)

    assert result.nodes == 8
    assert result.seconds > 0
    assert result.per_node_ms > 0


@pytest.mark.asyncio
async def test_the_latency_agent_is_used_when_asked_for() -> None:
    # The realistic mode: without it the driver can only answer "how fast is the engine",
    # not "does the engine disappear behind a model call".
    module = _load()

    result = await module.measure("wide", 4, repeat=1, latency_ms=10)

    assert result.seconds >= 0.01


def test_the_report_names_the_spec_target(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load()

    module.report([module.Result("wide", 100, 0.005)], latency_ms=0)

    assert "SPEC target" in capsys.readouterr().out


def test_the_report_says_when_latency_was_simulated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A per-node figure that includes a sleep must not be read as engine overhead.
    module = _load()

    module.report([module.Result("wide", 100, 5.0)], latency_ms=50)

    assert "simulated model latency" in capsys.readouterr().out
