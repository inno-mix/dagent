"""The examples in `examples/` have to actually run, or they are documentation that lies.

They use only the fake agents, so this stays network-free like every other test here.
"""

import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest

EXAMPLES = sorted((pathlib.Path(__file__).parent.parent / "examples").glob("*.py"))
IDS = [path.name for path in EXAMPLES]


def _load(path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"dagent_example_{path.stem}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_there_are_examples_to_check() -> None:
    assert EXAMPLES


@pytest.mark.parametrize("path", EXAMPLES, ids=IDS)
def test_example_builds_a_valid_workflow(path: pathlib.Path) -> None:
    module = _load(path)

    workflow = module.build()

    assert workflow.nodes


@pytest.mark.asyncio
@pytest.mark.parametrize("path", EXAMPLES, ids=IDS)
async def test_example_runs_end_to_end(path: pathlib.Path) -> None:
    # The assertion is that main() completes: it executes the workflow and reads every
    # node's output back out of the store, so a broken engine raises here.
    module = _load(path)

    await module.main()
