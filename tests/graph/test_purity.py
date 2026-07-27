"""Phase 1 acceptance: "No async, no I/O in this phase".

`dagent.graph` and `dagent.models` are the layer the property tests depend on, and they
are only cheap to property-test because they are pure. This scans their AST rather than
trusting review, so the guarantee survives every later phase.
"""

import ast
import pathlib

import pytest

import dagent

PURE_PACKAGES = ("graph", "models")
PACKAGE_ROOT = pathlib.Path(dagent.__file__).parent
REPO_ROOT = PACKAGE_ROOT.parent

# I/O, concurrency, and the sources of non-determinism AGENTS.md rule 4 puts behind a seam.
BANNED_IMPORTS = frozenset(
    {
        "asyncio",
        "io",
        "os",
        "pathlib",
        "random",
        "secrets",
        "shutil",
        "socket",
        "subprocess",
        "tempfile",
        "threading",
        "time",
        "uuid",
    }
)
BANNED_BUILTIN_CALLS = frozenset({"open", "input"})
# `datetime` the type is fine; reading the clock is not.
BANNED_METHOD_CALLS = frozenset({"now", "utcnow", "monotonic", "perf_counter"})

FILES = sorted(path for package in PURE_PACKAGES for path in (PACKAGE_ROOT / package).rglob("*.py"))
IDS = [str(path.relative_to(REPO_ROOT)) for path in FILES]


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_the_scan_actually_covers_files() -> None:
    assert FILES


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_module_defines_nothing_async(path: pathlib.Path) -> None:
    tree = _parse(path)

    coroutines = [node.name for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)]
    awaits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Await | ast.AsyncFor | ast.AsyncWith)
    ]

    assert not coroutines, f"{path.name} defines coroutines {coroutines}"
    assert not awaits, f"{path.name} contains await/async-for/async-with"


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_module_imports_nothing_that_does_io_or_reads_the_clock(path: pathlib.Path) -> None:
    imported: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    offenders = sorted(imported & BANNED_IMPORTS)

    assert not offenders, f"{path.name} imports {offenders}; this layer must stay pure"


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_module_performs_no_io_and_reads_no_clock(path: pathlib.Path) -> None:
    offenders: list[str] = []
    for node in ast.walk(_parse(path)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTIN_CALLS:
            offenders.append(node.func.id)
        elif isinstance(node.func, ast.Attribute) and node.func.attr in BANNED_METHOD_CALLS:
            offenders.append(node.func.attr)

    assert not offenders, f"{path.name} calls {offenders}; this layer must stay deterministic"
