"""The package skeleton matches AGENTS.md §10.

If a package moves, this fails and the §10 map has to be updated in the same change.
"""

import importlib
import pathlib

import pytest

import dagent

PACKAGE_ROOT = pathlib.Path(dagent.__file__).parent

# AGENTS.md §10, verbatim.
PACKAGES = [
    "models",
    "graph",
    "runtime",
    "policy",
    "store",
    "agents",
    "observability",
]
MODULES = ["errors", "cli"]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_named_in_agents_md_exists_and_imports(package: str) -> None:
    assert (PACKAGE_ROOT / package / "__init__.py").is_file()
    importlib.import_module(f"dagent.{package}")


@pytest.mark.parametrize("module", MODULES)
def test_module_named_in_agents_md_exists_and_imports(module: str) -> None:
    assert (PACKAGE_ROOT / f"{module}.py").is_file()
    importlib.import_module(f"dagent.{module}")


def test_package_ships_type_information() -> None:
    # AGENTS.md §5 makes typing mandatory; py.typed is what makes it visible downstream.
    assert (PACKAGE_ROOT / "py.typed").is_file()


def test_tests_mirror_the_package_layout() -> None:
    tests_root = pathlib.Path(__file__).parent
    for package in ("models", "graph", "runtime", "store", "agents"):
        assert (tests_root / package).is_dir(), f"tests/{package}/ should mirror dagent/{package}/"
