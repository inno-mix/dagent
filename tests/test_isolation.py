"""The two boundaries that define this project, enforced mechanically.

AGENTS.md rule 3: the core imports no model SDK. AGENTS.md §7 and DR-2: no orchestration
framework appears anywhere. Both are checked by walking the AST of every source file
rather than by importing, so the guarantee holds for modules no test happens to import
and keeps holding as later phases add files — Phase 2's `runtime/executor.py` is covered
by this test before it is written.
"""

import ast
import pathlib
import re
import tomllib

import pytest

import dagent

PACKAGE_ROOT = pathlib.Path(dagent.__file__).parent
REPO_ROOT = PACKAGE_ROOT.parent

# Only dagent/agents/ may import these (ARCHITECTURE.md §2).
MODEL_SDKS = frozenset(
    {
        "anthropic",
        "boto3",
        "cohere",
        "google.genai",
        "google.generativeai",
        "groq",
        "litellm",
        "mistralai",
        "ollama",
        "openai",
        "replicate",
        "transformers",
        "vertexai",
    }
)

# The same SDKs under their PyPI distribution names, which are not always the import
# name: `google-genai` installs `google.genai`, so matching import names alone would let
# the Gemini SDK into the base install unnoticed.
MODEL_SDK_DISTRIBUTIONS = frozenset(
    {
        "anthropic",
        "boto3",
        "cohere",
        "google-cloud-aiplatform",
        "google-genai",
        "google-generativeai",
        "groq",
        "litellm",
        "mistralai",
        "ollama",
        "openai",
        "replicate",
        "transformers",
    }
)

# Nothing may import these, agents included. The point of the project is that the
# engine was built, not glued (DR-2).
ORCHESTRATION_FRAMEWORKS = frozenset(
    {
        "airflow",
        "autogen",
        "celery",
        "crewai",
        "dagster",
        "haystack",
        "langchain",
        "langchain_community",
        "langchain_core",
        "langgraph",
        "llama_index",
        "prefect",
        "semantic_kernel",
        "temporalio",
    }
)

ALL_FILES = sorted(PACKAGE_ROOT.rglob("*.py"))
CORE_FILES = [p for p in ALL_FILES if "agents" not in p.relative_to(PACKAGE_ROOT).parts]


def _ids(paths: list[pathlib.Path]) -> list[str]:
    return [str(p.relative_to(REPO_ROOT)) for p in paths]


def _imported_modules(path: pathlib.Path) -> set[str]:
    """Every absolute module name imported by a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module)
    return imported


def _banned_match(module: str, banned: frozenset[str]) -> str | None:
    """Return the banned prefix this module falls under, if any.

    Matching is on dotted prefixes so `openai.types` is caught by `openai` while
    `google.protobuf` is not caught by `google.generativeai`.
    """
    parts = module.split(".")
    for depth in range(len(parts), 0, -1):
        candidate = ".".join(parts[:depth])
        if candidate in banned:
            return candidate
    return None


def _requirement_name(requirement: str) -> str:
    """Strip version specifiers and extras from a requirement string."""
    return re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].strip().lower()


def _declared_requirements() -> list[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements: list[str] = list(data["project"]["dependencies"])
    for group in data.get("dependency-groups", {}).values():
        requirements.extend(item for item in group if isinstance(item, str))
    return requirements


def test_the_scan_actually_covers_files() -> None:
    # A parametrised test over an empty list passes silently; this is the tripwire.
    assert CORE_FILES
    assert ALL_FILES


def test_the_core_packages_named_in_agents_md_exist() -> None:
    for package in ("graph", "runtime", "policy", "store"):
        assert (PACKAGE_ROOT / package).is_dir()


@pytest.mark.parametrize("path", CORE_FILES, ids=_ids(CORE_FILES))
def test_core_module_imports_no_model_sdk(path: pathlib.Path) -> None:
    offenders = {
        module: banned
        for module in _imported_modules(path)
        if (banned := _banned_match(module, MODEL_SDKS)) is not None
    }
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} imports a model SDK {offenders}. "
        "Only dagent/agents/ may do that (AGENTS.md rule 3)."
    )


@pytest.mark.parametrize("path", CORE_FILES, ids=_ids(CORE_FILES))
def test_core_module_makes_no_http_calls(path: pathlib.Path) -> None:
    # Stronger than "no model SDK": nothing outside agents/ should be talking to a
    # provider at all. If HTTP appears in the core, a vendor has leaked in without
    # needing an SDK to do it.
    offenders = sorted(_imported_modules(path) & {"httpx", "requests", "aiohttp", "urllib3"})

    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} imports {offenders}; provider transport belongs "
        "in dagent/agents/ (AGENTS.md rule 3, §4)."
    )


@pytest.mark.parametrize("path", ALL_FILES, ids=_ids(ALL_FILES))
def test_no_module_imports_an_orchestration_framework(path: pathlib.Path) -> None:
    offenders = {
        module: banned
        for module in _imported_modules(path)
        if (banned := _banned_match(module, ORCHESTRATION_FRAMEWORKS)) is not None
    }
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} imports an orchestration framework {offenders}. "
        "The engine is built here, not glued (AGENTS.md §7, DR-2)."
    )


def test_no_model_sdk_is_declared_as_a_dependency() -> None:
    # A model SDK belongs to an agent's optional extra, never to the base install.
    for requirement in _declared_requirements():
        name = _requirement_name(requirement)
        assert name not in MODEL_SDK_DISTRIBUTIONS, f"{requirement!r} is a model SDK"
        assert _banned_match(name, MODEL_SDKS) is None, f"{requirement!r} is a model SDK"


def test_the_dependency_guard_knows_the_gemini_sdk_under_both_its_names() -> None:
    # google-genai on PyPI, google.genai on import. Miss either and the SDK the agents
    # will actually use is the one SDK the guard cannot see.
    assert _requirement_name("google-genai>=1.0") in MODEL_SDK_DISTRIBUTIONS
    assert _banned_match("google.genai", MODEL_SDKS) == "google.genai"
    assert _banned_match("google.genai.types", MODEL_SDKS) == "google.genai"
    # ...without banning every package that happens to live under `google`.
    assert _banned_match("google.protobuf", MODEL_SDKS) is None


def test_no_orchestration_framework_is_declared_as_a_dependency() -> None:
    for requirement in _declared_requirements():
        name = _requirement_name(requirement).replace("-", "_")
        assert _banned_match(name, ORCHESTRATION_FRAMEWORKS) is None, (
            f"{requirement!r} is an orchestration framework"
        )


# The dependency arrows the architecture actually rests on. A package may import itself,
# anything below it, and nothing above it. `agents/` is absent on purpose: it sits at the
# top and may depend on all of them, which is the whole point of rule 3.
ALLOWED_DEPENDENCIES = {
    "models": frozenset({"errors", "models"}),
    "graph": frozenset({"errors", "models", "graph"}),
    "policy": frozenset({"errors", "models", "policy"}),
    "store": frozenset({"errors", "models", "store"}),
    # `observability` sits above `store` — the run inspector reads one — and below
    # `runtime`, which emits through it. Signals only ever travel one way: nothing here
    # can change what a run does, which is why the executor may depend on it at all.
    "observability": frozenset({"errors", "models", "store", "observability"}),
    "runtime": frozenset(
        {"errors", "models", "graph", "policy", "store", "observability", "runtime"}
    ),
}

LAYERED_FILES = [
    path
    for path in ALL_FILES
    if path.relative_to(PACKAGE_ROOT).parts[0].removesuffix(".py") in ALLOWED_DEPENDENCIES
]


@pytest.mark.parametrize("path", LAYERED_FILES, ids=_ids(LAYERED_FILES))
def test_module_imports_nothing_from_a_layer_above_it(path: pathlib.Path) -> None:
    # Not style: `dagent/runtime/__init__.py` imports the executor, so a policy module
    # importing anything under `dagent.runtime` makes the whole package unimportable.
    # This says so at the boundary instead of at the circular-import traceback.
    layer = path.relative_to(PACKAGE_ROOT).parts[0].removesuffix(".py")
    allowed = ALLOWED_DEPENDENCIES[layer]

    offenders = sorted(
        module
        for module in _imported_modules(path)
        if module.startswith("dagent.") and module.split(".")[1].removesuffix(".py") not in allowed
    )

    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} is in the {layer!r} layer and imports {offenders}, "
        f"which sits above it. {layer!r} may import: {sorted(allowed)}."
    )


def test_the_layering_scan_actually_covers_files() -> None:
    assert LAYERED_FILES
    covered = {path.relative_to(PACKAGE_ROOT).parts[0] for path in LAYERED_FILES}
    assert covered == set(ALLOWED_DEPENDENCIES)


OPTIONAL_DEPENDENCIES = {
    "asyncpg": "dagent/store/postgres.py",
    # OpenTelemetry's *API* is a base dependency and does nothing on its own; the SDK and
    # the exporters are the `otel` extra. Only `setup.py` may reach for them, and only
    # when called — which is what lets every other module emit signals unconditionally.
    "opentelemetry.sdk": "dagent/observability/setup.py",
    "opentelemetry.exporter": "dagent/observability/setup.py",
    "prometheus_client": "dagent/observability/setup.py",
}
"""Packages that are extras, and the one module allowed to import each.

`asyncpg` ships under `dagent[postgres]`. If any other module imports it — the CLI
naming the store in a help string, say — then a plain `pip install dagent` produces a
package that cannot be imported at all, and SPEC's "v1 runs with zero external services"
becomes false in the most embarrassing way possible. The same reasoning covers the
OpenTelemetry SDK: an engine that always emits spans must not require the machinery that
collects them.

Matched on dotted prefixes, so `opentelemetry.api` stays allowed everywhere while
`opentelemetry.sdk` does not.
"""


@pytest.mark.parametrize("path", ALL_FILES, ids=_ids(ALL_FILES))
def test_an_optional_dependency_is_imported_only_where_it_is_allowed(
    path: pathlib.Path,
) -> None:
    relative = str(path.relative_to(REPO_ROOT))
    offenders = sorted(
        module
        for module in _imported_modules(path)
        if (owner := _optional_owner(module)) is not None and owner != relative
    )

    assert not offenders, (
        f"{relative} imports the optional dependency {offenders}. Only "
        f"{sorted({_optional_owner(m) for m in offenders})} may, or the base install "
        "stops importing."
    )


def _optional_owner(module: str) -> str | None:
    """The module allowed to import this one, if it is an extra."""
    parts = module.split(".")
    for depth in range(len(parts), 0, -1):
        owner = OPTIONAL_DEPENDENCIES.get(".".join(parts[:depth]))
        if owner is not None:
            return owner
    return None


def test_no_optional_dependency_is_declared_in_the_base_install() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = {_requirement_name(requirement) for requirement in data["project"]["dependencies"]}

    leaked = {name for name in base if _optional_owner(name.replace("-", ".")) is not None}
    assert not leaked, f"an extra leaked into the base install: {sorted(leaked)}"


def test_the_opentelemetry_api_is_a_base_dependency() -> None:
    # The half of the split that makes the other half safe: without the API in the base
    # install, every module that emits a span would need the extra after all.
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = {_requirement_name(requirement) for requirement in data["project"]["dependencies"]}

    assert "opentelemetry-api" in base
    assert "opentelemetry-sdk" not in base
