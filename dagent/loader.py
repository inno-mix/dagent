"""Loading a workflow from YAML — the other half of FR-1.

A workflow is definable two ways: in Python with ``WorkflowBuilder``, and in a file. Both
end at the same frozen, validated ``Workflow``, because this loader builds through the
same builder rather than around it. There is no second set of rules for files.

Lives at the package root rather than in ``dagent/graph`` because reading a file is I/O,
and ``graph`` is proven pure by ``tests/graph/test_purity.py``. Parsing is separated from
reading for the same reason: :func:`load_workflow` is pure and takes text.

    name: research
    nodes:
      - id: seed_a
        agent: constant
        params:
          value: "the topic to research"
      - id: research_a
        agent: researcher
        inputs:
          topic: seed_a
        policy:
          max_attempts: 3
          timeout_s: 30
"""

from __future__ import annotations

import pathlib
from typing import Any

import yaml

from dagent.errors import ValidationError
from dagent.graph.builder import WorkflowBuilder
from dagent.models.workflow import Policy, Workflow

__all__ = ["load_workflow", "load_workflow_file"]

_NODE_KEYS = frozenset({"id", "agent", "depends_on", "inputs", "params", "policy"})


def load_workflow_file(path: str | pathlib.Path) -> Workflow:
    """Read and parse a workflow definition from a YAML file.

    Raises:
        ValidationError: If the file is missing, unreadable, or does not describe a
            valid workflow.
    """
    file = pathlib.Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"cannot read workflow file {str(file)!r}: {exc}") from exc
    return load_workflow(text, source=str(file))


def load_workflow(text: str, *, source: str = "<string>") -> Workflow:
    """Parse a workflow definition from YAML text.

    Pure: no file access, so this stays cheap to test and safe to call on untrusted input
    (``safe_load`` never constructs arbitrary Python objects).

    Args:
        text: The YAML document.
        source: Name used in error messages to say where the problem is.

    Returns:
        A frozen, fully validated workflow — the same thing ``WorkflowBuilder`` produces.

    Raises:
        ValidationError: On malformed YAML, a missing or misspelled key, or any graph
            rule the builder enforces.
    """
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValidationError(f"{source}: invalid YAML: {exc}") from exc

    if not isinstance(document, dict):
        raise ValidationError(f"{source}: expected a mapping at the top level")

    unknown = sorted(set(document) - {"name", "nodes"})
    if unknown:
        raise ValidationError(f"{source}: unknown top-level key(s): {', '.join(unknown)}")

    name = document.get("name")
    if not isinstance(name, str) or not name:
        raise ValidationError(f"{source}: 'name' is required and must be a non-empty string")

    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        raise ValidationError(f"{source}: 'nodes' is required and must be a list")

    builder = WorkflowBuilder(name)
    for index, entry in enumerate(nodes):
        _add_node(builder, entry, source=source, index=index)
    return builder.build()


def _add_node(builder: WorkflowBuilder, entry: Any, *, source: str, index: int) -> None:
    """Add one YAML node entry to the builder, or explain precisely what is wrong."""
    where = f"{source}: nodes[{index}]"
    if not isinstance(entry, dict):
        raise ValidationError(f"{where}: expected a mapping")

    unknown = sorted(set(entry) - _NODE_KEYS)
    if unknown:
        raise ValidationError(f"{where}: unknown key(s): {', '.join(unknown)}")

    node_id = entry.get("id")
    if not isinstance(node_id, str):
        raise ValidationError(f"{where}: 'id' is required and must be a string")

    agent = entry.get("agent")
    if not isinstance(agent, str):
        raise ValidationError(f"{where} ({node_id}): 'agent' is required and must be a string")

    depends_on = entry.get("depends_on", [])
    if not isinstance(depends_on, list) or not all(isinstance(dep, str) for dep in depends_on):
        raise ValidationError(f"{where} ({node_id}): 'depends_on' must be a list of node ids")

    inputs = entry.get("inputs", {})
    if not isinstance(inputs, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in inputs.items()
    ):
        raise ValidationError(
            f"{where} ({node_id}): 'inputs' must map a local name to an upstream node id"
        )

    params = entry.get("params", {})
    if not isinstance(params, dict) or not all(isinstance(key, str) for key in params):
        raise ValidationError(f"{where} ({node_id}): 'params' must be a mapping with string keys")

    builder.add_node(
        node_id,
        agent,
        depends_on=depends_on,
        inputs=inputs,
        params=params,
        policy=_policy(entry.get("policy"), where=f"{where} ({node_id})"),
    )


def _policy(raw: Any, *, where: str) -> Policy | None:
    """Build a node's policy override, converting pydantic's complaint into ours."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValidationError(f"{where}: 'policy' must be a mapping")
    try:
        return Policy(**raw)
    except Exception as exc:
        raise ValidationError(f"{where}: invalid policy: {exc}") from exc
