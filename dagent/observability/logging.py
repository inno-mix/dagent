"""Structured logs, correlated by ``run_id`` and ``node_id`` (FR-9).

Two decisions shape this module, and both are about being a library rather than an
application.

**Silent unless asked.** ``structlog`` unconfigured prints every level straight to stdout,
which for a library means an import that starts spraying into somebody's terminal — and
into the CLI's own report. So the first call to :func:`get_logger` installs a
stdlib-routing configuration *if nobody else has configured one*, and the standard library
emits nothing without a handler. An application that configures ``structlog`` itself, before
or after, wins outright: an explicit call always replaces this default.

Only WARNING and above escape that, through ``logging``'s last-resort handler, which is
ordinary Python behaviour and the level at which silence would be the wrong default
anyway.

**Correlation through context, not arguments.** ``run_id`` and ``node_id`` are bound into
``structlog``'s context variables, which ``asyncio`` copies into every task the executor
spawns. A log line written deep inside an agent therefore carries the node that caused it
without the agent having been handed anything, and without every function in between
growing a parameter it does not otherwise want.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog

__all__ = ["bind_node", "bind_run", "configure", "get_logger"]

LOGGER_NAME = "dagent"


def get_logger(name: str = LOGGER_NAME) -> Any:
    """Return a bound logger, quiet by default.

    Typed loosely on purpose: ``structlog``'s bound loggers accept arbitrary keyword
    arguments by design, and pinning a protocol here would either forbid that or restate
    ``structlog``'s own types badly.
    """
    if not structlog.is_configured():
        _silence()
    return structlog.get_logger(name)


def _silence() -> None:
    """Route to the standard library, and install no handler.

    Deliberately not ``basicConfig``: that would add a handler, and adding a handler is
    the decision this function exists to avoid making on an application's behalf.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.render_to_log_kwargs,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def configure(*, level: int | str = logging.INFO, json: bool = False) -> None:
    """Turn dagent's logging on.

    Called by the CLI and by any application that wants output. Safe to call more than
    once, and safe never to call.

    Args:
        level: Minimum level to emit.
        json: Render each line as JSON rather than as a console-friendly key/value line.
            JSON when something else will parse the output; the readable renderer when a
            person will.
    """
    logging.basicConfig(format="%(message)s", level=level, force=True)

    renderer: Any = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level) if isinstance(level, str) else level
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


@contextmanager
def bind_run(run_id: str, workflow: str) -> Iterator[None]:
    """Tag every log line inside this block with the run that produced it."""
    tokens = structlog.contextvars.bind_contextvars(run_id=run_id, workflow=workflow)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)


@contextmanager
def bind_node(node_id: str, agent: str, attempt: int) -> Iterator[None]:
    """Tag every log line inside this block with the node attempt that produced it.

    Nested inside :func:`bind_run` by the executor, so a line carries both. Reset on the
    way out rather than left bound, or a node's identity would leak into whatever the
    scheduler did next.
    """
    tokens = structlog.contextvars.bind_contextvars(node_id=node_id, agent=agent, attempt=attempt)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
