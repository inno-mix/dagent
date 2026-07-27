"""The ``dagent`` command-line entrypoint.

Commands arrive with the phase that gives them something to do: ``run`` and ``submit``
with the real agents (Phase 3), ``resume`` with durable state (Phase 5), and ``inspect``
with the run inspector (Phase 7).
"""

from __future__ import annotations

import typer

from dagent import __version__

app = typer.Typer(
    name="dagent",
    help="A durable, async DAG execution engine for multi-agent AI workflows.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Group the subcommands.

    Without this, Typer collapses a single-command app into a bare entrypoint, and
    ``dagent version`` would stop working the moment a second command is added.
    """


@app.command()
def version() -> None:
    """Print the installed dagent version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
