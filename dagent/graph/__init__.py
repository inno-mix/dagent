"""Pure functions over a workflow: validation, topological order, ready set.

No async and no I/O anywhere in this package — that is what keeps the graph layer
property-testable and the whole engine replayable. Arrives in Phase 1.
"""
