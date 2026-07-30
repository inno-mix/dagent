# AGENTS.md

The operating contract for this repository lives in **[`docs/AGENTS.md`](docs/AGENTS.md)**.
Read it before writing any code, together with:

- [`docs/SPEC.md`](docs/SPEC.md) — what Dagent does.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how it is built, and why.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — the ordered build plan and its acceptance criteria.

This file exists only so tooling that looks for `AGENTS.md` at the repository root finds
the contract. It is a pointer, not a second source of truth.

[`CLAUDE.md`](CLAUDE.md) is the same kind of thing for Claude Code, with one addition: it
carries the orientation and the traps that are not rules — which commands to run, which
invariants span several files, and what has previously cost time. It restates no rule from
`docs/AGENTS.md`, deliberately, because a duplicated rule is a rule that drifts.
