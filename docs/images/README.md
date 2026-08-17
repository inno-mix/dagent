# Demo screenshots

Terminal captures of the offline demos in [`DEMO.md`](../../DEMO.md), for slides, the README,
and anywhere a run needs showing rather than describing.

Every image is a real command, run and captured — the output is the process's own, through a
pty so that Rich emits its actual colour. Nothing is mocked up or hand-edited.

Regenerate with [`capture.sh`](capture.sh) (needs `freeze`, and `sips` from macOS):

```bash
brew install charmbracelet/tap/freeze
./docs/images/capture.sh all            # or a single shot name
```

| Image | Demo | Command captured |
|---|---|---|
| `00-cli-surface.png` | — | `dagent --help` |
| `01-diamond.png` | 1 | `python examples/diamond.py` |
| `01-fan-out.png` | 1 | `python examples/fan_out.py` |
| `02-validate-ok.png` | 2, 7 | `dagent validate examples/research_dynamic.yaml` |
| `02-cycle-rejected.png` | 2 | `dagent validate /tmp/cycle.yaml` — the cycle is named |
| `03-failure-skip-downstream.png` | 3 | `--on-failure skip_downstream` |
| `04-research-stub.png` | 4 | `--provider stub`, summary only |
| `04-research-stub-outputs.png` | 4 | the same run with every node output |
| `05-budget-exceeded.png` | 5 | `--max-tokens 20` → run ends `budget_exceeded` |
| `05-concurrency-caps.png` | 5 | `--max-concurrency 4 --provider-concurrency 2` |
| `09-benchmark.png` | 9 | `python benchmarks/load.py` |

## What is not here

Demos 6 (crash and resume) and 8 (distributed workers) need Postgres and Redis, and demo 8
needs three terminals at once. They are not captured — a screenshot of a single terminal
would not show what those demos are actually claiming.

Demos 4 and 7 are captured against `--provider stub` rather than a live model, so these
images cost nothing to regenerate and are stable across runs. `02-validate-ok.png` doubles as
demo 7's validate step: it is the same command, showing the two-node graph before the planner
grows it.

## The two stray log lines

`03-failure-skip-downstream.png` and `05-budget-exceeded.png` show `node.attempt_failed` and
`node.failed` above the CLI's own report. That is the known wrinkle documented at the foot of
`DEMO.md` — `WARNING` escapes because the library installs no log handler and Python's
`logging.lastResort` writes it to stderr. It is left in rather than cropped out: the
screenshots show what the command actually prints.
