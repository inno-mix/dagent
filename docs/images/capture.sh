#!/usr/bin/env bash
# Regenerate the demo screenshots in this directory.
#
# Every image is a real command from DEMO.md, run and captured — nothing here is
# mocked up. Only the offline demos are covered: no Docker, no provider key, no
# spend. Demos 6 and 8 need Postgres and Redis and are not captured.
#
#   ./docs/images/capture.sh all          # every shot
#   ./docs/images/capture.sh 09-benchmark # just one
#
# Requires: freeze (brew install charmbracelet/tap/freeze) and sips (macOS).

set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT="$REPO/docs/images"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cd "$REPO" || exit 1

# The dagent CLI renders through Rich, which only emits colour and box-drawing
# when it believes it is talking to a terminal. Capturing through a pipe would
# throw all of that away, so each command is run on a pty of a fixed size —
# fixed so that every screenshot wraps at the same column.
cat > "$WORK/ptyrun.py" <<'PY'
import fcntl, os, pty, struct, sys, termios

COLS, ROWS = 100, 50

pid, fd = pty.fork()
if pid == 0:
    os.execvp(sys.argv[1], sys.argv[1:])
    os._exit(127)

fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))

chunks = []
while True:
    try:
        data = os.read(fd, 65536)
    except OSError:
        break
    if not data:
        break
    chunks.append(data)
os.close(fd)
os.waitpid(pid, 0)

out = b"".join(chunks).replace(b"\r\n", b"\n").replace(b"\r", b"")
sys.stdout.buffer.write(out)
PY

FREEZE_OPTS=(
  --language ansi --window --wrap 100 --theme charm
  --font.family "JetBrains Mono" --font.size 14 --line-height 1.2
  --padding "20,30" --margin 18
  --border.radius 8 --border.width 1 --border.color "#3a3a3a"
  --shadow.blur 22 --shadow.x 0 --shadow.y 10
)

shot() {
  local name=$1; shift
  local cmd="$*"
  local ansi="$WORK/$name.ansi"

  {
    printf '\033[38;5;114m$\033[0m %s\n\n' "$cmd"
    # Drop every control char except ESC (033), tab (011) and newline (012):
    # freeze rejects the rest as illegal XML when it builds the SVG.
    python3 "$WORK/ptyrun.py" /bin/bash -c "$cmd" 2>&1 \
      | LC_ALL=C tr -d '\000-\010\013\014\015\016-\032\034-\037'
  } > "$ansi"

  freeze "$ansi" "${FREEZE_OPTS[@]}" -o "$OUT/$name.png" >/dev/null 2>&1 || {
    echo "freeze failed for $name" >&2; return 1
  }
  # freeze renders at a high DPI; 1800px keeps the text crisp at a quarter of the bytes.
  sips -Z 1800 "$OUT/$name.png" >/dev/null 2>&1
  printf '  %s\n' "$name.png"
}

# Demos 2 and 3 need workflow files that DEMO.md asks the reader to write out.
# They go to the same /tmp paths the doc names, so the captured command line
# matches what a reader following DEMO.md would actually type.
fixtures() {
  cat > /tmp/cycle.yaml <<'YAML'
name: broken
nodes:
  - id: a
    agent: fake
    inputs: {x: c}
  - id: b
    agent: fake
    inputs: {x: a}
  - id: c
    agent: fake
    inputs: {x: b}
YAML
  cat > /tmp/failing.yaml <<'YAML'
name: failure-modes
nodes:
  - id: seed
    agent: constant
    params: {value: hello}
  - id: ok
    agent: echo
    inputs: {x: seed}
  - id: bad
    agent: constant          # no `value` param -> fails, and not retryable
    inputs: {x: seed}
  - id: downstream
    agent: echo
    inputs: {x: bad}
YAML
}

all() {
  fixtures
  shot 00-cli-surface               "uv run dagent --help"
  shot 01-diamond                   "uv run python examples/diamond.py"
  shot 01-fan-out                   "uv run python examples/fan_out.py"
  shot 02-validate-ok               "uv run dagent validate examples/research_dynamic.yaml"
  shot 02-cycle-rejected            "uv run dagent validate /tmp/cycle.yaml"
  shot 03-failure-skip-downstream   "uv run dagent run /tmp/failing.yaml --run-id fm-skip --provider none --on-failure skip_downstream --no-show-outputs"
  shot 04-research-stub             "uv run dagent run examples/research.yaml --run-id r1-dry --provider stub --no-show-outputs"
  shot 04-research-stub-outputs     "uv run dagent run examples/research.yaml --run-id r1-dry --provider stub"
  shot 05-budget-exceeded           "uv run dagent run examples/research.yaml --run-id budget-1 --provider stub --max-tokens 20 --no-show-outputs"
  shot 05-concurrency-caps          "uv run dagent run examples/research.yaml --run-id conc-1 --provider stub --max-concurrency 4 --provider-concurrency 2 --no-show-outputs"
  shot 09-benchmark                 "uv run python benchmarks/load.py"
}

if [ $# -eq 0 ]; then
  echo "usage: $0 all | <shot-name>" >&2
  exit 2
fi
"$@"
