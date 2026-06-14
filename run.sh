#!/usr/bin/env bash
# Boot the MLX inference sidecar (Python) + the TypeScript REPL together.
#
#   ./run.sh                 # tools on, default model/port
#   ./run.sh --no-tools      # pass-through flags go to the REPL
#
# Env overrides: DIFG_PORT (8088), DIFG_MODEL (4-bit), DIFG_PYTHON (.venv/bin/python).
# If a sidecar is already serving on the port, it is reused (and left running).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${DIFG_PORT:-8088}"
MODEL="${DIFG_MODEL:-mlx-community/diffusiongemma-26B-A4B-it-4bit}"
PYTHON="${DIFG_PYTHON:-$ROOT/.venv/bin/python}"
BASE="http://127.0.0.1:$PORT"

SIDECAR_PID=""
cleanup() {
  # Only stop a sidecar WE started (reused ones are left alone).
  if [[ -n "$SIDECAR_PID" ]] && kill -0 "$SIDECAR_PID" 2>/dev/null; then
    echo "Stopping inference sidecar (pid $SIDECAR_PID)..."
    kill "$SIDECAR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if curl -sf "$BASE/info" >/dev/null 2>&1; then
  echo "Reusing inference sidecar already serving at $BASE"
else
  [[ -x "$PYTHON" ]] || { echo "Python not found at $PYTHON (set DIFG_PYTHON)." >&2; exit 1; }
  echo "Starting inference sidecar ($MODEL) on $BASE ..."
  "$PYTHON" "$ROOT/infer_server.py" "$MODEL" "$PORT" &
  SIDECAR_PID=$!
  # Model load + warmup can take a while; poll /info until ready.
  for _ in $(seq 1 180); do
    curl -sf "$BASE/info" >/dev/null 2>&1 && break
    kill -0 "$SIDECAR_PID" 2>/dev/null || { echo "Sidecar exited during startup." >&2; exit 1; }
    sleep 1
  done
  curl -sf "$BASE/info" >/dev/null 2>&1 || { echo "Sidecar not ready in time." >&2; exit 1; }
fi

echo "Launching TypeScript REPL ..."
cd "$ROOT/ts"
[[ -d node_modules ]] || npm install
# Foreground (no exec) so the EXIT trap fires and cleans up the sidecar afterwards.
node_modules/.bin/tsx src/chat_stream.ts --server "$BASE" "$@"
