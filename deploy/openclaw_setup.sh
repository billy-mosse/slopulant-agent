#!/usr/bin/env bash
# OpenClaw side of the deployment (idempotent; safe to re-run):
#   - agent "watcher" (Qwen3-Coder-Next on Ollama) writes the note for each duplicate alert
#   - scheduled job "slopulant-watcher" runs one watcher tick every 30 s from this repo
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${WATCHER_PYTHON:-$REPO/.venv/bin/python}"
NODE_BIN="${NODE_BIN:-$(dirname "$(command -v openclaw 2>/dev/null || ls -d "$HOME"/.local/opt/node-*/bin/openclaw | head -1)")}"
export PATH="$NODE_BIN:$PATH"
CMD="cd $REPO && $PY -m watcher.main --once"

if ! openclaw agents list 2>/dev/null | grep -q '^- watcher'; then
  openclaw agents add watcher --non-interactive --workspace "$HOME/.openclaw/workspace-watcher" --model ollama/coder-next:latest
fi

JOB_ID="$(openclaw cron list --all --json | python3 -c 'import json,sys; print(next((j["id"] for j in json.load(sys.stdin)["jobs"] if j["name"] == "slopulant-watcher"), ""))')"
if [ -z "$JOB_ID" ]; then
  openclaw cron add --name slopulant-watcher --every 30s --no-deliver --timeout-seconds 3600 --command "$CMD"
else
  openclaw cron edit "$JOB_ID" --command "$CMD" --enable \
    --description "Poll PRs, extract topics (Qwen), generate candidates; watcher.clm_worker judges them (one tick; ticks never overlap)"
fi
openclaw cron list --all --json | python3 -c 'import json,sys; j=next(j for j in json.load(sys.stdin)["jobs"] if j["name"] == "slopulant-watcher"); print(j["name"], "->", j["payload"].get("argv"))'
