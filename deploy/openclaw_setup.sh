#!/usr/bin/env bash
# OpenClaw side of the deployment (idempotent; safe to re-run):
#   - agent "watcher" (Qwen3-Coder-Next on Ollama): drafts the alerts and answers @slop-factory
#     in Discord; its instructions are deploy/openclaw/watcher/*.md
#   - Discord channel (plugin @openclaw/discord), routing and the read-only MCP tools:
#     deploy/openclaw_config.py
#   - scheduled jobs:
#       slopulant-watcher   every 30 s     one watcher tick (PRs -> topics -> candidates)
#       slopulant-discord   every 30 s     post new alerts, sync 👍/👎 (dupcheck_discord openclaw-sync)
#       slopulant-reindex   06:00 New York  regenerate topics for every folder on main
#                                           (registered disabled; enable with --enable-reindex)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${WATCHER_PYTHON:-$REPO/.venv/bin/python}"
DPY="$REPO/discord_worker/.venv/bin/python"
NODE_BIN="${NODE_BIN:-$(dirname "$(command -v openclaw 2>/dev/null || ls -d "$HOME"/.local/opt/node-*/bin/openclaw | head -1)")}"
export PATH="$NODE_BIN:$PATH"
ENABLE_REINDEX=0; [ "${1:-}" = "--enable-reindex" ] && ENABLE_REINDEX=1

# --- agent
if ! openclaw agents list 2>/dev/null | grep -q '^- watcher'; then
  openclaw agents add watcher --non-interactive --workspace "$HOME/.openclaw/workspace-watcher" --model ollama/coder-next:latest
fi
WS="$HOME/.openclaw/workspace-watcher"
rm -f "$WS/BOOTSTRAP.md"   # the onboarding ritual; this agent has a fixed identity
cp "$REPO"/deploy/openclaw/watcher/*.md "$WS/"

# --- Discord channel, routing, MCP tools, tool limits
openclaw plugins list 2>/dev/null | grep -qi discord || openclaw plugins install @openclaw/discord
"$PY" "$REPO/deploy/openclaw_config.py" | openclaw config patch --stdin

# --- scheduled jobs
job_id() { openclaw cron list --all --json | python3 -c "import json,sys; print(next((j['id'] for j in json.load(sys.stdin)['jobs'] if j['name'] == '$1'), ''))"; }
upsert() {  # name, description, command, schedule args...
  local name="$1" desc="$2" cmd="$3"; shift 3
  local id; id="$(job_id "$name")"
  if [ -z "$id" ]; then
    openclaw cron add --name "$name" --description "$desc" --no-deliver --timeout-seconds 3600 --command "$cmd" "$@" >/dev/null
  else
    openclaw cron edit "$id" --description "$desc" --command "$cmd" "$@" >/dev/null
  fi
}
upsert slopulant-watcher "Poll PRs, extract topics (Qwen), generate candidates; the CLM worker judges them (one tick; ticks never overlap)" \
  "cd $REPO && $PY -m watcher.main --once" --every 30s
upsert slopulant-discord "Post new duplicate alerts to #slop-factory and sync 👍/👎 reviews" \
  "cd $REPO/discord_worker && $DPY -m dupcheck_discord openclaw-sync" --every 30s
upsert slopulant-reindex "Daily: regenerate topics for every folder on main with Qwen (cache bypassed)" \
  "cd $REPO && $PY -m watcher.reindex" --cron "0 6 * * *" --tz America/New_York
openclaw cron enable "$(job_id slopulant-watcher)" >/dev/null
openclaw cron enable "$(job_id slopulant-discord)" >/dev/null
if [ "$ENABLE_REINDEX" = 1 ]; then openclaw cron enable "$(job_id slopulant-reindex)" >/dev/null
else openclaw cron disable "$(job_id slopulant-reindex)" >/dev/null; fi

openclaw cron list --all --json | python3 -c "$(cat <<'PY'
import json, sys
for j in json.load(sys.stdin)["jobs"]:
    if j["name"].startswith("slopulant-"):
        s = j["schedule"]
        when = f"every {s.get('everyMs', 0) // 1000}s" if s.get("kind") == "every" else f"{s.get('expr')} {s.get('tz', '')}"
        print(f"{j['name']:20s} {'enabled ' if j['enabled'] else 'disabled'} {when}")
PY
)"
