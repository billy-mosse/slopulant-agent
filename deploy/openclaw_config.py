"""Prints the OpenClaw config patch for the Discord side of the pipeline (JSON, for
`openclaw config patch --stdin`). Secrets and ids come from the local .env files, never
from this repo:

  discord_worker/.env : DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID
  .env                : DISCORD_GUILD_ID

What it sets up:
  - OpenClaw owns the Discord connection (the slop-factory bot), mention-only: it answers
    when someone @mentions it in #slop-factory or in a thread under it, nowhere else; no
    DMs, no other bots, no join introduction, reactions don't wake the agent (votes are
    read by `python -m dupcheck_discord openclaw-sync`).
  - Discord routes to the `watcher` agent (the same agent that drafts the alerts).
  - `watcher` gets only the read-only `slopulant` MCP tools (watcher/mcp_server.py):
    thread messages are untrusted, so no shell, files, web or config tools.

    python deploy/openclaw_config.py | openclaw config patch --stdin
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def env(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def main():
    discord_env, root_env = env(ROOT / "discord_worker" / ".env"), env(ROOT / ".env")
    token, channel, guild = discord_env.get("DISCORD_BOT_TOKEN"), discord_env.get("DISCORD_CHANNEL_ID"), root_env.get("DISCORD_GUILD_ID")
    if not (token and channel and guild):
        sys.exit("need DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID in discord_worker/.env and DISCORD_GUILD_ID in .env")
    patch = {
        "channels": {"discord": {
            "enabled": True,
            "token": token,
            "intents": {"messageContent": False},  # mention-only: no privileged intent needed
            "allowBots": False,
            "joinIntro": False,
            "dmPolicy": "disabled",
            "groupPolicy": "allowlist",
            "replyToMode": "first",
            "historyLimit": 20,
            "guilds": {guild: {
                "requireMention": True,
                "reactionNotifications": "off",
                "channels": {channel: {
                    "enabled": True,
                    "requireMention": True,
                    "requireMentionInBotThreads": True,
                    "includeThreadStarter": True,
                    "users": ["*"],
                    "systemPrompt": (
                        "This is #slop-factory, where the duplicate-effort alerts are posted. People @mention you "
                        "in an alert's thread to ask about it. Follow the 'Answering in Discord' section of your "
                        "AGENTS.md: look the alert up with the slopulant tools before answering."),
                }},
            }},
        }},
        "bindings": [{"type": "route", "agentId": "watcher", "match": {"channel": "discord"}}],
        "mcp": {"servers": {"slopulant": {
            "command": str(ROOT / ".venv" / "bin" / "python"),
            "args": ["-m", "watcher.mcp_server"],
            "cwd": str(ROOT),
            "enabled": True,
            "requestTimeoutMs": 30000,
        }}},
        "agents": {"entries": {"watcher": {"tools": {
            "profile": "minimal",
            "alsoAllow": ["slopulant__*"],
            "deny": ["gateway", "group:runtime", "group:fs", "group:web", "group:ui", "group:automation", "group:nodes"],
        }}}},
    }
    json.dump(patch, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
