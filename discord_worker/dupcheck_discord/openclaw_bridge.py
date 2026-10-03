"""Discord delivery and feedback through OpenClaw (which owns the Discord connection).

One pass (`python -m dupcheck_discord openclaw-sync`, run every 30 s by the OpenClaw job
`slopulant-discord`):

  1. import finished verdicts from the watcher's tables (source.sync_source)
  2. post each new duplicate alert as a Discord card via `openclaw message send
     --presentation`, seed 👍/👎, and record the message id (exactly the old bot's
     delivery bookkeeping: store.ready / mark_sent / mark_failed)
  3. reconcile 👍/👎 on recent alerts via `openclaw message reactions` into
     discord_feedback (per reviewer; the bot's own seed reactions are ignored)

Threads and @mentions are handled by OpenClaw itself (routed to the `watcher` agent).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from .source import sync_source
from .store import Store

LOG = logging.getLogger(__name__)
EMOJIS = ("👍", "👎")


def _openclaw() -> str:
    found = os.environ.get("OPENCLAW_BIN") or shutil.which("openclaw")
    if not found:
        found = next(iter(sorted(Path.home().glob(".local/opt/node-*/bin/openclaw"))), None)
    if not found:
        raise RuntimeError("openclaw CLI not found; set OPENCLAW_BIN")
    return str(found)


def _cli(*args: str, timeout: int = 60) -> dict:
    """Runs `openclaw message ...` and returns its JSON (stdout may start with log lines)."""
    binary = _openclaw()
    env = {**os.environ, "PATH": os.path.dirname(binary) + os.pathsep + os.environ.get("PATH", "")}
    proc = subprocess.run([binary, "message", *args, "--channel", "discord", "--json"],
                          capture_output=True, text=True, timeout=timeout, env=env)
    out = proc.stdout
    start = out.find("{")
    if proc.returncode != 0 or start < 0:
        raise RuntimeError(f"openclaw message {args[0]} failed ({proc.returncode}): {(proc.stderr or out)[-500:]}")
    return json.loads(out[start:])


def presentation(decision: dict) -> dict:
    """The alert card: facts from data, plus Qwen's 'what overlaps' and 'next step'."""
    pr = decision["pr_number"]
    title = f"PR #{pr}: {decision['pr_title'].replace('[TEST] ', '')}"
    title = f"[{title}]({decision['pr_url']})" if decision.get("pr_url") else title
    blocks = [{"type": "text", "text": f"### ⚠️ Possible duplicate · {title}"}]
    if decision.get("overlap"):
        blocks.append({"type": "text", "text": f"**What overlaps:** {decision['overlap']}\n**Suggested next step:** {decision['next_step']}"})
    else:
        blocks.append({"type": "text", "text": decision["reason"]})
    blocks.append({"type": "divider"})
    for match in decision["matches"][:5]:
        name = f"[`{match['project_name']}`]({match['url']})" if match.get("url") else f"`{match['project_name']}`"
        facts = [line for line in (match.get("evidence") or "").splitlines() if not line.startswith("Matched PR topic")]
        topic = (match.get("topic") or "").split(" — ")[0]
        blocks.append({"type": "text", "text": f"**Matches** {name}" + (f" · {topic}" if topic else "")
                       + ("\n" + " · ".join(facts) if facts else "")})
    if len(decision["matches"]) > 5:
        blocks.append({"type": "context", "text": f"+{len(decision['matches']) - 5} more matches"})
    blocks.append({"type": "divider"})
    author = decision.get("author_name") or decision["author_login"]
    folders = ", ".join(f"`{f}`" for f in decision.get("folder_names") or [])
    commit = f" · commit `{decision['head_sha'][:7]}`" if decision.get("head_sha") else ""
    blocks.append({"type": "context", "text": f"By {author} · {folders}{commit} · react 👍 if this is a real duplicate, "
                                             f"👎 if not · @mention me in a thread to ask about it"})
    # Hidden marker so a delivery can be traced from the message (and recovered if needed).
    blocks.append({"type": "context", "text": f"`{decision['decision_id']}`"})
    return {"blocks": blocks}


def deliver(store: Store, channel_id: str) -> int:
    sent = 0
    for decision in store.ready(channel_id):
        store.mark_attempt(decision["decision_id"], channel_id)
        try:
            result = _cli("send", "--target", f"channel:{channel_id}", "--presentation", json.dumps(presentation(decision)))
            message_id = result.get("messageId") or (result.get("payload") or {}).get("result", {}).get("receipt", {}).get("primaryPlatformMessageId")
            if not message_id:
                raise RuntimeError("send returned no message id")
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            LOG.warning("Delivery failed for %s; retrying later: %s", decision["decision_id"], exc)
            store.mark_failed(decision["decision_id"], channel_id, str(exc)[:500], retry_after=30)
            continue
        store.mark_sent(decision["decision_id"], channel_id, str(message_id))
        LOG.info("Sent %s as message %s", decision["decision_id"], message_id)
        sent += 1
        for emoji in EMOJIS:  # seed the two review buttons
            try:
                _cli("react", "--target", f"channel:{channel_id}", "--message-id", str(message_id), "--emoji", emoji)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                LOG.warning("Seeding %s failed: %s", emoji, exc)
    return sent


def sync_reactions(store: Store, channel_id: str, reviewer_ids: frozenset[str] = frozenset(), limit: int = 10) -> int:
    """Full reaction snapshot per recent alert -> discord_feedback (humans only). Each openclaw call
    costs a few seconds of CLI startup, so only the most recent alerts are reconciled."""
    synced = 0
    bot_ids: set[str] = set()
    for row in store.notifications(channel_id, limit=limit):
        try:
            result = _cli("reactions", "--target", f"channel:{channel_id}", "--message-id", row["message_id"])
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            LOG.warning("Reading reactions of %s failed: %s", row["message_id"], exc)
            continue
        votes = []
        for reaction in (result.get("payload") or {}).get("reactions") or []:
            emoji = (reaction.get("emoji") or {}).get("name")
            if emoji not in EMOJIS:
                continue
            for user in reaction.get("users") or []:
                uid = str(user.get("id"))
                if user.get("bot") or user.get("username") == "slop-factory":
                    bot_ids.add(uid)
                    continue
                if not reviewer_ids or uid in reviewer_ids:
                    votes.append((uid, emoji))
        store.replace_reactions(row["message_id"], channel_id, votes)
        synced += 1
    return synced


def run_once(store: Store, repository: str, channel_id: str, reviewer_ids: frozenset[str] = frozenset()) -> dict:
    imported = sync_source(store, repository)
    sent = deliver(store, channel_id)
    reviewed = sync_reactions(store, channel_id, reviewer_ids)
    return {"imported": imported, "sent": sent, "reactions_synced": reviewed}
