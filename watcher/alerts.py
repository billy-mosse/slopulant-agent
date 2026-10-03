"""The alert: an OpenClaw agent turn writes a short note for the PR author, and the
watcher posts it as a single "[oc]" comment on the PR (edited in place on re-scores).

Links, owners and verdicts in the comment are built here from data, never by the
model, so they can't be hallucinated; the agent only writes the summary paragraph."""
import json
import os
import shutil
import subprocess
import tempfile

from . import config, github

BRIEF = """You are the overlap watcher for an ML monorepo. A pull request was just analysed.
Write a short, friendly note (at most 4 sentences, plain text, no markdown headers, no links)
to the PR author. Say which existing systems they are likely duplicating or depending on,
name the owners to talk to, and suggest one concrete next step. If nothing relevant was
found, say so in one sentence. Do not invent systems, owners or facts beyond the list below.

PR #{number}: {title} (author: {author})
Folders changed: {folders}

Findings:
{findings}"""


def _findings_text(findings):
    if not findings:
        return "- none"
    return "\n".join(
        f"- {f['pr_folder']} [{f['pr_topic']}] vs {f['repo_id']} [{f['repo_topic']}], owner {f['owner'] or 'unknown'}: "
        f"{f['relation']} (confidence {f['confidence']:.2f}; {f['reason']})"
        + (f"; shared tables {f['dataflow']}" if f["dataflow"] else "")
        for f in findings
    )


def _openclaw_bin():
    if os.path.isabs(config.OPENCLAW_BIN):
        return config.OPENCLAW_BIN if os.path.exists(config.OPENCLAW_BIN) else None
    return shutil.which(config.OPENCLAW_BIN)


def agent_summary(pr, folders, findings):
    """(text, author_by). Runs one OpenClaw agent turn; falls back to a template."""
    brief = BRIEF.format(number=pr["number"], title=pr["title"], author=pr.get("commit_author") or pr["author"],
                         folders=", ".join(folders) or "-", findings=_findings_text(findings))
    binary = _openclaw_bin()
    if binary:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(brief)
            path = f.name
        try:
            proc = subprocess.run(
                [binary, "agent", "--agent", config.OPENCLAW_AGENT, "--session-id", f"slopulant-pr-{pr['number']}",
                 "--message-file", path, "--json"],
                capture_output=True, text=True, timeout=config.OPENCLAW_TIMEOUT,
                # openclaw is a node script: make sure its node is on PATH (cron shells have a bare PATH)
                env={**os.environ, "PATH": os.path.dirname(binary) + os.pathsep + os.environ.get("PATH", "")},
            )
            data = json.loads(proc.stdout) if proc.stdout.strip() else {}
            meta = (data.get("result") or {}).get("meta") or {}
            text = meta.get("finalAssistantVisibleText") or next(
                (p.get("text") for p in (data.get("result") or {}).get("payloads") or [] if p.get("text")), None)
            if text and text.strip():
                return text.strip(), f"openclaw:{config.OPENCLAW_AGENT}"
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
            pass
        finally:
            os.unlink(path)
    return _fallback(findings), "fallback"


def _fallback(findings):
    dupes = [f for f in findings if f["relation"] in ("duplicate", "partial")]
    deps = [f for f in findings if f["relation"] in ("upstream", "downstream") or f["dataflow"]]
    if not dupes and not deps:
        return "No overlapping or connected systems found for this PR."
    parts = []
    if dupes:
        parts.append("This looks like it overlaps with " + ", ".join(
            f"{f['repo_id']} (owner {f['owner'] or 'unknown'})" for f in dupes) + "; worth a chat before going further.")
    if deps:
        parts.append("It connects to " + ", ".join(f"{f['repo_id']}" for f in deps) + " through shared tables.")
    return " ".join(parts)


RELATION_LABEL = {
    "duplicate": "🟥 duplicate", "partial": "🟧 partial duplicate", "upstream": "🟦 reads from it",
    "downstream": "🟦 feeds it", "unrelated": "⬜ unrelated",
}


def comment_body(pr, base_sha, summary, findings, author_by):
    repo = f"https://github.com/{config.GITHUB_REPO}"
    lines = [f"{github.COMMENT_PREFIX} **Overlap watcher** · base `{base_sha[:7]}` · head `{pr['head_sha'][:7]}`", "", summary, ""]
    if findings:
        lines += ["| your folder → topic | existing system → topic | owner | verdict | why |", "|---|---|---|---|---|"]
        for f in findings:
            why = f["reason"].replace("|", "/")
            if f["dataflow"]:
                why += f" · shared tables: `{f['dataflow'].split(':', 1)[1]}`"
            lines.append(
                f"| [`{f['pr_folder']}/`]({repo}/tree/{pr['head_sha']}/{f['pr_folder']}) → {f['pr_topic']} "
                f"| [`{f['repo_id']}/`]({repo}/tree/{base_sha}/{f['repo_id']}) → {f['repo_topic']} "
                f"| {f['owner'] or '?'} | {RELATION_LABEL.get(f['relation'], f['relation'])} ({f['confidence']:.2f}) | {why} |"
            )
    who = "OpenClaw agent" if author_by.startswith("openclaw") else "template (OpenClaw unavailable)"
    lines += ["", f"<sub>Candidates: top-{config.TOP_K} similar topics above {config.MIN_CANDIDATE_SCORE} plus shared-table links · "
              f"verdicts: dummy LLM classifier · note: {who}. This comment is updated on every push.</sub>"]
    return "\n".join(lines)
