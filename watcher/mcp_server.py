"""Read-only MCP server (stdio) for the OpenClaw `watcher` agent: what it may look at when
someone @mentions slop-factory in Discord.

Everything here reads data/watcher.db (opened read-only) or the monorepo clone. Nothing
writes, nothing runs user-supplied commands; paths are validated and outputs capped.
Thread messages are untrusted, so this is the agent's entire tool surface in Discord.

Tools:
  alert_for_message(message_id)   the alert a Discord message/thread belongs to (a thread
                                  started from an alert has the alert's message id)
  recent_alerts(limit)            the latest alerts posted to Discord
  pr_analysis(pr_number)          topics, candidates and verdicts for a PR's latest commit
  system_topics(folder)           what an existing system on main does (its topics)
  pr_code(pr_number, contains)    the code a PR changed (files + contents), optionally filtered
  system_code(folder, contains)   the code of an existing system on main
  search_code(query, ref, folder) grep for a string/function name on main or a PR's head
  list_files(folder, ref)         files in a folder (ref: "main" or a PR number)
  read_file(path, ref)            one file (capped), e.g. "fake_review_detection/train.py"

Arguments are forgiving (pr_number for ref, folder+filename for path, ...) because the local
model guesses names; errors include the expected call.

    python -m watcher.mcp_server      (OpenClaw starts it; JSON-RPC over stdin/stdout)
"""
import json
import re
import sqlite3
import sys

from . import config, gitrepo

MAX_CHARS = 12_000
SAFE_PATH = re.compile(r"^[A-Za-z0-9_.\-/]+$")


def _db():
    conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn, sql, params=()):
    return [dict(r) for r in conn.execute(sql, params)]


def _clip(text, n=MAX_CHARS):
    return text if len(text) <= n else text[:n] + f"\n… [truncated, {len(text) - n} more characters]"


def _main_sha(conn):
    row = conn.execute("SELECT value FROM kv WHERE key LIKE 'indexed_base_sha:%' ORDER BY rowid DESC LIMIT 1").fetchone()
    return row[0] if row else "origin/main"


def _topics(conn, folder):
    return _rows(conn, """
        SELECT t.name, t.description, t.keywords, t.inputs AS reads, t.outputs AS writes FROM topics t
        JOIN folders f USING (tree_sha, llm_model, embed_model)
        WHERE f.folder = ? AND f.created_at = (SELECT max(created_at) FROM folders WHERE folder = ?)
        ORDER BY t.idx""", (folder, folder))


def _ref(conn, ref):
    """'main' -> indexed main commit; a PR number -> its head commit."""
    ref = str(ref or "main").strip().lstrip("#")
    if ref == "main":
        return _main_sha(conn)
    if ref.isdigit():
        row = conn.execute("SELECT head_sha FROM prs WHERE number = ?", (int(ref),)).fetchone()
        if row:
            return row[0]
    raise ValueError("ref must be 'main' or a PR number")


def alert_for_message(message_id):
    conn = _db()
    row = conn.execute(
        """SELECT d.payload_json, n.message_id, n.like_count, n.dislike_count, n.sent_at
           FROM discord_notifications n JOIN discord_decisions d USING (decision_id)
           WHERE n.message_id = ?""", (str(message_id).strip(),)).fetchone()
    if row is None:
        return {"found": False, "hint": "No alert with that message id. Use recent_alerts to list them."}
    p = json.loads(row["payload_json"])
    out = {"found": True, "message_id": row["message_id"], "posted_at": row["sent_at"],
           "reviews": {"thumbs_up": row["like_count"], "thumbs_down": row["dislike_count"]}}
    out.update({k: p.get(k) for k in ("pr_number", "pr_title", "pr_url", "author_login", "author_name", "head_sha",
                                       "base_sha", "folder_names", "reason", "model_version")})
    out["matches"] = [{k: m.get(k) for k in ("project_name", "topic", "evidence", "url")} for m in p.get("matches", [])]
    out["verdicts"] = pr_analysis(p["pr_number"])["verdicts"]
    return out


def recent_alerts(limit=10):
    conn = _db()
    rows = _rows(conn, """SELECT n.message_id, n.sent_at, d.pr_number, json_extract(d.payload_json, '$.pr_title') AS pr_title,
                                 json_extract(d.payload_json, '$.matches') AS matches
                          FROM discord_notifications n JOIN discord_decisions d USING (decision_id)
                          WHERE n.status = 'sent' ORDER BY n.sent_at DESC LIMIT ?""", (max(1, min(int(limit), 25)),))
    for r in rows:
        r["matched_systems"] = [m["project_name"] for m in json.loads(r.pop("matches") or "[]")]
    return rows


def pr_analysis(pr_number):
    conn = _db()
    pr = conn.execute("SELECT number, title, author, url, state, head_sha FROM prs WHERE number = ?", (int(pr_number),)).fetchone()
    if pr is None:
        return {"found": False}
    scores = _rows(conn, """SELECT pr_folder, repo_id, pr_topic, repo_topic, round(score, 3) AS similarity, kw_match AS shared_keywords,
                                   dataflow AS shared_tables, candidate FROM latest_scores
                            WHERE pr_number = ? AND (candidate = 1 OR rank <= 3) ORDER BY pr_folder, rank""", (int(pr_number),))
    verdicts = _rows(conn, """SELECT d.pr_folder, d.repo_id, d.relation, d.is_duplicate, round(d.confidence, 3) AS score,
                                     d.threshold, d.classifier, d.reason
                              FROM decisions d JOIN latest_scores s USING (pr_number, head_sha, base_sha, pr_folder, repo_id)
                              WHERE d.pr_number = ?""", (int(pr_number),))
    return {"found": True, "pr": dict(pr), "pr_topics": {f: _topics(conn, f) for f in sorted({s["pr_folder"] for s in scores})},
            "compared_with": scores, "verdicts": verdicts,
            "how_to_read": "candidate=1 pairs were judged. relation 'duplicate' = CLM said the two solve the same problem "
                           "(score >= threshold); upstream/downstream = they share a table."}


def system_topics(folder):
    conn = _db()
    topics = _topics(conn, folder)
    if not topics:
        return {"found": False, "hint": "Unknown folder. Folder names look like 'fake_review_detection'."}
    owner = None
    try:
        owner = gitrepo.owner(_main_sha(conn), folder)
    except Exception:
        pass
    return {"found": True, "folder": folder, "owner": owner, "topics": topics}


def _numbered(text):
    """'  74| code' per line, so answers can cite exact line numbers."""
    return "\n".join(f"{i:4d}| {line}" for i, line in enumerate(text.splitlines(), 1))


def _files_with_code(sha, paths, contains=None, per_file=8_000, total=24_000):
    """[{path, lines, content}] for these paths at a commit, capped so a reply fits the model's context."""
    out, used = [], 0
    for path in paths:
        if contains and contains.lower() not in path.lower():
            continue
        try:
            text = gitrepo.git("show", f"{sha}:{path}")
        except Exception:
            continue
        room = min(per_file, total - used)
        if room <= 500:
            out.append({"path": path, "lines": text.count("\n") + 1, "content": "(not included: reply size limit; use read_file)"})
            continue
        out.append({"path": path, "lines": text.count("\n") + 1, "content": _clip(_numbered(text), room)})
        used += min(len(text), room)
    return out


def pr_code(pr_number, contains=None):
    conn = _db()
    pr = conn.execute("SELECT number, title, head_sha FROM prs WHERE number = ?", (int(pr_number),)).fetchone()
    if pr is None:
        return {"found": False, "hint": "Unknown PR number. recent_alerts lists PRs with alerts."}
    base = _main_sha(conn)
    merge_base = gitrepo.git("merge-base", base, pr["head_sha"]).strip()
    changed = [p for p in gitrepo.git("diff", "--name-only", merge_base, pr["head_sha"]).splitlines() if p]
    files = _files_with_code(pr["head_sha"], changed, contains)
    return {"found": True, "pr": pr["number"], "title": pr["title"], "head_sha": pr["head_sha"],
            "changed_files": changed, "files": files,
            "note": "Code at the PR's head commit, as 'line| code'. Cite line numbers; in code blocks drop the 'N| ' prefix."}


def system_code(folder, contains=None):
    if not SAFE_PATH.match(folder) or ".." in folder:
        raise ValueError("bad folder name")
    sha = _main_sha(_db())
    paths = gitrepo.git("ls-tree", "-r", "--name-only", sha, "--", f"{folder.strip('/')}/").splitlines()
    if not paths:
        return {"found": False, "hint": "No such folder on main. Folder names look like 'fake_review_detection'."}
    return {"found": True, "folder": folder, "ref": "main", "files": _files_with_code(sha, paths, contains)}


def search_code(query, ref="main", folder=None):
    query = str(query)
    if not query.strip() or len(query) > 200 or "\n" in query:
        raise ValueError("query must be one line, 1-200 characters")
    sha = _ref(_db(), ref)
    args = ["grep", "-n", "-I", "-F", "-i", "--max-count=20", "-e", query, sha]
    if folder:
        if not SAFE_PATH.match(folder) or ".." in folder:
            raise ValueError("bad folder name")
        args += ["--", f"{folder.strip('/')}/"]
    try:
        hits = gitrepo.git(*args).splitlines()
    except Exception:
        hits = []  # git grep exits 1 when nothing matches
    hits = [h.split(":", 1)[1] if h.startswith(sha) else h for h in hits]  # drop the "<sha>:" prefix
    return {"ref": ref, "query": query, "matches": hits[:60], "truncated": len(hits) > 60}


def list_files(folder, ref="main"):
    if not SAFE_PATH.match(folder) or ".." in folder:
        raise ValueError("bad folder name")
    sha = _ref(_db(), ref)
    lines = gitrepo.git("ls-tree", "-r", "--name-only", sha, "--", f"{folder.strip('/')}/").splitlines()
    return {"ref": ref, "files": lines[:200]}


def read_file(path, ref="main"):
    if not SAFE_PATH.match(path) or ".." in path or path.startswith("/"):
        raise ValueError("bad path")
    sha = _ref(_db(), ref)
    return {"ref": ref, "path": path, "content": _clip(_numbered(gitrepo.git("show", f"{sha}:{path}")))}


ALIASES = {  # what the model tends to call the parameters -> what the function takes
    "pr": "pr_number", "pr_id": "pr_number", "number": "pr_number", "repo_id": "folder", "system": "folder",
    "repo": "folder", "folder_name": "folder", "id": "message_id", "thread_id": "message_id", "channel_id": "message_id",
    "file": "path", "file_path": "path", "filepath": "path", "q": "query", "pattern": "query", "text": "query",
    "filter": "contains", "path_contains": "contains", "branch": "ref",
}


def _normalize_args(name, args):
    """Maps guessed parameter names onto the real ones and drops anything unknown."""
    fn, _, props, _ = TOOLS[name]
    args = {ALIASES.get(k, k): v for k, v in (args or {}).items()}
    if "filename" in args and "path" in props:  # folder + filename -> path
        args["path"] = f"{str(args.get('folder', '')).strip('/')}/{args.pop('filename')}".lstrip("/")
    if "pr_number" in args and "ref" in props and "pr_number" not in props:  # read_file/list_files on a PR
        args["ref"] = str(args.pop("pr_number"))
    return {k: v for k, v in args.items() if k in props}


def _signature(name):
    _, _, props, req = TOOLS[name]
    return f"{name}(" + ", ".join(k if k in req else f"{k}=..." for k in props) + ")"


TOOLS = {
    "alert_for_message": (alert_for_message, "Look up the duplicate alert a Discord message belongs to. In a thread "
                          "started from an alert, pass the thread's id (it equals the alert's message id).",
                          {"message_id": {"type": "string", "description": "Discord message or thread id"}}, ["message_id"]),
    "recent_alerts": (recent_alerts, "List the most recent duplicate alerts posted to Discord.",
                      {"limit": {"type": "integer", "description": "1-25, default 10"}}, []),
    "pr_analysis": (pr_analysis, "Topics, compared systems, similarity scores and CLM verdicts for a PR's latest commit.",
                    {"pr_number": {"type": "integer"}}, ["pr_number"]),
    "system_topics": (system_topics, "What an existing system (top-level folder on main) does: its topics, tables, owner.",
                      {"folder": {"type": "string"}}, ["folder"]),
    "pr_code": (pr_code, "The code a PR changed: changed file list plus contents at the PR's head commit. Use for "
                "questions about the PR's code.",
                {"pr_number": {"type": "integer"},
                 "contains": {"type": "string", "description": "optional: only files whose path contains this"}},
                ["pr_number"]),
    "system_code": (system_code, "The code of an existing system (top-level folder) on main: file list plus contents.",
                    {"folder": {"type": "string", "description": "e.g. ab_test_analysis"},
                     "contains": {"type": "string", "description": "optional: only files whose path contains this"}},
                    ["folder"]),
    "search_code": (search_code, "Find where a string, function or table name appears (case-insensitive grep), on main "
                    "or on a PR's head commit. Returns path:line: text.",
                    {"query": {"type": "string"}, "ref": {"type": "string", "description": "'main' (default) or a PR number"},
                     "folder": {"type": "string", "description": "optional: limit to one top-level folder"}},
                    ["query"]),
    "list_files": (list_files, "List files in a folder on main or on a PR's head commit.",
                   {"folder": {"type": "string"}, "ref": {"type": "string", "description": "'main' (default) or a PR number"}},
                   ["folder"]),
    "read_file": (read_file, "Read one source file (first 12k characters) on main or on a PR's head commit.",
                  {"path": {"type": "string", "description": "e.g. fake_review_detection/train.py"},
                   "ref": {"type": "string", "description": "'main' (default) or a PR number"}}, ["path"]),
}


def handle(msg):
    method, params = msg.get("method"), msg.get("params") or {}
    if method == "initialize":
        return {"protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}}, "serverInfo": {"name": "slopulant", "version": "1.0"}}
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [{"name": name, "description": desc,
                           "inputSchema": {"type": "object", "properties": props, "required": req},
                           "annotations": {"readOnlyHint": True}}
                          for name, (_, desc, props, req) in TOOLS.items()]}
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in TOOLS:
            return {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}
        try:
            result = TOOLS[name][0](**_normalize_args(name, args))
            return {"content": [{"type": "text", "text": _clip(json.dumps(result, ensure_ascii=False, default=str), 30_000)}]}
        except Exception as e:  # report, never crash the server; say how to call it right
            return {"content": [{"type": "text", "text": f"error: {e}. Call it as {_signature(name)}."}], "isError": True}
    raise LookupError(method)


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        if "id" not in msg:  # notification (e.g. notifications/initialized)
            continue
        try:
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": handle(msg)}
        except LookupError:
            reply = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": f"method not found: {msg.get('method')}"}}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
