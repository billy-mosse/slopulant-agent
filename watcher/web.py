"""Demo dashboard: live PRs, demo controls (open / merge / close real PRs, reset),
git history graph, activity log and a database browser.

    python -m watcher.web               # dashboard only; the OpenClaw job runs the watcher
    python -m watcher.web --watcher     # also run the watcher loop in-process (laptop dev)
"""
import argparse
import json
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from clm_dupe import decision as clm_decision

from . import config, db, demo, gitrepo
from . import main as watcher

STATIC = Path(__file__).parent / "static"
PORT = 8765
DB_TABLES = ("history", "prs", "pr_queue", "folders", "topics", "scores", "decisions", "alerts", "events", "kv",
             "discord_decisions", "discord_notifications", "discord_feedback")  # discord_*: the Discord worker's
BLOB_COLUMNS = {"embedding", "func_embeddings"}
_catalog_cache = {"at": 0, "data": None}


def public_folder(entry):
    if entry is None:
        return None
    return {
        "folder": entry["folder"], "n_chunks": entry["n_chunks"], "func_names": entry["func_names"],
        "topics": [{k: t[k] for k in ("name", "description", "keywords", "inputs", "outputs")} for t in entry["topics"]],
    }


def _rows(cur):
    return [dict(r) for r in cur.fetchall()]


def state():
    conn = db.connect()
    kv = {r["key"]: r["value"] for r in conn.execute("SELECT * FROM kv")}
    base_sha = kv.get(db.index_key())
    prs = []
    for pr in conn.execute("SELECT * FROM prs ORDER BY CASE state WHEN 'open' THEN 0 ELSE 1 END, number DESC LIMIT 40"):
        pr = dict(pr)
        queue = conn.execute("SELECT status, error FROM pr_queue WHERE pr_number = ? ORDER BY enqueued_at DESC LIMIT 1",
                             (pr["number"],)).fetchone()
        pr["status"] = queue["status"] if queue else ("new" if pr["state"] == "open" else pr["state"])
        pr["error"] = queue["error"] if queue else None
        scores = _rows(conn.execute("SELECT * FROM latest_scores WHERE pr_number = ? ORDER BY pr_folder, score DESC",
                                    (pr["number"],)))
        decisions = {(d["pr_folder"], d["repo_id"]): d for d in _rows(conn.execute(
            "SELECT * FROM decisions WHERE pr_number = ? ORDER BY ts DESC", (pr["number"],)))}
        folders = {}
        for r in scores:
            f = folders.setdefault(r["pr_folder"], {"folder": r["pr_folder"], "rows": [], "entry": None})
            r["decision"] = decisions.get((r["pr_folder"], r["repo_id"])) if r["candidate"] else None
            f["rows"].append(r)
        if scores:
            try:
                trees = gitrepo.folders(scores[0]["head_sha"])
                for f in folders.values():
                    f["entry"] = public_folder(db.get_folder(conn, trees.get(f["folder"], "")))
            except Exception:
                pass
        alert = conn.execute("SELECT * FROM alerts WHERE pr_number = ? ORDER BY ts DESC LIMIT 1", (pr["number"],)).fetchone()
        pr["alert"] = dict(alert) if alert else None
        pr["discord"] = discord_alert(conn, pr["number"], scores[0]["head_sha"] if scores else pr["head_sha"])
        pr["folders"] = list(folders.values())
        pr["commit_author"] = gitrepo.commit_author(pr["head_sha"])
        prs.append(pr)
    index = []
    if base_sha:
        try:
            for folder, tree_sha in gitrepo.folders(base_sha).items():
                index.append(public_folder(db.get_folder(conn, tree_sha)) or {"folder": folder, "pending": True})
        except Exception:
            pass
    return {
        "repo": config.GITHUB_REPO, "base_sha": base_sha, "now": time.time(),
        "last_tick": float(kv["last_tick"]) if kv.get("last_tick") else None,
        "tick_running_since": float(kv["tick_started"]) if kv.get("tick_started") else None,
        "last_tick_seconds": kv.get("last_tick_seconds"),
        "poll_seconds": config.POLL_SECONDS, "llm": {k: v for k, v in config.llm().items() if k != "api_key"},
        "profiles": {name: {"label": p["label"], "available": bool(p["api_key"]) or name == "local"} for name, p in config.PROFILES.items()},
        "embed_model": config.EMBED_MODEL, "indexed": bool(base_sha),
        "top_k": config.TOP_K, "min_candidate_score": config.MIN_CANDIDATE_SCORE,
        "baseline": demo.baseline(conn), "prs": prs, "index": index,
    }


def catalog(force=False):
    if force or not _catalog_cache["data"] or time.time() - _catalog_cache["at"] > 20:
        _catalog_cache.update(at=time.time(), data=demo.catalog())
    return _catalog_cache["data"]


def history():
    """main's recent first-parent history + open PR branches, annotated with PR info."""
    conn = db.connect()
    base_sha = gitrepo.fetch_base()
    prs = _rows(conn.execute("SELECT * FROM prs"))
    by_merge = {p["merge_sha"]: p for p in prs if p.get("merge_sha")}
    latest_scored = {}
    for e in _rows(conn.execute("SELECT * FROM events WHERE kind = 'scored' ORDER BY id")):
        latest_scored[e["pr_number"]] = json.loads(e["data"] or "{}")

    def annotate(pr):
        info = latest_scored.get(pr["number"], {})
        return {"number": pr["number"], "title": pr["title"], "branch": pr["branch"], "state": pr["state"],
                "url": pr["url"], "topics": info.get("topics", {}), "duplicates": info.get("duplicates", []),
                "connected": info.get("connected", []), "scored": bool(info)}

    commits = gitrepo.history(base_sha, n=18)
    for c in commits:
        pr = by_merge.get(c["sha"])
        c["pr"] = annotate(pr) if pr else None
    open_branches = []
    for pr in prs:
        if pr["state"] != "open":
            continue
        try:
            gitrepo.fetch_pr(pr["number"])
            fork = gitrepo.merge_base(base_sha, pr["head_sha"])
            open_branches.append({**annotate(pr), "fork": fork,
                                  "commits": gitrepo.branch_commits(base_sha, pr["head_sha"])[:6]})
        except Exception:
            continue
    return {"base_sha": base_sha, "commits": commits, "open": open_branches,
            "baseline": (demo.baseline(conn) or {}).get("main_sha")}


def table(name, limit=50, offset=0, q=None):
    if name not in DB_TABLES:
        raise ValueError(f"unknown table {name}")
    conn = db.connect()
    cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({name})")]
    where, params = "", []
    if q:
        text_cols = [c for c in cols if c not in BLOB_COLUMNS]
        where = " WHERE " + " OR ".join(f"CAST({c} AS TEXT) LIKE ?" for c in text_cols)
        params = [f"%{q}%"] * len(text_cols)
    order = " ORDER BY rowid DESC"
    total = conn.execute(f"SELECT count(*) FROM {name}{where}", params).fetchone()[0]
    rows = []
    for r in conn.execute(f"SELECT * FROM {name}{where}{order} LIMIT ? OFFSET ?", [*params, limit, offset]):
        row = {}
        for c in cols:
            v = r[c]
            row[c] = f"<{len(v)} bytes>" if isinstance(v, (bytes, memoryview)) else v
        rows.append(row)
    return {"table": name, "columns": cols, "rows": rows, "total": total, "offset": offset, "limit": limit}


def tables():
    conn = db.connect()
    present = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    return [{"name": t, "count": conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]} for t in DB_TABLES if t in present]


def discord_alert(conn, pr_number, head_sha):
    """The Discord alert for a PR commit, from the Discord worker's tables in the same database:
    {status: pending | sent | failed | clean | waiting, url, likes, dislikes, sent_at}, or None."""
    try:
        row = conn.execute(
            """SELECT d.status AS decision_status, d.is_duplicate, n.status, n.channel_id, n.message_id,
                      n.like_count, n.dislike_count, n.sent_at, n.last_error
               FROM discord_decisions d LEFT JOIN discord_notifications n ON n.decision_id = d.decision_id
               WHERE d.repository = ? AND d.pr_number = ? AND d.head_sha = ?
               ORDER BY n.status = 'sent' DESC, d.rowid DESC LIMIT 1""",
            (config.GITHUB_REPO, pr_number, head_sha)).fetchone()
    except sqlite3.OperationalError:  # the Discord worker hasn't created its tables yet
        return None
    if row is None:
        return None
    if row["decision_status"] != "completed":
        status = "waiting"
    elif not row["is_duplicate"]:
        status = "clean"
    else:
        status = row["status"] or "pending"
    url = (f"https://discord.com/channels/{config.DISCORD_GUILD_ID}/{row['channel_id']}/{row['message_id']}"
           if config.DISCORD_GUILD_ID and row["message_id"] else None)
    return {"status": status, "url": url, "message_id": row["message_id"], "likes": row["like_count"] or 0,
            "dislikes": row["dislike_count"] or 0, "sent_at": row["sent_at"], "error": row["last_error"]}


def events(limit=80):
    return _rows(db.connect().execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)))


def try_code(folder, code):
    """Scores pasted code against main with the active model, without any PR."""
    from . import classifier, topics
    conn = db.connect()
    base_sha = db.get_kv(conn, db.index_key())
    if not base_sha:
        raise RuntimeError("main hasn't been indexed with this model yet; wait for the watcher's next tick")
    started = time.time()
    folder = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in folder.strip()) or "my_system"
    entry = topics.build_folder(folder, [(f"{folder}/main.py", code)])
    rows = topics.rank_against_base(conn, base_sha, folder, entry)
    base_folders = gitrepo.folders(base_sha)
    for r in [r for r in rows if r["candidate"]]:
        other = db.get_folder(conn, base_folders[r["repo_id"]])
        r["decision"] = classifier.classify(
            classifier.as_input(entry, classifier.topic_by_name(entry, r["pr_topic"])),
            classifier.as_input(other, classifier.topic_by_name(other, r["repo_topic"])), r)
    return {"entry": public_folder(entry), "rows": rows, "seconds": round(time.time() - started, 1),
            "llm": config.llm()["label"]}


PRESETS = ("review_toxicity", "customer_profiles", "inventory_forecast", "related_items", "fraud_scoring", "image_thumbnails")


def presets():
    """Example snippets for the Try-it tab (from eval/snippets; expected answers included)."""
    folder = Path(__file__).resolve().parent.parent / "eval" / "snippets"
    expected = json.loads((folder / "expected.json").read_text()) if (folder / "expected.json").exists() else {}
    return [{"name": n, "code": (folder / f"{n}.py").read_text(), "expected": expected.get(n, {}).get("duplicate", [])}
            for n in PRESETS if (folder / f"{n}.py").exists()]


def overview():
    """Team view: KPIs, recent analyses, most re-built systems, team overlap matrix."""
    from collections import Counter, defaultdict
    from statistics import median
    from . import teams as teams_mod
    conn = db.connect()
    base_sha = db.get_kv(conn, db.index_key())
    info = teams_mod.load(base_sha) if base_sha else {"teams": {}, "folder_team": {}, "person_team": {}}
    name = lambda key: teams_mod.team_name(info, key)
    rows = _rows(conn.execute("SELECT * FROM history ORDER BY alerted_at DESC"))
    latest = {}
    for r in rows:  # latest analysis per branch
        latest.setdefault(r["branch"], r)
    analyses = list(latest.values())
    for a in analyses:
        a["folders"] = json.loads(a["folders"]); a["findings"] = json.loads(a["findings"])
        a["author_team_name"] = name(a["author_team"])
        a["discord"] = discord_alert(conn, a["pr_number"], a["head_sha"]) if a["pr_number"] else None
        for f in a["findings"]:
            f["owner_team_name"] = name(f["owner_team"])
    overl = lambda a: [f for f in a["findings"] if f["relation"] in ("duplicate", "partial")]
    links = lambda a: [f for f in a["findings"] if f["relation"] in ("upstream", "downstream")]
    flagged = [a for a in analyses if overl(a)]
    cross = [(a, f) for a in analyses for f in overl(a) if f["owner_team"] and f["owner_team"] != a["author_team"]]
    alert_secs = [a["alerted_at"] - a["detected_at"] for a in analyses if a["detected_at"] and a["source"] == "live"]
    rebuilt = Counter(f["repo_id"] for a in analyses for f in overl(a))
    owners = {f["repo_id"]: (f["owner"], f["owner_team_name"]) for a in analyses for f in a["findings"]}
    matrix = Counter((a["author_team_name"], f["owner_team_name"]) for a in analyses for f in overl(a))
    per_team = defaultdict(lambda: {"prs": 0, "flagged": 0, "rebuilt_from_them": 0, "links": 0})
    for a in analyses:
        t = per_team[a["author_team_name"]]; t["prs"] += 1; t["flagged"] += bool(overl(a)); t["links"] += len(links(a))
        for f in overl(a):
            if f["owner_team"] != a["author_team"]:  # only re-builds by someone outside the owning team
                per_team[f["owner_team_name"]]["rebuilt_from_them"] += 1
    folders = {}
    if base_sha:
        for folder, tree in gitrepo.folders(base_sha).items():
            entry = db.get_folder(conn, tree)
            folders[folder] = {"team": name(info["folder_team"].get(folder)), "topics": [t["name"] for t in entry["topics"]] if entry else [],
                               "rebuilt": rebuilt.get(folder, 0)}
    return {
        "repo": config.GITHUB_REPO, "base_sha": base_sha, "llm": config.llm()["label"],
        "kpis": {
            "analysed": len(analyses), "flagged": len(flagged), "cross_team": len(cross),
            "links": sum(len(links(a)) for a in analyses),
            "median_alert_seconds": round(median(alert_secs)) if alert_secs else None,
            "systems": len(folders), "topics": sum(len(f["topics"]) for f in folders.values()),
            "live": sum(a["source"] == "live" for a in analyses), "backfill": sum(a["source"] == "backfill" for a in analyses),
        },
        "teams": [{"key": k, "name": t.get("name", k), "lead": t.get("lead"), "members": t.get("members") or [],
                   "folders": t.get("folders") or [], **per_team[t.get("name", k)]} for k, t in info["teams"].items()],
        "analyses": analyses[:60],
        "rebuilt": [{"folder": f, "count": c, "owner": owners.get(f, (None, None))[0], "team": owners.get(f, (None, None))[1]}
                    for f, c in rebuilt.most_common(8)],
        "matrix": [{"from": a, "to": b, "count": c} for (a, b), c in matrix.most_common()],
        "folders": folders,
    }


CLASSIFIER_LABEL = (f"CLM-v0.1-8B · p ≥ {clm_decision.THRESHOLD}" if not config.CLASSIFIER
                    else "dummy LLM classifier (CLASSIFIER=1)")
PIPELINE_KINDS = ("stage", "pr_opened", "pr_merged", "pr_closed", "reset", "action", "error")


def pipeline():
    """Everything the live pipeline view needs: recent pipeline events, tick status,
    open PRs, latest verdicts/alerts for the PR in focus."""
    conn = db.connect()
    kv = {r["key"]: r["value"] for r in conn.execute("SELECT * FROM kv")}
    marks = ",".join("?" * len(PIPELINE_KINDS))
    events = _rows(conn.execute(f"SELECT * FROM events WHERE kind IN ({marks}) ORDER BY id DESC LIMIT 200", PIPELINE_KINDS))
    for e in events:
        e["data"] = json.loads(e["data"]) if e["data"] else {}
    open_prs = _rows(conn.execute("SELECT number, title, branch, url, head_sha, state FROM prs WHERE state = 'open' ORDER BY number DESC"))
    focus = next((e["pr_number"] for e in events if e["kind"] == "stage" and e["pr_number"]), None)
    findings = []
    if focus:
        findings = _rows(conn.execute(
            "SELECT d.repo_id, d.relation, d.confidence, d.reason, d.pr_folder, d.is_duplicate FROM decisions d "
            "JOIN latest_scores s USING (pr_number, head_sha, base_sha, pr_folder, repo_id) "
            "WHERE d.pr_number = ? ORDER BY d.is_duplicate DESC, d.confidence DESC",
            (focus,)))
        for f in findings:
            f["owner"] = None
    alert = conn.execute("SELECT summary, comment_url, author_by FROM alerts WHERE pr_number = ? ORDER BY ts DESC LIMIT 1",
                         (focus,)).fetchone() if focus else None
    focus_pr = conn.execute("SELECT number, title, branch, url, state, head_sha FROM prs WHERE number = ?", (focus,)).fetchone() if focus else None
    discord = discord_alert(conn, focus, focus_pr["head_sha"]) if focus_pr else None
    return {
        "now": time.time(), "repo": config.GITHUB_REPO, "base_sha": kv.get(db.index_key()),
        "tick_running_since": float(kv["tick_started"]) if kv.get("tick_started") else None,
        "last_tick": float(kv["last_tick"]) if kv.get("last_tick") else None, "poll_seconds": config.POLL_SECONDS,
        "llm": config.llm()["label"], "classifier": CLASSIFIER_LABEL, "agent": config.OPENCLAW_AGENT,
        "top_k": config.TOP_K, "floor": config.MIN_CANDIDATE_SCORE,
        "events": events, "open_prs": open_prs, "focus": dict(focus_pr) if focus_pr else None,
        "findings": findings, "alert": dict(alert) if alert else None, "discord": discord, "baseline": demo.baseline(conn),
    }


def run_tick_now():
    """Kicks a watcher tick in the background (the lock keeps it from overlapping)."""
    subprocess.Popen([sys.executable, "-m", "watcher.main", "--once"], cwd=str(Path(__file__).resolve().parent.parent),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


ACTIONS = {
    "open": lambda b: {"pr": demo.open_branch(b["branch"])[0]},
    "merge": lambda b: {"result": demo.merge(int(b["number"]))},
    "close": lambda b: {"pr": demo.close(int(b["number"]))},
    "baseline": lambda b: {"baseline": demo.save_baseline(db.connect())},
    "reset": lambda b: {"steps": demo.reset(db.connect())},
    "tick": lambda b: {"started": True},
    "model": lambda b: (config.set_llm_profile(b["profile"]), {"llm": config.llm()["label"]})[1],
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if url.path in ("/demo", "/demo.html"):
                return self._send(200, (STATIC / "demo.html").read_bytes(), "text/html; charset=utf-8")
            routes = {
                "/api/state": state,
                "/api/catalog": lambda: catalog(force=q.get("force") == "1"),
                "/api/history": history,
                "/api/tables": tables,
                "/api/events": events,
                "/api/presets": presets,
                "/api/pipeline": pipeline,
                "/api/overview": overview,
                "/api/table": lambda: table(q.get("name", "topics"), int(q.get("limit", 50)), int(q.get("offset", 0)), q.get("q")),
            }
            if url.path not in routes:
                return self._send(404, {"error": "not found"})
            self._send(200, routes[url.path]())
        except Exception as e:
            self._send(500, {"error": str(e), "trace": traceback.format_exc()[-1500:]})

    def do_POST(self):
        name = urlparse(self.path).path.rsplit("/", 1)[-1]
        if self.path == "/api/try":
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                return self._send(200, try_code(body.get("folder", ""), body.get("code", "")))
            except Exception as e:
                return self._send(500, {"error": str(e)})
        if not self.path.startswith("/api/action/") or name not in ACTIONS:
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            result = ACTIONS[name](body)
            db.add_event(db.connect(), "action", f"dashboard: {name} {json.dumps(body) if body else ''}".strip())
            _catalog_cache["data"] = None
            run_tick_now()  # pick the change up right away instead of waiting for the next tick
            self._send(200, result)
        except Exception as e:
            self._send(500, {"error": str(e)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watcher", action="store_true", help="also run the watcher loop in-process")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    watcher.setup_logging()
    gitrepo.ensure_clone()
    db.connect()
    if args.watcher:
        threading.Thread(target=watcher.run, daemon=True).start()
    print(f"dashboard on http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
