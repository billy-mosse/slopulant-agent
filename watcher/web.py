"""Demo dashboard: live PRs, demo controls (open / merge / close real PRs, reset),
git history graph, activity log and a database browser.

    python -m watcher.web               # dashboard only; the OpenClaw job runs the watcher
    python -m watcher.web --watcher     # also run the watcher loop in-process (laptop dev)
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import config, db, demo, gitrepo
from . import main as watcher

STATIC = Path(__file__).parent / "static"
PORT = 8765
DB_TABLES = ("prs", "pr_queue", "folders", "topics", "scores", "decisions", "alerts", "events", "kv")
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
    return [{"name": t, "count": conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]} for t in DB_TABLES]


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
            routes = {
                "/api/state": state,
                "/api/catalog": lambda: catalog(force=q.get("force") == "1"),
                "/api/history": history,
                "/api/tables": tables,
                "/api/events": events,
                "/api/presets": presets,
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
