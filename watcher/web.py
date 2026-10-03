"""Local demo UI: runs the watcher loop in a background thread and serves a
dashboard of PRs, their candidates, and a "try it" box for scoring pasted code.

    python -m watcher.web            # http://localhost:8765
    python -m watcher.web --no-watcher   # UI only (another process runs the loop)
"""
import argparse
import json
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import cards, config, db, gitrepo
from . import main as watcher

STATIC = Path(__file__).parent / "static"
PORT = 8765


def public_card(card):
    if card is None:
        return None
    return {k: card[k] for k in ("folder", "description", "inputs", "outputs", "keywords", "func_names")}


def commit_author(sha):
    try:
        return gitrepo.git("log", "-1", "--format=%an", sha).strip()
    except Exception:
        return None


def state():
    conn = db.connect()
    base_sha = db.get_kv(conn, "indexed_base_sha")
    last_tick = db.get_kv(conn, "last_tick")

    index = []
    if base_sha:
        for folder, tree_sha in gitrepo.folders(base_sha).items():
            index.append(public_card(db.get_card(conn, tree_sha)) or {"folder": folder, "pending": True})

    prs = []
    for pr in conn.execute("SELECT * FROM prs ORDER BY open DESC, number DESC"):
        queue = conn.execute(
            "SELECT status, error FROM pr_queue WHERE pr_number = ? ORDER BY enqueued_at DESC LIMIT 1",
            (pr["number"],),
        ).fetchone()
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM latest_scores WHERE pr_number = ? ORDER BY pr_folder, rank", (pr["number"],)
        )]
        folders = {}
        for r in rows:
            folders.setdefault(r["pr_folder"], {"folder": r["pr_folder"], "rows": [], "card": None})
            folders[r["pr_folder"]]["rows"].append(r)
        scored_sha = rows[0]["head_sha"] if rows else None
        if scored_sha:
            try:
                trees = gitrepo.folders(scored_sha)
                for f in folders.values():
                    f["card"] = public_card(db.get_card(conn, trees.get(f["folder"], "")))
            except Exception:
                pass
        prs.append({
            **dict(pr),
            "commit_author": commit_author(pr["head_sha"]),
            "status": queue["status"] if queue else "new",
            "error": queue["error"] if queue else None,
            "scored_sha": scored_sha,
            "folders": list(folders.values()),
        })

    return {
        "repo": config.GITHUB_REPO,
        "base_sha": base_sha,
        "last_tick": float(last_tick) if last_tick else None,
        "now": time.time(),
        "poll_seconds": config.POLL_SECONDS,
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "top_k": config.TOP_K,
        "min_candidate_score": config.MIN_CANDIDATE_SCORE,
        "index": index,
        "prs": prs,
    }


def try_code(folder, code):
    """Scores pasted code as if it were a new folder in a PR against main."""
    conn = db.connect()
    base_sha = db.get_kv(conn, "indexed_base_sha")
    if not base_sha:
        raise RuntimeError("main hasn't been indexed yet; wait for the first watcher pass")
    folder = re.sub(r"[^a-zA-Z0-9_\-]", "_", folder.strip()) or "my_system"
    started = time.time()
    card = cards.build_card(folder, [(f"{folder}/main.py", code)])
    rows = cards.rank_against_base(conn, base_sha, folder, card)
    return {"card": public_card(card), "rows": rows, "seconds": round(time.time() - started, 1)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=float).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            try:
                self._send(200, state())
            except Exception as e:
                self._send(500, {"error": str(e), "trace": traceback.format_exc()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/try":
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self._send(200, try_code(body.get("folder", ""), body.get("code", "")))
        except Exception as e:
            self._send(500, {"error": str(e)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-watcher", action="store_true")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    watcher.setup_logging()
    gitrepo.ensure_clone()
    if not args.no_watcher:
        threading.Thread(target=watcher.run, daemon=True).start()
    print(f"demo on http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
