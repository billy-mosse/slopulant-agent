"""SQLite state: PR queue, cached folder topics/embeddings, and scores."""
import json
import sqlite3
import time

import numpy as np

from . import config

# Bump when SCHEMA changes: everything here is derived state, so old tables are dropped.
SCHEMA_VERSION = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS pr_queue (
    pr_number   INTEGER NOT NULL,
    head_sha    TEXT    NOT NULL,
    -- base sha + topics/scoring version: a PR is re-scored when main moves or these change
    context     TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',  -- pending | running | done | error
    error       TEXT,
    enqueued_at REAL    NOT NULL,
    claimed_at  REAL,
    PRIMARY KEY (pr_number, head_sha, context)
);

-- Keyed by git tree hash (+ models): identical folder contents => identical topics, on any branch.
CREATE TABLE IF NOT EXISTS folders (
    tree_sha    TEXT NOT NULL,
    llm_model   TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    folder      TEXT NOT NULL,
    n_chunks    INTEGER NOT NULL,   -- >1 when the folder was too big for one LLM call
    func_names  TEXT NOT NULL,      -- JSON list, row i of func_embeddings (evidence only)
    func_embeddings BLOB NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (tree_sha, llm_model, embed_model)
);

-- 1..N topics per folder: one per distinct model / pipeline / tool.
CREATE TABLE IF NOT EXISTS topics (
    tree_sha    TEXT NOT NULL,
    llm_model   TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    name        TEXT NOT NULL,
    description TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    keywords    TEXT NOT NULL,   -- JSON list of cleaned technical keywords
    inputs      TEXT NOT NULL,   -- JSON list of tables read
    outputs     TEXT NOT NULL,   -- JSON list of tables written
    PRIMARY KEY (tree_sha, llm_model, embed_model, idx)
);

CREATE TABLE IF NOT EXISTS scores (
    ts          REAL    NOT NULL,
    pr_number   INTEGER NOT NULL,
    head_sha    TEXT    NOT NULL,
    pr_folder   TEXT    NOT NULL,   -- folder touched by the PR
    repo_id     TEXT    NOT NULL,   -- other folder on the base branch
    base_sha    TEXT    NOT NULL,
    score       REAL    NOT NULL,   -- similarity of the best topic pair: mean(desc, kw)
    desc_score  REAL    NOT NULL,   -- cosine of the two topics' description embeddings
    kw_score    REAL    NOT NULL,   -- TF-IDF cosine of the two topics' keywords
    kw_match    TEXT,               -- shared keywords, rarest first
    pr_topic    TEXT NOT NULL,      -- best-matching topic in the PR folder
    repo_topic  TEXT NOT NULL,      -- ...and in the other folder
    code_score  REAL    NOT NULL,   -- best cosine between any two functions (evidence only)
    code_match  TEXT,               -- which functions matched, e.g. "a.py:f ~ b.py:g"
    dataflow    TEXT,               -- "upstream:<tables>" | "downstream:<tables>" | NULL
    rank        INTEGER NOT NULL,   -- 1 = best match for this pr_folder
    candidate   INTEGER NOT NULL,   -- 1 if surfaced: dataflow link, or top-K similar above the floor
    PRIMARY KEY (pr_number, head_sha, pr_folder, repo_id)
);

CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);

-- PR metadata from the last poll, for display.
CREATE TABLE IF NOT EXISTS prs (
    number    INTEGER PRIMARY KEY,
    title     TEXT NOT NULL,
    author    TEXT NOT NULL,
    branch    TEXT NOT NULL,
    url       TEXT NOT NULL,
    head_sha  TEXT NOT NULL,
    state     TEXT NOT NULL,     -- open | merged | closed
    merged_at TEXT,
    merge_sha TEXT
);

-- What happened, for the dashboard's activity log and the history graph.
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    kind      TEXT NOT NULL,     -- pr_opened | pr_merged | pr_closed | scored | indexed | alert | reset | error ...
    pr_number INTEGER,
    message   TEXT NOT NULL,
    data      TEXT               -- JSON
);

-- Duplicate classifier verdicts (dummy LLM classifier now; CLM classifier later).
CREATE TABLE IF NOT EXISTS decisions (
    pr_number   INTEGER NOT NULL,
    head_sha    TEXT    NOT NULL,
    base_sha    TEXT    NOT NULL,
    pr_folder   TEXT    NOT NULL,
    repo_id     TEXT    NOT NULL,
    relation    TEXT    NOT NULL,  -- duplicate | partial | upstream | downstream | unrelated
    is_duplicate INTEGER NOT NULL,
    confidence  REAL    NOT NULL,
    reason      TEXT    NOT NULL,
    classifier  TEXT    NOT NULL,
    ts          REAL    NOT NULL,
    PRIMARY KEY (pr_number, head_sha, base_sha, pr_folder, repo_id)
);

-- The alert written by the OpenClaw agent, as posted to the PR.
CREATE TABLE IF NOT EXISTS alerts (
    pr_number   INTEGER NOT NULL,
    head_sha    TEXT    NOT NULL,
    base_sha    TEXT    NOT NULL,
    summary     TEXT    NOT NULL,   -- the agent's paragraph
    body        TEXT    NOT NULL,   -- full comment markdown
    author_by   TEXT    NOT NULL,   -- openclaw:<agent> | fallback
    comment_url TEXT,
    ts          REAL    NOT NULL,
    PRIMARY KEY (pr_number, head_sha, base_sha)
);

-- Scores from the most recent completed run of each open-or-closed PR.
CREATE VIEW IF NOT EXISTS latest_scores AS
SELECT s.* FROM scores s
JOIN (
    SELECT pr_number, head_sha, substr(context, 1, instr(context, '|') - 1) AS base_sha
    FROM pr_queue q
    WHERE status = 'done' AND enqueued_at = (
        SELECT max(enqueued_at) FROM pr_queue q2 WHERE q2.pr_number = q.pr_number AND q2.status = 'done'
    )
) latest USING (pr_number, head_sha, base_sha);
"""


def connect():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # the dashboard reads while the watcher writes
    if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        conn.execute("DROP VIEW IF EXISTS latest_scores")
        for table in ("pr_queue", "folder_cards", "folders", "topics", "scores", "kv", "prs", "events", "decisions", "alerts"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.executescript(SCHEMA)
    return conn


def enqueue(conn, pr_number, head_sha, base_sha):
    """Returns True if this (PR commit, base commit, card version) hasn't been seen."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO pr_queue (pr_number, head_sha, context, enqueued_at) VALUES (?, ?, ?, ?)",
        (pr_number, head_sha, f"{base_sha}|{llm_cache_key()}|s{config.SCORING_VERSION}", time.time()),
    )
    conn.commit()
    return cur.rowcount == 1


def pending(conn):
    return conn.execute(
        "SELECT pr_number, head_sha, context FROM pr_queue WHERE status = 'pending' ORDER BY enqueued_at"
    ).fetchall()


def claim(conn, item):
    """Marks a queue item as running; False if another run got it first."""
    cur = conn.execute(
        "UPDATE pr_queue SET status = 'running', claimed_at = ? "
        "WHERE pr_number = ? AND head_sha = ? AND context = ? AND status = 'pending'",
        (time.time(), item["pr_number"], item["head_sha"], item["context"]),
    )
    conn.commit()
    return cur.rowcount == 1


def requeue_stale(conn):
    """Items left 'running' by a crashed run go back to pending."""
    cur = conn.execute(
        "UPDATE pr_queue SET status = 'pending', claimed_at = NULL WHERE status = 'running' AND claimed_at < ?",
        (time.time() - config.STALE_RUN_SECONDS,),
    )
    conn.commit()
    return cur.rowcount


def mark(conn, item, status, error=None):
    conn.execute(
        "UPDATE pr_queue SET status = ?, error = ? WHERE pr_number = ? AND head_sha = ? AND context = ?",
        (status, error, item["pr_number"], item["head_sha"], item["context"]),
    )
    conn.commit()


def add_event(conn, kind, message, pr_number=None, **data):
    conn.execute("INSERT INTO events (ts, kind, pr_number, message, data) VALUES (?, ?, ?, ?, ?)",
                 (time.time(), kind, pr_number, message, json.dumps(data) if data else None))
    conn.commit()


def put_decisions(conn, rows):
    conn.executemany(
        "INSERT OR REPLACE INTO decisions VALUES (:pr_number, :head_sha, :base_sha, :pr_folder, :repo_id, "
        ":relation, :is_duplicate, :confidence, :reason, :classifier, :ts)", rows)
    conn.commit()


def put_alert(conn, row):
    conn.execute(
        "INSERT OR REPLACE INTO alerts VALUES (:pr_number, :head_sha, :base_sha, :summary, :body, :author_by, :comment_url, :ts)",
        row)
    conn.commit()


def llm_cache_key():
    return f"{config.llm()['model']}#topics-v{config.TOPICS_VERSION}"


def index_key():
    """kv key for "main has been indexed at sha X" with the active model."""
    return f"indexed_base_sha:{llm_cache_key()}"


def get_folder(conn, tree_sha):
    """{folder, n_chunks, topics: [...], func_names, func_embeddings} or None."""
    key = (tree_sha, llm_cache_key(), config.EMBED_MODEL)
    row = conn.execute("SELECT * FROM folders WHERE tree_sha = ? AND llm_model = ? AND embed_model = ?", key).fetchone()
    if row is None:
        return None
    entry = {"folder": row["folder"], "n_chunks": row["n_chunks"], "func_names": json.loads(row["func_names"])}
    funcs = np.frombuffer(row["func_embeddings"], dtype=np.float32)
    entry["func_embeddings"] = funcs.reshape(len(entry["func_names"]), -1) if entry["func_names"] else funcs
    entry["topics"] = [
        {
            "name": t["name"], "description": t["description"],
            "embedding": np.frombuffer(t["embedding"], dtype=np.float32),
            **{k: json.loads(t[k]) for k in ("keywords", "inputs", "outputs")},
        }
        for t in conn.execute(
            "SELECT * FROM topics WHERE tree_sha = ? AND llm_model = ? AND embed_model = ? ORDER BY idx", key
        )
    ]
    return entry


def put_folder(conn, tree_sha, entry):
    key = (tree_sha, llm_cache_key(), config.EMBED_MODEL)
    conn.execute("DELETE FROM topics WHERE tree_sha = ? AND llm_model = ? AND embed_model = ?", key)
    conn.executemany(
        "INSERT INTO topics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (*key, i, t["name"], t["description"], np.asarray(t["embedding"], dtype=np.float32).tobytes(),
             json.dumps(t["keywords"]), json.dumps(t["inputs"]), json.dumps(t["outputs"]))
            for i, t in enumerate(entry["topics"])
        ],
    )
    conn.execute(
        "INSERT OR REPLACE INTO folders VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (*key, entry["folder"], entry["n_chunks"], json.dumps(entry["func_names"]),
         np.asarray(entry["func_embeddings"], dtype=np.float32).tobytes(), time.time()),
    )
    conn.commit()


def put_scores(conn, rows):
    conn.executemany(
        "INSERT OR REPLACE INTO scores VALUES (:ts, :pr_number, :head_sha, :pr_folder, :repo_id, :base_sha, :score, :desc_score, :kw_score, :kw_match, :pr_topic, :repo_topic, :code_score, :code_match, :dataflow, :rank, :candidate)",
        rows,
    )
    conn.commit()


def upsert_prs(conn, prs):
    """Insert/update PR rows; returns {number: previous_state} for PRs whose state changed."""
    changed = {}
    for pr in prs:
        row = conn.execute("SELECT state FROM prs WHERE number = ?", (pr["number"],)).fetchone()
        if row is None or row["state"] != pr["state"]:
            changed[pr["number"]] = row["state"] if row else None
        conn.execute(
            "INSERT OR REPLACE INTO prs VALUES (:number, :title, :author, :branch, :url, :head_sha, :state, :merged_at, :merge_sha)",
            {"merged_at": None, "merge_sha": None, **pr},
        )
    conn.commit()
    return changed


def get_kv(conn, key):
    row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_kv(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, value))
    conn.commit()
