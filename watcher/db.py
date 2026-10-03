"""SQLite state: PR queue, cached folder descriptions/embeddings, and scores."""
import json
import sqlite3
import time

import numpy as np

from . import config

# Bump when SCHEMA changes: everything here is derived state, so old tables are dropped.
SCHEMA_VERSION = 6

SCHEMA = """
CREATE TABLE IF NOT EXISTS pr_queue (
    pr_number   INTEGER NOT NULL,
    head_sha    TEXT    NOT NULL,
    -- base sha + card version: a PR is re-scored when main moves or the cards change
    context     TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',  -- pending | done | error
    error       TEXT,
    enqueued_at REAL    NOT NULL,
    PRIMARY KEY (pr_number, head_sha, context)
);

-- Keyed by git tree hash (+ models): identical folder contents => identical card, on any branch.
CREATE TABLE IF NOT EXISTS folder_cards (
    tree_sha    TEXT NOT NULL,
    folder      TEXT NOT NULL,
    description TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    inputs      TEXT NOT NULL,   -- JSON list of tables read
    outputs     TEXT NOT NULL,   -- JSON list of tables written
    keywords    TEXT NOT NULL,   -- JSON list of cleaned technical keywords
    func_names  TEXT NOT NULL,   -- JSON list, row i of func_embeddings
    func_embeddings BLOB NOT NULL,
    llm_model   TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (tree_sha, llm_model, embed_model)
);

CREATE TABLE IF NOT EXISTS scores (
    ts          REAL    NOT NULL,
    pr_number   INTEGER NOT NULL,
    head_sha    TEXT    NOT NULL,
    pr_folder   TEXT    NOT NULL,   -- folder touched by the PR
    repo_id     TEXT    NOT NULL,   -- other folder on the base branch
    base_sha    TEXT    NOT NULL,
    score       REAL    NOT NULL,   -- similarity: mean(card, kw)
    card_score  REAL    NOT NULL,   -- cosine of capability-card embeddings
    kw_score    REAL    NOT NULL,   -- TF-IDF cosine of technical keywords
    kw_match    TEXT,               -- shared keywords, rarest first
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
    number   INTEGER PRIMARY KEY,
    title    TEXT NOT NULL,
    author   TEXT NOT NULL,
    branch   TEXT NOT NULL,
    url      TEXT NOT NULL,
    head_sha TEXT NOT NULL,
    open     INTEGER NOT NULL
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
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        conn.execute("DROP VIEW IF EXISTS latest_scores")
        for table in ("pr_queue", "folder_cards", "scores", "kv", "prs"):
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


def mark(conn, item, status, error=None):
    conn.execute(
        "UPDATE pr_queue SET status = ?, error = ? WHERE pr_number = ? AND head_sha = ? AND context = ?",
        (status, error, item["pr_number"], item["head_sha"], item["context"]),
    )
    conn.commit()


def llm_cache_key():
    return f"{config.LLM_MODEL}#cards-v{config.CARDS_VERSION}"


def get_card(conn, tree_sha):
    row = conn.execute(
        "SELECT * FROM folder_cards WHERE tree_sha = ? AND llm_model = ? AND embed_model = ?",
        (tree_sha, llm_cache_key(), config.EMBED_MODEL),
    ).fetchone()
    if row is None:
        return None
    card = dict(row)
    card["embedding"] = np.frombuffer(row["embedding"], dtype=np.float32)
    for key in ("inputs", "outputs", "keywords", "func_names"):
        card[key] = json.loads(row[key])
    funcs = np.frombuffer(row["func_embeddings"], dtype=np.float32)
    card["func_embeddings"] = funcs.reshape(len(card["func_names"]), -1) if card["func_names"] else funcs
    return card


def put_card(conn, tree_sha, card):
    conn.execute(
        "INSERT OR REPLACE INTO folder_cards VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            tree_sha, card["folder"], card["description"],
            np.asarray(card["embedding"], dtype=np.float32).tobytes(),
            json.dumps(card["inputs"]), json.dumps(card["outputs"]), json.dumps(card["keywords"]),
            json.dumps(card["func_names"]),
            np.asarray(card["func_embeddings"], dtype=np.float32).tobytes(),
            llm_cache_key(), config.EMBED_MODEL, time.time(),
        ),
    )
    conn.commit()


def put_scores(conn, rows):
    conn.executemany(
        "INSERT OR REPLACE INTO scores VALUES (:ts, :pr_number, :head_sha, :pr_folder, :repo_id, :base_sha, :score, :card_score, :kw_score, :kw_match, :code_score, :code_match, :dataflow, :rank, :candidate)",
        rows,
    )
    conn.commit()


def sync_prs(conn, prs):
    """Upserts the currently open PRs; everything else is marked closed."""
    conn.execute("UPDATE prs SET open = 0")
    conn.executemany(
        "INSERT OR REPLACE INTO prs VALUES (:number, :title, :author, :branch, :url, :head_sha, 1)", prs
    )
    conn.commit()


def get_kv(conn, key):
    row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_kv(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, value))
    conn.commit()
