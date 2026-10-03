"""SQLite state: PR queue, cached folder topics/embeddings, and scores."""
import json
import sqlite3
import time

import numpy as np

from . import config

try:  # vector search inside SQLite (https://github.com/asg017/sqlite-vec); optional
    import sqlite_vec
except ImportError:
    sqlite_vec = None

# Topic description embeddings, indexed for nearest-neighbour search. rowid = topics.rowid;
# llm_model/embed_model are metadata columns so a query only sees the active models' topics.
VEC_TABLE = """CREATE VIRTUAL TABLE IF NOT EXISTS topic_vec USING vec0(
    llm_model text, embed_model text, embedding float[{dim}] distance_metric=cosine)"""

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

-- One row per analysed PR commit, for the team view. Kept across demo resets.
CREATE TABLE IF NOT EXISTS history (
    branch       TEXT    NOT NULL,
    head_sha     TEXT    NOT NULL,
    base_sha     TEXT    NOT NULL,
    pr_number    INTEGER,
    title        TEXT,
    author       TEXT,
    author_team  TEXT,
    folders      TEXT    NOT NULL,   -- JSON {folder: [topic names]}
    findings     TEXT    NOT NULL,   -- JSON [{repo_id, repo_topic, owner, owner_team, relation, confidence, reason}]
    n_candidates INTEGER NOT NULL,
    comment_url  TEXT,
    detected_at  REAL,
    alerted_at   REAL    NOT NULL,
    source       TEXT    NOT NULL,   -- live | backfill
    PRIMARY KEY (branch, head_sha, base_sha)
);

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
    classifier  TEXT    NOT NULL,  -- CLM: clm_dupe.decision.VERSION; "dataflow" for shared-table links
    ts          REAL    NOT NULL,
    threshold   REAL,              -- CLM: is_duplicate = confidence >= threshold (NULL otherwise)
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
    # Additive migrations (no SCHEMA_VERSION bump, so cached topics survive).
    if "threshold" not in {r["name"] for r in conn.execute("PRAGMA table_info(decisions)")}:
        conn.execute("ALTER TABLE decisions ADD COLUMN threshold REAL")
    _load_vec(conn)
    return conn


def _load_vec(conn):
    """Loads sqlite-vec and makes sure topic_vec mirrors topics (backfills on first use)."""
    if sqlite_vec is None:
        return
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.OperationalError):
        return
    if vector_search(conn):
        n_vec = conn.execute("SELECT count(*) FROM topic_vec").fetchone()[0]
        if n_vec == conn.execute("SELECT count(*) FROM topics").fetchone()[0]:
            return
        conn.execute("DELETE FROM topic_vec")
    rows = conn.execute("SELECT rowid, llm_model, embed_model, embedding FROM topics").fetchall()
    if rows:
        _ensure_vec_table(conn, len(rows[0]["embedding"]) // 4)
        conn.executemany("INSERT INTO topic_vec (rowid, llm_model, embed_model, embedding) VALUES (?, ?, ?, ?)",
                         [tuple(r) for r in rows])
        conn.commit()


def _ensure_vec_table(conn, dim):
    conn.execute(VEC_TABLE.format(dim=dim))


def _vec_loaded(conn):
    try:
        conn.execute("SELECT vec_version()")
        return True
    except sqlite3.OperationalError:
        return False


def vector_search(conn):
    """True if this connection has sqlite-vec loaded and the topic_vec index exists."""
    return _vec_loaded(conn) and conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'topic_vec'").fetchone() is not None


def topic_similarities(conn, embedding):
    """{topics.rowid: cosine similarity} of one embedding to every topic of the active
    models: a k-nearest-neighbour query on topic_vec with k = all of them."""
    key = (llm_cache_key(), config.EMBED_MODEL)
    k = conn.execute("SELECT count(*) FROM topic_vec WHERE llm_model = ? AND embed_model = ?", key).fetchone()[0]
    if not k:
        return {}
    rows = conn.execute(
        "SELECT rowid, distance FROM topic_vec WHERE embedding MATCH ? AND k = ? AND llm_model = ? AND embed_model = ?",
        (np.asarray(embedding, dtype=np.float32).tobytes(), k, *key))
    return {r["rowid"]: 1.0 - r["distance"] for r in rows}


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
        "INSERT OR REPLACE INTO decisions (pr_number, head_sha, base_sha, pr_folder, repo_id, relation, "
        "is_duplicate, confidence, reason, classifier, ts, threshold) VALUES (:pr_number, :head_sha, :base_sha, "
        ":pr_folder, :repo_id, :relation, :is_duplicate, :confidence, :reason, :classifier, :ts, :threshold)",
        [{"threshold": None, **r} for r in rows])
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
            "rowid": t["rowid"], "name": t["name"], "description": t["description"],
            "embedding": np.frombuffer(t["embedding"], dtype=np.float32),
            **{k: json.loads(t[k]) for k in ("keywords", "inputs", "outputs")},
        }
        for t in conn.execute(
            "SELECT rowid, * FROM topics WHERE tree_sha = ? AND llm_model = ? AND embed_model = ? ORDER BY idx", key
        )
    ]
    return entry


def put_folder(conn, tree_sha, entry):
    key = (tree_sha, llm_cache_key(), config.EMBED_MODEL)
    vec = _vec_loaded(conn)
    if vector_search(conn):
        conn.execute("DELETE FROM topic_vec WHERE rowid IN (SELECT rowid FROM topics WHERE tree_sha = ? AND llm_model = ? "
                     "AND embed_model = ?)", key)
    conn.execute("DELETE FROM topics WHERE tree_sha = ? AND llm_model = ? AND embed_model = ?", key)
    for i, t in enumerate(entry["topics"]):
        blob = np.asarray(t["embedding"], dtype=np.float32).tobytes()
        cur = conn.execute(
            "INSERT INTO topics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*key, i, t["name"], t["description"], blob,
             json.dumps(t["keywords"]), json.dumps(t["inputs"]), json.dumps(t["outputs"])))
        if vec:  # keep the vector index in step with topics
            _ensure_vec_table(conn, len(blob) // 4)
            conn.execute("INSERT INTO topic_vec (rowid, llm_model, embed_model, embedding) VALUES (?, ?, ?, ?)",
                         (cur.lastrowid, key[1], key[2], blob))
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


def put_history(conn, row):
    conn.execute(
        "INSERT OR REPLACE INTO history VALUES (:branch, :head_sha, :base_sha, :pr_number, :title, :author, :author_team, "
        ":folders, :findings, :n_candidates, :comment_url, :detected_at, :alerted_at, :source)", row)
    conn.commit()


def get_kv(conn, key):
    row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_kv(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, value))
    conn.commit()
