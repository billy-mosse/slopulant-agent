-- This inbox is separate from the flagger's own tables. Decisions are immutable.
CREATE TABLE IF NOT EXISTS discord_decisions (
    decision_id TEXT PRIMARY KEY,
    repository TEXT NOT NULL,
    pr_number INTEGER NOT NULL CHECK (pr_number > 0),
    head_sha TEXT NOT NULL,
    revision_key TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'failed')),
    is_duplicate INTEGER NOT NULL CHECK (is_duplicate IN (0, 1)),
    pr_state TEXT NOT NULL CHECK (pr_state IN ('open', 'closed', 'merged')),
    model_version TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS discord_decisions_pr
    ON discord_decisions (repository, pr_number);

CREATE TABLE IF NOT EXISTS discord_notifications (
    decision_id TEXT NOT NULL REFERENCES discord_decisions(decision_id),
    channel_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    message_id TEXT,
    like_count INTEGER NOT NULL DEFAULT 0 CHECK (like_count >= 0),
    dislike_count INTEGER NOT NULL DEFAULT 0 CHECK (dislike_count >= 0),
    last_error TEXT,
    next_retry_at REAL,
    last_attempt_at TEXT,
    sent_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (decision_id, channel_id),
    UNIQUE (channel_id, message_id)
);

CREATE TABLE IF NOT EXISTS discord_feedback (
    decision_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    thumbs_up INTEGER NOT NULL DEFAULT 0 CHECK (thumbs_up IN (0, 1)),
    thumbs_down INTEGER NOT NULL DEFAULT 0 CHECK (thumbs_down IN (0, 1)),
    vote INTEGER CHECK (vote IS NULL OR vote IN (-1, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (decision_id, channel_id, user_id),
    FOREIGN KEY (decision_id, channel_id)
        REFERENCES discord_notifications(decision_id, channel_id)
);

-- Count reaction flags, including both reactions from an undecided reviewer.
-- These updates share the feedback write's transaction and roll back with it.
CREATE TRIGGER IF NOT EXISTS discord_feedback_counts_insert
AFTER INSERT ON discord_feedback
BEGIN
    UPDATE discord_notifications
    SET like_count = (
        SELECT COALESCE(SUM(f.thumbs_up), 0) FROM discord_feedback AS f
        WHERE f.decision_id = NEW.decision_id AND f.channel_id = NEW.channel_id
    ), dislike_count = (
        SELECT COALESCE(SUM(f.thumbs_down), 0) FROM discord_feedback AS f
        WHERE f.decision_id = NEW.decision_id AND f.channel_id = NEW.channel_id
    )
    WHERE decision_id = NEW.decision_id AND channel_id = NEW.channel_id;
END;

CREATE TRIGGER IF NOT EXISTS discord_feedback_counts_update
AFTER UPDATE ON discord_feedback
BEGIN
    UPDATE discord_notifications
    SET like_count = (
        SELECT COALESCE(SUM(f.thumbs_up), 0) FROM discord_feedback AS f
        WHERE f.decision_id = discord_notifications.decision_id
          AND f.channel_id = discord_notifications.channel_id
    ), dislike_count = (
        SELECT COALESCE(SUM(f.thumbs_down), 0) FROM discord_feedback AS f
        WHERE f.decision_id = discord_notifications.decision_id
          AND f.channel_id = discord_notifications.channel_id
    )
    WHERE (decision_id = OLD.decision_id AND channel_id = OLD.channel_id)
       OR (decision_id = NEW.decision_id AND channel_id = NEW.channel_id);
END;

CREATE TRIGGER IF NOT EXISTS discord_feedback_counts_delete
AFTER DELETE ON discord_feedback
BEGIN
    UPDATE discord_notifications
    SET like_count = (
        SELECT COALESCE(SUM(f.thumbs_up), 0) FROM discord_feedback AS f
        WHERE f.decision_id = OLD.decision_id AND f.channel_id = OLD.channel_id
    ), dislike_count = (
        SELECT COALESCE(SUM(f.thumbs_down), 0) FROM discord_feedback AS f
        WHERE f.decision_id = OLD.decision_id AND f.channel_id = OLD.channel_id
    )
    WHERE decision_id = OLD.decision_id AND channel_id = OLD.channel_id;
END;

-- A bridge back to the watcher's tables: one review covers the whole alert, i.e. every
-- (pr_folder, repo_id) pair in source_decision_keys at (pr_number, head_sha, base_sha).
-- A null vote means neither reaction or opposing reactions, not a negative label.
-- Dropped and recreated on every start so its columns follow this file.
DROP VIEW IF EXISTS discord_flagger_feedback;
CREATE VIEW discord_flagger_feedback AS
SELECT f.decision_id, f.user_id, f.channel_id, n.message_id, f.vote,
       CASE WHEN f.vote = 1 THEN 1 WHEN f.vote = -1 THEN 0 ELSE NULL END AS is_good,
       d.repository, d.pr_number, NULLIF(d.head_sha, '') AS head_sha,
       json_extract(d.payload_json, '$.base_sha') AS base_sha,
       json_extract(d.payload_json, '$.source_decision_keys') AS source_decision_keys,
       json_extract(d.payload_json, '$.source_dup_ids') AS source_dup_ids,
       json_extract(d.payload_json, '$.source_revision') AS source_revision,
       d.payload_json, f.created_at, f.updated_at
FROM discord_feedback AS f
JOIN discord_decisions AS d ON d.decision_id = f.decision_id
JOIN discord_notifications AS n
  ON n.decision_id = f.decision_id AND n.channel_id = f.channel_id;
