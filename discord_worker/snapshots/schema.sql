-- Schema from the committed GB10 database snapshot. Data is in my_database.db.

CREATE TABLE Commited (
            topic_id INTEGER PRIMARY KEY,
            commited_topic TEXT NOT NULL,
            topic_desc TEXT NOT NULL,
            owner TEXT NOT NULL
        , folder_name TEXT);

CREATE TABLE discord_decisions (
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

CREATE TABLE discord_feedback (
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

CREATE TABLE discord_notifications (
    decision_id TEXT NOT NULL REFERENCES discord_decisions(decision_id),
    channel_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    message_id TEXT,
    last_error TEXT,
    next_retry_at REAL,
    last_attempt_at TEXT,
    sent_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')), like_count INTEGER NOT NULL DEFAULT 0 CHECK (like_count >= 0), dislike_count INTEGER NOT NULL DEFAULT 0 CHECK (dislike_count >= 0),
    PRIMARY KEY (decision_id, channel_id),
    UNIQUE (channel_id, message_id)
);

CREATE TABLE dup_cg (
            dup_id INTEGER PRIMARY KEY,
            pr_id INTEGER NOT NULL,
            t_id INTEGER NOT NULL REFERENCES Commited(topic_id),
            incumbent_topic TEXT NOT NULL,
            pr_topic TEXT NOT NULL
        );

CREATE TABLE dupe_decision (
            dup_id INTEGER PRIMARY KEY REFERENCES dup_cg(dup_id),
            boolean INTEGER NOT NULL CHECK (boolean IN (0, 1))
        );

CREATE TABLE new_prs (
            pr_id INTEGER NOT NULL,
            topic_id INTEGER NOT NULL,
            folder_name TEXT NOT NULL,
            topic_desc TEXT NOT NULL,
            owner TEXT NOT NULL,
            timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (pr_id, topic_id)
        );

CREATE INDEX discord_decisions_pr
    ON discord_decisions (repository, pr_number);

CREATE TRIGGER discord_feedback_counts_delete
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

CREATE TRIGGER discord_feedback_counts_insert
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

CREATE TRIGGER discord_feedback_counts_update
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

CREATE VIEW discord_flagger_feedback AS
SELECT f.decision_id, f.user_id, f.channel_id, n.message_id, f.vote,
       CASE WHEN f.vote = 1 THEN 1 WHEN f.vote = -1 THEN 0 ELSE NULL END AS is_good,
       d.repository, d.pr_number, NULLIF(d.head_sha, '') AS head_sha,
       json_extract(d.payload_json, '$.source_dup_ids') AS source_dup_ids,
       json_extract(d.payload_json, '$.source_revision') AS source_revision,
       d.payload_json, f.created_at, f.updated_at
FROM discord_feedback AS f
JOIN discord_decisions AS d ON d.decision_id = f.decision_id
JOIN discord_notifications AS n
  ON n.decision_id = f.decision_id AND n.channel_id = f.channel_id;
