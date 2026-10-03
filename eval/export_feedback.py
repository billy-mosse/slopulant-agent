"""Export Discord reviews (👍/👎 on duplicate alerts) as labelled CLM pairs.

Each review covers one alert, i.e. every flagged (PR folder, existing folder) pair in it.
One output row per (review, pair): the reviewer's label next to the CLM score, threshold and
the candidate generator's signals, so clm_dupe/calibration/ can re-pick the threshold on
real reviews. Only positives are ever reviewed (clean PRs get no alert), so this measures
precision; recall still comes from eval/for_jesse.sqlite.

    python -m eval.export_feedback                       # CSV to stdout
    python -m eval.export_feedback --out data/feedback.csv
"""
import argparse
import csv
import json
import sqlite3
import sys

from watcher import config

QUERY = """
SELECT f.decision_id, f.user_id, f.is_good, f.updated_at AS reviewed_at, f.pr_number, f.head_sha, f.base_sha,
       f.source_decision_keys
FROM discord_flagger_feedback f
WHERE f.is_good IS NOT NULL AND f.source_decision_keys IS NOT NULL
ORDER BY f.updated_at
"""

PAIR = """
SELECT d.relation, d.confidence AS clm_score, d.threshold, d.classifier,
       s.score AS cg_score, s.desc_score, s.kw_score, s.dataflow, s.pr_topic, s.repo_topic
FROM decisions d JOIN scores s USING (pr_number, head_sha, base_sha, pr_folder, repo_id)
WHERE d.pr_number = ? AND d.head_sha = ? AND d.base_sha = ? AND d.pr_folder = ? AND d.repo_id = ?
"""

COLUMNS = ["decision_id", "user_id", "is_good", "reviewed_at", "pr_number", "head_sha", "base_sha", "pr_folder",
           "repo_id", "relation", "clm_score", "threshold", "classifier", "cg_score", "desc_score", "kw_score",
           "dataflow", "pr_topic", "repo_topic"]


def rows(conn):
    for review in conn.execute(QUERY):
        for key in json.loads(review["source_decision_keys"]):
            pr_folder, repo_id = key.split("/", 1)
            pair = conn.execute(PAIR, (review["pr_number"], review["head_sha"], review["base_sha"], pr_folder, repo_id)).fetchone()
            yield {**dict(review), "pr_folder": pr_folder, "repo_id": repo_id, **(dict(pair) if pair else {})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(config.DB_PATH))
    parser.add_argument("--out", help="CSV path (default: stdout)")
    args = parser.parse_args()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out = open(args.out, "w", newline="") if args.out else sys.stdout
    writer = csv.DictWriter(out, COLUMNS, extrasaction="ignore")
    writer.writeheader()
    n = 0
    for row in rows(conn):
        writer.writerow(row)
        n += 1
    print(f"{n} labelled pair(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
