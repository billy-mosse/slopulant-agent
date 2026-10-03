import argparse
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger("search.autocomplete")

INPUT_TABLE = "search.query_logs"
OUTPUT_TABLE = "search.autocomplete_index"
DECAY_RATE = 0.05
CANDIDATE_LIMIT = 8
MIN_SCORE_THRESHOLD = 2.5
CLICK_BONUS = 0.7
PREFIX_MAX_LEN = 20

FILTER_WORDS = {"fuck", "shit", "bitch", "cunt", "porn", "nsfw", "dick", "pussy", "nazi", "slut"}
CHAR_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "$": "s", "@": "a"})

RAW_QUERY = f"""
SELECT LOWER(TRIM(query_text)) AS phrase,
       event_date::date AS day,
       COUNT(*) AS impressions,
       SUM((clicked_sku IS NOT NULL)::int) AS clicks
FROM {INPUT_TABLE}
WHERE event_date >= CURRENT_DATE - INTERVAL '120 days'
GROUP BY 1, 2
"""


@dataclass(frozen=True)
class Candidate:
    term: str
    weight: float


def clean_text(s: str) -> str:
    s = re.sub(r"[^a-z0-9 '&\-]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def contains_profanity(s: str) -> bool:
    normalized = s.translate(CHAR_MAP)
    tokens = set(normalized.split())
    return bool(tokens & FILTER_WORDS) or any(prof in normalized.replace(" ", "") for prof in FILTER_WORDS if len(prof) > 4)


def canonical_form(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "sses", "xes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us")) and len(token) > 3:
        return token[:-1]
    return token


def normalize_key(phrase: str) -> str:
    return " ".join(canonical_form(w) for w in phrase.replace("-", " ").split())


def temporal_weight(days_old: float) -> float:
    return math.exp(-DECAY_RATE * days_old)


def process_logs(df: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    df = df.assign(phrase=df.phrase.map(clean_text))
    df = df[
        df.phrase.str.len().between(2, 60) &
        ~df.phrase.map(contains_profanity)
    ]
    age = (pd.Timestamp(cutoff) - pd.to_datetime(df.day)).dt.days.clip(lower=0)
    decay_factors = age.map(temporal_weight)
    df["imp_weight"] = df.impressions * decay_factors
    df["click_weight"] = df.clicks * decay_factors
    grouped = df.groupby("phrase", as_index=False)[["imp_weight", "click_weight"]].sum()
    grouped = grouped[grouped.imp_weight >= MIN_SCORE_THRESHOLD]
    conversion = (grouped.click_weight + 1.0) / (grouped.imp_weight + 2.0)
    grouped["score"] = grouped.imp_weight.map(math.log1p) * ((1 - CLICK_BONUS) + CLICK_BONUS * conversion)
    return deduplicate(grouped)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(group_key=df.phrase.map(normalize_key))
    ranked = df.sort_values("score", ascending=False).drop_duplicates("group_key")
    group_scores = df.groupby("group_key").score.sum()
    return ranked.assign(score=ranked.group_key.map(group_scores))[["phrase", "score"]]


class TrieNode:
    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.candidates: list[Candidate] = []

    def add(self, cand: Candidate, limit: int = CANDIDATE_LIMIT * 2) -> None:
        self.candidates.append(cand)
        if len(self.candidates) > limit:
            self.candidates.sort(key=lambda x: -x.weight)
            self.candidates = self.candidates[:limit]


def build_trie(df: pd.DataFrame, cutoff: date) -> TrieNode:
    root = TrieNode()
    for _, row in process_logs(df, cutoff).iterrows():
        cand = Candidate(row.phrase, row.score)
        node = root
        for ch in row.phrase[:PREFIX_MAX_LEN]:
            node = node.children.setdefault(ch, TrieNode())
            node.add(cand)
    return root


def prune_trie(node: TrieNode, limit: int = CANDIDATE_LIMIT) -> None:
    if node.candidates:
        node.candidates.sort(key=lambda x: -x.weight)
        node.candidates = node.candidates[:limit]
    for child in node.children.values():
        prune_trie(child, limit)


def traverse(node: TrieNode, prefix: str = ""):
    for ch, child in node.children.items():
        yield prefix + ch, [(c.term, round(c.weight, 4)) for c in child.candidates]
        yield from traverse(child, prefix + ch)


def fetch_prefix(node: TrieNode, prefix: str) -> list[Candidate]:
    for ch in clean_text(prefix):
        if ch not in node.children:
            return []
        node = node.children[ch]
    return node.candidates


def main():
    parser = argparse.ArgumentParser(description="Generate autocomplete index")
    parser.add_argument("--db", required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = create_engine(args.db)
    cutoff = datetime.fromisoformat(args.date).date()
    raw = pd.read_sql(RAW_QUERY, engine)
    trie = build_trie(raw, cutoff)
    prune_trie(trie)

    records = [
        {"prefix": p, "payload": json.dumps(s), "generated_on": cutoff}
        for p, s in traverse(trie)
    ]
    result = pd.DataFrame(records)
    logger.info("Built %d prefixes; example 'duv': %s", len(result), [c.term for c in fetch_prefix(trie, "duv")])

    schema, tbl = OUTPUT_TABLE.split(".")
    result.to_sql(tbl, engine, schema=schema, if_exists="replace", index=False, chunksize=10_000)


if __name__ == "__main__":
    main()
