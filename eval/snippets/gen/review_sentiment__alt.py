from __future__ import annotations

import argparse
import logging
import re
import unicodedata

import pandas as pd
from sqlalchemy import create_engine, text

from lexicon import (ASPECTS, INTENSIFIERS, NEGATION_FACTOR, NEGATORS, POLARITY,
                     SCOPE_BREAKERS)

log = logging.getLogger("review_sentiment")

SPAN = 6
BACKTRACK = 4
TOKEN_PATTERN = re.compile(r"\b[a-z]+(?:'[a-z]+)?\b|[.,;!?]")
COMPOUND_MAX = max(len(p.split()) for p in set(POLARITY) | set(INTENSIFIERS) | {t for ts in ASPECTS.values() for t in ts})

ASPECT_MAP = {term: aspect for aspect, terms in ASPECTS.items() for term in terms}


def clean(text: str | None) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text).casefold()
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    return s


def segment(s: str) -> list[str]:
    raw = TOKEN_PATTERN.findall(s)
    tokens, idx = [], 0
    while idx < len(raw):
        for length in range(min(COMPOUND_MAX, len(raw) - idx), 0, -1):
            phrase = " ".join(raw[idx : idx + length])
            if phrase in POLARITY or phrase in INTENSIFIERS or phrase in ASPECT_MAP:
                tokens.append(phrase)
                idx += length
                break
        else:
            tokens.append(raw[idx])
            idx += 1
    return tokens


def valence(tokens: list[str], pos: int) -> float:
    base = POLARITY.get(tokens[pos], 0.0)
    modifier = 1.0
    for k in range(pos - 1, max(-1, pos - 1 - BACKTRACK), -1):
        token = tokens[k]
        if token in SCOPE_BREAKERS:
            break
        if token in INTENSIFIERS:
            modifier *= INTENSIFIERS[token]
        elif token in NEGATORS:
            modifier *= NEGATION_FACTOR
            break
    return max(-1.0, min(1.0, base * modifier))


def extract_aspects(tokens: list[str]) -> list[tuple[str, float]]:
    scores = []
    for idx, token in enumerate(tokens):
        aspect = ASPECT_MAP.get(token)
        if not aspect:
            continue
        weighted_sum, weight_total = 0.0, 0.0
        for j in range(max(0, idx - SPAN), min(len(tokens), idx + SPAN + 1)):
            if tokens[j] not in POLARITY:
                continue
            dist_weight = 1.0 / (1.0 + abs(idx - j))
            weighted_sum += dist_weight * valence(tokens, j)
            weight_total += dist_weight
        if weight_total:
            scores.append((aspect, weighted_sum / weight_total))
    return scores


def process_reviews(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in data.iterrows():
        content = clean(f"{row.title or ''} {row.body or ''}")
        for aspect, score in extract_aspects(segment(content)):
            records.append((row.review_id, row.sku, aspect, score))
    return pd.DataFrame(records, columns=["review_id", "sku", "aspect", "value"])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    per_review = df.groupby(["review_id", "sku", "aspect"], as_index=False)["value"].mean()
    per_review["positive"] = per_review["value"] > 0.15
    per_review["negative"] = per_review["value"] < -0.15
    summary = per_review.groupby(["sku", "aspect"]).agg(
        count=("review_id", "nunique"),
        avg_val=("value", "mean"),
        pos_rate=("positive", "mean"),
        neg_rate=("negative", "mean")
    ).reset_index()
    return summary


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--min-count", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    engine = create_engine(args.dsn)

    reviews = pd.read_sql(text("SELECT review_id, sku, title, body FROM reviews.approved"), engine)
    mentions = process_reviews(reviews)
    log.info("Processed %d reviews into %d aspect-value pairs", len(reviews), len(mentions))
    summary = summarize(mentions)
    summary = summary[summary["count"] >= args.min_count]
    summary.to_sql("aspect_sentiment", engine, schema="reviews", if_exists="replace", index=False)


if __name__ == "__main__":
    run()
