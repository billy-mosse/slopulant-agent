from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple

import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("search_autocomplete")


@dataclass(frozenset=True)
class AutocompleteCandidate:
    """Represents a search term with its relevance score."""
    text: str
    relevance_score: float


class SearchTermProcessor:
    """Normalizes, cleans, and deduplicates user search queries for indexing."""

    BLOCKLIST = {"fuck", "shit", "bitch", "cunt", "porn", "nsfw", "dick", "pussy", "nazi", "slut"}
    CHAR_SUBSTITUTIONS = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "$": "s", "@": "a"})

    @classmethod
    def sanitize(cls, raw_input: str) -> str:
        """Apply consistent text cleaning: lowercase, strip, normalize whitespace and invalid chars."""
        cleaned = raw_input.lower().strip()
        cleaned = re.sub(r"[^a-z0-9 '&\-]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def is_inappropriate(cls, processed: str) -> bool:
        """Check if the query contains blocked terms (including leetspeak variants)."""
        normalized = processed.translate(cls.CHAR_SUBSTITUTIONS)
        tokens = set(normalized.split())
        if tokens & cls.BLOCKLIST:
            return True
        no_spaces = normalized.replace(" ", "")
        return any(blocked in no_spaces for blocked in cls.BLOCKLIST if len(blocked) > 4)

    @classmethod
    def normalize_inflection(cls, token: str) -> str:
        """Convert plural forms to singular (basic heuristic)."""
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith(("ches", "shes", "sses", "xes")):
            return token[:-2]
        if token.endswith("s") and not token.endswith(("ss", "us")) and len(token) > 3:
            return token[:-1]
        return token

    @classmethod
    def make_dedup_signature(cls, query: str) -> str:
        """Generate a canonical form to group semantically similar queries."""
        parts = query.replace("-", " ").split()
        return " ".join(cls.normalize_inflection(p) for p in parts)


class ScoringEngine:
    """Compute time-decayed relevance scores for search queries."""

    HALF_LIFE = 14.0
    MINIMAL_DECAYED_VOLUME = 3.0
    CLICK_THRESHOLD_WEIGHT = 0.6
    TOP_N = 8
    MAX_TOKEN_LENGTH = 20

    @staticmethod
    def compute_decay_factor(age_in_days: float) -> float:
        """Calculate weight decay based on exponential half-life."""
        return math.exp(-math.log(2) * age_in_days / ScoringEngine.HALF_LIFE)

    @classmethod
    def process_dataframe(cls, historical_data: pd.DataFrame, cutoff_date: date) -> pd.DataFrame:
        """Transform raw logs into scored candidates suitable for autocomplete indexing."""
        df = historical_data.assign(text=historical_data.query_text.map(cls.sanitize))
        df = df[
            df.text.str.len().between(2, 60) &
            ~df.text.map(cls.is_inappropriate)
        ]
        df["age_days"] = (pd.Timestamp(cutoff_date) - pd.to_datetime(df.event_date)).dt.days.clip(lower=0)
        decay_weights = df.age_days.map(cls.compute_decay_factor)
        df["weighted_searches"] = df.searches * decay_weights
        df["weighted_clicks"] = df.clicks * decay_weights

        aggregated = df.groupby("text")[["weighted_searches", "weighted_clicks"]].sum().reset_index()
        aggregated = aggregated[aggregated.weighted_searches >= cls.MINIMAL_DECAYED_VOLUME]

        success_rate = (aggregated.weighted_clicks + 1.0) / (aggregated.weighted_searches + 2.0)
        log_volume = aggregated.weighted_searches.map(math.log1p)
        aggregated["score"] = log_volume * ((1 - cls.CLICK_THRESHOLD_WEIGHT) + cls.CLICK_THRESHOLD_WEIGHT * success_rate)

        return cls._resolve_semantic_duplicates(aggregated)

    @staticmethod
    def sanitize(text: str) -> str:
        return SearchTermProcessor.sanitize(text)

    @staticmethod
    def is_inappropriate(text: str) -> bool:
        return SearchTermProcessor.is_inappropriate(text)

    @staticmethod
    def _resolve_semantic_duplicates(scored: pd.DataFrame) -> pd.DataFrame:
        """Group semantically equivalent queries and assign aggregate scores."""
        dedup_key = scored.text.map(SearchTermProcessor.make_dedup_signature)
        scored = scored.assign(group_key=dedup_key)
        best_per_group = scored.sort_values("score", ascending=False).drop_duplicates("group_key")
        group_totals = scored.groupby("group_key").score.sum()
        return best_per_group.assign(score=best_per_group.group_key.map(group_totals))[["text", "score"]]


class AutocompleteIndexBuilder:
    """Maintains a prefix-indexed trie for efficient autocomplete lookups."""

    def __init__(self, max_suggestions_per_node: int = 8, max_prefix_length: int = 20):
        self._root: Dict = {}
        self._max_suggestions = max_suggestions_per_node
        self._max_prefix = max_prefix_length

    def _truncate_suggestions(self, bucket: List[AutocompleteCandidate]) -> List[AutocompleteCandidate]:
        return sorted(bucket, key=lambda c: -c.relevance_score)[:self._max_suggestions]

    def insert(self, candidate: AutocompleteCandidate) -> None:
        node = self._root
        text = candidate.text[:self._max_prefix]
        for char in text:
            node = node.setdefault(char, {})
            top_candidates = node.setdefault("candidates", [])
            top_candidates.append(candidate)
            if len(top_candidates) > self._max_suggestions * 2:
                node["candidates"] = self._truncate_suggestions(top_candidates)

    def finalize(self, current_node: Dict | None = None) -> None:
        node = current_node or self._root
        if "candidates" in node:
            node["candidates"] = self._truncate_suggestions(node["candidates"])
        for key, child in node.items():
            if key != "candidates":
                self.finalize(child)

    def query_prefix(self, prefix: str) -> List[AutocompleteCandidate]:
        node = self._root
        normalized_prefix = SearchTermProcessor.sanitize(prefix)
        for char in normalized_prefix:
            if char not in node:
                return []
            node = node[char]
        return node.get("candidates", [])

    def yield_all_paths(self, current_node: Dict | None = None, prefix: str = ""):
        node = current_node or self._root
        for char, subtree in node.items():
            if char == "candidates":
                continue
            new_prefix = prefix + char
            suggestions = subtree.get("candidates", [])
            yield new_prefix, [(c.text, round(c.relevance_score, 4)) for c in suggestions]
            yield from self.yield_all_paths(subtree, new_prefix)


def construct_index(historical_events: pd.DataFrame, reference_date: date) -> AutocompleteIndexBuilder:
    """Build and finalize an autocomplete index from processed historical search data."""
    processor = ScoringEngine()
    scored_candidates = processor.process_dataframe(historical_events, reference_date)
    index = AutocompleteIndexBuilder()
    for _, row in scored_candidates.iterrows():
        index.insert(AutocompleteCandidate(text=row.text, relevance_score=row.score))
    index.finalize()
    return index


def main():
    parser = argparse.ArgumentParser(description="Generate search autocomplete index from historical logs")
    parser.add_argument("--database-url", required=True, help="SQL connection string")
    parser.add_argument("--as-of", default=date.today().isoformat(), help="Reference date for decay calculations")
    args = parser.parse_args()

    engine = create_engine(args.database_url)
    reference_date = date.fromisoformat(args.as_of)
    query_sql = """
        SELECT LOWER(TRIM(query_text)) AS query_text,
               event_date,
               COUNT(*) AS searches,
               SUM(CASE WHEN clicked_sku IS NOT NULL THEN 1 ELSE 0 END) AS clicks
        FROM search.query_logs
        WHERE event_date >= CURRENT_DATE - INTERVAL '120 days'
        GROUP BY 1, 2
    """
    raw_data = pd.read_sql(query_sql, engine)
    index = construct_index(raw_data, reference_date)

    output_records = [
        {"prefix": p, "suggestions": json.dumps(s), "generation_date": reference_date}
        for p, s in index.yield_all_paths()
    ]

    log_output = pd.DataFrame(output_records)
    schema, table = "search.autocomplete_index".split(".")
    log_output.to_sql(table, engine, schema=schema, if_exists="replace", index=False, chunksize=10_000)

    logger.info("Autocomplete index built: %d prefixes generated", len(log_output))
    sample = index.query_prefix("duv")
    logger.info("Sample query 'duv' returned %d candidates", len(sample))


if __name__ == "__main__":
    main()
