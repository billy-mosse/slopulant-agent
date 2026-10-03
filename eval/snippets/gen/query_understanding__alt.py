from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine

# SQL to aggregate historical click-throughs by query and category
HISTORICAL_CTR_SQL = """
SELECT 
  LOWER(TRIM(q.query_text)) AS q_text,
  p.category_id,
  COUNT(*) AS ctc
FROM search.query_logs q
INNER JOIN catalog.predicted_category p ON p.sku = q.clicked_sku
WHERE q.clicked_sku IS NOT NULL
  AND q.event_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY q.query_text, p.category_id
"""

# Regex patterns for price-related signals
PRICE_PATTERNS = [
    (re.compile(r"\b(?:under|below|less than)\s*\$?(\d+)"), "upper_bound"),
    (re.compile(r"\b(?:over|above|more than)\s*\$?(\d+)"), "lower_bound"),
    (re.compile(r"\$(\d+)\s*-\s*\$?(\d+)"), "price_interval"),
]

# Known attribute values by dimension
ATTRIBUTE_CATALOG = {
    "bed_size": ["twin xl", "twin", "full", "queen", "california king", "king"],
    "fabric": ["linen", "cotton", "percale", "sateen", "bamboo", "silk", "wool", "velvet", "jute"],
}

# Configuration
MAX_CATS = 3
MIN_CONFIDENCE = 0.05


@dataclass
class SearchIntent:
    original: str
    normalized: str
    category_scores: list[tuple[str, float]] = field(default_factory=list)
    constraints: dict[str, float | str] = field(default_factory=dict)


def infer_category_probs(clicks: pd.DataFrame, smoothing: float = 1.0) -> dict[str, list[tuple[str, float]]]:
    result: dict[str, list[tuple[str, float]]] = {}
    for q, group in clicks.groupby("q_text"):
        scores = group.set_index("category_id")["ctc"].astype(float) + smoothing
        normalized = scores / scores.sum()
        ranked = normalized.sort_values(ascending=False)
        result[q] = [
            (cat, round(p, 4))
            for cat, p in ranked.items()
            if p >= MIN_CONFIDENCE
        ][:MAX_CATS]
    return result


def extract_price_constraints(text: str) -> dict[str, float]:
    bounds: dict[str, float] = {}
    for pattern, label in PRICE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if label == "price_interval":
            bounds["min_price"], bounds["max_price"] = float(match.group(1)), float(match.group(2))
        else:
            bounds[label] = float(match.group(1))
    return bounds


def detect_attributes(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for dim, options in ATTRIBUTE_CATALOG.items():
        for val in options:
            if re.search(rf"\b{re.escape(val)}\b", text):
                found[dim] = val
                break
    return found


def clean_query(text: str) -> str:
    for pattern, _ in PRICE_PATTERNS:
        text = pattern.sub("", text)
    return re.sub(r"\s+", " ", text.strip())


class IntentEngine:
    def __init__(self, category_map: dict[str, list[tuple[str, float]]]):
        self.category_map = category_map
        self.token_index: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for q, cats in category_map.items():
            for token in q.split():
                self.token_index[token].extend(cats)

    def infer_categories(self, text: str) -> list[tuple[str, float]]:
        if text in self.category_map:
            return self.category_map[text]
        aggregate: dict[str, float] = defaultdict(float)
        for token in text.split():
            for cat, score in self.token_index.get(token, []):
                aggregate[cat] += score
        total = sum(aggregate.values()) or 1.0
        return sorted(
            ((c, round(v / total, 4)) for c, v in aggregate.items()),
            key=lambda x: -x[1]
        )[:MAX_CATS]

    def analyze(self, raw: str, cleaned: str) -> SearchIntent:
        constraints = {**extract_price_constraints(cleaned), **detect_attributes(cleaned)}
        return SearchIntent(
            original=raw,
            normalized=cleaned,
            category_scores=self.infer_categories(clean_query(cleaned)),
            constraints=constraints,
        )


def build_pipeline(dsn: str, min_freq: int, run_date: str) -> None:
    engine = create_engine(dsn)
    freq_df = pd.read_sql(
        """
        SELECT LOWER(TRIM(query_text)) AS q_text, COUNT(*) AS cnt
        FROM search.query_logs
        WHERE event_date >= CURRENT_DATE - INTERVAL '90 days'
        GROUP BY 1
        HAVING COUNT(*) >= %s
        """,
        engine,
        params=(min_freq,),
    )
    click_df = pd.read_sql(HISTORICAL_CTR_SQL, engine)
    click_df["q_text"] = click_df.q_text.str.lower()

    # Build vocabulary and correct queries
    vocab = Vocabulary.from_queries(list(zip(freq_df.q_text, freq_df.cnt)))
    corrector = SpellCorrector(vocab)
    click_df["q_text"] = click_df.q_text.map(corrector.correct)

    # Build intent engine
    cat_probs = infer_category_probs(
        click_df.groupby(["q_text", "category_id"], as_index=False)["ctc"].sum()
    )
    engine = IntentEngine(cat_probs)

    # Process all queries
    rows = []
    for q in freq_df.q_text:
        intent = engine.analyze(q, corrector.correct(q))
        rows.append({
            "query_text": intent.original,
            "canonical_query": intent.normalized,
            "category_distribution": str(intent.category_scores),
            "filters": str(intent.constraints),
            "run_timestamp": run_date,
        })

    pd.DataFrame(rows).to_sql(
        "query_intents",
        engine,
        schema="search",
        if_exists="append",
        index=False,
    )


# --- Spell correction (reimplementation with alternative heuristics) ---

ALPHABET = set("abcdefghijklmnopqrstuvwxyz'-")
TOKENIZER = re.compile(r"[a-z0-9'\-]+")

EDIT_COSTS = {0: 1.0, 1: 0.07, 2: 0.003}
MIN_WORD_SUPPORT = 3


class Vocabulary:
    def __init__(self, freqs: Counter):
        self.freqs = Counter({w: c for w, c in freqs.items() if c >= MIN_WORD_SUPPORT})
        self.total = sum(self.freqs.values()) or 1

    @classmethod
    def from_queries(cls, data: list[tuple[str, int]]) -> "Vocabulary":
        counter = Counter()
        for q, n in data:
            for tok in re.findall(TOKENIZER, q.lower()):
                counter[tok] += n
        return cls(counter)

    def likelihood(self, word: str) -> float:
        return self.freqs.get(word, 0) / self.total

    def known(self, words: set[str]) -> set[str]:
        return {w for w in words if w in self.freqs}


class SpellCorrector:
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        self.cache = {}

    def variants(self, word: str) -> dict[str, int]:
        if word in self.vocab.freqs:
            return {word: 0}
        one = self.vocab.known(self._one_edit(word))
        if one:
            return {w: 1 for w in one}
        two = self.vocab.known(self._two_edit(word))
        return {w: 2 for w in two} if two else {}

    def _one_edit(self, word: str) -> set[str]:
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = {a + b[1:] for a, b in splits if b}
        transposes = {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}
        replaces = {a + c + b[1:] for a, b in splits if b for c in ALPHABET}
        inserts = {a + c + b for a, b in splits for c in ALPHABET}
        return deletes | transposes | replaces | inserts

    def _two_edit(self, word: str) -> set[str]:
        return {e2 for e1 in self._one_edit(word) for e2 in self._one_edit(e1)}

    def _score(self, orig: str, cand: str, edit_dist: int) -> float:
        return (
            math.log(self.vocab.likelihood(cand) + 1e-12)
            + math.log(EDIT_COSTS[min(edit_dist, 2)])
        )

    def fix(self, token: str) -> str:
        if token.isdigit() or len(token) <= 2:
            return token
        if token not in self.cache:
            variants = self.variants(token)
            if not variants:
                self.cache[token] = token
            else:
                self.cache[token] = max(variants, key=lambda c: self._score(token, c, variants[c]))
        return self.cache[token]

    def correct(self, query: str) -> str:
        return " ".join(self.fix(t) for t in re.findall(TOKENIZER, query.lower()))
