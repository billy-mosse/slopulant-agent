from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

import pandas as pd

from shared_utils import (ExperimentConfig, IdfTable, get_logger, load_config, normalize_query,
                          read_sql, require_columns, simple_stem, summarize_metrics, timed,
                          tokenize, write_table)

log = get_logger("search_exp.query_categorizer")

QUERY_LOGS = "search.query_logs"
PRODUCTS = "catalog.products"
OUTPUT = "search.query_categories"
VERSION = "qcat-tf-idf-v2"

TRAIN_QUERY = f"""
SELECT q.query_text, p.category, COUNT(*) AS clicks
FROM {QUERY_LOGS} q
JOIN {PRODUCTS} p ON p.product_id = q.clicked_product_id
WHERE q.event_ts >= CURRENT_DATE - (:lookback_days * INTERVAL '1 day')
  AND p.category IS NOT NULL
GROUP BY q.query_text, p.category
"""

SCORE_QUERY = f"""
SELECT query_text, COUNT(*) AS n_searches
FROM {QUERY_LOGS}
WHERE event_ts >= CURRENT_DATE - (:score_days * INTERVAL '1 day')
GROUP BY query_text
HAVING COUNT(*) >= :min_query_count
"""


@dataclass
class CategorizerConfig(ExperimentConfig):
    lookback_days: int = 180
    score_days: int = 30
    min_query_count: int = 3
    alpha: float = 0.3
    ngram_range: tuple[int, int] = (2, 4)
    min_label_share: float = 0.5
    min_confidence: float = 0.55
    holdout_frac: float = 0.1
    max_vocab: int = 300_000
    exclude_categories: list[str] = field(default_factory=lambda: ["gift cards", "unknown"])

    def validate(self) -> None:
        super().validate()
        if not 0 < self.alpha <= 10:
            raise ValueError("alpha out of range")
        if not 1 <= self.ngram_range[0] <= self.ngram_range[1] <= 6:
            raise ValueError("ngram_range must satisfy 1 <= n_min <= n_max <= 6")
        if not 0 <= self.holdout_frac < 0.5:
            raise ValueError("holdout_frac must be in [0, 0.5)")


@dataclass(frozen=True)
class LabeledExample:
    text: str
    category: str
    weight: float


@dataclass(frozen=True)
class InferenceResult:
    query: str
    ranked: tuple[tuple[str, float], ...]

    @property
    def top(self) -> tuple[str, float]:
        return self.ranked[0]

    @property
    def runner_up(self) -> tuple[str | None, float]:
        return self.ranked[1] if len(self.ranked) > 1 else (None, 0.0)


class TFIDFClassifier:
    """TF-IDF weighted voting over character n-grams with Laplace smoothing."""

    def __init__(self, alpha: float = 0.3, n_min: int = 2, n_max: int = 4, max_vocab: int = 300_000):
        self.alpha, self.n_min, self.n_max, self.max_vocab = alpha, n_min, n_max, max_vocab
        self.class_counts: Counter = Counter()
        self.term_class_counts: dict[str, Counter] = defaultdict(Counter)
        self.class_totals: Counter = Counter()
        self.vocab: set[str] = set()
        self._log_prior: dict[str, float] = {}
        self._log_unseen: dict[str, float] = {}

    def _extract_features(self, text: str) -> Counter:
        s = normalize_query(text)
        padded = f" {s} "
        features = Counter()
        for n in range(self.n_min, self.n_max + 1):
            for i in range(len(padded) - n + 1):
                features[padded[i:i + n]] += 1
        return features

    def train(self, examples: Iterable[LabeledExample]) -> "TFIDFClassifier":
        df: Counter = Counter()
        for ex in examples:
            feats = self._extract_features(ex.text)
            self.class_counts[ex.category] += ex.weight
            for g, cnt in feats.items():
                self.term_class_counts[g][ex.category] += cnt * ex.weight
            df.update(feats.keys())
        self.vocab = {g for g, _ in df.most_common(self.max_vocab)}
        for term in list(self.term_class_counts):
            if term not in self.vocab:
                del self.term_class_counts[term]
        for cat, cnt in self.term_class_counts.items():
            self.class_totals[cat] = sum(cnt.values())
        total = sum(self.class_counts.values())
        v = len(self.vocab)
        self._log_prior = {c: math.log(w / total) for c, w in self.class_counts.items()}
        self._log_unseen = {c: -math.log(self.class_totals[c] + self.alpha * v) for c in self.term_class_counts}
        log.info("trained TF-IDF: classes=%d vocab=%d", len(self.term_class_counts), v)
        return self

    def _log_scores(self, text: str) -> dict[str, float]:
        feats = {g: n for g, n in self._extract_features(text).items() if g in self.vocab}
        scores = {}
        for cat, cnt in self.term_class_counts.items():
            denom = self._log_unseen[cat]
            s = self._log_prior[cat]
            for g, n in feats.items():
                s += n * (math.log(cnt.get(g, 0.0) + self.alpha) + denom)
            scores[cat] = s
        return scores

    def predict(self, text: str, top_k: int = 3) -> InferenceResult:
        ll = self._log_scores(text)
        m = max(ll.values())
        exp_scores = {c: math.exp(v - m) for c, v in ll.items()}
        z = sum(exp_scores.values())
        ranked = sorted(((c, p / z) for c, p in exp_scores.items()), key=lambda x: -x[1])[:top_k]
        return InferenceResult(text, tuple(ranked))


def load_training_data(cfg: CategorizerConfig) -> list[LabeledExample]:
    df = read_sql(TRAIN_QUERY, cfg.warehouse_uri, params={"lookback_days": cfg.lookback_days})
    require_columns(df, ["query_text", "category", "clicks"], "training pairs")
    df["query_norm"] = df["query_text"].map(normalize_query)
    df["category"] = df["category"].str.strip().str.lower()
    df = df[~df["category"].isin(cfg.exclude_categories) & (df["query_norm"] != "")]
    df = df.groupby(["query_norm", "category"], as_index=False)["clicks"].sum()
    share = df["clicks"] / df.groupby("query_norm")["clicks"].transform("sum")
    df = df[share >= cfg.min_label_share]
    log.info("training queries=%d categories=%d", df["query_norm"].nunique(), df["category"].nunique())
    return [LabeledExample(q, c, math.log1p(n)) for q, c, n in df[["query_norm", "category", "clicks"]].itertuples(index=False)]


def split_data(examples: Sequence[LabeledExample], frac: float, seed: int) -> tuple[list[LabeledExample], list[LabeledExample]]:
    queries = sorted({e.text for e in examples})
    rng = __import__("random").Random(seed)
    held = set(rng.sample(queries, int(len(queries) * frac))) if frac else set()
    return [e for e in examples if e.text not in held], [e for e in examples if e.text in held]


def evaluate_model(model: TFIDFClassifier, test: Sequence[LabeledExample], min_conf: float) -> dict[str, float]:
    if not test:
        return {}
    preds = [(e, model.predict(e.text)) for e in test]
    w = sum(e.weight for e, _ in preds)
    acc = sum(e.weight for e, p in preds if p.top[0] == e.category) / w
    hit_3 = sum(e.weight * 3 * precision_at_k([c for c, _ in p.ranked], {e.category}, 3) for e, p in preds) / w
    confident = [(e, p) for e, p in preds if p.top[1] >= min_conf]
    cov = len(confident) / len(preds)
    conf_acc = sum(p.top[0] == e.category for e, p in confident) / max(1, len(confident))
    return {"accuracy": acc, "hit_at_3": hit_3, "coverage_at_conf": cov, "precision_at_conf": conf_acc}


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for x in ranked[:k] if x in relevant) / k


def per_category_stats(model: TFIDFClassifier, test: Sequence[LabeledExample], top: int = 10) -> pd.DataFrame:
    rows = [(e.category, model.predict(e.text).top[0] == e.category) for e in test]
    df = pd.DataFrame(rows, columns=["category", "correct"])
    out = df.groupby("category")["correct"].agg(["mean", "size"]).rename(columns={"mean": "acc", "size": "n"})
    return out.sort_values("n", ascending=False).head(top)


def score_queries(model: TFIDFClassifier, cfg: CategorizerConfig) -> pd.DataFrame:
    params = {"score_days": cfg.score_days, "min_query_count": cfg.min_query_count}
    q = read_sql(SCORE_QUERY, cfg.warehouse_uri, params=params)
    queries = sorted({normalize_query(t) for t in q["query_text"]} - {""})
    rows = []
    with timed(log, f"scoring {len(queries)} queries"):
        for query in queries:
            p = model.predict(query)
            cat, conf = p.top
            ru, ru_conf = p.runner_up
            rows.append({"query_norm": query, "category": cat if conf >= cfg.min_confidence else None,
                         "confidence": round(conf, 4), "runner_up": ru, "runner_up_conf": round(ru_conf, 4),
                         "best_guess": cat})
    df = pd.DataFrame(rows)
    df["model_version"] = VERSION
    df["scored_at"] = datetime.now(timezone.utc)
    log.info("assigned a category to %.1f%% of queries", 100 * df["category"].notna().mean())
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Train TF-IDF classifier and write {OUTPUT}")
    p.add_argument("--config")
    p.add_argument("--lookback-days", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--min-confidence", type=float)
    p.add_argument("--holdout-frac", type=float)
    p.add_argument("--dry-run", action="store_true", default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--predict", nargs="*", metavar="QUERY", help="print predictions and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k not in {"config", "eval_only", "predict"}}
    cfg = load_config(CategorizerConfig, args.config, overrides)
    examples = load_training_data(cfg)
    train, test = split_data(examples, cfg.holdout_frac, cfg.seed)
    with timed(log, "fit"):
        model = TFIDFClassifier(cfg.alpha, cfg.ngram_range[0], cfg.ngram_range[1], cfg.max_vocab).train(train)
    summarize_metrics(evaluate_model(model, test, cfg.min_confidence), log, "holdout")
    if test:
        print(per_category_stats(model, test).to_string())
    if args.predict:
        for q in args.predict:
            print(q, "->", [(c, round(p, 3)) for c, p in model.predict(normalize_query(q)).ranked])
        return 0
    if args.eval_only:
        return 0
    model = TFIDFClassifier(cfg.alpha, cfg.ngram_range[0], cfg.ngram_range[1], cfg.max_vocab).train(examples)
    write_table(score_queries(model, cfg), OUTPUT, cfg.warehouse_uri, dry_run=cfg.dry_run, logger=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
