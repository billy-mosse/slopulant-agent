import argparse
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("product.homepage_optimizer")


class ModuleCandidate:
    """Represents a product under consideration for inclusion in a homepage module."""

    __slots__ = ("sku", "category_id", "trend_score", "similarity_score")

    def __init__(self, sku: str, category_id: int, trend_score: float, similarity_score: float = 0.0):
        self.sku = sku
        self.category_id = category_id
        self.trend_score = trend_score
        self.similarity_score = similarity_score


class HomepagePersonalizationEngine:
    """
    Personalizes the homepage experience by selecting and ranking modules
    (e.g., Trending, Because You Viewed, New Arrivals, Sale) and populating each
    with a curated set of products per customer, balancing relevance and diversity.
    """

    SLOTS_PER_MODULE = 12
    DIVERSIFICATION_WEIGHT = 0.7
    CATEGORY_AFFINITY_COEFFICIENT = 0.65
    TRENDING_SIGNAL_COEFFICIENT = 0.35
    COOL_START_THRESHOLD = 0.05
    SMOOTHING_FACTOR = 2.0

    def __init__(
        self,
        trending_data: pd.DataFrame,
        similarity_data: pd.DataFrame,
        merchandise_catalog: pd.DataFrame,
        category_order: List[int],
    ):
        self.category_id_to_index: Dict[int, int] = {cat_id: idx for idx, cat_id in enumerate(category_order)}
        self._build_lookup_tables(trending_data, merchandise_catalog)
        self.trending_pool = self._prepare_trending_candidates(trending_data)
        self.new_arrival_pool = self._prepare_merch_candidates(merchandise_catalog, "new_arrivals")
        self.sale_pool = self._prepare_merch_candidates(merchandise_catalog, "sale")
        self.user_similarities = self._build_similarity_index(similarity_data)

    def _build_lookup_tables(self, trending: pd.DataFrame, merch: pd.DataFrame):
        sku_to_cat = pd.concat([trending[["sku", "category_id"]], merch[["sku", "category_id"]]]).drop_duplicates()
        self.sku_category_map = dict(zip(sku_to_cat.sku, sku_to_cat.category_id))
        self.trending_velocity = dict(zip(trending.sku, trending.velocity_z))

    def _prepare_trending_candidates(self, df: pd.DataFrame) -> List[ModuleCandidate]:
        return [
            ModuleCandidate(row.sku, row.category_id, row.velocity_z)
            for row in df.itertuples()
        ]

    def _prepare_merch_candidates(self, df: pd.DataFrame, module_type: str) -> List[ModuleCandidate]:
        subset = df[df.module_type == module_type]
        return [
            ModuleCandidate(row.sku, row.category_id, self.trending_velocity.get(row.sku, 0.0))
            for row in subset.itertuples()
        ]

    def _build_similarity_index(self, sim_df: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        grouped = sim_df.groupby("source_sku")
        return {
            src: list(zip(targets, scores))
            for src, (targets, scores) in grouped[["target_sku", "score"]].agg(list).iteritems()
        }

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x / self.SMOOTHING_FACTOR))

    def _build_viewed_pool(self, recent_skus: List[str]) -> List[ModuleCandidate]:
        accumulation: Dict[str, float] = {}
        for seed in recent_skus:
            for candidate_sku, weight in self.user_similarities.get(seed, []):
                if candidate_sku not in recent_skus:
                    accumulation[candidate_sku] = accumulation.get(candidate_sku, 0.0) + weight
        return [
            ModuleCandidate(
                sku,
                self.sku_category_map.get(sku, -1),
                self.trending_velocity.get(sku, 0.0),
                weight,
            )
            for sku, weight in accumulation.items()
        ]

    def _compute_relevance_scores(
        self,
        candidates: List[ModuleCandidate],
        category_mix: Optional[np.ndarray],
    ) -> np.ndarray:
        trend_normalized = self._sigmoid(np.array([c.trend_score for c in candidates]))
        if category_mix is None:
            return trend_normalized

        affinity = np.array([
            category_mix[self.category_id_to_index[c.category_id]]
            if c.category_id in self.category_id_to_index else 0.0
            for c in candidates
        ])
        affinity = affinity / (affinity.max() if affinity.max() > 0 else 1.0)

        similarity_normalized = np.array([c.similarity_score for c in candidates])
        similarity_normalized = similarity_normalized / (similarity_normalized.max() if similarity_normalized.max() > 0 else 1.0)

        hybrid_affinity = np.maximum(affinity, similarity_normalized)
        return self.CATEGORY_AFFINITY_COEFFICIENT * hybrid_affinity + self.TRENDING_SIGNAL_COEFFICIENT * trend_normalized

    def _select_with_mmr(
        self,
        candidates: List[ModuleCandidate],
        relevance_scores: np.ndarray,
        target_count: int,
    ) -> List[ModuleCandidate]:
        selected_indices: List[int] = []
        remaining = set(range(len(candidates)))

        while remaining and len(selected_indices) < target_count:
            def score_candidate(idx: int) -> float:
                intra_category_penalty = max(
                    (1.0 if candidates[idx].category_id == candidates[j].category_id else 0.0)
                    for j in selected_indices
                ) if selected_indices else 0.0
                return self.DIVERSIFICATION_WEIGHT * relevance_scores[idx] - (1 - self.DIVERSIFICATION_WEIGHT) * intra_category_penalty

            best_idx = max(remaining, key=score_candidate)
            selected_indices.append(best_idx)
            remaining.discard(best_idx)

        return [candidates[i] for i in selected_indices]

    def generate_customer_plan(
        self,
        customer_category_mix: Optional[np.ndarray],
        recent_viewed_skus: List[str],
    ) -> List[Dict[str, object]]:
        cold_start = customer_category_mix is None or np.sum(customer_category_mix) < self.COOL_START_THRESHOLD
        candidate_pools = {"trending": self.trending_pool} if cold_start else {
            "trending": self.trending_pool,
            "because_you_viewed": self._build_viewed_pool(list(recent_viewed_skus)),
            "new_arrivals": self.new_arrival_pool,
            "sale": self.sale_pool,
        }

        module_scores = []
        for module_name in ("trending", "because_you_viewed", "new_arrivals", "sale"):
            pool = candidate_pools.get(module_name) or []
            if not pool:
                continue
            relevance = self._compute_relevance_scores(pool, None if cold_start else customer_category_mix)
            selected_candidates = self._select_with_mmr(pool, relevance, self.SLOTS_PER_MODULE)
            relevance_mean = float(np.mean(relevance[:len(selected_candidates)]))
            module_scores.append({
                "module_name": module_name,
                "skus": [c.sku for c in selected_candidates],
                "relevance_score": relevance_mean,
            })

        return sorted(module_scores, key=lambda m: -m["relevance_score"])


def execute_personalization_pipeline(
    warehouse_dsn: str,
    merch_parquet_path: str,
    category_list_path: str,
) -> None:
    engine = create_engine(warehouse_dsn)

    with open(category_list_path) as f:
        category_order = json.load(f)

    engine = create_engine(warehouse_dsn)
    trending_data = pd.read_sql("""
        SELECT sku, category_id, velocity_z
        FROM recs.trending
        WHERE computed_at = (SELECT MAX(computed_at) FROM recs.trending)
    """, engine)

    similarity_data = pd.read_sql("""
        SELECT source_sku, target_sku, score
        FROM recs.i2i
    """, engine)

    merch_catalog = pd.read_parquet(merch_parquet_path)
    merch_catalog["module_type"] = merch_catalog["module"].map({"new_arrivals": "new_arrivals", "sale": "sale"})

    personalizer = HomepagePersonalizationEngine(
        trending_data, similarity_data, merch_catalog, category_order
    )

    embeddings = pd.read_sql("""
        SELECT customer_id, category_mix, recent_skus
        FROM features.customer_embeddings
    """, engine)

    output_rows = []
    for profile in embeddings.itertuples():
        mix_array = np.array(profile.category_mix, dtype=np.float64) if profile.category_mix else None
        for rank_idx, mod_entry in enumerate(
            personalizer.generate_customer_plan(mix_array, list(profile.recent_skus or [])), start=1
        ):
            output_rows.append({
                "customer_id": profile.customer_id,
                "module": mod_entry["module_name"],
                "module_rank": rank_idx,
                "skus": json.dumps(mod_entry["skus"]),
            })

    output_df = pd.DataFrame(output_rows)
    output_df.to_sql("homepage_slots", engine, schema="recs", if_exists="replace", index=False, chunksize=50000)
    log.info("Personalization complete: %d customers processed, %d module assignments written.", len(output_df), len(output_df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute personalized homepage module assignments")
    parser.add_argument("--warehouse-dsn", required=True, help="Database connection string")
    parser.add_argument("--merchandise-feed", required=True, help="Path to merchandise parquet file")
    parser.add_argument("--category-order", required=True, help="Path to JSON list of category IDs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
    execute_personalization_pipeline(args.warehouse_dsn, args.merchandise_feed, args.category_order)


if __name__ == "__main__":
    main()
