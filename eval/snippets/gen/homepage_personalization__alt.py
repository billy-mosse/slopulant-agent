import argparse
import json
import logging
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sqlalchemy import create_engine

log = logging.getLogger("ui_homepage")

CATEGORY_POOL = ["trending", "view_based", "fresh_arrivals", "discounted"]
SLOT_COUNT = 12
DIVERSIFY_WEIGHT = 0.7
COMPONENT_WEIGHTS = (0.65, 0.35)
MIN_CATEGORY_SUPPORT = 0.05

TREND_QUERY = """
    SELECT sku, category_id, velocity_z 
    FROM recs.trending 
    WHERE computed_at = (SELECT MAX(computed_at) FROM recs.trending)
"""
VIEW_QUERY = "SELECT source_sku, target_sku, score FROM recs.i2i"
CUSTOMER_QUERY = "SELECT customer_id, category_mix, recent_skus FROM features.customer_embeddings"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return expit(x / 2.0)


def compute_score池(candidates: List[Tuple], mix: np.ndarray, cat_map: Dict[int, int]) -> np.ndarray:
    base_scores = np.zeros(len(candidates))
    aff_scores = np.zeros(len(candidates))
    for idx, (sku, cat_id, trend_val, _) in enumerate(candidates):
        base_scores[idx] = sigmoid(trend_val)
        if cat_id in cat_map and mix is not None:
            aff_scores[idx] = mix[cat_map[cat_id]]
    if mix is None or aff_scores.max() == 0:
        return base_scores
    aff_scores = aff_scores / max(aff_scores.max(), 1e-6)
    return COMPONENT_WEIGHTS[0] * aff_scores + COMPONENT_WEIGHTS[1] * base_scores


def mmr_dedup(candidates: List[Tuple], scores: np.ndarray, top_k: int, λ: float = DIVERSIFY_WEIGHT) -> List[int]:
    selection = []
    remaining = set(range(len(candidates)))
    while remaining and len(selection) < top_k:
        def utility(idx: int) -> float:
            cat = candidates[idx][1]
            max_sim = max((0 if candidates[i][1] != cat else 1 for i in selection), default=0)
            return λ * scores[idx] - (1 - λ) * max_sim
        best = max(remaining, key=utility)
        selection.append(best)
        remaining.remove(best)
    return selection


class HomepageEngine:
    def __init__(self, trend_df: pd.DataFrame, view_df: pd.DataFrame, merch_df: pd.DataFrame, cat_order: List[int]):
        self.cat_idx = {c: i for i, c in enumerate(cat_order)}
        self.cat_id_map = dict(zip(trend_df.sku, trend_df.category_id))
        self.cat_id_map.update(zip(merch_df.sku, merch_df.category_id))
        self.trend_map = dict(zip(trend_df.sku, trend_df.velocity_z))
        self.v2v = view_df.groupby("source_sku").apply(lambda g: list(zip(g.target_sku, g.score))).to_dict()
        self.pools = {
            "trending": [
                (row.sku, row.category_id, row.velocity_z, 0.0) 
                for row in trend_df.itertuples()
            ],
            "fresh_arrivals": [
                (row.sku, row.category_id, self.trend_map.get(row.sku, 0.0), 0.0)
                for row in merch_df[merch_df.type == "fresh_arrivals"].itertuples()
            ],
            "discounted": [
                (row.sku, row.category_id, self.trend_map.get(row.sku, 0.0), 0.0)
                for row in merch_df[merch_df.type == "discounted"].itertuples()
            ],
        }

    def derive_view_pool(self, recent_skus: List[str]) -> List[Tuple]:
        agg = {}
        for s in recent_skus:
            for t, sc in self.v2v.get(s, []):
                if t not in recent_skus:
                    agg[t] = agg.get(t, 0.0) + sc
        return [
            (sku, self.cat_id_map.get(sku, -1), self.trend_map.get(sku, 0.0), sc)
            for sku, sc in agg.items()
        ]

    def rank_customer(self, mix: np.ndarray, view_list: List[str]) -> List[Dict[str, Any]]:
        is_cold = mix is None or mix.sum() < MIN_CATEGORY_SUPPORT
        pool_subset = {"trending"} if is_cold else set(CATEGORY_POOL)
        scored_modules = []
        for mod in CATEGORY_POOL:
            if mod not in pool_subset:
                continue
            cand = self.pools.get(mod, [])
            if mod == "view_based":
                cand = self.derive_view_pool(view_list) or []
            if not cand:
                continue
            rel = compute_score池(cand, None if is_cold else mix, self.cat_idx)
            idx = mmr_dedup(cand, rel, SLOT_COUNT)
            scored_modules.append({
                "name": mod,
                "items": [cand[i][0] for i in idx],
                "avg_score": float(rel[idx].mean())
            })
        return sorted(scored_modules, key=lambda x: -x["avg_score"])


def run(dsn: str, merch_path: str, cat_json: str):
    engine = create_engine(dsn)
    categories = json.load(open(cat_json))
    trend_df = pd.read_sql(TREND_QUERY, engine)
    view_df = pd.read_sql(VIEW_QUERY, engine)
    merch_df = pd.read_parquet(merch_path)
    customer_df = pd.read_sql(CUSTOMER_QUERY, engine)
    engine = HomepageEngine(trend_df, view_df, merch_df, categories)

    results = []
    for row in customer_df.itertuples():
        mix = np.frombuffer(row.category_mix, dtype=np.float64) if row.category_mix else None
        for rank, mod in enumerate(engine.rank_customer(mix, row.recent_skus or []), 1):
            results.append({
                "customer_id": row.customer_id,
                "module": mod["name"],
                "module_rank": rank,
                "skus": json.dumps(mod["items"]),
            })
    out_df = pd.DataFrame(results)
    out_df.to_sql("homepage_slots", engine.raw_connection(), schema="recs", if_exists="replace", index=False, chunksize=50_000)
    log.info("Processed %d users → %d slot entries", len(customer_df), len(out_df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate homepage modules per user")
    parser.add_argument("--warehouse-dsn", required=True)
    parser.add_argument("--merch-data", required=True)
    parser.add_argument("--cat-ids", required=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parser.parse_args()
    run(args.warehouse_dsn, args.merch_data, args.cat_ids)
