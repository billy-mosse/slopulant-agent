from __future__ import annotations

import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from taxonomy import PARENT_MAP_L2, PARENT_MAP_L3, TAXONOMY_TREE

logger = logging.getLogger("inference")

SRC_TABLE = "catalog.products"
DEST_TABLE = "catalog.predicted_category"


def best_valid_child(probs: np.ndarray, labels: np.ndarray, allowed_parents: dict, target_parent: str) -> tuple[str | None, float]:
    """Select highest-probability child whose parent matches target_parent, renormalized."""
    valid_idx = np.array([allowed_parents.get(cls) == target_parent for cls in labels])
    if not valid_idx.any():
        return None, 0.0
    adjusted = probs * valid_idx
    mass = adjusted.sum()
    if mass <= 0:
        return None, 0.0
    idx = int(np.argmax(adjusted))
    return labels[idx], float(adjusted[idx] / mass)


def infer_tree(models: dict, features: pd.Series, min_leaf_conf: float) -> pd.DataFrame:
    l1_dist = models["dept"].predict_proba(features)
    l2_dist = models["cat"].predict_proba(features)
    l3_dist = models["subcat"].predict_proba(features)
    l1_classes = models["dept"].classes_
    l2_classes = models["cat"].classes_
    l3_classes = models["subcat"].classes_

    records = []
    for idx in range(len(features)):
        dept_idx = int(l1_dist[idx].argmax())
        dept = l1_classes[dept_idx]
        dept_conf = float(l1_dist[idx][dept_idx])

        cat, cat_conf = best_valid_child(l2_dist[idx], l2_classes, PARENT_MAP_L2, dept)
        if cat is None or cat_conf < min_leaf_conf:
            records.append((dept, None, None, dept_conf, cat_conf, None, 1))
            continue

        subcat, subcat_conf = best_valid_child(l3_dist[idx], l3_classes, PARENT_MAP_L3, cat)
        if subcat is None or subcat_conf < min_leaf_conf:
            records.append((dept, cat, None, dept_conf, cat_conf, subcat_conf, 2))
            continue

        records.append((dept, cat, subcat, dept_conf, cat_conf, subcat_conf, 3))

    return pd.DataFrame(records, columns=["dept", "cat", "subcat", "p_dept", "p_cat", "p_subcat", "depth"])


def main() -> None:
    argparser = argparse.ArgumentParser("Run hierarchical product classifier")
    argparser.add_argument("--db", required=True, help="Database connection string")
    argparser.add_argument("--checkpoint", required=True, help="Path to saved model bundle")
    argparser.add_argument("--min-prob", type=float, default=0.6, help="Threshold for child selection")
    opts = argparser.parse_args()

    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S", format="%(asctime)s %(levelname)s %(message)s")

    db = create_engine(opts.db)
    bundle = joblib.load(opts.checkpoint)
    data = pd.read_sql(f"SELECT sku, title, description FROM {SRC_TABLE} WHERE active", db)
    logger.info("processing %d active SKUs", len(data))

    # Combine title twice + description as weighted text signal
    text_signal = (data["title"].fillna("").str.cat(data["title"].fillna(""), sep=" ") + " " + data["description"].fillna("")).str.lower()

    results = infer_tree(bundle["estimators"], text_signal, opts.min_prob)
    results.insert(0, "sku", data["sku"].values)
    results["path"] = results[["dept", "cat", "subcat"]].apply(lambda row: " > ".join(row.dropna().astype(str)), axis=1)
    results["inference_ts"] = pd.Timestamp.utcnow()

    logger.info("depth breakdown: %s", results["depth"].value_counts(normalize=True).round(3).to_dict())

    schema, tbl = DEST_TABLE.split(".")
    results.to_sql(tbl, db, schema=schema, if_exists="replace", index=False)
    logger.info("stored %d predictions in %s", len(results), DEST_TABLE)


if __name__ == "__main__":
    main()
