import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial.distance import cdist
import sqlalchemy as sa

logger = logging.getLogger(__name__)

SRC_TABLE = "features.customer_embeddings"
DEST_TABLE = "marketing.segments"
COMPONENTS = 5
CLUSTER_RANGE = range(4, 11)
RANDOM_SEED = 42
PROFILE_FIELDS = ["recency_days", "orders_12m", "spend_12m", "share_bedding", "share_bath", "share_decor"]

CLUSTER_LABELS = [
    ("High-value lapsed", {"spend_12m": 0.5, "recency_days": 0.75, "op": "gt"}),
    ("Premium frequent", {"spend_12m": 1.0, "orders_12m": 1.0, "op": "gt"}),
    ("Bedding fans", {"share_bedding": 0.75, "op": "gt"}),
    ("Bath lovers", {"share_bath": 0.75, "op": "gt"}),
    ("Decor explorers", {"share_decor": 0.75, "orders_12m": 0.0, "op": "lt"}),
    ("Fresh faces", {"recency_days": -0.5, "orders_12m": -0.25, "op": "lt"}),
    ("One-time dormant", {"recency_days": 0.5, "orders_12m": -0.25, "op": "gt"}),
]
DEFAULT_LABEL = "Core steady"


def pull_embeddings(conn):
    q = f"""
        SELECT * FROM {SRC_TABLE}
        WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM {SRC_TABLE})
    """
    data = pd.read_sql(q, conn).set_index("customer_id")
    logger.info("Fetched %d records from %s", len(data), SRC_TABLE)
    return data


def extract_latent(df):
    vecs = [c for c in df.columns if c.startswith("emb_")]
    return df[vecs].values.astype(np.float32)


def best_cluster_count(X):
    rng = np.random.default_rng(RANDOM_SEED)
    subsamp = rng.choice(len(X), min(20_000, len(X)), replace=False)
    scores = {}
    for k in CLUSTER_RANGE:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)
        ch = calinski_harabasz_score(X[subsamp], labels[subsamp])
        scores[k] = ch
        logger.debug("k=%d CH-score=%.2f", k, ch)
    return max(scores, key=scores.get), scores


def assign_labels(profiles, cluster_ids):
    normed = (profiles - profiles.mean()) / profiles.std(ddof=0).clip(1e-6)
    centroids = normed.groupby(cluster_ids).mean()
    used = set()
    mapping = {}
    for cid, row in centroids.iterrows():
        label = DEFAULT_LABEL
        for name, cond in CLUSTER_LABELS:
            if cond["op"] == "gt" and row[cond["field"]] > cond["threshold"]:
                if name not in used:
                    label = name
                    break
            elif cond["op"] == "lt" and row[cond["field"]] < cond["threshold"]:
                if name not in used:
                    label = name
                    break
        if label in used:
            label = f"{label} #{cid}"
        used.add(label)
        mapping[cid] = label
    logger.info("Cluster labels assigned: %s", mapping)
    return mapping


def execute(dsn, commit=True):
    engine = sa.create_engine(dsn)
    raw = pull_embeddings(engine)
    latent = extract_latent(raw)

    scaler = RobustScaler()
    reduced = TruncatedSVD(n_components=COMPONENTS, random_state=RANDOM_SEED)
    transformed = scaler.fit_transform(latent)
    projected = reduced.fit_transform(transformed)
    logger.info("SVD variance captured: %s", np.round(reduced.explained_variance_ratio_, 3))

    optimal_k, _ = best_cluster_count(projected)
    model = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
    clusters = model.fit_predict(projected)

    label_map = assign_labels(raw[PROFILE_FIELDS], clusters)
    dists = cdist(projected, model._fit_X if hasattr(model, "_fit_X") else projected, metric="euclidean").diagonal()

    results = pd.DataFrame({
        "customer_id": raw.index,
        "segment_id": clusters,
        "segment_label": [label_map[c] for c in clusters],
        "centroid_dist": np.round(dists, 4),
        "k_value": optimal_k,
        "refresh_date": pd.Timestamp.today().normalize(),
    })

    print(results.groupby("segment_label").size().sort_values(ascending=False).to_string())

    if commit:
        schema, tbl = DEST_TABLE.split(".")
        results.to_sql(tbl, engine, schema=schema, if_exists="replace", index=False)
        logger.info("Persisted %d rows to %s", len(results), DEST_TABLE)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marketing cluster builder")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    execute(args.dsn, commit=not args.dry_run)
