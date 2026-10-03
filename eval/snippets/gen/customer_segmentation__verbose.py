"""Marketing segmentation pipeline for campaign targeting.

This module computes customer segments on a weekly basis to support targeted
marketing initiatives. It ingests the most recent customer feature vectors,
applies dimensionality reduction and clustering, then assigns each customer
to a descriptive segment based on behavioral centroids. The resulting
assignments are stored for downstream use in campaign planning and
personalization.

Segments are derived using a data-driven process: embeddings are standardized
and compressed to five principal components, then k-means is evaluated across
a range of cluster counts using silhouette scoring to identify the optimal
partition. Each cluster is interpreted by comparing its centroid profile
(recency, order frequency, spend, category preferences) against a curated
set of naming heuristics.

The output table includes segment identifiers, human-readable names, and a
distance metric indicating how centrally each customer sits within their
assigned group. Marketing teams should rely on segment names rather than
numeric IDs, as cluster counts may vary across refresh cycles.
"""
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_SOURCE = "features.customer_embeddings"
SEGMENT_OUTPUT = "marketing.segments"
PCA_COMPONENT_COUNT = 5
CLUSTER_CANDIDATES = range(4, 11)
SILHOUETTE_SUBSAMPLE = 20_000
SEED = 42

BEHAVIORAL_INDICATORS = [
    "recency_days",
    "orders_12m",
    "spend_12m",
    "share_bedding",
    "share_bath",
    "share_decor",
]

SEGMENT_DEFINITIONS = [
    ("Lapsed high spenders", {"spend_12m": (">", 0.5), "recency_days": (">", 0.75)}),
    ("VIP regulars", {"spend_12m": (">", 1.0), "orders_12m": (">", 1.0)}),
    ("Bedding loyalists", {"share_bedding": (">", 0.75)}),
    ("Bath enthusiasts", {"share_bath": (">", 0.75)}),
    ("Decor browsers", {"share_decor": (">", 0.75), "orders_12m": ("<", 0.0)}),
    ("New & curious", {"recency_days": ("<", -0.5), "orders_12m": ("<", -0.25)}),
    ("Dormant one-timers", {"recency_days": (">", 0.5), "orders_12m": ("<", -0.25)}),
]
DEFAULT_LABEL = "Steady mainstream"


class SegmentBuilder:
    def __init__(self, engine):
        self.engine = engine

    def _fetch_latest_features(self) -> pd.DataFrame:
        query = f"""
            SELECT *
            FROM {FEATURE_SOURCE}
            WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM {FEATURE_SOURCE})
        """
        data = pd.read_sql(query, self.engine).set_index("customer_id")
        logger.info("Fetched %d customer records from %s", len(data), FEATURE_SOURCE)
        return data

    def _extract_embeddings(self, features: pd.DataFrame) -> np.ndarray:
        embedding_columns = [col for col in features.columns if col.startswith("emb_")]
        return features[embedding_columns].to_numpy(dtype=np.float32)

    def _determine_optimal_clusters(self, reduced_features: np.ndarray) -> Tuple[int, Dict[int, float]]:
        rng = np.random.default_rng(SEED)
        indices = rng.choice(len(reduced_features), size=min(SILHOUETTE_SUBSAMPLE, len(reduced_features)), replace=False)
        scores = {}
        for k in CLUSTER_CANDIDATES:
            model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(reduced_features)
            scores[k] = silhouette_score(reduced_features[indices], model.labels_[indices])
            logger.info("k=%d silhouette=%.4f inertia=%.1f", k, scores[k], model.inertia_)
        best_k = max(scores, key=scores.get)
        return best_k, scores

    def _assign_segment_labels(self, behavior: pd.DataFrame, cluster_assignments: np.ndarray) -> Dict[int, str]:
        normalized = (behavior - behavior.mean()) / behavior.std(ddof=0).replace(0, 1)
        centroids = normalized.groupby(cluster_assignments).mean()
        label_map, reserved = {}, set()
        for cluster_id, profile in centroids.iterrows():
            candidate = DEFAULT_LABEL
            for label, conditions in SEGMENT_DEFINITIONS:
                if all(
                    (profile[attr] > threshold) if operator == ">" else (profile[attr] < threshold)
                    for attr, (operator, threshold) in conditions.items()
                ) and label not in reserved:
                    candidate = label
                    break
            if candidate in reserved:
                candidate = f"{candidate} {cluster_id}"
            reserved.add(candidate)
            label_map[cluster_id] = candidate
        logger.info("Final segment mapping: %s", label_map)
        return label_map

    def execute(self, persist: bool = True) -> pd.DataFrame:
        raw = self._fetch_latest_features()
        embeddings = self._extract_embeddings(raw)

        transformer = Pipeline([
            ("normalize", StandardScaler()),
            ("compress", PCA(n_components=PCA_COMPONENT_COUNT, random_state=SEED)),
        ])
        latent_space = transformer.fit_transform(embeddings)
        logger.info("PCA variance explained: %s", np.round(transformer["compress"].explained_variance_ratio_, 3))

        optimal_k, silhouette_history = self._determine_optimal_clusters(latent_space)
        logger.info("Selected k=%d (silhouette=%.4f)", optimal_k, silhouette_history[optimal_k])

        final_model = KMeans(n_clusters=optimal_k, n_init=20, random_state=SEED).fit(latent_space)
        segment_names = self._assign_segment_labels(raw[BEHAVIORAL_INDICATORS], final_model.labels_)
        radial_distances = np.linalg.norm(
            latent_space - final_model.cluster_centers_[final_model.labels_], axis=1
        )

        results = pd.DataFrame({
            "customer_id": raw.index,
            "segment_id": final_model.labels_,
            "segment_name": [segment_names[cid] for cid in final_model.labels_],
            "distance_to_centroid": np.round(radial_distances, 4),
            "k_value": optimal_k,
            "refresh_date": pd.Timestamp.today().normalize(),
        })

        logger.info("Segment distribution:\n%s", results.groupby("segment_name").size().sort_values(ascending=False).to_string())

        if persist:
            schema, table = SEGMENT_OUTPUT.split(".")
            results.to_sql(table, self.engine, schema=schema, if_exists="replace", index=False)
            logger.info("Persisted %d segment assignments to %s", len(results), SEGMENT_OUTPUT)

        return results
