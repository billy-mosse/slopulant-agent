import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SRC = "features.customer_embeddings"
DST = "marketing.segments"
K_MIN, K_MAX = 4, 10
PCA_DIM = 5
SAMP = 20_000
SEED = 42

PRF = ["recency_days", "orders_12m", "spend_12m", "share_bedding", "share_bath", "share_decor"]

RULES = [
    ("Lapsed high spenders", {"spend_12m": (">", 0.5), "recency_days": (">", 0.75)}),
    ("VIP regulars",         {"spend_12m": (">", 1.0), "orders_12m": (">", 1.0)}),
    ("Bedding loyalists",    {"share_bedding": (">", 0.75)}),
    ("Bath enthusiasts",     {"share_bath": (">", 0.75)}),
    ("Decor browsers",       {"share_decor": (">", 0.75), "orders_12m": ("<", 0.0)}),
    ("New & curious",        {"recency_days": ("<", -0.5), "orders_12m": ("<", -0.25)}),
    ("Dormant one-timers",   {"recency_days": (">", 0.5), "orders_12m": ("<", -0.25)}),
]

def _load(dsn):
    engine = dsn if isinstance(dsn, object) else __import__("sqlalchemy").create_engine(dsn)
    q = f"SELECT * FROM {SRC} WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM {SRC})"
    df = pd.read_sql(q, engine).set_index("customer_id")
    logger.info("Loaded %d rows", len(df))
    return df

def _embed(df):
    cols = [c for c in df.columns if c.startswith("emb_")]
    return df[cols].to_numpy(np.float32)

def _best_k(X):
    r = np.random.default_rng(SEED)
    idx = r.choice(len(X), min(SAMP, len(X)), replace=False)
    scores = {}
    for k in range(K_MIN, K_MAX+1):
        m = KMeans(k, n_init=10, random_state=SEED).fit(X)
        scores[k] = silhouette_score(X[idx], m.labels_[idx])
        logger.info("k=%d sil=%.4f", k, scores[k])
    return max(scores, key=scores.get), scores

def _name(p, lbl):
    z = (p - p.mean()) / p.std(ddof=0).replace(0, 1)
    centroids = z.groupby(lbl).mean()
    names, used = {}, set()
    for cid, row in centroids.iterrows():
        n = "Steady mainstream"
        for cand, conds in RULES:
            ok = all((row[c] > t) if op == ">" else (row[c] < t) for c, (op, t) in conds.items())
            if ok and cand not in used:
                n = cand
                break
        if n in used: n = f"{n} {cid}"
        used.add(n); names[cid] = n
    logger.info("Segments: %s", names)
    return names

def go(dsn, save=True):
    df = _load(dsn)
    X = _embed(df)
    pipe = Pipeline([("sc", StandardScaler()), ("pca", PCA(PCA_DIM, random_state=SEED))])
    Z = pipe.fit_transform(X)
    logger.info("Var ratio: %s", np.round(pipe["pca"].explained_variance_ratio_, 3))
    k, _ = _best_k(Z)
    km = KMeans(k, n_init=20, random_state=SEED).fit(Z)
    nms = _name(df[PRF], km.labels_)
    d = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
    out = pd.DataFrame({
        "customer_id": df.index,
        "segment_id": km.labels_,
        "segment_name": [nms[c] for c in km.labels_],
        "dist_to_ctr": np.round(d, 4),
        "k": k,
        "snapshot_date": pd.Timestamp.today().normalize(),
    })
    print(out.groupby("segment_name").size().sort_values(ascending=False).to_string())
    if save:
        sc, tb = DST.split(".")
        out.to_sql(tb, dsn if isinstance(dsn, object) else __import__("sqlalchemy").create_engine(dsn), schema=sc, if_exists="replace", index=False)
        logger.info("Wrote %d rows to %s", len(out), DST)
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    go(a.dsn, save=not a.dry)
