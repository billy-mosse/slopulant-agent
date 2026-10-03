import json
import logging
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker

logger = logging.getLogger("search_ranker")


def load_data(dsn: str, cfg: dict) -> dict:
    t = cfg["tables"]
    days = cfg["data"]["lookback_days"]
    max_pos = cfg["data"]["max_position"]
    engine = create_engine(dsn)

    logs = pd.read_sql(
        f"SELECT query_id, query_text, sku, position, clicked, title, category_id, price, "
        f"avg_rating, review_count, query_embedding FROM {t['click_logs']} "
        f"WHERE event_date >= CURRENT_DATE - {days} AND position <= {max_pos}",
        engine,
    )
    prod_emb = pd.read_sql(f"SELECT sku, embedding FROM {t['product_embeddings']}", engine)
    q_intents = pd.read_sql(f"SELECT query_text, top_categories FROM {t['query_intents']}", engine)

    logger.info("Loaded %d impressions, %d product embeddings, %d query intents", len(logs), len(prod_emb), len(q_intents))
    return {"logs": logs, "prod_emb": prod_emb, "intents": q_intents}


def cosine_sim(df: pd.DataFrame, prod: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing semantic similarity")
    prod_map = prod.set_index("sku")["embedding"].apply(np.array)
    query_map = query.set_index("query_text")["embedding"].apply(np.array)

    p_vecs = np.stack(df["sku"].map(prod_map).values)
    q_vecs = np.stack(df["query_text"].map(query_map).values)

    dot = np.einsum("ij,ij->i", p_vecs, q_vecs)
    norm = np.linalg.norm(p_vecs, axis=1) * np.linalg.norm(q_vecs, axis=1) + 1e-9
    df["semantic_sim"] = dot / norm
    return df


def title_bm25(df: pd.DataFrame, k1: float, b: float) -> pd.DataFrame:
    logger.info("Calculating BM25 scores")
    titles = df.drop_duplicates("sku").set_index("sku")["title"].fillna("").str.lower().str.split()
    n = len(titles)
    avg_len = titles.map(len).mean() or 1.0
    dfreq = defaultdict(int)
    for toks in titles:
        for t in set(toks):
            dfreq[t] += 1

    idf = {t: math.log((n - dfreq[t] + 0.5) / (dfreq[t] + 0.5) + 1) for t in dfreq}
    logger.info("Vocabulary size: %d, avg doc length: %.1f", len(idf), avg_len)

    def bm25_score(q: str, sku: str) -> float:
        toks = titles.get(sku, [])
        tf = defaultdict(int)
        for t in toks:
            tf[t] += 1
        denom = k1 * (1 - b + b * len(toks) / avg_len)
        score = 0.0
        for term in q.lower().split():
            if term in tf:
                score += idf.get(term, 0.0) * tf[term] * (k1 + 1) / (tf[term] + denom)
        return score

    df["title_score"] = [bm25_score(q, s) for q, s in zip(df["query_text"], df["sku"])]
    return df


def category_affinity(df: pd.DataFrame, intents: pd.DataFrame) -> pd.DataFrame:
    logger.info("Aggregating intent-category alignment")
    intent_map = {
        row.query_text: json.loads(row.top_categories)
        for row in intents.itertuples()
    }
    df["intent_match"] = [
        intent_map.get(q, {}).get(c, 0.0)
        for q, c in zip(df["query_text"], df["category_id"])
    ]
    coverage = (df["intent_match"] > 0).mean() * 100
    logger.info("Intent coverage: %.1f%%", coverage)
    return df


def ctr_adjusted(df: pd.DataFrame, prior_clicks: float, prior_impr: float) -> pd.DataFrame:
    logger.info("Applying Bayesian CTR smoothing")
    agg = df.groupby(["query_text", "sku"]).agg(
        clicks=("clicked", "sum"),
        impressions=("clicked", "count")
    ).reset_index()
    agg["smooth_ctr"] = (agg["clicks"] + prior_clicks) / (agg["impressions"] + prior_impr)
    df = df.merge(agg[["query_text", "sku", "smooth_ctr"]], on=["query_text", "sku"], how="left")
    df["smooth_ctr"].fillna(prior_clicks / prior_impr, inplace=True)
    return df


def normalize_price(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Normalizing price within query context")
    scaler = StandardScaler()
    df["price_norm"] = df.groupby("query_id")["price"].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).ravel() if len(x) > 1 else 0.0
    ).fillna(0.0)
    return df


def review_features(df: pd.DataFrame) -> pd.DataFrame:
    df["rating"] = df["avg_rating"].fillna(df["avg_rating"].median())
    df["log_reviews"] = np.log1p(df["review_count"].fillna(0))
    return df


def build_dataset(df: pd.DataFrame, cfg: dict) -> tuple:
    df = df.sort_values(["query_id", "position"])
    groups = df.groupby("query_id", sort=False).size().values

    # Inverse position weighting for clicks
    pos_rate = df.groupby("position")["clicked"].mean()
    pos_rate = pos_rate.rolling(3, min_periods=1, center=True).mean()
    prop = pos_rate / pos_rate.iloc[0]
    prop = prop.clip(lower=0.02)

    ipw = np.where(df["clicked"], 1.0 / df["position"].map(prop).fillna(prop.min()), 1.0)
    ipw = np.clip(ipw, 1.0, 20.0)

    features = ["semantic_sim", "title_score", "intent_match", "smooth_ctr", "price_norm", "rating", "log_reviews"]
    X = df[features].values
    y = df["clicked"].astype(int).values
    w = ipw

    logger.info("Dataset: %d samples, %d groups, mean IPW=%.2f", len(X), len(groups), w.mean())
    return X, y, w, groups


def train_ranker(X_train, y_train, w_train, groups_train,
                 X_val, y_val, w_val, groups_val, params: dict) -> XGBRanker:
    model = XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg@10",
        **{k: v for k, v in params.items() if k != "num_boost_round"}
    )
    model.fit(
        X_train, y_train, group=groups_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_group=[groups_val],
        sample_weight_eval_set=[w_val],
        verbose=False,
        early_stopping_rounds=params["early_stopping_rounds"]
    )
    return model


def main(dsn: str, config_path: str, output_dir: str) -> None:
    cfg = yaml.safe_load(open(config_path))
    frames = load_data(dsn, cfg)

    # Feature engineering
    q_emb = frames["logs"].drop_duplicates("query_text")[["query_text", "query_embedding"]].rename(
        columns={"query_embedding": "embedding"}
    )
    df = cosine_sim(frames["logs"].copy(), frames["prod_emb"], q_emb)
    df = title_bm25(df, cfg["features"]["bm25_k1"], cfg["features"]["bm25_b"])
    df = category_affinity(df, frames["intents"])
    df = ctr_adjusted(df, cfg["features"]["ctr_prior_clicks"], cfg["features"]["ctr_prior_impressions"])
    df = normalize_price(df)
    df = review_features(df)

    # Filter and split
    min_impr = cfg["data"]["min_impressions_per_query"]
    valid_queries = df.groupby("query_id").size()
    df = df[df["query_id"].isin(valid_queries[valid_queries >= min_impr].index)]

    cutoff = df["event_date"].max() - pd.Timedelta(days=cfg["validation"]["holdout_days"])
    train_df, val_df = df[df["event_date"] < cutoff], df[df["event_date"] >= cutoff]

    # Prepare datasets
    X_tr, y_tr, w_tr, g_tr = build_dataset(train_df, cfg)
    X_va, y_va, w_va, g_va = build_dataset(val_df, cfg)

    # Train
    params = cfg["lightgbm"].copy()
    params["num_boost_round"] = params.pop("num_boost_round", 600)
    params["early_stopping_rounds"] = params.pop("early_stopping_rounds", 50)

    model = train_ranker(X_tr, y_tr, w_tr, g_tr, X_va, y_va, w_va, g_va, params)
    best_iter = model.best_iteration
    ndcg = model.evals_result()["validation_0"]["ndcg@10"][best_iter - 1]

    # Save
    import os
    os.makedirs(output_dir, exist_ok=True)
    model.save_model(f"{output_dir}/model.json")

    feat_imp = {f: round(float(v), 2) for f, v in zip(
        ["semantic_sim", "title_score", "intent_match", "smooth_ctr", "price_norm", "rating", "log_reviews"],
        model.feature_importances_
    )}
    with open(f"{output_dir}/importance.json", "w") as f:
        json.dump(feat_imp, f, indent=2)

    logger.info("Trained model: best_iter=%d, val_ndcg@10=%.4f", best_iter, ndcg)
