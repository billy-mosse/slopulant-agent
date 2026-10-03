import json
import math
import logging
from collections import Counter

import numpy as np
import pandas as pd
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

FEAT_KEYS = [
    "emb_cos", "bm25", "cat_match", "smooth_ctr", "price_z", "rating", "log_rev"
]


def _cos_emb(df, pe, qe):
    p_emb = pe.set_index("sku")["embedding"].map(np.asarray)
    q_emb = qe.set_index("query_text")["embedding"].map(np.asarray)
    pv = np.stack(df["sku"].map(p_emb).values)
    qv = np.stack(df["query_text"].map(q_emb).values)
    df["emb_cos"] = ((pv * qv).sum(1) / (np.linalg.norm(pv, axis=1) * np.linalg.norm(qv, axis=1) + 1e-9))
    return df


def _bm25_t(df, k1, b):
    t = df.drop_duplicates("sku").set_index("sku")["title"].fillna("").str.lower().str.split()
    n = len(t)
    avgdl = t.map(len).mean() or 1.0
    dfreq = Counter(t for tok in t for t in set(tok))
    idf = {w: math.log((n - c + 0.5) / (c + 0.5) + 1) for w, c in dfreq.items()}
    def s(q, sk):
        toks = t.get(sk, [])
        tf = Counter(toks)
        norm = k1 * (1 - b + b * len(toks) / avgdl) if toks else k1
        return sum(idf.get(w, 0) * tf.get(w, 0) * (k1 + 1) / (tf.get(w, 0) + norm) for w in q.lower().split() if w in tf)
    df["bm25"] = [s(q, s_) for q, s_ in zip(df["query_text"], df["sku"])]
    return df


def _cat_match(df, intents):
    im = {r.query_text: json.loads(r.top_categories) for r in intents.itertuples()}
    df["cat_match"] = [im.get(q, {}).get(c, 0.0) for q, c in zip(df["query_text"], df["category_id"])]
    return df


def _sctr(df, clicks, a, b):
    agg = clicks.groupby(["query_text", "sku"]).agg(c=("clicked", "sum"), n=("clicked", "size")).reset_index()
    agg["smooth_ctr"] = (agg["c"] + a) / (agg["n"] + a + b)
    df = df.merge(agg[["query_text", "sku", "smooth_ctr"]], on=["query_text", "sku"], how="left")
    df["smooth_ctr"] = df["smooth_ctr"].fillna(a / (a + b))
    return df


def _price_z(df):
    g = df.groupby("query_id")["price"]
    df["price_z"] = ((df["price"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    return df


def _review(df):
    df["rating"] = df["avg_rating"].fillna(df["avg_rating"].median())
    df["log_rev"] = np.log1p(df["review_count"].fillna(0))
    return df


def prep(cfg, eng):
    r = cfg["tables"]
    days = cfg["data"]["lookback_days"]
    cl = pd.read_sql(f"SELECT * FROM {r['click_logs']} WHERE event_date >= CURRENT_DATE - {days} AND position <= {cfg['data']['max_position']}", eng)
    pe = pd.read_sql(f"SELECT sku, embedding FROM {r['product_embeddings']}", eng)
    qi = pd.read_sql(f"SELECT query_text, top_categories FROM {r['query_intents']}", eng)
    return {"cl": cl, "pe": pe, "qi": qi}


def build(ds, cfg):
    fc = cfg["features"]
    df = ds["cl"].copy()
    qe = df.drop_duplicates("query_text")[["query_text", "query_embedding"]].rename(columns={"query_embedding": "embedding"})
    df = _cos_emb(df, ds["pe"], qe)
    df = _bm25_t(df, fc["bm25_k1"], fc["bm25_b"])
    df = _cat_match(df, ds["qi"])
    df = _sctr(df, ds["cl"], fc["ctr_prior_clicks"], fc["ctr_prior_impressions"])
    df = _price_z(df)
    df = _review(df)
    return df


def _pos_curve(df):
    c = df.groupby("position")["clicked"].mean()
    c = c.rolling(3, min_periods=1, center=True).mean()
    return (c / c.iloc[0]).clip(lower=0.02)


def mk_lgb(df, prop):
    df = df.sort_values(["query_id", "position"])
    grps = df.groupby("query_id", sort=False).size().values
    ipw = np.where(df["clicked"], 1.0 / df["position"].map(prop).fillna(prop.min()).values, 1.0)
    ipw = np.clip(ipw, 1.0, 20.0)
    return lgb.Dataset(df[FEAT_KEYS], label=df["clicked"].astype(int), group=grps, weight=ipw)


def run(cfg, dsn, out):
    eng = create_engine(dsn)
    ds = prep(cfg, eng)
    df = build(ds, cfg)
    cnt = df.groupby("query_id").size()
    df = df[df["query_id"].isin(cnt[cnt >= cfg["data"]["min_impressions_per_query"]].index)]
    cutoff = df["event_date"].max() - pd.Timedelta(days=cfg["validation"]["holdout_days"])
    tr, va = df[df["event_date"] < cutoff], df[df["event_date"] >= cutoff]
    prop = _pos_curve(tr)
    params = {k: v for k, v in cfg["lightgbm"].items() if k not in ("num_boost_round", "early_stopping_rounds")}
    dtr, dva = mk_lgb(tr, prop), mk_lgb(va, prop)
    model = lgb.train(params, dtr, cfg["lightgbm"]["num_boost_round"],
                      [dva], callbacks=[lgb.early_stopping(cfg["lightgbm"]["early_stopping_rounds"]), lgb.log_evaluation(50)])
    log.info("Best iter %d, ndcg@10 %.4f", model.best_iteration, model.best_score["valid_0"]["ndcg@10"])
    (out / "ranker.txt").write_bytes(model.model_to_string().encode())
    imp = {k: round(v, 2) for k, v in zip(FEAT_KEYS, model.feature_importance("gain"))}
    (out / "fi.json").write_text(json.dumps(imp, indent=2))
    pd.DataFrame([{
        "model_path": str(out / "ranker.txt"), "best_iter": model.best_iteration,
        "ndcg@10": model.best_score["valid_0"]["ndcg@10"], "fi": json.dumps(imp),
        "ts": pd.Timestamp.utcnow()
    }]).to_sql(cfg["tables"]["output"].split(".")[1], eng, schema=cfg["tables"]["output"].split(".")[0], if_exists="append", index=False)
