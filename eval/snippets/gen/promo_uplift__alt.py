import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MARGIN = 22.0
DISCOUNT_COST = 1.80

HIST_Q = """
SELECT customer_id, campaign_id, treated, converted, sent_at
FROM marketing.promo_history
WHERE is_randomized = TRUE
"""

FEAT_Q = "SELECT * FROM features.customer_embeddings"
OUT_TBL = "marketing.promo_uplift"
SEED = 17
HOLDOUT_RATIO = 0.25
GB_PARAMS = dict(n_estimators=250, learning_rate=0.04, max_depth=6,
                 min_samples_leaf=80, subsample=0.8, random_state=SEED)


def ingest_data(db_url):
    eng = create_engine(db_url)
    hist = pd.read_sql(HIST_Q, eng, parse_dates=["sent_at"])
    feats = pd.read_sql(FEAT_Q, eng)
    hist = hist.sort_values("sent_at").drop_duplicates("customer_id", keep="last")
    data = hist.merge(feats, on="customer_id", how="inner")
    logger.info("loaded %d samples (treated %.1f%%, baseline conv %.2f%%)",
                len(data), 100 * data["treated"].mean(), 100 * data["converted"].mean())
    return data, feats


def select_features(df):
    exclude = {"customer_id", "campaign_id", "treated", "converted", "sent_at", "updated_at"}
    return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]


class DoubleML:
    """Meta-learner using regression on potential outcomes."""

    def __init__(self, params=None):
        self.params = params or GB_PARAMS
        self.treat_mod = GradientBoostingRegressor(**self.params)
        self.ctrl_mod = GradientBoostingRegressor(**self.params)
        self.cols = None

    def train(self, X, z, y):
        self.cols = list(X.columns)
        mask = z.astype(bool).values
        self.treat_mod.fit(X[mask], y[mask])
        self.ctrl_mod.fit(X[~mask], y[~mask])
        logger.info("trained outcome models: treat=%d, ctrl=%d obs",
                    mask.sum(), (~mask).sum())
        return self

    def score(self, X):
        X = X[self.cols]
        pred_t = self.treat_mod.predict(X)
        pred_c = self.ctrl_mod.predict(X)
        return pd.DataFrame({
            "p_treat": pred_t,
            "p_control": pred_c,
            "delta": pred_t - pred_c
        }, index=X.index)


def partition(df, frac=HOLDOUT_RATIO):
    strat = (df["treated"].astype(str) + "_" + df["converted"].astype(str))
    return train_test_split(df, test_size=frac, stratify=strat, random_state=SEED)


def cumulative_uplift(df, score="delta"):
    df = df.sort_values(score, ascending=False).reset_index(drop=True)
    treat = (df["treated"] == 1).astype(int).values
    conv = df["converted"].values
    n_t = np.cumsum(treat)
    n_c = np.cumsum(1 - treat)
    y_t = np.cumsum(conv * treat)
    y_c = np.cumsum(conv * (1 - treat))
    q = y_t - np.where(n_c > 0, y_c * n_t / n_c, 0.0)
    return pd.DataFrame({
        "prop": np.arange(1, len(df) + 1) / len(df),
        "qini": q,
        "score": df[score].values,
        "n_treat": n_t,
        "n_ctrl": n_c
    })


def normalized_auc(curve):
    base = curve["qini"].iloc[-1]
    baseline = curve["prop"] * base
    return float(np.trapz(curve["qini"] - baseline, curve["prop"]) / max(len(curve), 1))


def profit_optimize(curve, pop_size, margin=MARGIN, cost=DISCOUNT_COST):
    scale = pop_size / len(curve)
    incr = curve["qini"] * scale
    target_cnt = curve["prop"] * pop_size
    profit = incr * margin - target_cnt * cost
    idx = int(np.argmax(profit.values))
    return {
        "target_share": float(curve["prop"].iloc[idx]),
        "score_thresh": float(curve["score"].iloc[idx]),
        "inc_conversions": float(incr.iloc[idx]),
        "proj_profit": float(profit.iloc[idx]),
        "customers": int(target_cnt.iloc[idx])
    }


def decile_analysis(df):
    d = df.copy()
    d["decile"] = pd.qcut(d["delta"].rank(method="first", ascending=False), 10,
                          labels=range(1, 11), duplicates="drop")
    agg = d.groupby(["decile", "treated"])["converted"].mean().unstack(fill_value=0)
    agg.columns = ["ctrl_conv", "treat_conv"]
    agg["obs_delta"] = agg["treat_conv"] - agg["ctrl_conv"]
    agg["pred_delta"] = d.groupby("decile")["delta"].mean()
    return agg


def run_pipeline(db, model_file, holdout_file, pop_size, skip_write):
    data, feats = ingest_data(db)
    feats_cols = select_features(data)
    train, test = partition(data)

    learner = DoubleML(GB_PARAMS).train(train[feats_cols], train["treated"], train["converted"])
    learner.save = lambda p: __import__("joblib").dump(learner, p)
    learner.save(model_file)

    holdout_scores = test[["customer_id", "treated", "converted"]].join(learner.score(test[feats_cols]))
    holdout_scores.to_parquet(holdout_file, index=False)
    logger.info("holdout avg uplift: pred=%.4f, obs=%.4f",
                holdout_scores["delta"].mean(),
                holdout_scores.loc[holdout_scores.treated == 1, "converted"].mean()
                - holdout_scores.loc[holdout_scores.treated == 0, "converted"].mean())

    curve = cumulative_uplift(holdout_scores)
    logger.info("normalized AUUC: %.4f", normalized_auc(curve))

    print(decile_analysis(holdout_scores).round(4).to_string())

    opt = profit_optimize(curve, pop_size)
    logger.info("optimal policy: top %.1f%% (delta ≥ %.4f), %d customers, profit $%.0f",
                100 * opt["target_share"], opt["score_thresh"],
                opt["customers"], opt["proj_profit"])

    full_scores = feats[["customer_id"]].join(learner.score(feats[feats_cols].fillna(0)))
    full_scores["scored_at"] = pd.Timestamp.utcnow()
    logger.info("scored %d customers; %.1f%% positive delta",
                len(full_scores), 100 * (full_scores["delta"] > 0).mean())

    if not skip_write:
        schema, tbl = OUT_TBL.split(".")
        full_scores.to_sql(tbl, create_engine(db), schema=schema, if_exists="replace", index=False)
        logger.info("saved to %s", OUT_TBL)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--model-path", default="uplift_model.joblib")
    p.add_argument("--holdout-path", default="holdout_scored.parquet")
    p.add_argument("--pop-size", type=int, default=1_200_000)
    p.add_argument("--no-upload", action="store_true")
    args = p.parse_args()
    run_pipeline(args.dsn, args.model_path, args.holdout_path, args.pop_size, args.no_upload)
