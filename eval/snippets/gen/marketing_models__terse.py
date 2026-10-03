import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("promo")

DB_URI = "postgresql://analytics@warehouse/slopulent"
SQL = """
    SELECT customer_id, promo_id, sent_at, discount_pct, redeemed, last_order_date
    FROM marketing.promo_history
    WHERE sent_at >= CURRENT_DATE - INTERVAL '540 days'
"""
OUT_TBL = "coupon_propensity"
DEPTHS = [10, 15, 20, 25]
HOLDOUT = 60
FEATS = ["sends", "redems", "rate", "discount_pct", "days_since_order", "days_since_redem", "avg_depth"]


def pull(engine):
    df = pd.read_sql(SQL, engine, parse_dates=["sent_at", "last_order_date"])
    df.sort_values(["customer_id", "sent_at"], inplace=True)
    log.info("fetched %d rows, %d customers", len(df), df["customer_id"].nunique())
    return df


def enrich(df):
    g = df.groupby("customer_id")
    df["sends"] = g.cumcount()
    df["redems"] = g["redeemed"].cumsum() - df["redeemed"]
    df["rate"] = (df["redems"] + 1) / (df["sends"] + 4)
    df["days_since_order"] = (df["sent_at"] - df["last_order_date"]).dt.days.clip(lower=0)

    red_ts = df["sent_at"].where(df["redeemed"] == 1)
    df["prev_redem"] = red_ts.groupby(df["customer_id"]).shift(1).ffill()
    df["days_since_redem"] = (df["sent_at"] - df["prev_redem"]).dt.days

    depth = df["discount_pct"].where(df["redeemed"] == 1)
    df["avg_depth"] = depth.groupby(df["customer_id"]).transform(lambda s: s.shift(1).expanding().mean())
    return df


def mk_model():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), FEATS),
    ])
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000)
    return Pipeline([("pre", pre), ("clf", clf)])


def split(df, days=HOLDOUT):
    cut = df["sent_at"].max() - pd.Timedelta(days=days)
    return df[df["sent_at"] < cut], df[df["sent_at"] >= cut]


def eval_(m, d):
    p = m.predict_proba(d[FEATS])[:, 1]
    log.info("AUC=%.3f Brier=%.4f base=%.3f",
             roc_auc_score(d["redeemed"], p), brier_score_loss(d["redeemed"], p), d["redeemed"].mean())
    dec = pd.qcut(p, 10, labels=False, duplicates="drop")
    print(d.assign(dec=dec).groupby("dec")["redeemed"].mean().rename("rate"))


def latest(df, now):
    last = df.groupby("customer_id").tail(1).copy()
    last["sends"] += 1
    last["redems"] += last["redeemed"]
    last["rate"] = (last["redems"] + 1) / (last["sends"] + 4)
    last["days_since_order"] = (now - last["last_order_date"]).dt.days.clip(lower=0)
    last["days_since_redem"] = (now - last["prev_redem"]).dt.days
    return last


def score(m, st):
    rows = []
    for d in DEPTHS:
        s = st.assign(discount_pct=d)
        rows.append(pd.DataFrame({
            "customer_id": s["customer_id"].to_numpy(),
            "discount_pct": d,
            "p_redeem": m.predict_proba(s[FEATS])[:, 1],
        }))
    out = pd.concat(rows, ignore_index=True)
    out["scored_at"] = datetime.utcnow()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout", type=int, default=HOLDOUT)
    p.add_argument("--dry", action="store_true")
    args = p.parse_args()

    eng = create_engine(DB_URI)
    df = enrich(pull(eng))
    tr, te = split(df, args.holdout)

    m = mk_model().fit(tr[FEATS], tr["redeemed"])
    eval_(m, te)

    m.fit(df[FEATS], df["redeemed"])
    sc = score(m, latest(df, pd.Timestamp.utcnow().tz_localize(None)))
    log.info("scored %d customers x %d depths, mean p=%.3f",
             sc["customer_id"].nunique(), len(DEPTHS), np.mean(sc["p_redeem"]))

    if not args.dry:
        sc.to_sql(OUT_TBL, eng, schema="marketing", if_exists="replace", index=False)
        log.info("wrote %s", OUT_TBL)


if __name__ == "__main__":
    main()
