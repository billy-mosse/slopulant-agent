import numpy as np
import pandas as pd
import logging

log = logging.getLogger("churn")

HALF_LIFE = 90.0
TOP_CATS = ["bedding", "bath", "decor", "kitchen", "rugs", "lighting", "furniture"]
LOOKBACK = 730

ORD_Q = """
SELECT customer_id, order_id, order_ts, category_l1, quantity, net_revenue
FROM orders.lines
WHERE order_ts >= %(start)s AND order_ts < %(cutoff)s
  AND status NOT IN ('cancelled', 'fraud')
"""

SESS_Q = """
SELECT customer_id, session_id, event_ts, event_type
FROM sessions.events
WHERE event_ts >= %(start)s AND event_ts < %(cutoff)s
  AND customer_id IS NOT NULL
"""

def mk_feats(conn, cutoff):
    start = cutoff - pd.Timedelta(days=LOOKBACK)
    p = {"start": start, "cutoff": cutoff}
    log.info(f"orders {start.date()}–{cutoff.date()}")
    o = pd.read_sql(ORD_Q, conn, params=p, parse_dates=["order_ts"])
    log.info(f"loaded {len(o):,} lines, {o.customer_id.nunique():,} custs")

    ord = o.groupby(["customer_id", "order_id"], as_index=False).agg(
        ts=("order_ts", "min"), val=("net_revenue", "sum"), qty=("quantity", "sum")
    )
    rfm = ord.groupby("customer_id").agg(
        last_ts=("ts", "max"), first_ts=("ts", "min"),
        freq=("order_id", "nunique"), tot_val=("val", "sum"),
        avg_val=("val", "mean"), avg_qty=("qty", "mean")
    )
    rfm["recency"] = (cutoff - rfm["last_ts"]).dt.total_seconds() / 86400.0
    rfm["tenure"] = (cutoff - rfm["first_ts"]).dt.total_seconds() / 86400.0
    rfm["log_rec"] = np.log1p(rfm["recency"])
    rfm["log_freq"] = np.log1p(rfm["freq"])
    rfm["log_tot"] = np.log1p(rfm["tot_val"].clip(0))
    rfm["log_avg"] = np.log1p(rfm["avg_val"].clip(0))
    rfm["ord_mo"] = rfm["freq"] / np.maximum(rfm["tenure"] / 30.0, 1.0)

    ord = ord.sort_values(["customer_id", "ts"])
    ord["gap"] = ord.groupby("customer_id")["ts"].diff().dt.total_seconds() / 86400.0
    gaps = ord.groupby("customer_id")["gap"].agg(gap_m=("mean"), gap_s=("std"))
    rfm = rfm.join(gaps)
    rfm["rec_gap"] = rfm["recency"] / np.maximum(rfm["gap_m"].fillna(rfm["recency"]), 1.0)

    age = (cutoff - o["order_ts"]).dt.total_seconds() / 86400.0
    o["w"] = np.power(0.5, age / HALF_LIFE) * o["net_revenue"].clip(0)
    o["cat"] = o["category_l1"].where(o["category_l1"].isin(TOP_CATS), "other")
    cat = o.groupby(["customer_id", "cat"])["w"].sum().unstack(fill_value=0.0)
    tot = cat.sum(axis=1).replace(0, np.nan)
    share = cat.div(tot, axis=0).fillna(0.0).add_prefix("cat_")
    p = share.to_numpy().clip(1e-12, 1.0)
    share["ent"] = -(p * np.log(p)).sum(axis=1)
    share["decayed"] = tot.fillna(0.0)

    log.info("sessions")
    ev = pd.read_sql(SESS_Q, conn, params=p, parse_dates=["event_ts"])
    s = ev.groupby(["customer_id", "session_id"], as_index=False).agg(
        start=("event_ts", "min"),
        evts=("event_type", "size"),
        atc=("event_type", lambda x: (x == "add_to_cart").sum())
    )
    s["age"] = (cutoff - s["start"]).dt.total_seconds() / 86400.0
    sf = s.groupby("customer_id").agg(
        sess_rec=("age", "min"),
        sess_tot=("session_id", "nunique"),
        evts_ps=("evts", "mean"),
        atc_tot=("atc", "sum")
    )
    for w in (7, 30, 90):
        sf[f"sess_{w}d"] = s[s["age"] <= w].groupby("customer_id")["session_id"].nunique()
    sf = sf.fillna({f"sess_{w}d": 0 for w in (7, 30, 90)})
    sf["sess_trend"] = (sf["sess_30d"] + 1) / (sf["sess_90d"] / 3 + 1)
    sf["log_sess_rec"] = np.log1p(sf["sess_rec"])

    f = rfm.join(share, how="left").join(sf, how="left")
    f["sess_rec"] = f["sess_rec"].fillna(LOOKBACK)
    f["log_sess_rec"] = f["log_sess_rec"].fillna(np.log1p(LOOKBACK))
    f = f.fillna(0.0).drop(columns=["last_ts", "first_ts"])
    log.info(f"feats: {f.shape[0]:,} rows × {f.shape[1]} cols")
    return f

def mk_labels(conn, cutoff, h=90):
    end = cutoff + pd.Timedelta(days=h)
    sql = "SELECT DISTINCT customer_id FROM orders.lines WHERE order_ts >= %(start)s AND order_ts < %(cutoff)s"
    b = pd.read_sql(sql, conn, params={"start": cutoff, "cutoff": end})
    log.info(f"{len(b):,} buyers in {h}d label window")
    return pd.Index(b["customer_id"].unique(), name="customer_id")
