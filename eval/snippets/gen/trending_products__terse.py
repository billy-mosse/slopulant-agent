import argparse
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import islice

import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger("trend")

W = {"view": 1.0, "cart": 4.0, "buy": 10.0}
HALF = 6.0
H48 = 48
D28 = 28
K = 50
MIN_B = 20
MIN_R = 5
MIN_S = 10

Q1 = """
SELECT sku, category_id, event_type, event_ts
FROM sessions.events
WHERE event_type IN ('view', 'cart') AND event_ts >= :t0
"""
Q2 = """
SELECT sku, category_id, 'buy' AS event_type, order_ts AS event_ts
FROM orders.lines
WHERE order_ts >= :t0 AND status <> 'cancelled'
"""


def fetch(dsn, t0):
    since = t0 - timedelta(days=D28)
    eng = create_engine(dsn)
    with eng.connect() as c:
        ev = pd.read_sql(text(Q1), c, params={"t0": since})
        po = pd.read_sql(text(Q2), c, params={"t0": since})
    df = pd.concat([ev, po], ignore_index=True)
    df.event_ts = pd.to_datetime(df.event_ts)
    df["w"] = df.event_type.map(W)
    return df


def score(df, now):
    cut = now - timedelta(hours=H48)
    age = (now - df.event_ts).dt.total_seconds() / 3600
    df = df.assign(d=df.w * (0.5 ** (age / HALF)), r=df.event_ts >= cut)

    base = df[~df.r]
    nwin = (D28 * 24 - H48) / H48
    win = ((cut - base.event_ts).dt.total_seconds() // (H48 * 3600)).astype(int)
    per = base.assign(wi=win).groupby(["sku", "wi"]).w.sum()
    agg = defaultdict(lambda: [0.0, 0.0, 0])
    for (s, _), v in per.items():
        a = agg[s]
        a[0] += v
        a[1] += v * v
        a[2] += 1

    rec = df[df.r].groupby(["sku", "category_id"]).agg(d=("d", "sum"), n=("w", "size"))
    bn = base.groupby("sku").size()
    norm = HALF / math.log(2) / H48 * (1 - 0.5 ** (H48 / HALF))

    rows = []
    for (s, c), r in rec.iterrows():
        t, sq, _ = agg[s]
        mu = t / nwin
        var = max(sq / nwin - mu * mu, 0.0)
        sd = math.sqrt(var + mu + 1.0)
        cur = r.d / norm
        rows.append((s, c, cur, mu, (cur - mu) / sd, r.n, int(bn.get(s, 0))))
    return pd.DataFrame(rows, columns=["sku", "cat", "cur", "mu", "z", "rn", "bn"])


def filter(v, st):
    ok = (v.rn >= MIN_R) & (v.bn >= MIN_B) & (v.z > 0)
    if st is not None:
        ok &= v.sku.map(st).fillna(0) >= MIN_S
    log.info("kept %d/%d", ok.sum(), len(v))
    return v[ok]


def rank(v, k=K):
    out = []
    for cat, g in v.sort_values("z", ascending=False).groupby("cat", sort=False):
        for i, row in enumerate(islice(g.itertuples(), k), 1):
            out.append({"cat": cat, "sku": row.sku, "rank": i,
                        "z": round(row.z, 3), "cur": round(row.cur, 2),
                        "mu": round(row.mu, 2)})
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--stock")
    p.add_argument("--ts")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    now = datetime.fromisoformat(a.ts) if a.ts else datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    st = pd.read_parquet(a.stock).set_index("sku").units_on_hand if a.stock else None
    res = rank(filter(score(fetch(a.dsn, now), now), st))
    res["ts"] = now
    eng = create_engine(a.dsn)
    res.to_sql("trending", eng, schema="recs", if_exists="append", index=False)
    log.info("recs.trending: %d rows, %d cats", len(res), res.cat.nunique())


if __name__ == "__main__":
    main()
