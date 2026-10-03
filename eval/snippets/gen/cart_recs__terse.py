import argparse
import logging
from collections import Counter
from itertools import combinations

import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("cart_rules")

SPT = 15
CNF = 0.02
LFT = 1.5
SPS = 40
MAX_B = 30
TOP = 20

Q = """
SELECT order_id, sku, category_id
FROM orders.lines
WHERE order_date >= CURRENT_DATE - INTERVAL '365 days'
  AND status NOT IN ('cancelled', 'returned')
"""


def _baskets(df, col):
    g = df.groupby("order_id")[col].apply(lambda s: sorted(set(s)))
    return [x for x in g if 1 < len(x) <= MAX_B]


def _freq(b):
    n = len(b)
    c1 = Counter(i for x in b for i in x)
    c2 = Counter(p for x in b for p in combinations(x, 2))
    rows = []
    for (a, b_), cnt in c2.items():
        if cnt < SPT: continue
        for x, y in ((a, b_), (b_, a)):
            conf = cnt / c1[x]
            lift = conf / (c1[y] / n)
            if conf >= CNF and lift >= LFT:
                rows.append((x, y, cnt / n, conf, lift))
    log.info("%d baskets, %d pairs, %d rules", n, sum(v >= SPT for v in c2.values()), len(rows))
    return pd.DataFrame(rows, columns=["a", "c", "sup", "conf", "lift"])


def _cat_backoff(df, ir, cr, sk2c):
    cnt = df.groupby("sku").order_id.nunique()
    sp = cnt[cnt < SPS].index
    top = (df.groupby(["category_id", "sku"]).order_id.nunique()
           .reset_index().sort_values("order_id", ascending=False)
           .groupby("category_id").head(3))
    best = top.groupby("category_id").sku.apply(list).to_dict()
    crg = cr.groupby("a")
    rows = []
    for sku in sp:
        cat = sk2c.get(sku)
        if cat not in crg.groups: continue
        for r in crg.get_group(cat).itertuples():
            for t in best.get(r.c, []):
                rows.append((sku, t, r.sup, r.conf, r.lift * 0.8))
    log.info("backoff: %d rules for %d items", len(rows), len(sp))
    back = pd.DataFrame(rows, columns=ir.columns).assign(level="category")
    return pd.concat([ir.assign(level="item"), back], ignore_index=True)


def _prune_same_cat(rules, sk2c):
    same = rules.a.map(sk2c) == rules.c.map(sk2c)
    log.info("dropped %d same-cat rules", same.sum())
    return rules[~same]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    eng = create_engine(args.dsn)

    df = pd.read_sql(Q, eng)
    sk2c = df.drop_duplicates("sku").set_index("sku").category_id.to_dict()
    ir = _freq(_baskets(df, "sku"))
    cr = _freq(_baskets(df, "category_id"))
    cr = cr[cr.a != cr.c]

    r = _prune_same_cat(_cat_backoff(df, ir, cr, sk2c), sk2c)
    r = (r.sort_values("lift", ascending=False)
         .drop_duplicates(["a", "c"])
         .groupby("a").head(TOP))
    r["built_at"] = pd.Timestamp.utcnow()
    r.to_sql("cart_addons", eng, schema="recs", if_exists="replace", index=False)
    log.info("wrote %d rules", len(r))


if __name__ == "__main__":
    main()
