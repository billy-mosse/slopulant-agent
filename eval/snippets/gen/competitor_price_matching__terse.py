import argparse
import logging
import re
import string
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine

log = logging.getLogger("cmp_match")

SQL_PRODS = "SELECT sku, title, brand, category, size, color, list_price FROM catalog.products WHERE is_active"
SQL_LISTS = "SELECT listing_id, competitor, title, brand, category, price, scraped_at FROM pricing.competitor_listings WHERE scraped_at >= CURRENT_DATE - INTERVAL '7 days'"
OUT_TBL = "pricing.competitor_matches"

THRESH = 0.78
SIG = 0.45
W_T, W_A, W_P = 0.70, 0.15, 0.15

PAT = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:inches?|in\b|\")"), r"\1in"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:centimeters?|cm)\b"), r"\1cm"),
    (re.compile(r"(\d+)\s*(?:thread\s*count|tc)\b"), r"\1tc"),
    (re.compile(r"(\d+)\s*(?:pieces?|pcs?|pc)\b"), r"\1pc"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:ounces?|oz)\b"), r"\1oz"),
]
SZ_V = {"twin", "full", "queen", "king", "cal king", "standard", "euro"}
CL_V = {"white", "ivory", "black", "grey", "gray", "navy", "blue", "green", "sage", "blush", "pink", "beige", "taupe", "charcoal", "natural", "linen"}
RP = re.compile(f"[{re.escape(string.punctuation.replace(chr(34), ''))}]")

def norm_t(t):
    t = (t or "").lower().replace("grey", "gray").replace("california king", "cal king")
    for p, r in PAT: t = p.sub(r, t)
    t = RP.sub(" ", t).replace('"', " ")
    return re.sub(r"\s+", " ", t).strip()

def tok_r(a, b):
    sa, sb = set(a.split()), set(b.split())
    i = " ".join(sorted(sa & sb))
    da = " ".join(sorted(sa - sb))
    db = " ".join(sorted(sb - sa))
    return max(SequenceMatcher(None, x, y).ratio() for x, y in ((i, f"{i} {da}".strip()), (i, f"{i} {db}".strip()), (f"{i} {da}".strip(), f"{i} {db}".strip())))

def get_attr(t, v):
    h = [w for w in v if re.search(rf"\b{re.escape(w)}\b", t)]
    return max(h, key=len) if h else None

def attr_s(ours, col, t):
    s = []
    for o, v in ((ours, SZ_V), (col, CL_V)):
        th = get_attr(t, v)
        if not o or th is None: s.append(0.5)
        else: s.append(1.0 if norm_t(str(o)) == th else 0.0)
    return 0.0 if 0.0 in s else float(np.mean(s))

def price_s(p1, p2):
    if not p1 or not p2 or p1 <= 0 or p2 <= 0: return 0.0
    z = abs(np.log(p2 / p1) / SIG)
    return float(2 * stats.norm.sf(z))

def blk(d): return d["brand"].fillna("").str.lower().str.strip() + "|" + d["category"].fillna("").str.lower()

def run(prods, lists, th=THRESH):
    prods = prods.assign(n=prods["title"].map(norm_t), b=blk(prods))
    lists = lists.assign(n=lists["title"].map(norm_t), b=blk(lists))
    rows = []
    for b, grp in lists.groupby("b"):
        o = prods[prods["b"] == b]
        if o.empty: continue
        for c in grp.itertuples(index=False):
            best = None
            for p in o.itertuples(index=False):
                ts = tok_r(p.n, c.n)
                if ts < 0.5: continue
                as_ = attr_s(p.size, p.color, c.n)
                ps = price_s(p.list_price, c.price)
                sc = W_T * ts + W_A * as_ + W_P * ps
                if best is None or sc > best["sc"]:
                    best = {"sku": p.sku, "comp": c.competitor, "lid": c.listing_id,
                            "c_price": c.price, "o_price": p.list_price,
                            "ts": ts, "as": as_, "ps": ps, "sc": sc}
            if best and best["sc"] >= th: rows.append(best)
    df = pd.DataFrame(rows)
    if df.empty: return df
    df = df.sort_values("sc", ascending=False).drop_duplicates(["sku", "comp"])
    df["gap"] = df["c_price"] - df["o_price"]
    df["gap_pct"] = df["gap"] / df["o_price"]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--th", type=float, default=THRESH)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    eng = create_engine(args.dsn)
    prods = pd.read_sql(SQL_PRODS, eng)
    lists = pd.read_sql(SQL_LISTS, eng)
    log.info("loaded %d products, %d listings", len(prods), len(lists))

    res = run(prods, lists, args.th)
    if res.empty:
        log.warning("no matches above %.2f", args.th)
        return
    g = res["gap_pct"]
    log.info("%d matches; median gap %.1f%%, IQR [%.1f%%, %.1f%%]", len(res),
             100 * g.median(), 100 * g.quantile(0.25), 100 * g.quantile(0.75))
    if args.dry:
        print(res.head(30).to_string(index=False))
        return
    res["ts"] = pd.Timestamp.utcnow()
    sch, tbl = OUT_TBL.split(".")
    res.to_sql(tbl, eng, schema=sch, if_exists="replace", index=False)
    log.info("wrote %s", OUT_TBL)

if __name__ == "__main__":
    main()
