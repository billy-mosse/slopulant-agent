import argparse
import json
import logging
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("hp")

def sig(z): return 1.0 / (1.0 + np.exp(-z / 2.0))

def _mmr(cands, rel, k, λ=0.7):
    sel, rem = [], set(range(len(cands)))
    while rem and len(sel) < k:
        def g(i):
            sim = max((1.0 if cands[i].cat == cands[j].cat else 0.0 for j in sel), default=0.0)
            return λ * rel[i] - (1 - λ) * sim
        b = max(rem, key=g)
        sel.append(b)
        rem.remove(b)
    return sel

def _mk_cands(df, trend_z, cat_map, mod=None):
    if mod:
        df = df[df.mod == mod]
    return [(r.sku, r.cat, trend_z.get(r.sku, 0.0), 0.0) for r in df.itertuples()]

def _i2i_agg(i2i_df, seeds, cat_map, trend_z):
    agg = {}
    for s in seeds:
        for t, sc in i2i_df.get(s, []):
            if t not in seeds:
                agg[t] = agg.get(t, 0.0) + sc
    return [(t, cat_map.get(t, -1), trend_z.get(t, 0.0), sc) for t, sc in agg.items()]

def _score(cands, mix, idx):
    trend = sig(np.array([c[2] for c in cands]))
    if mix is None:
        return trend
    aff = np.array([mix[idx.get(c[1], -1)] if c[1] in idx else 0.0 for c in cands])
    aff = aff / (aff.max() or 1.0)
    prior = np.array([c[3] for c in cands])
    prior = prior / (prior.max() or 1.0)
    return 0.65 * np.maximum(aff, prior) + 0.35 * trend

def _rank(mix, seeds, pools, idx, cat_idx, i2i_map, trend_z):
    cold = mix is None or mix.sum() < 0.05
    p = {"trending": pools["trending"]}
    if not cold:
        p["because_viewed"] = _i2i_agg(i2i_map, seeds, cat_idx, trend_z)
        p["new_arr"] = pools["new_arr"]
        p["sale"] = pools["sale"]
    res = []
    for mod in ("trending", "because_viewed", "new_arr", "sale"):
        cands = p.get(mod, [])
        if not cands:
            continue
        rel = _score(cands, None if cold else mix, idx)
        pick = _mmr(cands, rel, 12)
        res.append((mod, [cands[i][0] for i in pick], float(rel[pick].mean())))
    return sorted(res, key=lambda x: -x[2])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--merch", required=True)
    p.add_argument("--cats", required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    eng = create_engine(a.dsn)
    cats = json.load(open(a.merch))
    cat_idx = {c: i for i, c in enumerate(cats)}
    idx = {c: i for i, c in enumerate(cats)}

    tr = pd.read_sql("SELECT sku,cat,vel FROM recs.trend WHERE ts=(SELECT max(ts) FROM recs.trend)", eng)
    i2i = pd.read_sql("SELECT src,tgt,score FROM recs.i2i", eng)
    merch = pd.read_parquet(a.merch)
    prof = pd.read_sql("SELECT cid,cat_mix,seeds FROM features.profiles", eng)

    trend_z = dict(zip(tr.sku, tr.vel))
    cat_map = {**dict(zip(merch.sku, merch.cat)), **dict(zip(tr.sku, tr.cat))}
    pools = {
        "trending": _mk_cands(tr, trend_z, cat_map),
        "new_arr": _mk_cands(merch, trend_z, cat_map, "new_arr"),
        "sale": _mk_cands(merch, trend_z, cat_map, "sale"),
    }
    i2i_map = i2i.groupby("src").apply(lambda g: list(zip(g.tgt, g.score))).to_dict()

    rows = []
    for r in prof.itertuples():
        mix = np.frombuffer(r.cat_mix, dtype=float) if r.cat_mix else None
        for rank, (mod, skus, str) in enumerate(_rank(mix, list(r.seeds or []), pools, idx, cat_idx, i2i_map, trend_z), 1):
            rows.append((r.cid, mod, rank, json.dumps(skus)))
    out = pd.DataFrame(rows, columns=["cid", "mod", "rank", "skus"])
    out.to_sql("hp_slots", eng, schema="recs", if_exists="replace", index=False, chunksize=50_000)
    log.info("done: %d customers, %d rows", len(prof), len(out))

if __name__ == "__main__":
    main()
