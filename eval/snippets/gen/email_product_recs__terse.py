import argparse
import logging
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

log = logging.getLogger("email_recs")

def _txt(s): 
    s = s.fillna("").str.lower()
    s = s.str.replace(r"<[^>]+>", " ", regex=True)
    s = s.str.replace(r"\b\d+\s*(?:tc|thread count)\b", " threadcount ", regex=True)
    return s.str.replace(r"[^a-z0-9 ]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()

def _run(db, dt, dry):
    prods = pd.read_sql("SELECT product_id, title, description, category_l1, is_active, in_stock FROM catalog.products", db)
    log.info(f"loaded {len(prods):,} products")
    prods["t"] = _txt(prods.title) + " " + _txt(prods.title) + " " + _txt(prods.description)
    prods = prods.reset_index(drop=True)
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.5, sublinear_tf=True, stop_words="english")
    M = normalize(vec.fit_transform(prods["t"]))
    pid2r = pd.Series(prods.index, index=prods.product_id)
    ok = (prods.is_active & prods.in_stock).to_numpy()

    lines = pd.read_sql("SELECT customer_id, product_id, order_ts FROM orders.lines WHERE order_ts >= %(start)s AND status = 'completed'", db, params={"start": dt - pd.Timedelta(days=730)}, parse_dates=["order_ts"])
    owned = lines.groupby("customer_id").product_id.agg(set)
    recent = lines[lines.order_ts >= dt - pd.Timedelta(days=30)]
    last = recent.sort_values("order_ts").groupby("customer_id").tail(1).set_index("customer_id")[["product_id", "order_ts"]]
    last = last[last.product_id.isin(pid2r.index)]
    log.info(f"{len(last):,} recent customers")

    anchors = last.product_id.unique()
    nb = {}
    for i in range(0, len(anchors), 2048):
        chunk = anchors[i:i+2048]
        rows = pid2r.loc[chunk].to_numpy()
        sim = (M[rows] @ M.T).toarray()
        sim[:, ~ok] = -1.0
        sim[np.arange(len(rows)), rows] = -1.0
        top = np.argpartition(-sim, kth=min(24, sim.shape[1]-1), axis=1)[:, :24]
        for j, pid in enumerate(chunk):
            idx = top[j][np.argsort(-sim[j, top[j]])]
            nb[pid] = [(prods.product_id.iat[k], float(sim[j, k])) for k in idx if sim[j, k] >= 0.08]
        log.info(f"anchors {i:,}-{i+len(chunk):,}")

    recs = []
    for cid, row in last.iterrows():
        o = owned.get(cid, set())
        cands = [(p, s) for p, s in nb.get(row.product_id, []) if p not in o][:6]
        for r, (p, s) in enumerate(cands, 1):
            recs.append((cid, row.product_id, p, r, round(s, 4)))
    out = pd.DataFrame(recs, columns=["customer_id", "anchor_id", "rec_id", "rank", "sim"])
    out["ts"] = dt

    log.info(f"{len(out):,} recs for {out.customer_id.nunique():,} customers")
    if not dry:
        out.to_sql("email_recs", db, schema="marketing", if_exists="replace", index=False, chunksize=50_000)
        log.info("wrote to marketing.email_recs")
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--asof", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--dry", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _run(sa.create_engine(args.dsn), pd.Timestamp(args.asof), args.dry)

if __name__ == "__main__":
    main()
