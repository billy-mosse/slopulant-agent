import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from taxonomy import PARENT_OF_L2, PARENT_OF_L3
from train import mk_text

logger = logging.getLogger("infer")

SRC = "catalog.products"
DST = "catalog.predicted_category"


def best_valid(probs, labels, par_map, par):
    mask = np.array([par_map.get(c) == par for c in labels])
    if not mask.any():
        return None, 0.0
    cand = probs * mask
    s = cand.sum()
    if s == 0:
        return None, 0.0
    idx = int(cand.argmax())
    return labels[idx], float(cand[idx] / s)


def run_hier(models, txt, thresh):
    p1, p2, p3 = (models[f"l{i}"].predict_proba(txt) for i in (1, 2, 3))
    c1, c2, c3 = (models[f"l{i}"].classes_ for i in (1, 2, 3))
    out = []
    for i in range(len(txt)):
        j = int(p1[i].argmax())
        lv1, c1v = c1[j], float(p1[i][j])
        lv2, c2v = best_valid(p2[i], c2, PARENT_OF_L2, lv1)
        if lv2 is None or c2v < thresh:
            out.append((lv1, None, None, c1v, c2v, None, 1))
            continue
        lv3, c3v = best_valid(p3[i], c3, PARENT_OF_L3, lv2)
        if lv3 is None or c3v < thresh:
            out.append((lv1, lv2, None, c1v, c2v, c3v, 2))
            continue
        out.append((lv1, lv2, lv3, c1v, c2v, c3v, 3))
    return pd.DataFrame(out, columns=["l1", "l2", "l3", "c1", "c2", "c3", "depth"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--mdl", required=True)
    p.add_argument("--min-child", type=float, default=0.55)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    eng = create_engine(a.dsn)
    bundle = joblib.load(a.mdl)
    df = pd.read_sql(f"SELECT sku, title, description FROM {SRC} WHERE is_active", eng)
    logger.info("scoring %d items", len(df))

    res = run_hier(bundle["models"], mk_text(df), a.min_child)
    res.insert(0, "sku", df["sku"].values)
    res["path"] = res[["l1", "l2", "l3"]].apply(lambda r: " > ".join(x for x in r if x), axis=1)
    res["ts"] = pd.Timestamp.utcnow()
    logger.info("depth dist: %s", res["depth"].value_counts(normalize=True).round(3).to_dict())

    sch, tbl = DST.split(".")
    res.to_sql(tbl, eng, schema=sch, if_exists="replace", index=False)
    logger.info("wrote %d rows to %s", len(res), DST)


if __name__ == "__main__":
    main()
