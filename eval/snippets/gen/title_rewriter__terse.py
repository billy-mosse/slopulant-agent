import argparse
import json
import logging
import re
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine

from templates import MAX_LEN, SlotConfig, get_cfg

log = logging.getLogger(__name__)

SRC_PROD = "catalog.products"
SRC_ATTR = "catalog.product_attributes"
SRC_COLR = "catalog.product_colors"
TGT_TBL = "catalog.seo_titles"

CONF_MIN = 0.75
SHARE_MIN = 0.30

SPAM = re.compile(
    r"(!{2,}|\bfree\s+shipping\b|\bsale\b|\bclearance\b|\bbest\s+seller\b|\bhot\s+deal\b|\blimited\s+time\b|\bnew!?\b)",
    re.I,
)
LITTLE = {"a", "an", "and", "the", "of", "in", "with", "for", "x", "by"}
UNIT = re.compile(r"^\d+(\.\d+)?(cm|in|tc)?$", re.I)


@dataclass
class OutRow:
    t: str
    f: bool
    m: str


def clean(t: str) -> str:
    t = SPAM.sub(" ", t or "")
    t = re.sub(r"\s*([,|-])\s*(?=[,|-]|$)", "", t)
    return re.sub(r"\s{2,}", " ", t).strip(" ,-|")


def uniq(t: str) -> str:
    seen, out = set(), []
    for w in t.split():
        k = re.sub(r"\W", "", w).lower()
        if k and k in seen:
            continue
        seen.add(k)
        out.append(w)
    return " ".join(out)


def caps(t: str) -> str:
    ws = t.split()
    for i, w in enumerate(ws):
        c = w.strip(",")
        if UNIT.match(c) or re.search(r"\d", c):
            continue
        if i > 0 and c.lower() in LITTLE:
            ws[i] = w.lower()
        else:
            ws[i] = w[:1].upper() + w[1:].lower()
    return " ".join(ws)


def fmt_val(k: str, v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    if k == "thread_count":
        return f"{int(v)} Thread Count"
    if k == "material" and isinstance(v, str) and v.startswith("{"):
        blend = json.loads(v)
        top = sorted(blend.items(), key=lambda kv: -(kv[1] or 0))
        return "-".join(x for x, _ in top[:2]) + (" Blend" if len(top) > 1 else "")
    if k == "dimensions" and isinstance(v, str) and v.startswith("{"):
        d = json.loads(v)
        return f"{round(d['width_cm'])}x{round(d['length_cm'])}cm"
    return str(v)


def fill(cfg: SlotConfig, s: dict[str, str | None]) -> str:
    active = {k: v for k, v in s.items() if v}
    drop = list(cfg.drop)
    while True:
        txt = cfg.pat
        for slot in cfg.sl:
            txt = txt.replace(f"{{{slot}}}", active.get(slot, ""))
        txt = caps(uniq(clean(re.sub(r"\s+", " ", txt))))
        if len(txt) <= MAX_LEN or not drop:
            return txt[:MAX_LEN].rstrip(" ,")
        active.pop(drop.pop(0), None)


def gen_title(r: pd.Series) -> OutRow:
    orig = clean(r["title"])
    cfg = get_cfg(r.get("category_l2"))
    s: dict[str, str | None] = {}
    for n in cfg.sl:
        c = r.get(f"{n}_confidence")
        if c is not None and not pd.isna(c) and c < CONF_MIN:
            s[n] = None
            continue
        s[n] = fmt_val(n, r.get(n))
    sh = r.get("share_1", 0)
    s["color"] = fmt_val("color", r.get("color_1")) if sh >= SHARE_MIN else None

    miss = [x for x in cfg.req if not s.get(x)]
    if miss:
        return OutRow(orig, True, "miss:" + ",".join(miss))
    return OutRow(fill(cfg, s), False, "ok")


def pull(e) -> pd.DataFrame:
    q = f"""
        SELECT *
        FROM {SRC_PROD}
        LEFT JOIN {SRC_ATTR} USING (sku)
        LEFT JOIN {SRC_COLR} USING (sku)
        WHERE is_active
    """
    return pd.read_sql(q, e)


def run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--dry", action="store_true")
    arg = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(msg)s")

    eng = create_engine(arg.dsn)
    df = pull(eng)
    log.info("loaded %d rows", len(df))

    rs = [gen_title(r) for _, r in df.iterrows()]
    out = pd.DataFrame({
        "sku": df["sku"],
        "orig": df["title"],
        "seo": [x.t for x in rs],
        "tpl": df["category_l2"].fillna("base"),
        "orig_kept": [x.f for x in rs],
        "msg": [x.m for x in rs],
    })
    log.info("rewrote %.1f%%", 100 * (~out["orig_kept"]).mean())

    if arg.dry:
        print(out.sample(min(25, len(out))).to_string(index=False))
        return
    sch, tbl = TGT_TBL.split(".")
    out.to_sql(tbl, eng, schema=sch, if_exists="replace", index=False)
    log.info("wrote %d rows to %s", len(out), TGT_TBL)


if __name__ == "__main__":
    run()
