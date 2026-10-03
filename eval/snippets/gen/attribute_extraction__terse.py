import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import create_engine

from gazetteers import care_gaz, material_gaz, pattern_gaz, size_gaz

log = logging.getLogger(__name__)

IN2CM = 2.54

BL_RE = re.compile(r"(\d{1,3})\s*%\s*([a-z][a-z \-]{1,24}?)(?=\s*(?:\d{1,3}\s*%|[,/;&]|\band\b|$))", re.I)
DM_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:\"|in\.?)?\s*[x×]\s*(\d{1,3}(?:\.\d+)?)\s*(\"|in\.?|cm)\b", re.I)
TH_RE = re.compile(r"\b([1-9]\d{1,3})\s*(?:-|\s)?(?:thread[\s\-]?count|tc)\b", re.I)


def _norm(t: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t or "")).strip()


def _mat(t: str) -> dict | None:
    blends = {}
    for m in BL_RE.finditer(t):
        c = material_gaz.canonical(m.group(2))
        if c:
            blends[c] = blends.get(c, 0) + int(m.group(1))
    if blends:
        return blends, 0.95 if sum(blends.values()) == 100 else 0.7
    hits = {material_gaz.canonical(h[0]) for h in material_gaz.find_all(t)}
    hits.discard(None)
    if hits:
        return {c: None for c in sorted(hits)}, 0.85 if len(hits) == 1 else 0.7
    return None


def _gaz_first(g, t: str, d: str, c: tuple) -> tuple | None:
    for s, x, k in (("t", t, c[0]), ("d", d, c[1])):
        hits = g.find_all(x)
        if hits:
            vals = {g.canonical(h[0]) for h in hits}
            vals.discard(None)
            return next(iter(vals)), k if len(vals) == 1 else k / 2, s
    return None


def _dim(t: str) -> dict | None:
    m = DM_RE.search(t)
    if m:
        w, l, u = float(m.group(1)), float(m.group(2)), m.group(3).lower()
        f = 1.0 if u == "cm" else IN2CM
        return {"width_cm": round(w * f, 1), "length_cm": round(l * f, 1)}, 0.9
    return None


def _tc(t: str) -> int | None:
    m = TH_RE.search(t)
    if m:
        v = int(m.group(1))
        return v, 0.95 if 150 <= v <= 1500 else 0.4
    return None


def _care(d: str) -> list | None:
    hits = {care_gaz.canonical(h[0]) for h in care_gaz.find_all(d)}
    hits.discard(None)
    if hits:
        return sorted(hits), 0.4 if {"dry_clean_only", "machine_wash_cold"} <= hits else 0.8
    return None


def _run(sku: str, ti: str, de: str, th: dict) -> dict:
    ti, de = _norm(ti), _norm(de)
    r = {"sku": sku}
    for n, f in (
        ("material", lambda: _mat(ti) or _mat(de)),
        ("size", lambda: _gaz_first(size_gaz, ti, de, (0.9, 0.7))),
        ("dimensions", lambda: _dim(ti) or _dim(de)),
        ("thread_count", lambda: _tc(ti) or _tc(de)),
        ("care", lambda: _care(de)),
        ("pattern", lambda: _gaz_first(pattern_gaz, ti, de, (0.85, 0.6))),
    ):
        v = f()
        ok = v and v[1] >= th.get(n, 0.5)
        r[n] = json.dumps(v[0]) if isinstance(v[0], (dict, list)) else v[0] if ok else None
        r[f"{n}_confidence"] = round(v[1], 2) if v else None
        r[f"{n}_source"] = v[2] if v and len(v) == 3 else None
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    p.add_argument("--dsn", required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(a.config.read_text())
    sql = f"SELECT {', '.join(cfg['input']['columns'])} FROM {cfg['input']['table']} WHERE {cfg['input']['where']}"
    eng = create_engine(a.dsn)

    frames = []
    for chunk in pd.read_sql(sql, eng, chunksize=cfg["batch_size"]):
        frames.append(pd.DataFrame([_run(r.sku, r.title, r.description, cfg["thresholds"]) for r in chunk.itertuples()]))
        log.info("processed %d rows", sum(len(f) for f in frames))

    out = pd.concat(frames, ignore_index=True)
    out["run_date"] = pd.Timestamp.utcnow().date()
    log.info("coverage: %s", {k: f"{out[k].notna().mean():.1%}" for k in cfg["thresholds"]})

    sch, tbl = cfg["output"]["table"].split(".")
    out.to_sql(tbl, eng, schema=sch, if_exists="replace", index=False)
    log.info("wrote %d rows to %s", len(out), cfg["output"]["table"])


if __name__ == "__main__":
    main()
