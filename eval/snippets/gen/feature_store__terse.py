import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd
import yaml

log = logging.getLogger("mats")

DEF_PATH = Path(__file__).parent / "defs.yml"
FUNCS = {"count", "count_distinct", "sum", "avg"}
NULL_TOL = 0.05

class DB(Protocol):
    def exec(self, q: str, p: dict | None = None) -> pd.DataFrame: ...
    def upsert(self, t: str, df: pd.DataFrame, part: str) -> None: ...

@dataclass(frozen=True)
class Agg:
    col: str
    fn: str
    out: str

@dataclass(frozen=True)
class Src:
    tbl: str
    ts: str
    aggs: list[Agg]
    flt: str | None = None

@dataclass(frozen=True)
class View:
    nm: str
    key: str
    sink: str
    wins: list[int]
    ttl: int
    srcs: list[Src]
    static_tbl: str | None = None
    static_cols: list[str] = field(default_factory=list)

class BadDef(ValueError): pass

def _load(path: Path) -> list[View]:
    cfg = yaml.safe_load(path.read_text())
    d = cfg.get("defaults", {})
    out = []
    for v in cfg["feature_views"]:
        srcs = []
        for s in v["sources"]:
            ags = [Agg(**a) for a in s["aggregations"]]
            bad = [a.fn for a in ags if a.fn not in FUNCS]
            if bad: raise BadDef(f"{v['name']}: {bad}")
            srcs.append(Src(s["table"], s["timestamp_column"], ags, s.get("filter")))
        st = v.get("static") or {}
        out.append(View(
            v["name"], v["entity"], v["sink"],
            v.get("windows", d["windows"]),
            v.get("ttl_days", d["ttl_days"]),
            srcs,
            st.get("table"),
            st.get("columns", []),
        ))
    log.info("loaded %d views", len(out))
    return out

def _pull(db: DB, s: Src, k: str, d: date, w: int) -> pd.DataFrame:
    q = f"SELECT * FROM {s.tbl} WHERE {s.ts} >= %(st)s AND {s.ts} < %(en)s"
    df = db.exec(q, {"st": d - timedelta(days=w), "en": d})
    if s.flt: df = df.query(s.flt)
    return df

def _roll(df: pd.DataFrame, s: Src, k: str, d: date, w: int) -> pd.DataFrame:
    cut = pd.Timestamp(d - timedelta(days=w))
    ts = pd.to_datetime(df[s.ts])
    sub = df[(ts >= cut) & (ts < pd.Timestamp(d))]
    grp = sub.groupby(k)
    res = {}
    for a in s.aggs:
        nm = f"{a.out}_{w}d"
        fn = {"count": grp[a.col].count, "count_distinct": grp[a.col].nunique,
              "sum": grp[a.col].sum, "avg": grp[a.col].mean}[a.fn]
        res[nm] = fn()
    return pd.DataFrame(res)

def _build(db: DB, v: View, d: date) -> pd.DataFrame:
    wmax = max(v.wins)
    parts = []
    for s in v.srcs:
        df = _pull(db, s, v.key, d, wmax)
        log.info("%s: %d rows from %s", v.nm, len(df), s.tbl)
        parts += [_roll(df, s, v.key, d, w) for w in v.wins]
    feat = pd.concat(parts, axis=1).fillna(0)
    if v.static_tbl:
        st = db.exec(f"SELECT {v.key}, {', '.join(v.static_cols)} FROM {v.static_tbl}")
        feat = feat.join(st.set_index(v.key), how="left")
    feat = feat.reset_index().rename(columns={"index": v.key})
    feat["as_of"] = d
    feat["expires"] = d + timedelta(days=v.ttl)
    return feat

def _chk(v: View, df: pd.DataFrame) -> None:
    if df[v.key].isna().any(): raise ValueError(f"{v.nm}: null keys")
    if df[v.key].duplicated().any(): raise ValueError(f"{v.nm}: dup keys")
    exp = {f"{a.out}_{w}d" for s in v.srcs for a in s.aggs for w in v.wins}
    miss = exp - set(df.columns)
    if miss: raise ValueError(f"{v.nm}: missing {sorted(miss)}")
    nr = df.isna().mean()
    bad = nr[nr > NULL_TOL]
    if not bad.empty: raise ValueError(f"{v.nm}: null rate {bad.to_dict()}")

def go(db: DB, d: date, only: list[str] | None = None, dry: bool = False) -> None:
    for v in _load(DEF_PATH):
        if only and v.nm not in only: continue
        df = _build(db, v, d)
        _chk(v, df)
        log.info("%s: %d rows, %d cols", v.nm, len(df), df.shape[1])
        if not dry: db.upsert(v.sink, df, d.isoformat())

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    p.add_argument("--view", action="append")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from warehouse import connect
    go(connect(), args.as_of, args.view, args.dry_run)

if __name__ == "__main__":
    main()
