from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from minhash import near_dupes

EDGE_TYPES = {"dev": "shared_device", "ip": "shared_ip", "txt": "near_duplicate_text"}
BURST_HRS = 48
BURST_MIN = 4
YOUNG_DAYS = 30
YOUNG_FRAC = 0.5
MAX_IP_SIZE = 25
RING_MIN = 3

REV_Q = """SELECT review_id, reviewer_id, sku, rating, body, created_at, account_created_at
FROM reviews.raw WHERE created_at >= :since"""
DEV_Q = "SELECT reviewer_id, device_hash, ip_address FROM reviews.reviewer_devices"


class DSU:
    def __init__(self):
        self.p: Dict[str, str] = {}

    def root(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def join(self, a: str, b: str):
        ra, rb = self.root(a), self.root(b)
        if ra != rb:
            self.p[rb] = ra


def mk_edges(rv: pd.DataFrame, dv: pd.DataFrame) -> Dict[Tuple[str, str], Set[str]]:
    e: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for col, tag in (("device_hash", "dev"), ("ip_address", "ip")):
        for _, grp in dv.dropna(subset=[col]).groupby(col)["reviewer_id"]:
            m = grp.unique()
            if len(m) < 2 or (tag == "ip" and len(m) > MAX_IP_SIZE):
                continue
            for x in m[1:]:
                e[tuple(sorted((m[0], x)))].add(tag)
    o = dict(zip(rv["review_id"], rv["reviewer_id"]))
    for a, b, _ in near_dupes(dict(zip(rv["review_id"], rv["body"].fillna("")))):
        if (ra := o[a]) != o[b]:
            e[tuple(sorted((ra, o[b])))].add("txt")
    return e


def burst_check(df: pd.DataFrame) -> List[Dict]:
    hits = []
    for sku, g in df[df["rating"] == 5].sort_values("created_at").groupby("sku"):
        ts = g["created_at"].tolist()
        lo = 0
        for hi in range(len(ts)):
            while ts[hi] - ts[lo] > pd.Timedelta(hours=BURST_HRS):
                lo += 1
            if hi - lo + 1 >= BURST_MIN:
                hits.append({"sku": sku, "five_star_in_48h": hi - lo + 1,
                             "start": str(ts[lo]), "end": str(ts[hi])})
                break
    return hits


def go(rv: pd.DataFrame, dv: pd.DataFrame) -> pd.DataFrame:
    edges = mk_edges(rv, dv)
    dsu = DSU()
    for a, b in edges:
        dsu.join(a, b)
    comps: Dict[str, Set[str]] = defaultdict(set)
    for n in dsu.p:
        comps[dsu.root(n)].add(n)

    rows = []
    for m in comps.values():
        if len(m) < RING_MIN:
            continue
        sub = rv[rv["reviewer_id"].isin(m)]
        bursts = burst_check(sub)
        age = (sub.groupby("reviewer_id")["created_at"].min()
               - sub.groupby("reviewer_id")["account_created_at"].first()).dt.days
        young = float((age <= YOUNG_DAYS).mean()) if len(age) else 0.0
        if not bursts or young < YOUNG_FRAC:
            continue
        cnt = defaultdict(int)
        for (a, b), lbls in edges.items():
            if a in m:
                for l in lbls:
                    cnt[l] += 1
        rows.append({"members": sorted(m), "evidence": {
            "edge_types": dict(cnt), "bursts": bursts, "new_account_share": round(young, 2)}})
    out = pd.DataFrame(rows)
    if len(out):
        out.insert(0, "ring_id", [f"ring_{i:05d}" for i in range(len(out))])
    return out


def main():
    import argparse, logging, json
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--since", default="2026-06-01")
    args = ap.parse_args()
    eng = create_engine(args.dsn)
    rv = pd.read_sql(text(REV_Q), eng, params={"since": args.since},
                     parse_dates=["created_at", "account_created_at"])
    dv = pd.read_sql(text(DEV_Q), eng)
    rings = go(rv, dv)
    logging.info("flagged %d rings covering %d reviewers", len(rings),
                 sum(len(m) for m in rings["members"]) if len(rings) else 0)
    if len(rings):
        rings["members"] = rings["members"].map(json.dumps)
        rings["evidence"] = rings["evidence"].map(json.dumps)
        rings.to_sql("suspicious_rings", eng, schema="reviews", if_exists="append", index=False)


if __name__ == "__main__":
    main()
