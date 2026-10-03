from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger("fraud_networks")

WINDOW_HOURS = 48
MIN_BURST_COUNT = 4
YOUNG_ACCOUNT_DAYS = 30
YOUNG_ACCOUNT_THRESHOLD = 0.5
MAX_SHARED_IP_SIZE = 25
MIN_CLUSTER_SIZE = 3

REVIEWS_QUERY = """
SELECT review_id, reviewer_id, sku, rating, body, created_at, account_created_at
FROM reviews.raw WHERE created_at >= :start_date
"""
DEVICES_QUERY = "SELECT reviewer_id, device_id, ip_hash FROM reviews.reviewer_devices"


class DisjointSet:
    def __init__(self):
        self._roots: dict[str, str] = {}

    def _root(self, x: str) -> str:
        if x not in self._roots:
            self._roots[x] = x
        while self._roots[x] != x:
            self._roots[x] = self._roots[self._roots[x]]
            x = self._roots[x]
        return x

    def merge(self, u: str, v: str) -> None:
        ru, rv = self._root(u), self._root(v)
        if ru != rv:
            self._roots[rv] = ru

    def groups(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for node in self._roots:
            r = self._root(node)
            out.setdefault(r, set()).add(node)
        return out


def _tokenize(text: str) -> list[str]:
    return [w for w in text.lower().split() if w.isalnum() or "'" in w]


def _shingle_set(text: str, k: int = 3) -> set[tuple[str, ...]]:
    tokens = _tokenize(text)
    if len(tokens) < k:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def _hash_perm(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(100)


def _minhash_signature(shingles: set[tuple[str, ...]], seed: int) -> np.ndarray:
    if not shingles:
        return np.full(100, np.inf)
    hashes = np.array([hash(frozenset(s)) for s in shingles], dtype=np.float64)
    perm = _hash_perm(seed)
    sig = np.full(100, np.inf)
    for h in hashes:
        idx = int(h % 100)
        sig[perm[idx]] = min(sig[perm[idx]], h)
    return sig


def _jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _candidate_pairs(texts: dict[str, str]) -> list[tuple[str, str]]:
    shingles_map = {k: _shingle_set(v) for k, v in texts.items()}
    sigs = {k: _minhash_signature(v, 42) for k, v in shingles_map.items() if v}
    buckets: dict[tuple[int, ...], list[str]] = {}
    for key, sig in sigs.items():
        band = tuple(sig[:20].astype(int))
        buckets.setdefault(band, []).append(key)
    pairs = set()
    for group in buckets.values():
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pairs.add(tuple(sorted((group[i], group[j]))))
    return [p for p in pairs if _jaccard(shingles_map[p[0]], shingles_map[p[1]]) >= 0.6]


def _find_bursts(df: pd.DataFrame) -> list[dict]:
    bursts = []
    five_star = df[df["rating"] == 5].sort_values("created_at")
    for sku, group in five_star.groupby("sku"):
        times = group["created_at"].tolist()
        left = 0
        for right in range(len(times)):
            while times[right] - times[left] > pd.Timedelta(hours=WINDOW_HOURS):
                left += 1
            if right - left + 1 >= MIN_BURST_COUNT:
                bursts.append({
                    "sku": sku,
                    "count": right - left + 1,
                    "start": times[left].isoformat(),
                    "end": times[right].isoformat()
                })
                break
    return bursts


def _build_connections(reviews: pd.DataFrame, devices: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    connections: dict[tuple[str, str], set[str]] = {}
    for col, label in [("device_id", "device_match"), ("ip_hash", "ip_match")]:
        grouped = devices.dropna(subset=[col]).groupby(col)["reviewer_id"].apply(list)
        for reviewers in grouped:
            if len(reviewers) < 2:
                continue
            if label == "ip_match" and len(reviewers) > MAX_SHARED_IP_SIZE:
                continue
            anchor = reviewers[0]
            for r in reviewers[1:]:
                key = tuple(sorted((anchor, r)))
                connections.setdefault(key, set()).add(label)
    text_map = dict(zip(reviews["review_id"], reviews["body"].fillna("")))
    owner_map = dict(zip(reviews["review_id"], reviews["reviewer_id"]))
    for a_id, b_id in _candidate_pairs(text_map):
        ra, rb = owner_map[a_id], owner_map[b_id]
        if ra != rb:
            key = tuple(sorted((ra, rb)))
            connections.setdefault(key, set()).add("text_similarity")
    return connections


def scan_for_rings(reviews: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    conn = _build_connections(reviews, devices)
    ds = DisjointSet()
    for (u, v), _ in conn.items():
        ds.merge(u, v)
    clusters = ds.groups().values()

    results = []
    for cluster in clusters:
        if len(cluster) < MIN_CLUSTER_SIZE:
            continue
        sub = reviews[reviews["reviewer_id"].isin(cluster)]
        burst_list = _find_bursts(sub)
        if not burst_list:
            continue
        ages = (sub.groupby("reviewer_id")["created_at"].min() -
                sub.groupby("reviewer_id")["account_created_at"].first()).dt.days
        young_ratio = (ages <= YOUNG_ACCOUNT_DAYS).mean()
        if young_ratio < YOUNG_ACCOUNT_THRESHOLD:
            continue
        edge_counts = Counter()
        for (u, v), labels in conn.items():
            if u in cluster:
                edge_counts.update(labels)
        results.append({
            "reviewers": sorted(cluster),
            "signals": {
                "link_types": dict(edge_counts),
                "burst_events": burst_list,
                "young_account_ratio": round(young_ratio, 2)
            }
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df.insert(0, "cluster_id", [f"cluster_{i:05d}" for i in range(len(df))])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--start", default="2026-06-01")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    engine = create_engine(args.db)
    reviews = pd.read_sql(text(REVIEWS_QUERY), engine, params={"start_date": args.start},
                          parse_dates=["created_at", "account_created_at"])
    devices = pd.read_sql(text(DEVICES_QUERY), engine)

    rings = scan_for_rings(reviews, devices)
    log.info("detected %d rings with %d reviewers total", len(rings),
             sum(len(r) for r in rings["reviewers"]) if len(rings) else 0)

    if not rings.empty:
        rings["reviewers"] = rings["reviewers"].map(json.dumps)
        rings["signals"] = rings["signals"].map(json.dumps)
        rings.to_sql("fraud_clusters", engine, schema="reviews", if_exists="append", index=False)


if __name__ == "__main__":
    main()
