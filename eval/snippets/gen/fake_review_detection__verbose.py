"""Module for identifying coordinated synthetic review activity patterns.

This functionality supports our fraud intelligence program by detecting clusters
of accounts that collaborate to inflate product ratings. A coordinated group is
identified when multiple accounts demonstrate behavioral alignment through
shared infrastructure signals (device fingerprints, network endpoints), text
similarities, and coordinated rating patterns—specifically, a concentrated burst
of five-star ratings for the same product within a short timeframe, combined
with a high proportion of recently created accounts in the group.

Outputs are passed downstream to our trust-and-safety workflow systems for
investigation and remediation.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

import pandas as pd
from sqlalchemy import create_engine, text

from integrity.minhash import CandidatePairFinder

logger = logging.getLogger("fraud.intelligence")

BURST_DURATION = pd.Timedelta(hours=48)
MINIMUM_BURST_VOLUME = 4
ACCOUNT_YOUNG_WINDOW_DAYS = 30
RECENT_ACCOUNT_THRESHOLD = 0.5
IP_BURST_FILTER = 25
MINIMAL_CLUSTER_SIZE = 3

SQL_REVIEWS = """
SELECT review_identifier, account_identifier, product_identifier, star_rating,
       content_text, submission_timestamp, registration_timestamp
FROM review_platform.raw_reviews
WHERE submission_timestamp >= :cutoff_date
"""

SQL_DEVICES = """
SELECT account_identifier, device_fingerprint, network_endpoint
FROM review_platform.account_metadata
"""


class AccountGraph:
    def __init__(self):
        self._links: dict[str, str] = {}

    def _root(self, node: str) -> str:
        self._links.setdefault(node, node)
        while self._links[node] != node:
            self._links[node] = self._links[self._links[node]]
            node = self._links[node]
        return node

    def merge(self, node_a: str, node_b: str) -> None:
        root_a, root_b = self._root(node_a), self._root(node_b)
        if root_a != root_b:
            self._links[root_b] = root_a

    def clusters(self) -> dict[str, set[str]]:
        grouping: dict[str, set[str]] = defaultdict(set)
        for account in self._links:
            grouping[self._root(account)].add(account)
        return grouping


def link_candidates(reviews: pd.DataFrame, devices: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    adjacency: dict[tuple[str, str], set[str]] = defaultdict(set)

    for signal, label in (("device_fingerprint", "device_fingerprint"), ("network_endpoint", "network_endpoint")):
        filtered = devices[devices[signal].notna()]
        for fingerprint, group in filtered.groupby(signal)["account_identifier"]:
            if len(group) < 2:
                continue
            if label == "network_endpoint" and len(group) > IP_BURST_FILTER:
                continue
            seed = next(iter(group))
            for peer in group:
                if peer == seed:
                    continue
                pair = tuple(sorted((seed, peer)))
                adjacency[pair].add(label)

    reviewer_map = dict(zip(reviews["review_identifier"], reviews["content_text"].fillna("")))
    owner_map = dict(zip(reviews["review_identifier"], reviews["account_identifier"]))

    finder = CandidatePairFinder(min_jaccard=0.6)
    for left_id, right_id, _ in finder.matches(reviewer_map):
        left_author, right_author = owner_map[left_id], owner_map[right_id]
        if left_author != right_author:
            pair = tuple(sorted((left_author, right_author)))
            adjacency[pair].add("text_similarity")

    return adjacency


def _detect_bursts(reviews: pd.DataFrame) -> list[dict]:
    bursts = []
    high_rated = reviews[reviews["star_rating"] == 5].sort_values("submission_timestamp")
    for product, subset in high_rated.groupby("product_identifier"):
        timestamps = subset["submission_timestamp"].tolist()
        left = 0
        for right in range(len(timestamps)):
            while timestamps[right] - timestamps[left] > BURST_DURATION:
                left += 1
            if right - left + 1 >= MINIMUM_BURST_VOLUME:
                bursts.append({
                    "product": product,
                    "high_ratings_in_window": right - left + 1,
                    "window_start": str(timestamps[left]),
                    "window_end": str(timestamps[right])
                })
                break
    return bursts


def assess_accounts(reviews: pd.DataFrame, clusters: dict[str, set[str]]) -> list[dict]:
    results = []
    for cluster_id, cluster_accounts in clusters.items():
        if len(cluster_accounts) < MINIMAL_CLUSTER_SIZE:
            continue
        subset = reviews[reviews["account_identifier"].isin(cluster_accounts)]

        burst_signals = _detect_bursts(subset)
        if not burst_signals:
            continue

        account_tenure = (subset.groupby("account_identifier")["submission_timestamp"].min()
                          - subset.groupby("account_identifier")["registration_timestamp"].first())
        new_account_ratio = (account_tenure.dt.days <= ACCOUNT_YOUNG_WINDOW_DAYS).mean()
        if new_account_ratio < RECENT_ACCOUNT_THRESHOLD:
            continue

        link_counts = defaultdict(int)
        for (a, b), labels in link_candidates(reviews, reviews.merge(
            pd.DataFrame({"account_identifier": list(cluster_accounts),
                          "device_fingerprint": [None] * len(cluster_accounts),
                          "network_endpoint": [None] * len(cluster_accounts)}),
            on="account_identifier"
        )).items():
            if a in cluster_accounts:
                for k in labels:
                    link_counts[k] += 1

        results.append({
            "cluster_accounts": sorted(cluster_accounts),
            "signals": {
                "linkage_types": dict(link_counts),
                "rating_bursts": burst_signals,
                "recent_account_fraction": round(new_account_ratio, 2)
            }
        })
    return results


def identify_coordinated_activity(review_table: pd.DataFrame, device_table: pd.DataFrame) -> pd.DataFrame:
    graph = AccountGraph()
    for pair, _ in link_candidates(review_table, device_table).items():
        graph.merge(pair[0], pair[1])

    clusters = graph.clusters()
    findings = assess_accounts(review_table, clusters)

    output = pd.DataFrame(findings)
    if len(output):
        output.insert(0, "coordinated_group_id",
                      [f"COORD-{i:06d}" for i in range(len(output))])
    return output


def execute(dsn: str, cutoff: str) -> None:
    logging.basicConfig(level=logging.INFO)
    engine = create_engine(dsn)

    review_data = pd.read_sql(
        text(SQL_REVIEWS),
        engine,
        params={"cutoff_date": cutoff},
        parse_dates=["submission_timestamp", "registration_timestamp"]
    )
    device_data = pd.read_sql(text(SQL_DEVICES), engine)
    findings = identify_coordinated_activity(review_data, device_data)

    logger.info("identified %d clusters encompassing %d accounts",
                len(findings),
                sum(len(g["cluster_accounts"]) for g in findings.to_dict("records")))

    if len(findings):
        findings["cluster_accounts"] = findings["cluster_accounts"].map(json.dumps)
        findings["signals"] = findings["signals"].map(json.dumps)
        findings.to_sql("coordinated_activity_clusters", engine, schema="trust_safety",
                        if_exists="append", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect coordinated synthetic review patterns")
    parser.add_argument("--database", required=True, help="PostgreSQL connection string")
    parser.add_argument("--since", default="2026-06-01", help="Start date for analysis window")
    args = parser.parse_args()
    execute(args.database, args.since)
