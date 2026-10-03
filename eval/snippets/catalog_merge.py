"""Collapse listings that are the same physical item (merchant re-uploads)."""
from collections import defaultdict


def tokens(title):
    return sorted(title.lower().replace("-", " ").replace(",", " ").split())


def token_sort_ratio(a, b):
    ta, tb = tokens(a), tokens(b)
    common = len(set(ta) & set(tb))
    return 2 * common / (len(set(ta)) + len(set(tb)))


def merge_groups(listings, min_ratio=0.85):
    by_brand = defaultdict(list)
    for l in listings:
        by_brand[l["brand"]].append(l)
    pairs = []
    for items in by_brand.values():
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                if token_sort_ratio(a["title"], b["title"]) >= min_ratio:
                    pairs.append((a["id"], b["id"]))
    return pairs
