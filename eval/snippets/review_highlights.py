"""Pros/cons blurb on the PDP built from customer reviews."""
from collections import defaultdict

PROS = {"soft", "cozy", "warm", "sturdy", "beautiful"}
CONS = {"thin", "itchy", "shrank", "faded", "pilled"}


def highlights(reviews):
    by_item = defaultdict(lambda: {"pros": defaultdict(int), "cons": defaultdict(int), "stars": []})
    for r in reviews:
        h = by_item[r["item_id"]]
        h["stars"].append(r["stars"])
        for w in r["text"].lower().split():
            w = w.strip(".,!?")
            if w in PROS:
                h["pros"][w] += 1
            elif w in CONS:
                h["cons"][w] += 1
    return {
        item: {
            "stars": round(sum(h["stars"]) / len(h["stars"]), 1),
            "pros": sorted(h["pros"], key=h["pros"].get, reverse=True)[:2],
            "cons": sorted(h["cons"], key=h["cons"].get, reverse=True)[:2],
        }
        for item, h in by_item.items()
    }
