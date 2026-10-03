"""Customer profile vectors: recency, frequency, spend and category mix."""
import math
from collections import Counter

CATS = ["bedding", "bath", "decor", "kitchen", "furniture"]


def profile(orders, today):
    last = min(today - o["date"] for o in orders)
    spend = sum(o["total"] for o in orders)
    mix = Counter(o["category"] for o in orders)
    share = [mix[c] / len(orders) for c in CATS]
    return [math.exp(-last / 30), math.log1p(len(orders)), math.log1p(spend)] + share


def build_profiles(orders_by_customer, today):
    return {c: profile(o, today) for c, o in orders_by_customer.items()}
