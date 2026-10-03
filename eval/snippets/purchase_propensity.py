"""Live p(purchase) for the current browsing session."""
import math


def featurize(session):
    views = [e for e in session["events"] if e["type"] == "pageview"]
    carts = sum(e["type"] == "cart_add" for e in session["events"])
    secs = sum(e.get("seconds", 0) for e in views)
    return [len(views), carts, math.log1p(secs), int(session.get("returning", False))]


def propensity(session, w=(0.2, 1.4, 0.3, 0.5), b=-3.2):
    z = b + sum(wi * xi for wi, xi in zip(w, featurize(session)))
    return 1 / (1 + math.exp(-z))
