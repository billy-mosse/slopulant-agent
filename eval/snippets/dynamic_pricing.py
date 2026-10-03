"""Recommend a launch price for new SKUs from their attributes (ridge regression)."""
import numpy as np

ATTRS = ["linen", "cotton", "duvet", "towel", "pillow", "bath", "bed"]


def x(attrs):
    s = {a.lower().rstrip("s") for a in attrs}
    return np.array([1.0] + [float(a in s) for a in ATTRS])


def train(history, lam=1.0):
    X = np.stack([x(h["attrs"]) for h in history])
    y = np.array([h["price"] for h in history])
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def recommend(w, attrs):
    return round(float(x(attrs) @ w), 2)
