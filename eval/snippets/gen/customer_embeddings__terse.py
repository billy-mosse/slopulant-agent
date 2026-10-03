import numpy as np

CATS = ["bedding", "bath", "decor", "kitchen", "bath"]
HALF_LIFE = 90

def _vec(lines):
    v = np.zeros(len(CATS))
    for r in lines:
        if r["cat"] in CATS:
            v[CATS.index(r["cat"])] += 0.5 ** (r["days"] / HALF_LIFE)
    if v.sum():
        v /= v.sum()
    r = min(r["days"] for r in lines)
    f = len(lines)
    m = sum(r["amt"] for r in lines)
    rfm = np.array([np.exp(-r / 30), np.log1p(f), np.log1p(m)])
    return np.hstack([rfm, v])

def _group(lines):
    grp = {}
    for r in lines:
        grp.setdefault(r["cid"], []).append(r)
    return {c: _vec(xs) for c, xs in grp.items()}

if __name__ == "__main__":
    data = [
        {"cid": "C1", "sku": "BED-001", "cat": "bedding", "amt": 189.0, "days": 10},
        {"cid": "C1", "sku": "BTH-010", "cat": "bath", "amt": 34.0, "days": 200},
    ]
    print({k: v.round(3).tolist() for k, v in _group(data).items()})
