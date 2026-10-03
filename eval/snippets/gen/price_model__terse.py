import numpy as np

F = ["bedding", "bath", "linen", "cotton", "towel", "duvet", "pillow"]

def _norm(t):
    t = t.strip().lower()
    return {"linens": "linen", "towels": "towel", "bed": "bedding", "pillows": "pillow"}.get(t, t)

def _vec(tags):
    s = {_norm(x) for x in tags}
    return np.array([1.0] + [1.0 if f in s else 0.0 for f in F])

def _train(d):
    X = np.stack([_vec(r["t"]) for r in d])
    y = np.array([r["p"] for r in d])
    return np.linalg.lstsq(X, y, rcond=None)[0]

def _score(w, tags):
    return float(_vec(tags) @ w)

if __name__ == "__main__":
    hist = [
        {"t": ["Bed", "Linens", "duvet"], "p": 189.0},
        {"t": ["bed", "cotton", "duvet"], "p": 129.0},
        {"t": ["bath", "Towels", "cotton"], "p": 34.0},
        {"t": ["bed", "linen", "Pillows"], "p": 59.0},
    ]
    w = _train(hist)
    print(round(_score(w, ["bed", "linen", "duvet"]), 2))
