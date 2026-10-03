import numpy as np

D = 128

def _agg_sess(evts, embs):
    v = [e for e in evts if e["event"] == "view"]
    a = sum(e["event"] == "add_to_cart" for e in evts)
    d = sum(e.get("dwell_s", 0) for e in v)
    r = [embs[e["sku"]] for e in v if e["sku"] in embs]
    p = np.mean(r, axis=0) if r else np.zeros(D)
    return np.hstack([[len(v), a, np.log1p(d)], p])

def _score(f, w, b=-3.0):
    return 1.0 / (1.0 + np.exp(-(np.dot(f, w) + b)))

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    emb = {"BED-001": rng.normal(size=D), "BED-003": rng.normal(size=D)}
    evts = [
        {"sku": "BED-001", "event": "view", "dwell_s": 40},
        {"sku": "BED-003", "event": "view", "dwell_s": 15},
        {"sku": "BED-001", "event": "add_to_cart"},
    ]
    wgt = np.zeros(3 + D)
    wgt[:3] = [0.3, 1.5, 0.2]
    print(round(_score(_agg_sess(evts, emb), wgt), 3))
