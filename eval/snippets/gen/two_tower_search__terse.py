import hashlib
import numpy as np

D = 128

def _hash_vec(q):
    v = np.zeros(D)
    for w in (x for x in q.lower().split() if len(x) > 2):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % D] += 1.0 if (h >> 8) & 1 else -1.0
    return v / (np.linalg.norm(v) or 1.0)

def rank(q, items, n=10):
    qv = _hash_vec(q)
    scores = [(k, qv @ v) for k, v in items.items()]
    return sorted(scores, key=lambda p: -p[1])[:n]

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    db = {f"SKU-{i}": rng.normal(size=D) for i in range(5)}
    print(rank("linen duvet", db, 3))
