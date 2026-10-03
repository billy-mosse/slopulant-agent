"""Search box: embed the user's query and retrieve the nearest products."""
import numpy as np


class QueryEncoder:
    def __init__(self, vocab, dim=128, seed=0):
        rng = np.random.default_rng(seed)
        self.table = {w: rng.normal(size=dim) for w in vocab}

    def encode(self, query):
        vs = [self.table[w] for w in query.lower().split() if w in self.table]
        v = np.mean(vs, axis=0) if vs else np.zeros(next(iter(self.table.values())).shape)
        return v / (np.linalg.norm(v) or 1.0)


def retrieve(query, encoder, item_matrix, item_ids, k=20):
    q = encoder.encode(query)
    scores = item_matrix @ q
    top = np.argsort(-scores)[:k]
    return [(item_ids[i], float(scores[i])) for i in top]
