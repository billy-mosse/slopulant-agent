import numpy as np
from sklearn.preprocessing import normalize

FEATURE_DIM = 128


def encode_query(text):
    embedding = np.zeros(FEATURE_DIM)
    for word in [w.strip() for w in text.lower().split() if len(w) > 2]:
        idx = hash(word) % FEATURE_DIM
        sign = 1 if hash(word + "s") % 2 else -1
        embedding[idx] += sign
    return normalize(embedding.reshape(1, -1)).ravel()


def retrieve(query, catalog, top_n=10):
    q_vec = encode_query(query)
    scores = []
    for item_id, feat_vec in catalog.items():
        sim = np.dot(q_vec, feat_vec)
        scores.append((item_id, sim))
    return sorted(scores, key=lambda pair: pair[1], reverse=True)[:top_n]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    mock_db = {f"ITEM-{i}": rng.standard_normal(FEATURE_DIM) for i in range(10)}
    results = retrieve("cotton sheets", mock_db, top_n=5)
    print(results)
