import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PRODUCT_DIM = 64


def tokenize(text):
    tokens = text.lower().split()
    return [t.strip() for t in tokens if len(t.strip()) > 2]


def encode_products(dataset):
    docs = [f"{p['title']} {p['description']}" for p in dataset]
    vectorizer = TfidfVectorizer(tokenizer=tokenize, max_features=PRODUCT_DIM)
    feats = vectorizer.fit_transform(docs)
    return feats.tocsr()


def nearest_neighbors(dataset, qty=5):
    codes = encode_products(dataset)
    dists = cosine_similarity(codes)
    sku_list = [p["sku"] for p in dataset]
    results = {}
    for idx, s in enumerate(sku_list):
        ranks = np.argsort(-dists[idx])
        matches = [
            (sku_list[r], float(dists[idx, r]))
            for r in ranks
            if r != idx
        ][:qty]
        results[s] = matches
    return results


if __name__ == "__main__":
    sample = [
        {"sku": "BED-001", "title": "Queen linen duvet cover", "description": "Stonewashed linen, oat"},
        {"sku": "BED-003", "title": "Linen pillowcase set", "description": "Stonewashed linen, oat, set of 2"},
        {"sku": "BTH-010", "title": "Turkish cotton bath towel", "description": "600gsm, white"},
    ]
    print(nearest_neighbors(sample, qty=2))
