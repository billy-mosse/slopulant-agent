"""TF-IDF product vectors (title, description, tags) for downstream models."""
import math
from collections import Counter


def doc(product):
    words = (product["title"] + " " + product["description"]).lower().split()
    return [w for w in words if len(w) > 2] + ["#" + t for t in product.get("tags", [])]


def tfidf_vectors(products):
    docs = {p["sku"]: Counter(doc(p)) for p in products}
    df = Counter(t for d in docs.values() for t in d)
    n = len(docs)
    vecs = {}
    for sku, tf in docs.items():
        v = {t: c * math.log(n / df[t]) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs[sku] = {t: x / norm for t, x in v.items()}
    return vecs
