import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

FEATURE_DIM = 128


def _prep_text(title, desc, tag_list):
    tags = " ".join(f"tag_{t}" for t in tag_list)
    return f"{title} {desc} {tags}".lower()


def _encode_items(item_list, tag_map):
    texts = [_prep_text(i["title"], i["description"], tag_map.get(i["sku"], [])) for i in item_list]
    vectorizer = TfidfVectorizer(
        max_features=FEATURE_DIM,
        ngram_range=(1, 2),
        token_pattern=r"\b[a-z]{3,}\b",
        sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(texts)
    normalized = normalize(matrix, norm="l2")
    return {item_list[i]["sku"]: normalized[i].toarray().flatten() for i in range(len(item_list))}


def construct(product_records, sku_tags):
    return _encode_items(product_records, sku_tags)


if __name__ == "__main__":
    catalog = [
        {"sku": "BED-001", "title": "Queen linen duvet cover", "description": "Stonewashed linen, oat"},
        {"sku": "BED-002", "title": "Queen linen duvet", "description": "Stone washed linen in oat color"},
    ]
    tags = {"BED-001": ["bedding", "linen"], "BED-002": ["bedding", "linen"]}
    result = construct(catalog, tags)
    for sku, vec in result.items():
        print(sku, vec[:4].round(3).tolist())
