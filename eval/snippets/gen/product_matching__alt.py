from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

SIMILARITY_CUTOFF = 0.82

def normalize(s):
    return " ".join(s.strip().lower().replace("_", " ").split())

def vectorize_titles(items):
    clean_titles = [normalize(item["title"]) for item in items]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    return vectorizer.fit_transform(clean_titles), items

def compute_pairwise_scores(matrix, items):
    results = []
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            if items[i]["brand"] != items[j]["brand"]:
                continue
            title_sim = cosine_similarity(matrix[i], matrix[j])[0, 0]
            p1, p2 = items[i]["price"], items[j]["price"]
            price_sim = min(p1, p2) / max(p1, p2) if max(p1, p2) > 0 else 0.0
            combined = 0.75 * title_sim + 0.25 * price_sim
            if combined >= SIMILARITY_CUTOFF:
                results.append((items[i]["sku"], items[j]["sku"], round(combined, 3)))
    return results

def detect_duplicates(product_list):
    if len(product_list) < 2:
        return []
    tfidf_matrix, _ = vectorize_titles(product_list)
    return compute_pairwise_scores(tfidf_matrix, product_list)

if __name__ == "__main__":
    test_data = [
        {"sku": "BED-001", "title": "Queen Linen Duvet Cover - Oat", "brand": "Hearth", "price": 189.0},
        {"sku": "BED-002", "title": "Queen linen duvet cover oat", "brand": "Hearth", "price": 179.0},
        {"sku": "BTH-010", "title": "Turkish Cotton Bath Towel", "brand": "Loom", "price": 34.0},
    ]
    print(detect_duplicates(test_data))
