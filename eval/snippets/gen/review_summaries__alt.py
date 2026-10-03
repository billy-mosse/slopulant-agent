import re
from collections import defaultdict

KEY_TOPICS = {"cozy", "rough", "hue", "fit", "shrinkage", "heat", "craft", "value", "fuzzing"}


def preprocess(text):
    text = re.sub(r"https?://\S+", "", text.lower())
    return re.sub(r"[^a-z\s]", " ", text)


def compile_insights(reviews):
    grouped = defaultdict(list)
    for item in reviews:
        grouped[item["sku"]].append(item)

    results = {}
    for sku, items in grouped.items():
        mean_score = sum(i["rating"] for i in items) / len(items)
        mentions = []
        for review in items:
            tokens = preprocess(review["text"]).split()
            mentions.extend(t for t in tokens if t in KEY_TOPICS)
        freq = sorted(set(mentions), key=lambda x: -mentions.count(x))[:3]
        aspect_str = ", ".join(freq) if freq else "no specific details"
        results[sku] = {
            "avg_rating": round(mean_score, 2),
            "summary": f"Highlights: {aspect_str}"
        }
    return results


if __name__ == "__main__":
    sample = [
        {"sku": "BED-001", "rating": 5, "text": "Very cozy hue, excellent fit"},
        {"sku": "BED-001", "rating": 4, "text": "Warm and cozy, slight shrinkage"},
    ]
    print(compile_insights(sample))
