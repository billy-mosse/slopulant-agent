import re
from collections import Counter

SPAM_PATTERNS = [r"https?://\S+", r"www\."]
PROFANITY_TERMS = {"scam", "garbage", "idiot"}
CAPS_THRESHOLD = 0.7


def preprocess(text):
    text = re.sub(r"https?://\S+|www\.\S+", " LINK ", text, flags=re.IGNORECASE)
    return re.sub(r"[^a-zA-Z\s]", " ", text).lower()


def detect_shouting(original):
    alpha_chars = [ch for ch in original if ch.isalpha()]
    if not alpha_chars:
        return False
    upper_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
    return upper_ratio > CAPS_THRESHOLD


def assess(item):
    content = item["text"]
    cleaned = preprocess(content)
    tokens = set(cleaned.split())

    if any(re.search(pat, content) for pat in SPAM_PATTERNS):
        return "spam_link"
    if tokens & PROFANITY_TERMS:
        return "profanity"
    if detect_shouting(content):
        return "shouting"
    return None


def process(dataset):
    keep, hold = [], []
    for entry in dataset:
        verdict = assess(entry)
        if verdict:
            hold.append((entry["review_id"], verdict))
        else:
            keep.append(entry)
    return keep, hold


if __name__ == "__main__":
    sample = [
        {"review_id": 1, "sku": "BED-001", "rating": 5, "text": "Lovely duvet"},
        {"review_id": 2, "sku": "BED-001", "rating": 1, "text": "SCAM buy at http://cheap.example"},
    ]
    good, bad = process(sample)
    print("Approved:", good)
    print("Held:", bad)
