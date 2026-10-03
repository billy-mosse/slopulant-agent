import re

URL = re.compile(r"(https?://|www\.)\S+", re.I)
SLURS = ["idiot", "garbage", "scam", "moron"]


def toxicity(text):
    score = 0.0
    if URL.search(text):
        score += 0.6
    low = text.lower()
    score += 0.3 * sum(w in low for w in SLURS)
    caps = sum(ch.isupper() for ch in text) / max(1, sum(ch.isalpha() for ch in text))
    if caps > 0.6:
        score += 0.4
    return min(score, 1.0)


def publishable(reviews, cutoff=0.5):
    return [r for r in reviews if toxicity(r["body"]) < cutoff]
