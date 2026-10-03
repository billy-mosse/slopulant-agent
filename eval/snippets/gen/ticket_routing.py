"""
Customer support ticket classifier for home-goods e-commerce.
Routes tickets to appropriate teams using keyword rules (priority) and Naive Bayes fallback.

Input: raw_tickets (schema.raw_tickets.ticket_id, schema.raw_tickets.subject, schema.raw_tickets.body)
Output: classified_tickets (schema.classified_tickets.ticket_id, schema.classified_tickets.team, schema.classified_tickets.confidence)
"""

import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

TEAM_KEYWORDS = {
    "shipping": ["ship", "deliver", "tracking", "delay", "lost", "package"],
    "returns": ["return", "refund", "exchange", "wrong item", "defective"],
    "product": ["quality", "material", "size", "color", "break", "leak"],
    "billing": ["charge", "invoice", "payment", "price", "discount", "coupon"]
}

# Preload model and vectorizer if available
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ticket_classifier.joblib")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "ticket_vectorizer.joblib")

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
else:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    model = MultinomialNB()

def _rule_based_classify(text: str) -> tuple[str, float]:
    text_lower = text.lower()
    for team, keywords in TEAM_KEYWORDS.items():
        if any(re.search(r"\b" + re.escape(kw) + r"\b", text_lower) for kw in keywords):
            return team, 0.95
    return "general", 0.0

def classify_ticket(subject: str, body: str) -> tuple[str, float]:
    text = f"{subject} {body}"
    team, confidence = _rule_based_classify(text)
    if team != "general":
        return team, confidence
    try:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        proba = max(model.predict_proba(X)[0])
        return pred, proba
    except Exception:
        return "general", 0.0
