class QueryIntentAnalyzer:
    """
    Analyzes user search queries to extract high-level intent signals including likely product categories,
    price constraints, and attribute preferences. This component powers downstream search personalization
    and ranking by transforming raw queries into structured signals that reflect user intent patterns
    observed from historical click behavior and linguistic cues.
    """

    def __init__(self, category_probabilities: dict[str, list[tuple[str, float]]]) -> None:
        self._category_distribution = category_probabilities
        self._token_based_backoff: dict[str, list[tuple[str, float]]] = {}
        self._build_backoff_index()

    def _build_backoff_index(self) -> None:
        for query, category_list in self._category_distribution.items():
            for token in query.split():
                self._token_based_backoff.setdefault(token, []).extend(category_list)

    def infer_categories(self, query: str) -> list[tuple[str, float]]:
        normalized = query.lower()
        if normalized in self._category_distribution:
            return self._category_distribution[normalized]

        category_scores: dict[str, float] = {}
        for token in normalized.split():
            for category, score in self._token_based_backoff.get(token, []):
                category_scores[category] += score

        if not category_scores:
            return []

        total = sum(category_scores.values())
        normalized_scores = [(cat, round(score / total, 4)) for cat, score in category_scores.items()]
        return sorted(normalized_scores, key=lambda x: -x[1])[:3]

    def extract_price_constraints(self, text: str) -> dict[str, float]:
        constraints: dict[str, float] = {}
        patterns = [
            (r"\b(?:under|below|less than)\s*\$?(\d+)", "max_price"),
            (r"\b(?:over|above|more than)\s*\$?(\d+)", "min_price"),
            (r"\$(\d+)\s*-\s*\$?(\d+)", "range"),
        ]
        for regex, label in patterns:
            match = re.search(regex, text, re.IGNORECASE)
            if not match:
                continue
            if label == "range":
                constraints["min_price"], constraints["max_price"] = float(match.group(1)), float(match.group(2))
            else:
                constraints[label] = float(match.group(1))
        return constraints

    def extract_attribute_signals(self, text: str) -> dict[str, str]:
        signals: dict[str, str] = {}
        attribute_options = {
            "size": ["twin xl", "twin", "full", "queen", "california king", "king"],
            "material": ["linen", "cotton", "percale", "sateen", "bamboo", "silk", "wool", "velvet", "jute"],
        }
        for attr_type, candidates in attribute_options.items():
            for candidate in candidates:
                if re.search(rf"\b{re.escape(candidate)}\b", text, re.IGNORECASE):
                    signals[attr_type] = candidate
                    break
        return signals

    def clean_query(self, text: str) -> str:
        cleaned = text
        for regex, _ in [
            (r"\b(?:under|below|less than)\s*\$?(\d+)", ""),
            (r"\b(?:over|above|more than)\s*\$?(\d+)", ""),
            (r"\$(\d+)\s*-\s*\$?(\d+)", ""),
        ]:
            cleaned = re.sub(regex, "", cleaned, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", cleaned).strip()

    def analyze(self, original: str, normalized: str) -> dict:
        cleaned = self.clean_query(normalized)
        return {
            "original_query": original,
            "normalized_query": normalized,
            "inferred_categories": self.infer_categories(cleaned),
            "price_constraints": self.extract_price_constraints(normalized),
            "attribute_signals": self.extract_attribute_signals(normalized),
        }
