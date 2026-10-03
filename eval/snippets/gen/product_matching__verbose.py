class DuplicateListingAnalyzer:
    """
    Identifies potential duplicate product listings in the catalog that likely
    represent the same physical item uploaded multiple times by merchants.
    This module enables the catalog team to consolidate redundant entries
    into single canonical SKUs, improving data quality and reducing operational
    overhead in inventory management.
    """

    SIMILARITY_THRESHOLD = 0.85

    def _normalize_title(self, raw_title: str) -> str:
        """Standardize title casing, spacing, and punctuation for comparison."""
        normalized = raw_title.lower().replace("-", " ")
        return " ".join(normalized.split())

    def _compute_pairwise_match_score(self, first_listing: dict, second_listing: dict) -> float:
        """Calculate similarity between two listings using brand, title, and price."""
        if first_listing["brand"].casefold() != second_listing["brand"].casefold():
            return 0.0

        title_similarity = SequenceMatcher(
            None,
            self._normalize_title(first_listing["title"]),
            self._normalize_title(second_listing["title"]),
        ).ratio()

        price_ratio = min(first_listing["price"], second_listing["price"]) / max(
            first_listing["price"], second_listing["price"]
        )

        return 0.8 * title_similarity + 0.2 * price_ratio

    def identify_duplicate_candidates(self, product_records: list[dict]) -> list[tuple[str, str, float]]:
        """Return all listing pairs with high similarity scores for review."""
        candidates = []
        for listing_a, listing_b in combinations(product_records, 2):
            match_score = self._compute_pairwise_match_score(listing_a, listing_b)
            if match_score >= self.SIMILARITY_THRESHOLD:
                candidates.append(
                    (
                        listing_a["sku"],
                        listing_b["sku"],
                        round(match_score, 3),
                    )
                )
        return candidates
