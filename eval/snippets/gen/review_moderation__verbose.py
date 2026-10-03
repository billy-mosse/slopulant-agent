class ReviewContentPolicyEnforcer:
    """ReviewContentPolicyEnforcer ensures customer feedback aligns with community guidelines
    before publication. It analyzes review text for prohibited patterns—including
    external links, offensive terminology, and excessive capitalization—and routes
    each submission to either the approved or flagged queue accordingly.

    The system processes raw review entries and produces two distinct collections:
    one containing reviews that meet all standards for public display, and another
    documenting reviews that require manual review, each accompanied by a specific
    policy violation identifier.
    """

    PROHIBITED_TERMS = {"scam", "garbage", "idiot"}
    URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)

    def __init__(self):
        self._approved_reviews = []
        self._flagged_submissions = []

    def _assess_text_compliance(self, content: str) -> str | None:
        """Evaluate review text against defined content policies."""
        normalized = self.URL_PATTERN.sub(" <url> ", content.lower())
        normalized = re.sub(r"[^a-z<>\s]", " ", normalized)

        if "<url>" in normalized:
            return "contains_external_link"
        if any(term in normalized.split() for term in self.PROHIBITED_TERMS):
            return "prohibited_language"
        alpha_chars = [ch for ch in content if ch.isalpha()]
        if alpha_chars and sum(ch.isupper() for ch in alpha_chars) / len(alpha_chars) > 0.7:
            return "excessive_capitalization"
        return None

    def process(self, review_batch: list[dict]) -> tuple[list[dict], list[tuple[int, str]]]:
        """Distribute reviews into approved or flagged collections based on policy review."""
        for entry in review_batch:
            violation = self._assess_text_compliance(entry["text"])
            if violation:
                self._flagged_submissions.append((entry["review_id"], violation))
            else:
                self._approved_reviews.append(entry)
        return self._approved_reviews, self._flagged_submissions
