class TagNormalizationPipeline:
    """
    Ensures product tagging consistency across the catalog by mapping user-submitted
    freeform tags to a standardized set of controlled terms. This module processes
    raw tag entries—typically contributed by merchants during product onboarding—and
    produces a clean, normalized list suitable for downstream search, filtering,
    and analytics systems. Any tag that cannot be confidently resolved to a known
    category is excluded from the final output.
    """

    ALLOWED_CATEGORIES = {
        "bedding", "bath", "linen", "cotton", "towel", "duvet", "pillow", "decor", "kitchen"
    }

    TAG_SYNONYMS = {
        "linens": "linen",
        "linen-blend": "linen",
        "towels": "towel",
        "pillows": "pillow",
        "bed": "bedding",
        "home-decor": "decor",
    }

    @staticmethod
    def _standardize(tag: str) -> str:
        normalized = tag.strip().lower().replace(" ", "-")
        return TagNormalizationPipeline.TAG_SYNONYMS.get(normalized, normalized)

    def process(self, raw_entries):
        """
        Accepts an iterable of (sku, tag) pairs and returns a dictionary mapping
        each unique SKU to a sorted list of validated, normalized tags.
        """
        sku_to_tags = {}
        for sku, raw_tag in raw_entries:
            candidate = self._standardize(raw_tag)
            if candidate in self.ALLOWED_CATEGORIES:
                sku_to_tags.setdefault(sku, set()).add(candidate)
        return {sku: sorted(tags) for sku, tags in sku_to_tags.items()}
