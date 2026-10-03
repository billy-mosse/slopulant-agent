class ReviewInsightGenerator:
    """Generates concise, customer-centric summaries that highlight overall satisfaction
    and recurring product attributes across verified reviews. Each summary includes
    the rounded average star rating and the top three most-mentioned qualities,
    helping shoppers quickly grasp what real users value or note about an item."""

    maintainable_aspects = {
        "soft", "scratchy", "color", "size", "shrink", "warm", "quality", "price", "pilling"
    }

    def create_summary(self, review_records):
        """Aggregate reviews by product identifier and compute averaged satisfaction
        plus frequent aspect mentions for the final narrative."""
        grouped_by_product = {}
        for record in review_records:
            key = record["product_identifier"]
            grouped_by_product.setdefault(key, []).append(record)

        summaries = {}
        for identifier, submissions in grouped_by_product.items():
            average_score = round(
                sum(submission["star_rating"] for submission in submissions) / len(submissions), 2
            )
            extracted_words = [
                word for submission in submissions
                for word in self._extract_terms(submission["review_body"])
                if word in self.maintainable_aspects
            ]
            aspect_counts = Counter(extracted_words)
            leading_aspects = ", ".join(
                term for term, _ in aspect_counts.most_common(3)
            ) or "no specific attributes highlighted"
            summaries[identifier] = {
                "average_rating": average_score,
                "narrative": f"Customers mention: {leading_aspects}"
            }
        return summaries

    def _extract_terms(self, raw_text):
        cleaned = re.sub(r"http\S+", "", raw_text.lower())
        cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
        return cleaned.split()

    def __call__(self, review_data):
        return self.create_summary(review_data)


if __name__ == "__main__":
    sample = [
        {"product_identifier": "BED-001", "star_rating": 5, "review_body": "So soft, great color!"},
        {"product_identifier": "BED-001", "star_rating": 3, "review_body": "Soft but did shrink after washing"},
    ]
    generator = ReviewInsightGenerator()
    print(generator(sample))
