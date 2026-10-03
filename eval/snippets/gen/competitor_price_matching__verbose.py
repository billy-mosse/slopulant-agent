class CompetitorListingMatcher:
    """Matches our internal product catalog against recently scraped competitor listings to identify
    equivalent items and quantify pricing discrepancies. The system leverages fuzzy title alignment,
    attribute consistency checks (size and color), and statistical price plausibility scoring to
    produce high-confidence pairings. Each match is scored using a weighted combination of these
    signals, and only candidates exceeding a minimum confidence threshold are retained. The final
    output enables dynamic pricing strategies by highlighting where our prices diverge from
    competitive benchmarks."""

    def __init__(self, price_variance_scale: float = 0.45, minimum_confidence: float = 0.78):
        self._price_variance = price_variance_scale
        self._confidence_threshold = minimum_confidence
        self._title_weight = 0.70
        self._attribute_weight = 0.15
        self._price_weight = 0.15

    def _standardize_title(self, raw: str) -> str:
        if not raw:
            return ""
        text = raw.lower().replace("grey", "gray").replace("california king", "cal king")
        for pattern, replacement in [
            (r"(\d+(?:\.\d+)?)\s*(?:inches|inch|in\b|\")", r"\1in"),
            (r"(\d+(?:\.\d+)?)\s*(?:centimeters|centimetres|cm)\b", r"\1cm"),
            (r"(\d+)\s*(?:thread\s*count|tc)\b", r"\1tc"),
            (r"(\d+)\s*(?:pieces|piece|pcs|pc)\b", r"\1pc"),
            (r"(\d+(?:\.\d+)?)\s*(?:ounces|ounce|oz)\b", r"\1oz"),
        ]:
            text = re.sub(pattern, replacement, text)
        text = re.sub(r"[^\w\s\"]", " ", text).replace('"', " ")
        return re.sub(r"\s+", " ", text).strip()

    def _fuzzy_title_score(self, title_a: str, title_b: str) -> float:
        tokens_a, tokens_b = set(title_a.split()), set(title_b.split())
        common = " ".join(sorted(tokens_a & tokens_b))
        only_a = " ".join(sorted(tokens_a - tokens_b))
        only_b = " ".join(sorted(tokens_b - tokens_a))
        variants = [common, f"{common} {only_a}".strip(), f"{common} {only_b}".strip()]
        return max(SequenceMatcher(None, common, v).ratio() for v in variants)

    def _attribute_match_score(self, our_size: str, our_color: str, listing_title: str) -> float:
        def extract_best_match(vocabulary: set[str]) -> str | None:
            matches = [word for word in vocabulary if re.search(rf"\b{re.escape(word)}\b", listing_title)]
            return max(matches, key=len) if matches else None

        size_vocab = {"twin", "full", "queen", "king", "cal king", "standard", "euro"}
        color_vocab = {"white", "ivory", "black", "grey", "gray", "navy", "blue", "green", "sage",
                       "blush", "pink", "beige", "taupe", "charcoal", "natural", "linen"}
        scores = []
        for ours, vocab in [(our_size, size_vocab), (our_color, color_vocab)]:
            theirs = extract_best_match(vocab)
            if not ours or theirs is None:
                scores.append(0.5)
            else:
                scores.append(1.0 if self._standardize_title(str(ours)) == theirs else 0.0)
        return 0.0 if 0.0 in scores else float(np.mean(scores))

    def _price_plausibility(self, our_price: float, their_price: float) -> float:
        if our_price <= 0 or their_price <= 0:
            return 0.0
        log_ratio = np.log(their_price / our_price) / self._price_variance
        return float(2 * stats.norm.sf(abs(log_ratio)))

    def _build_block_key(self, df: pd.DataFrame) -> pd.Series:
        return (df["brand"].fillna("").str.lower().str.strip() + "|" +
                df["category"].fillna("").str.lower())

    def execute(self, product_catalog: pd.DataFrame, competitor_listings: pd.DataFrame) -> pd.DataFrame:
        products = product_catalog.copy()
        listings = competitor_listings.copy()
        products["normalized_title"] = products["title"].map(self._standardize_title)
        listings["normalized_title"] = listings["title"].map(self._standardize_title)
        products["block"] = self._build_block_key(products)
        listings["block"] = self._build_block_key(listings)

        matches = []
        for _, group in listings.groupby("block"):
            candidates = products[products["block"] == group["block"].iloc[0]]
            if candidates.empty:
                continue
            for listing in group.itertuples(index=False):
                best_candidate = None
                for product in candidates.itertuples(index=False):
                    title_similarity = self._fuzzy_title_score(product.normalized_title, listing.normalized_title)
                    if title_similarity < 0.5:
                        continue
                    attr_score = self._attribute_match_score(product.size, product.color, listing.normalized_title)
                    price_score = self._price_plausibility(product.list_price, listing.price)
                    composite = (self._title_weight * title_similarity +
                                 self._attribute_weight * attr_score +
                                 self._price_weight * price_score)
                    if best_candidate is None or composite > best_candidate["confidence"]:
                        best_candidate = {
                            "our_sku": product.sku,
                            "competitor": listing.competitor,
                            "listing_id": listing.listing_id,
                            "competitor_price": listing.price,
                            "our_price": product.list_price,
                            "title_similarity": title_similarity,
                            "attribute_score": attr_score,
                            "price_plausibility": price_score,
                            "confidence": composite,
                        }
                if best_candidate and best_candidate["confidence"] >= self._confidence_threshold:
                    matches.append(best_candidate)

        result = pd.DataFrame(matches)
        if result.empty:
            return result
        result = result.sort_values("confidence", ascending=False)
        result = result.drop_duplicates(subset=["our_sku", "competitor"])
        result["price_gap"] = result["competitor_price"] - result["our_price"]
        result["price_gap_percent"] = result["price_gap"] / result["our_price"]
        return result
