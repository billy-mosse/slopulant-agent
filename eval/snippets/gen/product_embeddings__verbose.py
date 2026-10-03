class ProductFeatureAssembler:
    """
    Constructs semantic embeddings for each product using its title, description,
    and curated tag set. The resulting dense representations capture key product
    attributes and are intended for downstream recommendation and search systems.
    """

    def __init__(self, embedding_dimension: int = 128):
        self.dimension = embedding_dimension

    def _extract_significant_terms(self, raw_text: str) -> list[str]:
        cleaned = raw_text.lower().replace(",", " ")
        return [term for term in cleaned.split() if len(term) > 2]

    def _compute_hashed_vector(self, terms: list[str]) -> np.ndarray:
        vector = np.zeros(self.dimension)
        for term in terms:
            hash_digest = hashlib.md5(term.encode()).hexdigest()
            hash_int = int(hash_digest, 16)
            index = hash_int % self.dimension
            sign = 1.0 if (hash_int >> 8) % 2 == 0 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def generate_embeddings(self, product_records: list[dict], sku_to_tags: dict[str, list[str]]) -> dict[str, np.ndarray]:
        embeddings = {}
        for record in product_records:
            sku = record["sku"]
            content_terms = self._extract_significant_terms(record["title"])
            content_terms.extend(self._extract_significant_terms(record["description"]))
            tag_terms = [f"tag:{tag}" for tag in sku_to_tags.get(sku, [])]
            combined_terms = content_terms + tag_terms
            embeddings[sku] = self._compute_hashed_vector(combined_terms)
        return embeddings
