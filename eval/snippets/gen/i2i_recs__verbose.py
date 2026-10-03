class ProductSimilarityEngine:
    """
    Generates personalized product pairings for the product detail page experience.

    This module computes semantic similarity between items using a lightweight
    text-based embedding derived from product titles and descriptions. The
    resulting similarity scores power the "You might also like" section, enabling
    fast, self-contained recommendations without external dependencies.

    Each product is represented as a normalized 128-dimensional vector, where
    word contributions are determined via hash-based feature hashing. Pairwise
    cosine similarities are computed across the catalog, and the top N most
    similar items are retained per product.
    """

    DIMENSIONALITY = 128

    def __init__(self, catalog_items):
        self._items = catalog_items
        self._sku_list = [item["sku"] for item in catalog_items]
        self._vectors = np.stack([self._compute_normalized_vector(item) for item in catalog_items])

    def _tokenize(self, text):
        return [token for token in text.lower().replace(",", " ").split() if len(token) > 2]

    def _compute_normalized_vector(self, item):
        vector = np.zeros(self.DIMENSIONALITY)
        combined_text = item["title"] + " " + item["description"]
        for token in self._tokenize(combined_text):
            hash_digest = hashlib.md5(token.encode()).hexdigest()
            index = int(hash_digest, 16) % self.DIMENSIONALITY
            sign = 1 if (int(hash_digest, 16) >> 8) % 2 == 0 else -1
            vector[index] += sign
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def generate_top_matches(self, count=5):
        similarity_matrix = self._vectors @ self._vectors.T
        results = {}
        for idx, sku in enumerate(self._sku_list):
            ranked_indices = np.argsort(-similarity_matrix[idx])
            candidates = [
                (self._sku_list[j], float(similarity_matrix[idx, j]))
                for j in ranked_indices
                if j != idx
            ]
            results[sku] = candidates[:count]
        return results
