class TwoTowerRankingEngine:
    """Provides relevance-ranked product suggestions by matching user queries
    against a precomputed item embedding repository using a lightweight
    vector-space query representation.

    The query tower constructs a normalized feature vector from the input
    string by hashing tokens into a fixed-dimensional space and aggregating
    signed contributions. The item tower leverages a shared embedding store
    (sku → vector) and computes cosine-similarity proxies via dot products.
    Results are returned in descending order of relevance score.
    """

    DIMENSIONALITY = 128

    def __init__(self, item_embedding_store):
        self._item_store = item_embedding_store

    def _encode_query(self, natural_language_input):
        vector = np.zeros(self.DIMENSIONALITY, dtype=np.float32)
        for token in natural_language_input.lower().split():
            if len(token) <= 2:
                continue
            hash_digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            hash_int = int(hash_digest, 16)
            index = hash_int % self.DIMENSIONALITY
            sign = 1.0 if (hash_int >> 8) & 1 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0.0 else vector

    def retrieve_top_matches(self, user_query, limit=10):
        query_vector = self._encode_query(user_query)
        candidates = []
        for sku, item_vector in self._item_store.items():
            score = float(np.dot(query_vector, item_vector))
            candidates.append((sku, score))
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return candidates[:limit]
