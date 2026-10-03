class ProductCategoryPredictor:
    """
    Assigns hierarchical category paths to active products using a trained
    three-tier classification pipeline. The system enforces taxonomy consistency
    by ensuring child categories only appear under their designated parents,
    and truncates predictions when confidence falls below a configurable threshold.
    """

    def __init__(self, model_bundle_path: str, database_dsn: str, minimum_child_confidence: float = 0.55):
        self._database_url = database_dsn
        self._minimum_child_confidence = minimum_child_confidence
        self._models_bundle = joblib.load(model_bundle_path)
        self._level_models = self._models_bundle["models"]
        self._level_classes = {lvl: self._level_models[lvl].classes_ for lvl in ("l1", "l2", "l3")}

    def _select_best_valid_child(self, probabilities: np.ndarray, candidate_labels: np.ndarray, parent_label: str, parent_map: dict) -> tuple[str | None, float]:
        """Restricts prediction to children of the given parent, renormalizes probabilities, and returns the top candidate."""
        valid_mask = np.array([parent_map.get(label) == parent_label for label in candidate_labels])
        if not np.any(valid_mask):
            return None, 0.0
        adjusted_probs = probabilities * valid_mask
        total_mass = adjusted_probs.sum()
        if total_mass == 0.0:
            return None, 0.0
        normalized_probs = adjusted_probs / total_mass
        best_index = int(np.argmax(normalized_probs))
        return candidate_labels[best_index], float(normalized_probs[best_index])

    def _infer_category_path(self, product_texts: pd.Series) -> pd.DataFrame:
        """Runs inference across all three taxonomy levels with hierarchical constraints."""
        l1_probs = self._level_models["l1"].predict_proba(product_texts)
        l2_probs = self._level_models["l2"].predict_proba(product_texts)
        l3_probs = self._level_models["l3"].predict_proba(product_texts)

        results = []
        for idx in range(len(product_texts)):
            l1_label, l1_confidence = self._level_classes["l1"][int(l1_probs[idx].argmax())], float(l1_probs[idx].max())
            l2_label, l2_confidence = self._select_best_valid_child(l2_probs[idx], self._level_classes["l2"], l1_label, PARENT_OF_L2)
            if l2_label is None or l2_confidence < self._minimum_child_confidence:
                results.append((l1_label, None, None, l1_confidence, l2_confidence, None, 1))
                continue
            l3_label, l3_confidence = self._select_best_valid_child(l3_probs[idx], self._level_classes["l3"], l2_label, PARENT_OF_L3)
            if l3_label is None or l3_confidence < self._minimum_child_confidence:
                results.append((l1_label, l2_label, None, l1_confidence, l2_confidence, l3_confidence, 2))
                continue
            results.append((l1_label, l2_label, l3_label, l1_confidence, l2_confidence, l3_confidence, 3))

        return pd.DataFrame(results, columns=["department", "category", "subcategory", "confidence_l1", "confidence_l2", "confidence_l3", "depth"])

    def execute(self) -> None:
        """Processes all active products, writes predictions to the output table."""
        engine = create_engine(self._database_url)
        input_query = "SELECT sku, title, description FROM catalog.products WHERE is_active"
        raw_data = pd.read_sql(input_query, engine)
        logger.info("processing %d active products", len(raw_data))

        text_features = (raw_data["title"].fillna("") + " " + raw_data["title"].fillna("") + " " + raw_data["description"].fillna("")).str.lower()
        predictions = self._infer_category_path(text_features)
        predictions.insert(0, "sku", raw_data["sku"].values)
        predictions["full_path"] = predictions[["department", "category", "subcategory"]].apply(
            lambda row: " > ".join(str(x) for x in row if pd.notna(x)), axis=1
        )
        predictions["timestamp"] = pd.Timestamp.utcnow()
        logger.info("depth distribution: %s", predictions["depth"].value_counts(normalize=True).round(3).to_dict())

        predictions.to_sql("predicted_category", engine, schema="catalog", if_exists="replace", index=False)
        logger.info("stored %d predictions in catalog.predicted_category", len(predictions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hierarchical product categorization")
    parser.add_argument("--dsn", required=True, help="Database connection string")
    parser.add_argument("--model", required=True, help="Path to trained model bundle")
    parser.add_argument("--child-threshold", type=float, default=0.55, help="Minimum confidence for child-level predictions")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    predictor = ProductCategoryPredictor(args.model, args.dsn, args.child_threshold)
    predictor.execute()
