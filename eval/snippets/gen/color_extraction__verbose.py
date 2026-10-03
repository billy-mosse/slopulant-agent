class ProductColorAnalyzer:
    """
    Identifies the two most representative merchandising colors for each product
    based on aggregated visual analysis across all active product images.
    The system processes raw pixel data, filters out background noise, and applies
    k-means clustering in perceptually uniform CIE Lab space to derive dominant
    color signals. These are then mapped to our standardized color vocabulary using
    Delta-E distance minimization. The resulting color shares are averaged across
    images per SKU and ranked to produce the top two color assignments.
    """

    def __init__(self, database_connection_string: str, maximum_samples_per_image: int = 20_000) -> None:
        self._engine = create_engine(database_connection_string)
        self._subsample_limit = maximum_samples_per_image
        self._rng = np.random.default_rng(seed=7)

    def _convert_to_lab(self, rgb_values: np.ndarray) -> np.ndarray:
        """Transform sRGB pixel values to CIE Lab (D65 illuminant) for perceptual analysis."""
        linearized = np.where(
            rgb_values <= 0.04045,
            rgb_values / 12.92,
            ((rgb_values + 0.055) / 1.055) ** 2.4
        )
        xyz = linearized @ M_RGB2XYZ.T
        normalized = xyz / WHITE_D65
        cubic_root = np.where(normalized > 216 / 24389, np.cbrt(normalized), (24389 / 27 * normalized + 16) / 116)
        L = 116 * cubic_root[:, 1] - 16
        a = 500 * (cubic_root[:, 0] - cubic_root[:, 1])
        b = 200 * (cubic_root[:, 1] - cubic_root[:, 2])
        return np.column_stack([L, a, b])

    def _extract_foreground_pixels(self, image_array: np.ndarray) -> np.ndarray:
        """Discard near-white background pixels and subsample remaining foreground data."""
        flat_pixels = image_array[..., :3].reshape(-1, 3)
        foreground_mask = ~((flat_pixels >= 235).all(axis=1))
        filtered = flat_pixels[foreground_mask]
        if len(filtered) > self._subsample_limit:
            indices = self._rng.choice(len(filtered), self._subsample_limit, replace=False)
            filtered = filtered[indices]
        return filtered

    def _cluster_colors(self, lab_pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Perform k-means++ initialization followed by Lloyd iterations to find dominant color clusters."""
        initial_centroids = [lab_pixels[self._rng.integers(len(lab_pixels))]]
        for _ in range(3):
            distances_sq = ((lab_pixels[:, None, :] - np.array(initial_centroids)[None, :]) ** 2).sum(axis=-1).min(axis=1)
            next_idx = self._rng.choice(len(lab_pixels), p=distances_sq / distances_sq.sum())
            initial_centroids.append(lab_pixels[next_idx])
        centroids = np.array(initial_centroids)
        for _ in range(8):
            assignments = ((lab_pixels[:, None, :] - centroids[None, :]) ** 2).sum(axis=-1).argmin(axis=1)
            updated = np.array([
                lab_pixels[assignments == j].mean(axis=0) if np.any(assignments == j) else centroids[j]
                for j in range(4)
            ])
            if np.allclose(updated, centroids, atol=1e-3):
                break
            centroids = updated
        return centroids, assignments

    def _analyze_single_image(self, image_data: np.ndarray) -> dict[str, float]:
        """Compute color distribution for one product image, returning named color shares."""
        pixels = self._extract_foreground_pixels(image_data)
        if len(pixels) < 40:
            return {}
        lab_space = self._convert_to_lab(pixels.astype(np.float64) / 255.0)
        centroids, labels = self._cluster_colors(lab_space)
        shares = np.bincount(labels, minlength=4) / len(lab_space)
        aggregated: dict[str, float] = defaultdict(float)
        for centroid, share in zip(centroids, shares):
            name, _ = nearest(centroid)
            aggregated[name] += share
        return aggregated

    def _aggregate_sku_colors(self, image_batch: list[np.ndarray]) -> list[tuple[str, float]]:
        """Combine color signals across all images for a given SKU, returning top two named colors."""
        combined: dict[str, float] = defaultdict(float)
        for image in image_batch:
            per_image = self._analyze_single_image(image)
            for color_name, contribution in per_image.items():
                combined[color_name] += contribution / len(image_batch)
        return sorted(combined.items(), key=lambda item: -item[1])[:2]

    def execute(self) -> None:
        """Orchestrate the full pipeline: load images, compute color signatures, persist results."""
        query = "SELECT sku, pixels, height, width FROM catalog.product_images WHERE is_active ORDER BY sku"
        raw_data = pd.read_sql(query, self._engine)
        log.info("Retrieved %d image records", len(raw_data))

        results = []
        for sku, group in raw_data.groupby("sku"):
            images = [
                np.frombuffer(row.pixels, np.uint8).reshape(int(row.height), int(row.width), 3)
                for row in group.itertuples()
            ]
            top_colors = self._aggregate_sku_colors(images)
            padded = top_colors + [(None, 0.0)] * 2
            results.append({
                "sku": sku,
                "primary_color": padded[0][0],
                "primary_share": round(padded[0][1], 3),
                "secondary_color": padded[1][0],
                "secondary_share": round(padded[1][1], 3),
            })

        output_frame = pd.DataFrame(results)
        output_frame.to_sql("product_colors", self._engine, schema="catalog", if_exists="replace", index=False)
        log.info("Successfully persisted color assignments for %d SKUs", len(output_frame))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute dominant product colors from image data")
    parser.add_argument("--dsn", required=True, help="Database connection string")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    analyzer = ProductColorAnalyzer(args.dsn)
    analyzer.execute()
