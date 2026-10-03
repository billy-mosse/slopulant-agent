from __future__ import annotations

import argparse
import logging
from collections import Counter

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.cluster import KMeans

log = logging.getLogger("product_color_analyzer")

SRC_TABLE = "catalog.product_images"
DEST_TABLE = "catalog.product_colors"

CLUSTER_COUNT = 4
MAX_ITER = 10
WHITE_THRESHOLD = 235
PIXEL_LIMIT = 20_000
RANDOM_STATE = 7

# sRGB to XYZ conversion matrix (D65)
RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
D65_WHITE = np.array([0.95047, 1.0, 1.08883])


def gamma_correct(c: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma expansion."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab_batch(rgb: np.ndarray) -> np.ndarray:
    """Convert batch of sRGB uint8 values to CIE Lab (D65)."""
    xyz = gamma_correct(rgb.astype(np.float64) / 255.0) @ RGB_TO_XYZ.T
    normalized = xyz / D65_WHITE
    f = np.where(normalized > 216 / 24389, np.cbrt(normalized), (24389 / 27 * normalized + 16) / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.column_stack([L, a, b])


def extract_foreground(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Filter out near-white pixels and subsample."""
    flat = image[..., :3].reshape(-1, 3)
    mask = ~((flat >= WHITE_THRESHOLD).all(axis=1))
    filtered = flat[mask]
    if len(filtered) > PIXEL_LIMIT:
        indices = rng.choice(len(filtered), PIXEL_LIMIT, replace=False)
        filtered = filtered[indices]
    return filtered


def get_palette_name(lab_point: np.ndarray) -> tuple[str, float]:
    """Find closest named color in internal palette using Euclidean Lab distance."""
    palette = {
        "white": (97.0, 0.0, 1.5),
        "ivory": (95.0, -1.0, 9.0),
        "oat": (82.0, 2.0, 14.0),
        "sand": (74.0, 4.0, 20.0),
        "camel": (60.0, 12.0, 32.0),
        "rust": (45.0, 35.0, 40.0),
        "terracotta": (52.0, 30.0, 30.0),
        "blush": (82.0, 14.0, 8.0),
        "rose": (62.0, 32.0, 10.0),
        "burgundy": (28.0, 35.0, 12.0),
        "mustard": (68.0, 5.0, 58.0),
        "sage": (68.0, -12.0, 14.0),
        "olive": (48.0, -8.0, 30.0),
        "forest": (32.0, -22.0, 10.0),
        "eucalyptus": (60.0, -18.0, 2.0),
        "sky": (78.0, -6.0, -16.0),
        "denim": (45.0, 0.0, -25.0),
        "navy": (22.0, 6.0, -30.0),
        "lavender": (72.0, 12.0, -18.0),
        "plum": (32.0, 25.0, -12.0),
        "dove": (76.0, 0.0, 1.0),
        "stone": (62.0, 1.0, 5.0),
        "slate": (48.0, -3.0, -8.0),
        "charcoal": (30.0, 0.0, -1.0),
        "black": (12.0, 0.0, 0.0),
        "walnut": (35.0, 12.0, 18.0),
    }
    names = list(palette.keys())
    lab_arr = np.array(list(palette.values()))
    diff = lab_arr - lab_point
    dists = np.sqrt((diff ** 2).sum(axis=1))
    idx = int(dists.argmin())
    return names[idx], float(dists[idx])


def analyze_image(image: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """Compute color distribution for a single image."""
    fg = extract_foreground(image, rng)
    if len(fg) < CLUSTER_COUNT * 10:
        return {}
    lab_data = rgb_to_lab_batch(fg)
    model = KMeans(n_clusters=CLUSTER_COUNT, n_init=1, max_iter=MAX_ITER, random_state=rng.integers(0, 10000))
    labels = model.fit_predict(lab_data)
    counts = np.bincount(labels, minlength=CLUSTER_COUNT)
    shares = counts / len(labels)
    color_counts: Counter = Counter()
    for lab, share in zip(model.cluster_centers_, shares):
        name, _ = get_palette_name(lab)
        color_counts[name] += share
    return dict(color_counts)


def aggregate_colors(images: list[np.ndarray], rng: np.random.Generator) -> list[tuple[str, float]]:
    """Average color shares across multiple images for a SKU."""
    totals: Counter = Counter()
    for img in images:
        for name, share in analyze_image(img, rng).items():
            totals[name] += share / len(images)
    return sorted(totals.items(), key=lambda x: -x[1])[:2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = create_engine(args.dsn)
    query = f"SELECT sku, pixels, height, width FROM {SRC_TABLE} WHERE is_active ORDER BY sku"
    if args.max_rows:
        query += f" LIMIT {args.max_rows}"
    df = pd.read_sql(query, engine)
    log.info("Loaded %d image records", len(df))

    rng = np.random.default_rng(RANDOM_STATE)
    results = []
    for sku, group in df.groupby("sku"):
        images = [
            np.frombuffer(row.pixels, np.uint8).reshape(int(row.height), int(row.width), 3)
            for row in group.itertuples()
        ]
        top_colors = aggregate_colors(images, rng)
        top_colors += [(None, 0.0)] * (2 - len(top_colors))
        results.append({
            "sku": sku,
            "primary_color": top_colors[0][0],
            "primary_share": round(top_colors[0][1], 3),
            "secondary_color": top_colors[1][0],
            "secondary_share": round(top_colors[1][1], 3),
        })

    output = pd.DataFrame(results)
    schema, table = DEST_TABLE.split(".")
    output.to_sql(table, engine, schema=schema, if_exists="replace", index=False)
    log.info("Saved %d SKUs to %s", len(output), DEST_TABLE)


if __name__ == "__main__":
    main()
