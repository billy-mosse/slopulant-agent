"""Module for evaluating product image eligibility for main product display areas.

This system computes a holistic quality score (0–100) for every active product image using
four perceptually meaningful attributes: sharpness, tonal balance, compositional cleanliness,
and resolution adequacy. Scoring results power downstream merchandising workflows by flagging
images that require replacement or manual review.

Each signal is normalized to a 0–1 range and aggregated via weighted component scoring,
producing both an overall score and human-readable annotation strings indicating specific
deficiencies.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


@dataclass
class ImageAssessment:
    """Encapsulates detailed metrics and a final verdict for one product image."""
    identifier: str
    product_sku: str
    image_height: int
    image_width: int
    overall_rating: float
    annotations: list[str]


class ImageAssessmentEngine:
    """Performs perceptual quality evaluation on decoded RGB images."""

    _KERNEL_LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    _LUMINANCE_COEFFS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    _SHARPNESS_THRESHOLD = 120.0
    _AVERAGE_LUMINANCE_TARGET = 0.6
    _AVERAGE_LUMINANCE_TOLERANCE = 0.6
    _CLIPPED_PIXEL_FRACTION_MAX = 0.08
    _WHITE_BORDER_THRESHOLD = 0.92
    _BORDER_WIDTH_FRACTION = 0.06
    _MINIMAL_DIMENSION_PX = 1000
    _ASPECT_RATIO_MIN = 0.75
    _ASPECT_RATIO_MAX = 1.34

    _WEIGHTS = {"sharpness": 0.35, "tonal": 0.25, "frame_clarity": 0.25, "dimensions": 0.15}

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    def evaluate(self, image_id: str, sku: str, raw_bytes: bytes, h: int, w: int) -> ImageAssessment:
        """Produce a full quality assessment for one decoded image."""
        image_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3))
        luminance_channel = self._extract_luminance(image_array)

        sharpness_metric = self._compute_sharpness(luminance_channel)
        tonal_metrics = self._analyze_tonal_balance(luminance_channel)
        frame_clarity_metric = self._measure_border_cleanliness(luminance_channel, h, w)
        dimension_score = self._assess_resolution(h, w)

        normalized_scores = {
            "sharpness": self._map_sharpness(sharpness_metric),
            "tonal": self._map_tonal(*tonal_metrics),
            "frame_clarity": self._map_frame_clarity(frame_clarity_metric),
            "dimensions": self._map_dimensions(h, w),
        }

        final_score = 100.0 * sum(
            self._WEIGHTS[key] * value for key, value in normalized_scores.items()
        )
        annotations = self._diagnose_issues(
            sharpness_metric, *tonal_metrics, frame_clarity_metric, h, w
        )

        return ImageAssessment(
            identifier=image_id,
            product_sku=sku,
            image_height=h,
            image_width=w,
            overall_rating=round(final_score, 1),
            annotations=annotations,
        )

    def _extract_luminance(self, rgb: np.ndarray) -> np.ndarray:
        """Converts 8-bit RGB to single-channel luminance in [0, 1]."""
        return (rgb[..., :3] / 255.0) @ self._LUMINANCE_COEFFS

    def _compute_sharpness(self, luminance: np.ndarray) -> float:
        """Quantifies edge content via Laplacian variance on the luminance channel."""
        response = self._convolve(luminance * 255.0, self._KERNEL_LAPLACIAN)
        return float(response.var())

    def _convolve(self, signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Performs 2D valid convolution using sliding-window einsum optimization."""
        kh, kw = kernel.shape
        windows = np.lib.stride_tricks.sliding_window_view(signal, (kh, kw))
        return np.einsum("ijkl,kl->ij", windows, kernel[::-1, ::-1])

    def _analyze_tonal_balance(self, luminance: np.ndarray) -> tuple[float, float]:
        """Returns mean luminance and clipped-pixel fraction."""
        clipped = (luminance <= 0.01) | (luminance >= 0.99)
        return float(luminance.mean()), float(clipped.mean())

    def _measure_border_cleanliness(self, luminance: np.ndarray, height: int, width: int) -> float:
        """Computes fraction of near-white pixels within an outer margin."""
        margin = max(1, int(min(height, width) * self._BORDER_WIDTH_FRACTION))
        mask = np.zeros_like(luminance, dtype=bool)
        mask[:margin, :], mask[-margin:, :] = True, True
        mask[:, :margin], mask[:, -margin:] = True, True
        return float((luminance[mask] >= self._WHITE_BORDER_THRESHOLD).mean())

    def _assess_resolution(self, height: int, width: int) -> float:
        """Evaluates minimum side length and aspect ratio compliance."""
        min_side_ratio = min(height, width) / self._MINIMAL_DIMENSION_PX
        aspect = width / height
        aspect_bonus = 1.0 if self._ASPECT_RATIO_MIN <= aspect <= self._ASPECT_RATIO_MAX else 0.6
        return min_side_ratio * aspect_bonus

    def _map_sharpness(self, raw: float) -> float:
        return float(np.clip(np.log1p(raw) / np.log1p(4 * self._SHARPNESS_THRESHOLD), 0, 1))

    def _map_tonal(self, mean: float, clipped: float) -> float:
        exposure_fidelity = 1.0 - abs(mean - self._AVERAGE_LUMINANCE_TARGET) / self._AVERAGE_LUMINANCE_TOLERANCE
        exposure_fidelity = max(0.0, min(1.0, exposure_fidelity))
        clipping_penalty = 1.0 - clipped / (2 * self._CLIPPED_PIXEL_FRACTION_MAX)
        clipping_penalty = max(0.0, min(1.0, clipping_penalty))
        return exposure_fidelity * clipping_penalty

    def _map_frame_clarity(self, ratio: float) -> float:
        return float(np.clip(ratio / 0.85, 0, 1))

    def _map_dimensions(self, height: int, width: int) -> float:
        return float(self._assess_resolution(height, width))

    def _diagnose_issues(
        self,
        sharpness: float,
        mean: float,
        clipped: float,
        border_white_ratio: float,
        height: int,
        width: int,
    ) -> list[str]:
        diagnostics = []
        if sharpness < self._SHARPNESS_THRESHOLD:
            diagnostics.append("BLURRY")
        if mean < 0.35 or clipped > self._CLIPPED_PIXEL_FRACTION_MAX:
            diagnostics.append("DARK")
        if border_white_ratio < 0.85:
            diagnostics.append("BUSY_BACKGROUND")
        aspect = width / height
        if min(height, width) < self._MINIMAL_DIMENSION_PX or not (
            self._ASPECT_RATIO_MIN <= aspect <= self._ASPECT_RATIO_MAX
        ):
            diagnostics.append("LOW_RES")
        return diagnostics


def fetch_candidate_images(dsn: str, maximum: int | None = None) -> pd.DataFrame:
    """Retrieves active images from persistent storage for processing."""
    query = (
        "SELECT image_id, sku, pixels, height, width "
        "FROM catalog.product_images WHERE is_active"
    )
    if maximum is not None:
        query += f" LIMIT {maximum}"
    engine = create_engine(dsn)
    return pd.read_sql(query, engine)


def process_batch(engine: ImageAssessmentEngine, df: pd.DataFrame) -> pd.DataFrame:
    """Apply assessment pipeline to a dataframe of images."""
    results = []
    for row in df.itertuples():
        assessment = engine.evaluate(
            row.image_id,
            row.sku,
            row.pixels,
            int(row.height),
            int(row.width),
        )
        results.append(assessment)
    return pd.DataFrame([
        {
            "image_id": r.identifier,
            "sku": r.product_sku,
            "overall_rating": r.overall_rating,
            "height": r.image_height,
            "width": r.image_width,
            "reasons": ",".join(r.annotations),
        }
        for r in results
    ])


def persist_scores(df: pd.DataFrame, dsn: str) -> None:
    """Store computed assessments back to persistent storage."""
    engine = create_engine(dsn)
    df.to_sql("image_quality_scores", engine, schema="catalog", if_exists="replace", index=False)


def main() -> None:
    """Orchestrate the nightly image quality evaluation."""
    parser = argparse.ArgumentParser(
        description="Execute product image quality evaluation and store outcomes."
    )
    parser.add_argument("--dsn", required=True, help="Database connection string")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of processed images")
    parser.add_argument("--dry-run", action="store_true", help="Print preview instead of persisting")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("image_quality_evaluator")

    logger.info("loading candidate images")
    raw_images = fetch_candidate_images(args.dsn, args.max_images)
    logger.info("loaded %d images", len(raw_images))

    engine = ImageAssessmentEngine()
    scores = process_batch(engine, raw_images)
    logger.info("computed scores: mean=%.1f, flagged=%d", scores.overall_rating.mean(), (scores.reasons != "").sum())

    if args.dry_run:
        logger.info("dry-run mode – preview:\n%s", scores.head(20).to_string(index=False))
        return

    persist_scores(scores, args.dsn)
    logger.info("wrote %d assessment records", len(scores))


if __name__ == "__main__":
    main()
