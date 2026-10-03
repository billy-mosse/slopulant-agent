import argparse
import logging
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger("img_assessment")

INPUT_TBL = "catalog.product_images"
OUTPUT_TBL = "catalog.image_quality_scores"

Y_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
KERNEL_SCHARR = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.float32)

THRESH_BLUR = 110.0
THRESH_DARK = 0.38
THRESH_OVEREXPOSE = 0.07
THRESH_WHITE = 0.9
BORDER_RATIO = 0.05
THRESH_WHITE_BORDER = 0.8
MIN_DIM = 1000
ASPECT_MIN, ASPECT_MAX = 0.75, 1.34

WEIGHTS = {"sharpness": 0.35, "brightness": 0.25, "border_cleanliness": 0.25, "dimensions": 0.15}


@dataclass
class Assessment:
    image_id: str
    sku: str
    sharpness: float
    brightness: float
    overexposure: float
    border_whiteness: float
    width: int
    height: int
    final_score: float
    flags: str


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB to float [0,1] grayscale."""
    return (rgb @ Y_WEIGHTS).astype(np.float32) / 255.0


def edge_response(gray: np.ndarray) -> float:
    """Compute gradient magnitude variance using Scharr kernel."""
    gx = cv2.filter2D(gray, -1, KERNEL_SCHARR)
    gy = cv2.filter2D(gray, -1, KERNEL_SCHARR.T)
    mag = np.hypot(gx, gy)
    return float(mag.var())


def brightness_metrics(gray: np.ndarray) -> tuple[float, float]:
    """Return mean brightness and fraction of saturated pixels."""
    saturated = (gray <= 0.02) | (gray >= 0.98)
    return float(gray.mean()), float(saturated.mean())


def border_analysis(gray: np.ndarray) -> float:
    """Fraction of border pixels that are near-white."""
    h, w = gray.shape
    pad = max(1, int(min(h, w) * BORDER_RATIO))
    border = np.zeros_like(gray, dtype=bool)
    border[:pad, :] = border[-pad:, :] = True
    border[:, :pad] = border[:, -pad:] = True
    return float(gray[border].mean() >= THRESH_WHITE)


def normalize_components(sharp: float, bright: float, over: float, white: float, h: int, w: int) -> dict:
    """Map raw metrics to [0,1] scores."""
    s = np.clip(np.log1p(sharp) / np.log1p(4 * THRESH_BLUR), 0, 1)
    b = np.clip(1 - abs(bright - 0.62) / 0.6, 0, 1) * np.clip(1 - over / (2 * THRESH_OVEREXPOSE), 0, 1)
    c = np.clip(white / THRESH_WHITE_BORDER, 0, 1)
    ar = w / h
    d = np.clip(min(h, w) / MIN_DIM, 0, 1) * (1.0 if ASPECT_MIN <= ar <= ASPECT_MAX else 0.65)
    return {"sharpness": s, "brightness": b, "border_cleanliness": c, "dimensions": d}


def flag_issues(sharp: float, bright: float, over: float, white: float, h: int, w: int) -> list[str]:
    flags = []
    if sharp < THRESH_BLUR:
        flags.append("BLURRY")
    if bright < THRESH_DARK or over > THRESH_OVEREXPOSE:
        flags.append("DARK")
    if white < THRESH_WHITE_BORDER:
        flags.append("BUSY_BACKGROUND")
    ar = w / h
    if min(h, w) < MIN_DIM or not (ASPECT_MIN <= ar <= ASPECT_MAX):
        flags.append("LOW_RES")
    return flags


def assess(image_id: str, sku: str, raw: np.ndarray) -> Assessment:
    h, w = raw.shape[:2]
    gray = to_grayscale(raw)
    sharp = edge_response(gray)
    bright, over = brightness_metrics(gray)
    white = border_analysis(gray)
    norms = normalize_components(sharp, bright, over, white, h, w)
    score = 100 * sum(WEIGHTS[k] * v for k, v in norms.items())
    flags = flag_issues(sharp, bright, over, white, h, w)
    return Assessment(
        image_id, sku, sharp, bright, over, white, w, h, round(score, 1), ",".join(flags)
    )


def fetch_images(conn, max_rows: int | None) -> pd.DataFrame:
    q = f"SELECT image_id, sku, data, height, width FROM {INPUT_TBL} WHERE active = true"
    if max_rows:
        q += f" LIMIT {max_rows}"
    return pd.read_sql(q, conn)


def unpack(row: pd.Series) -> np.ndarray:
    return np.frombuffer(row.data, dtype=np.uint8).reshape(int(row.height), int(row.width), 3)


def main() -> None:
    p = argparse.ArgumentParser(description="Assess product image quality.")
    p.add_argument("--dsn", required=True)
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--dry", action="store_true")
    opts = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = create_engine(opts.dsn)
    df = fetch_images(engine, opts.max)
    logger.info("fetched %d images", len(df))

    results = [assess(r.image_id, r.sku, unpack(r)).__dict__ for r in df.itertuples()]
    out = pd.DataFrame(results)
    flagged = (out.flags != "").sum()
    logger.info("average score %.1f, flagged %d", out.final_score.mean(), flagged)

    if opts.dry:
        print(out.head(20).to_string(index=False))
        return

    schema, tbl = OUTPUT_TBL.split(".")
    out.to_sql(tbl, engine, schema=schema, if_exists="replace", index=False)
    logger.info("stored %d records in %s", len(out), OUTPUT_TBL)


if __name__ == "__main__":
    main()
