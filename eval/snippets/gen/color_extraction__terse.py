import argparse
import logging
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from palette import closest_name

log = logging.getLogger("color_agg")

SRC_TBL = "catalog.product_images"
TGT_TBL = "catalog.product_colors"

CLUSTERS = 4
EM_ITERS = 8
WHITE_THRESH = 235
PIX_LIMIT = 20_000
RNG_SEED = 7

RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
D65 = np.array([0.95047, 1.0, 1.08883])


def _srgb_lin(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _rgb2lab(px: np.ndarray) -> np.ndarray:
    xyz = _srgb_lin(px.astype(np.float64) / 255.0) @ RGB2XYZ.T
    t = xyz / D65
    eps, k = 216 / 24389, 24389 / 27
    f = np.where(t > eps, np.cbrt(t), (k * t + 16) / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def _fg_px(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    px = img[..., :3].reshape(-1, 3)
    px = px[~(px >= WHITE_THRESH).all(axis=1)]
    if len(px) > PIX_LIMIT:
        idx = rng.choice(len(px), PIX_LIMIT, replace=False)
        px = px[idx]
    return px


def _kpp_init(pts: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    c = [pts[rng.integers(len(pts))]]
    for _ in range(1, k):
        d2 = ((pts[:, None] - np.array(c)[None]) ** 2).sum(axis=2).min(axis=1)
        c.append(pts[rng.choice(len(pts), p=d2 / d2.sum())])
    return np.array(c)


def _kmeans(pts: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cent = _kpp_init(pts, k, rng)
    for _ in range(iters):
        lbl = ((pts[:, None] - cent[None]) ** 2).sum(axis=2).argmin(axis=1)
        new = np.array([pts[lbl == j].mean(axis=0) if (lbl == j).any() else cent[j] for j in range(k)])
        if np.allclose(new, cent, atol=1e-3):
            break
        cent = new
    return cent, lbl


def _img_to_colors(img: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    px = _fg_px(img, rng)
    if len(px) < CLUSTERS * 10:
        return {}
    lab = _rgb2lab(px)
    cent, lbl = _kmeans(lab, CLUSTERS, EM_ITERS, rng)
    shares = np.bincount(lbl, minlength=CLUSTERS) / len(lab)
    acc: dict[str, float] = Counter()
    for c, s in zip(cent, shares):
        name, _ = closest_name(c)
        acc[name] += float(s)
    return dict(acc)


def _sku_colors(imgs: list[np.ndarray], rng: np.random.Generator) -> list[tuple[str, float]]:
    acc: dict[str, float] = Counter()
    for img in imgs:
        for name, s in _img_to_colors(img, rng).items():
            acc[name] += s / len(imgs)
    return sorted(acc.items(), key=lambda x: -x[1])[:2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--max_rows", type=int, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    eng = create_engine(args.dsn)
    q = f"SELECT sku, data, h, w FROM {SRC_TBL} WHERE active ORDER BY sku"
    if args.max_rows:
        q += f" LIMIT {args.max_rows}"
    df = pd.read_sql(q, eng)
    log.info("read %d rows", len(df))

    rng = np.random.default_rng(RNG_SEED)
    out = []
    for sku, grp in df.groupby("sku"):
        imgs = [np.frombuffer(r.data, np.uint8).reshape(int(r.h), int(r.w), 3) for r in grp.itertuples()]
        top = _sku_colors(imgs, rng)
        top += [(None, 0.0)] * (2 - len(top))
        out.append({
            "sku": sku,
            "color_1": top[0][0],
            "share_1": round(top[0][1], 3),
            "color_2": top[1][0],
            "share_2": round(top[1][1], 3),
        })

    res = pd.DataFrame(out)
    sch, tbl = TGT_TBL.split(".")
    res.to_sql(tbl, eng, schema=sch, if_exists="replace", index=False)
    log.info("wrote %d SKUs to %s", len(res), TGT_TBL)


if __name__ == "__main__":
    main()
