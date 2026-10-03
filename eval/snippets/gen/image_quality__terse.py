import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("img_qc")

SRC = "catalog.product_images"
DST = "catalog.image_quality_scores"

LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
YCO = np.array([0.2126, 0.7152, 0.0722], np.float32)

B_MIN, D_MIN, C_MAX = 120.0, 0.35, 0.08
W_T, B_F = 0.92, 0.06
W_S_MIN, M_S_PX = 0.85, 1000
A_R = (0.75, 1.34)

W = {"b": 0.35, "e": 0.25, "bg": 0.25, "r": 0.15}


@dataclass
class Q:
    id: str
    sku: str
    b: float
    m: float
    c: float
    w: float
    w_px: int
    h_px: int
    s: float
    r: str


def to_y(p: np.ndarray) -> np.ndarray:
    return (p[..., :3].astype(np.float32) / 255.0) @ YCO


def conv2d(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    s = np.lib.stride_tricks.sliding_window_view(x, k.shape)
    return np.einsum("ijkl,kl->ij", s, k[::-1, ::-1])


def b_var(y: np.ndarray) -> float:
    return float(conv2d(y * 255.0, LAP).var())


def e_stats(y: np.ndarray) -> tuple[float, float]:
    clp = (y <= 0.01) | (y >= 0.99)
    return float(y.mean()), float(clp.mean())


def w_share(y: np.ndarray) -> float:
    h, w = y.shape
    b = max(1, int(min(h, w) * B_F))
    m = np.zeros_like(y, dtype=bool)
    m[:b], m[-b:] = True, True
    m[:, :b], m[:, -b:] = True, True
    return float((y[m] >= W_T).mean())


def parts(bv: float, m: float, c: float, w: float, h: int, w_px: int) -> dict[str, float]:
    bl = np.clip(np.log1p(bv) / np.log1p(4 * B_MIN), 0, 1)
    ex = np.clip(1 - abs(m - 0.6) / 0.6, 0, 1) * np.clip(1 - c / (2 * C_MAX), 0, 1)
    bg = np.clip(w / W_S_MIN, 0, 1)
    ar = w_px / h
    res = np.clip(min(h, w_px) / M_S_PX, 0, 1) * (1.0 if A_R[0] <= ar <= A_R[1] else 0.6)
    return {"b": float(bl), "e": float(ex), "bg": float(bg), "r": float(res)}


def flags(bv: float, m: float, c: float, w: float, h: int, w_px: int) -> list[str]:
    o = []
    if bv < B_MIN: o.append("BLURRY")
    if m < D_MIN or c > C_MAX: o.append("DARK")
    if w < W_S_MIN: o.append("BUSY_BACKGROUND")
    ar = w_px / h
    if min(h, w_px) < M_S_PX or not (A_R[0] <= ar <= A_R[1]): o.append("LOW_RES")
    return o


def eval_one(iid: str, sku: str, px: np.ndarray) -> Q:
    h, w_px = px.shape[:2]
    y = to_y(px)
    bv = b_var(y)
    m, c = e_stats(y)
    w = w_share(y)
    ps = parts(bv, m, c, w, h, w_px)
    s = 100.0 * sum(W[k] * v for k, v in ps.items())
    rf = flags(bv, m, c, w, h, w_px)
    return Q(iid, sku, bv, m, c, w, w_px, h, round(s, 1), ",".join(rf))


def fetch(db, n: int | None) -> pd.DataFrame:
    q = f"SELECT image_id, sku, pixels, height, width FROM {SRC} WHERE is_active"
    if n: q += f" LIMIT {n}"
    return pd.read_sql(q, db)


def unpack(r: pd.Series) -> np.ndarray:
    return np.frombuffer(r.pixels, dtype=np.uint8).reshape(int(r.height), int(r.width), 3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db = create_engine(a.dsn)
    df = fetch(db, a.limit)
    log.info("loaded %d images", len(df))

    res = [eval_one(r.image_id, r.sku, unpack(r)).__dict__ for r in df.itertuples()]
    out = pd.DataFrame(res)
    log.info("avg score %.1f, flagged %d", out.s.mean(), (out.r != "").sum())

    if a.dry_run:
        print(out.head(20).to_string(index=False))
        return
    sch, tbl = DST.split(".")
    out.to_sql(tbl, db, schema=sch, if_exists="replace", index=False)
    log.info("wrote %d rows to %s", len(out), DST)


if __name__ == "__main__":
    main()
