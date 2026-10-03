import argparse
import logging
import re
import unicodedata
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine, text

from lexicon import ASPECTS, INTENSIFIERS, NEGATION, NEGATORS, SENTIMENTS, PUNCT

log = logging.getLogger("sent")

W = 5
B = 3
MAX_PHRASE = max(len(x.split()) for x in (*SENTIMENTS, *INTENSIFIERS, *(t for ts in ASPECTS.values() for t in ts)))
ASPECT_MAP = {t: a for a, ts in ASPECTS.items() for t in ts}

def norm(s: str | None) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s.lower())
    s = s.replace("’", "'").replace("‘", "'")
    return re.sub(r"(.)\1{2,}", r"\1\1", s)

def _toks(s: str) -> list[str]:
    raw = re.findall(r"[a-z]+(?:'[a-z]+)?|[.,;!?]", s)
    out, i = [], 0
    while i < len(raw):
        for n in range(min(MAX_PHRASE, len(raw)-i), 1, -1):
            cand = " ".join(raw[i:i+n])
            if cand in SENTIMENTS or cand in INTENSIFIERS or cand in ASPECT_MAP:
                out.append(cand); i += n; break
        else: out.append(raw[i]); i += 1
    return out

def _polar(ts: list[str], j: int) -> float:
    v = SENTIMENTS[ts[j]]
    mult = 1.0
    for k in range(j-1, max(-1, j-1-B), -1):
        t = ts[k]
        if t in PUNCT: break
        if t in INTENSIFIERS: mult *= INTENSIFIERS[t]
        elif t in NEGATORS: mult *= NEGATION; break
    return max(-1.0, min(1.0, v * mult))

def _aspect_scores(ts: list[str]) -> Iterable[tuple[str, float]]:
    for i, t in enumerate(ts):
        a = ASPECT_MAP.get(t)
        if not a: continue
        num, den = 0.0, 0.0
        for j in range(max(0, i-W), min(len(ts), i+W+1)):
            if ts[j] not in SENTIMENTS: continue
            w = 1.0 / (1 + abs(i-j))
            num += w * _polar(ts, j)
            den += w
        if den: yield a, num/den

def _score(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        txt = norm(f"{r.title or ''}. {r.body or ''}")
        for a, s in _aspect_scores(_toks(txt)):
            rows.append((r.review_id, r.sku, a, s))
    return pd.DataFrame(rows, columns=["review_id", "sku", "aspect", "score"])

def _agg(m: pd.DataFrame) -> pd.DataFrame:
    pr = m.groupby(["review_id","sku","aspect"], as_index=False)["score"].mean()
    pr["pos"] = pr["score"] > 0.15
    pr["neg"] = pr["score"] < -0.15
    g = pr.groupby(["sku","aspect"]).agg(
        mentions=("review_id","nunique"),
        avg=("score","mean"),
        pos_rate=("pos","mean"),
        neg_rate=("neg","mean")).reset_index()
    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--min", type=int, default=3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    eng = create_engine(args.dsn)

    df = pd.read_sql(text("SELECT review_id, sku, title, body FROM reviews.approved"), eng)
    m = _score(df)
    log.info("%d reviews → %d mentions", len(df), len(m))
    g = _agg(m).query(f"mentions >= {args.min}")
    g.to_sql("aspect_sentiment", eng, schema="reviews", if_exists="replace", index=False)

if __name__ == "__main__":
    main()
