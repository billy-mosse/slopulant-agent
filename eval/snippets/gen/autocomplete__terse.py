import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("auto")

SRC = "search.query_logs"
DST = "search.autocomplete_index"
HALF_LIFE = 14.0
K = 8
MIN_SCORE = 3.0
CLICK_BONUS = 0.6
MAX_LEN = 20

BAD = {"fuck", "shit", "bitch", "cunt", "porn", "nsfw", "dick", "pussy", "nazi", "slut"}
TBL = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "$": "s", "@": "a"})

Q = f"""
SELECT LOWER(TRIM(query_text)) AS q,
       event_date,
       COUNT(*) AS cnt,
       SUM((clicked_sku IS NOT NULL)::int) AS clk
FROM {SRC}
WHERE event_date >= CURRENT_DATE - INTERVAL '120 days'
GROUP BY 1, 2
"""


@dataclass(frozen=True)
class S:
    t: str
    v: float


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 '&\-]", " ", s.lower())).strip()


def bad_q(s: str) -> bool:
    t = s.translate(TBL)
    return any(w in t.split() for w in BAD) or any(b in t.replace(" ", "") for b in BAD if len(b) > 4)


def stem(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "sses", "xes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us")) and len(w) > 3:
        return w[:-1]
    return w


def canon(s: str) -> str:
    return " ".join(stem(w) for w in s.replace("-", " ").split())


def decay(d: float) -> float:
    return math.exp(-math.log(2) * d / HALF_LIFE)


def score(df: pd.DataFrame, ref: date) -> pd.DataFrame:
    df = df.assign(q=df.q.map(norm))
    df = df[(df.q.str.len().between(2, 60)) & (~df.q.map(bad_q))]
    age = (pd.Timestamp(ref) - pd.to_datetime(df.event_date)).dt.days.clip(lower=0)
    w = age.map(decay)
    df["ws"] = df.cnt * w
    df["wc"] = df.clk * w
    g = df.groupby("q")[["ws", "wc"]].sum().reset_index()
    g = g[g.ws >= MIN_SCORE]
    r = (g.wc + 1) / (g.ws + 2)
    g["v"] = g.ws.map(math.log1p) * ((1 - CLICK_BONUS) + CLICK_BONUS * r)
    return merge(g)


def merge(g: pd.DataFrame) -> pd.DataFrame:
    g = g.assign(k=g.q.map(canon))
    best = g.sort_values("v", ascending=False).drop_duplicates("k")
    pool = g.groupby("k").v.sum()
    return best.assign(v=best.k.map(pool))[["q", "v"]]


class T:
    def __init__(self, n: int = K):
        self.r = {}
        self.n = n

    def add(self, s: S) -> None:
        nd = self.r
        for c in s.t[:MAX_LEN]:
            nd = nd.setdefault(c, {})
            top = nd.setdefault("$", [])
            top.append(s)
            if len(top) > self.n * 2:
                top.sort(key=lambda x: -x.v)
                nd["$"] = top[: self.n]

    def trim(self, nd: dict | None = None) -> None:
        nd = self.r if nd is None else nd
        if "$" in nd:
            nd["$"] = sorted(nd["$"], key=lambda x: -x.v)[: self.n]
        for k, c in nd.items():
            if k != "$":
                self.trim(c)

    def get(self, p: str) -> list[S]:
        nd = self.r
        for c in norm(p):
            if c not in nd:
                return []
            nd = nd[c]
        return nd.get("$", [])

    def items(self, nd: dict | None = None, p: str = ""):
        nd = self.r if nd is None else nd
        for k, c in nd.items():
            if k == "$":
                continue
            x = p + k
            yield x, [(s.t, round(s.v, 4)) for s in c.get("$", [])]
            yield from self.items(c, x)


def mk(df: pd.DataFrame, ref: date) -> T:
    t = T()
    for _, r in score(df, ref).iterrows():
        t.add(S(r.q, r.v))
    t.trim()
    return t


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--as-of", default=date.today().isoformat())
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    eng = create_engine(a.dsn)
    ref = datetime.fromisoformat(a.as_of).date()
    trie = mk(pd.read_sql(Q, eng), ref)
    rows = [{"prefix": p, "suggestions": json.dumps(s), "built_at": ref} for p, s in trie.items()]
    log.info("%d prefixes; sample 'duv' -> %s", len(rows), [s.t for s in trie.get("duv")])
    sch, tbl = DST.split(".")
    pd.DataFrame(rows).to_sql(tbl, eng, schema=sch, if_exists="replace", index=False, chunksize=10_000)


if __name__ == "__main__":
    main()
