import re
import json
import argparse
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import pandas as pd
from sqlalchemy import create_engine

# --- constants ---
P_QRY = """
SELECT LOWER(TRIM(query_text)) query_text, COUNT(*) n
FROM search.query_logs
WHERE event_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY 1 HAVING COUNT(*) >= %s
"""

P_CLICKS = """
SELECT ql.query_text, pc.category_id, COUNT(*) clicks
FROM search.query_logs ql
JOIN catalog.predicted_category pc ON pc.sku = ql.clicked_sku
WHERE ql.clicked_sku IS NOT NULL
  AND ql.event_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY 1, 2
"""

MIN_CAT_FRAC = 0.05
MAX_CATS = 3
PRIOR = 1.0
MIN_TOK_CNT = 3

PrixRx = [
    (re.compile(r"\b(?:under|below|less than)\s*\$?(\d+)"), "max_price"),
    (re.compile(r"\b(?:over|above|more than)\s*\$?(\d+)"), "min_price"),
    (re.compile(r"\$(\d+)\s*-\s*\$?(\d+)"), "range"),
]

ATTRS = {
    "size": ["twin xl", "twin", "full", "queen", "california king", "king"],
    "material": ["linen", "cotton", "percale", "sateen", "bamboo", "silk", "wool", "velvet", "jute"],
}

ALPHA = set("abcdefghijklmnopqrstuvwxyz'-")
TOKEN_RE = re.compile(r"[a-z0-9'\-]+")
EDIT_PEN = {0: 1.0, 1: 0.08, 2: 0.004}

OUTPUT_TBL = "search.query_intents"


# --- intent extraction ---
@dataclass
class QIntent:
    raw: str
    norm: str
    cats: List[Tuple[str, float]] = field(default_factory=list)
    flt: Dict[str, object] = field(default_factory=dict)


def _price_flt(q: str) -> Dict[str, float]:
    out = {}
    for rx, kind in PrixRx:
        m = rx.search(q)
        if not m:
            continue
        if kind == "range":
            out["min_price"], out["max_price"] = float(m.group(1)), float(m.group(2))
        else:
            out[kind] = float(m.group(1))
    return out


def _attr_flt(q: str) -> Dict[str, str]:
    out = {}
    for k, vs in ATTRS.items():
        for v in vs:
            if re.search(rf"\b{re.escape(v)}\b", q):
                out[k] = v
                break
    return out


def _strip_flt(q: str) -> str:
    for rx, _ in PrixRx:
        q = rx.sub("", q)
    return re.sub(r"\s+", " ", q).strip()


def _cat_distr(clks: pd.DataFrame, prior: float = PRIOR) -> Dict[str, List[Tuple[str, float]]]:
    out: Dict[str, List[Tuple[str, float]]] = {}
    for q, grp in clks.groupby("query_text"):
        w = grp.set_index("category_id")["clicks"].astype(float) + prior
        w = w / w.sum()
        srt = w.sort_values(ascending=False)
        out[q] = [(c, round(p, 4)) for c, p in srt.items() if p >= MIN_CAT_FRAC][:MAX_CATS]
    return out


class Resolver:
    def __init__(self, dist: Dict[str, List[Tuple[str, float]]]):
        self.d = dist
        self.bk = defaultdict(list)
        for q, cats in dist.items():
            for tok in q.split():
                self.bk[tok].extend(cats)

    def cats(self, q: str) -> List[Tuple[str, float]]:
        if q in self.d:
            return self.d[q]
        agg = defaultdict(float)
        for t in q.split():
            for c, p in self.bk.get(t, []):
                agg[c] += p
        total = sum(agg.values()) or 1.0
        return sorted(((c, round(v / total, 4)) for c, v in agg.items()), key=lambda x: -x[1])[:MAX_CATS]

    def resolve(self, raw: str, corr: str) -> QIntent:
        filt = {**_price_flt(corr), **_attr_flt(corr)}
        return QIntent(raw, corr, self.cats(_strip_flt(corr)), filt)


# --- spelling ---
def tok(q: str) -> List[str]:
    return TOKEN_RE.findall(q.lower())


def edits1(w: str) -> set:
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    d = {l + r[1:] for l, r in splits if r}
    t = {l + r[1] + r[0] + r[2:] for l, r in splits if len(r) > 1}
    r = {l + c + r[1:] for l, r in splits if r for c in ALPHA}
    i = {l + c + r for l, r in splits for c in ALPHA}
    return d | t | r | i


def edits2(w: str) -> set:
    return {x2 for x1 in edits1(w) for x2 in edits1(x1)}


class Vocab:
    def __init__(self, cnt: Counter):
        self.c = Counter({w: c for w, c in cnt.items() if c >= MIN_TOK_CNT})
        self.t = sum(self.c.values()) or 1

    @classmethod
    def from_q(cls, q: List[Tuple[str, int]]) -> "Vocab":
        cnt = Counter()
        for s, n in q:
            for t in tok(s):
                cnt[t] += n
        return cls(cnt)

    def p(self, w: str) -> float:
        return self.c.get(w, 0) / self.t

    def k(self, ws) -> set:
        return {x for x in ws if x in self.c}


class Corrector:
    def __init__(self, v: Vocab):
        self.v = v
        self._c = lru_cache(maxsize=200_000)(self._c1)

    def cand(self, w: str) -> Dict[str, int]:
        if w in self.v.c:
            return {w: 0}
        r = {x: 1 for x in self.v.k(edits1(w))}
        if not r and len(w) > 4:
            r = {x: 2 for x in self.v.k(edits2(w))}
        return r

    def score(self, w: str, c: str, d: int) -> float:
        if d > 2:
            d = d
        logp = math.log(self.v.p(c) + 1e-12)
        return logp + math.log(EDIT_PEN[min(d, 2)])

    def _c1(self, w: str) -> str:
        if w.isdigit() or len(w) <= 2:
            return w
        cands = self.cand(w)
        if not cands:
            return w
        return max(cands, key=lambda x: self.score(w, x, cands[x]))

    def fix(self, q: str) -> str:
        return " ".join(self._c(t) for t in tok(q))


# --- pipeline ---
def go(db, min_n: int, date: str) -> pd.DataFrame:
    f = pd.read_sql(P_QRY, db, params=(min_n,))
    c = pd.read_sql(P_CLICKS, db)
    v = Vocab.from_q(list(zip(f.query_text, f.n)))
    cr = Corrector(v)
    c["query_text"] = c.query_text.str.lower().map(cr.fix)
    resolver = Resolver(_cat_distr(c.groupby(["query_text", "category_id"], as_index=False).clicks.sum()))

    rows = []
    for q in f.query_text:
        r = resolver.resolve(q, cr.fix(q))
        rows.append({
            "query_text": r.raw,
            "corrected_query": r.norm,
            "top_categories": json.dumps(r.cats),
            "attribute_filters": json.dumps(r.flt),
            "run_date": date,
        })
    out = pd.DataFrame(rows)
    logging.info("resolved %d queries, %d corrected", len(out), (out.query_text != out.corrected_query).sum())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--run-date", required=True)
    p.add_argument("--min-count", type=int, default=5)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    engine = create_engine(a.dsn)
    out = go(engine, a.min_count, a.run_date)
    schema, tbl = OUTPUT_TBL.split(".")
    out.to_sql(tbl, engine, schema=schema, if_exists="append", index=False)


if __name__ == "__main__":
    main()
