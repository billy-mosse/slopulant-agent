import argparse
import logging
import re
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import create_engine

logger = logging.getLogger("price_intelligence")

SQL_PRODUCTS = """
SELECT sku, name, brand, dept, dimensions, hue, msrp
FROM inventory.items
WHERE active = 1
"""

SQL_LISTINGS = """
SELECT id, vendor, name, brand, dept, amount, fetched_at
FROM market.scraped_offers
WHERE fetched_at >= NOW() - INTERVAL '7 days'
"""

RESULT_TABLE = "market.matched_offers"

SCORE_WEIGHTS = (0.68, 0.18, 0.14)
THRESHOLD = 0.81
LOG_STD = 0.42

SIZE_UNITS = {"twin", "full", "queen", "king", "cal king", "standard", "euro"}
COLOR_UNITS = {"white", "ivory", "black", "gray", "grey", "navy", "blue", "green", "sage",
               "blush", "pink", "beige", "taupe", "charcoal", "natural", "linen"}
SIZE_RE = re.compile(r"\b(\d+(?:\.\d+)?)(?:\s*(?:in|inch|inches|\"))?\b")
CM_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:cm|centimeter|centimeters)\b")
TC_RE = re.compile(r"\b(\d+)\s*(?:thread\s*count|tc)\b")
PC_RE = re.compile(r"\b(\d+)\s*(?:piece|pieces|pc|pcs)\b")
OZ_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:oz|ounce|ounces)\b")

UNIT_FIXES = [(SIZE_RE, r"\1in"), (CM_RE, r"\1cm"), (TC_RE, r"\1tc"),
              (PC_RE, r"\1pc"), (OZ_RE, r"\1oz")]
PUNCT_RE = re.compile(r"[^\w\s]")


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().replace("grey", "gray").replace("california king", "cal king")
    for pat, repl in UNIT_FIXES:
        s = pat.sub(repl, s)
    s = PUNCT_RE.sub(" ", s).replace('"', " ")
    return " ".join(s.split())


def jaccard_token_score(a: str, b: str) -> float:
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / union if union else 0.0


def extract_keyword(text: str, vocab: set) -> Optional[str]:
    candidates = [w for w in vocab if re.search(rf"\b{re.escape(w)}\b", text)]
    return max(candidates, key=len) if candidates else None


def attr_match(our_val: Optional[str], title: str, vocab: set) -> float:
    theirs = extract_keyword(title, vocab)
    if not our_val and theirs is None:
        return 0.5
    if not our_val or theirs is None:
        return 0.0
    return 1.0 if clean_text(str(our_val)) == theirs else 0.0


def price_consistency(base: float, comp: float) -> float:
    if base <= 0 or comp <= 0:
        return 0.0
    z = abs(np.log(comp / base)) / LOG_STD
    return float(2 * norm.sf(z))


def build_group_key(df: pd.DataFrame) -> pd.Series:
    return (df["brand"].fillna("").str.lower().str.strip() +
            "|" + df["dept"].fillna("").str.lower().str.strip())


def align_catalog(catalog: pd.DataFrame, offers: pd.DataFrame,
                  cutoff: float = THRESHOLD) -> pd.DataFrame:
    catalog = catalog.assign(
        clean_name=catalog["name"].map(clean_text),
        group=build_group_key(catalog)
    )
    offers = offers.assign(
        clean_name=offers["name"].map(clean_text),
        group=build_group_key(offers)
    )

    matches = []

    for grp, subset in offers.groupby("group"):
        candidates = catalog[catalog["group"] == grp]
        if candidates.empty:
            continue
        for row in subset.itertuples(index=False):
            top = None
            for prod in candidates.itertuples(index=False):
                sim_title = jaccard_token_score(prod.clean_name, row.clean_name)
                if sim_title < 0.45:
                    continue
                sim_attr = attr_match(prod.dimensions, row.clean_name, SIZE_UNITS)
                sim_attr *= attr_match(prod.hue, row.clean_name, COLOR_UNITS)
                sim_price = price_consistency(prod.msrp, row.amount)
                total = (SCORE_WEIGHTS[0] * sim_title +
                         SCORE_WEIGHTS[1] * sim_attr +
                         SCORE_WEIGHTS[2] * sim_price)
                if top is None or total > top["total"]:
                    top = {
                        "sku": prod.sku,
                        "vendor": row.vendor,
                        "offer_id": row.id,
                        "comp_price": row.amount,
                        "base_price": prod.msrp,
                        "title_sim": sim_title,
                        "attr_sim": sim_attr,
                        "price_sim": sim_price,
                        "total": total
                    }
            if top and top["total"] >= cutoff:
                matches.append(top)

    result = pd.DataFrame(matches)
    if result.empty:
        return result

    result = result.sort_values("total", ascending=False)
    result = result.drop_duplicates(subset=["sku", "vendor"])
    result["gap"] = result["comp_price"] - result["base_price"]
    result["gap_pct"] = result["gap"] / result["base_price"]
    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-reference internal SKUs with market offers")
    parser.add_argument("--db", required=True, help="Database connection string")
    parser.add_argument("--cutoff", type=float, default=THRESHOLD)
    parser.add_argument("--dry", action="store_true")
    opts = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    engine = create_engine(opts.db)
    catalog = pd.read_sql(SQL_PRODUCTS, engine)
    offers = pd.read_sql(SQL_LISTINGS, engine)
    logger.info("Loaded %d SKUs and %d offers", len(catalog), len(offers))

    aligned = align_catalog(catalog, offers, opts.cutoff)
    if aligned.empty:
        logger.warning("No matches found above threshold %.2f", opts.cutoff)
        return

    gap_stats = aligned["gap_pct"]
    logger.info("Found %d matches; median gap %.1f%%, 25/75 pctiles [%.1f%%, %.1f%%]",
                len(aligned),
                100 * gap_stats.median(),
                100 * gap_stats.quantile(0.25),
                100 * gap_stats.quantile(0.75))

    if opts.dry:
        print(aligned.head(20).to_string(index=False))
        return

    aligned["synced_at"] = pd.Timestamp.utcnow()
    schema, tbl = RESULT_TABLE.split(".")
    aligned.to_sql(tbl, engine, schema=schema, if_exists="replace", index=False)
    logger.info("Persisted results to %s", RESULT_TABLE)


if __name__ == "__main__":
    main()
