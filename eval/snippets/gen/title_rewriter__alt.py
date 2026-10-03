from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger(__name__)

SRC_PRODUCTS = "catalog.products"
SRC_ATTRS = "catalog.product_attributes"
SRC_COLORS = "catalog.product_colors"
DEST_TABLE = "catalog.seo_titles"

CONF_THRESHOLD = 0.75
COLOR_MIN_SHARE = 0.30
MAX_LEN = 70

STOPWORDS = {"a", "an", "and", "the", "of", "in", "with", "for", "x", "by"}
UNIT_PAT = re.compile(r"^\d+(\.\d+)?(cm|in|tc)?$", re.I)
NOISE_PAT = re.compile(
    r"(!{2,}|\bfree\s+shipping\b|\bsale\b|\bclearance\b|\bbest\s+seller\b|\bhot\s+deal\b|\blimited\s+time\b|\bnew!?\b)",
    re.I,
)


@dataclass
class TitleSpec:
    pattern: str
    must_have: tuple[str, ...]
    discard_order: tuple[str, ...]

    @property
    def fields(self) -> list[str]:
        return re.findall(r"\{(\w+)\}", self.pattern)


@dataclass
class OutputRow:
    seo: str
    fallback: bool
    note: str


def clean_text(s: str) -> str:
    s = NOISE_PAT.sub(" ", s or "")
    s = re.sub(r"\s*([,|-])\s*(?=[,|-]|$)", "", s)
    return re.sub(r"\s{2,}", " ", s).strip(" ,-|")


def dedup_tokens(s: str) -> str:
    seen = set()
    parts = []
    for t in s.split():
        key = re.sub(r"\W", "", t).lower()
        if key and key in seen:
            continue
        seen.add(key)
        parts.append(t)
    return " ".join(parts)


def smart_cap(s: str) -> str:
    tokens = s.split()
    for i, t in enumerate(tokens):
        core = t.strip(",")
        if UNIT_PAT.match(core) or re.search(r"\d", core):
            continue
        if i and core.lower() in STOPWORDS:
            tokens[i] = t.lower()
        else:
            tokens[i] = t.capitalize()
    return " ".join(tokens)


def normalize_value(field: str, val: object) -> str | None:
    if pd.isna(val) or val is None or val == "":
        return None
    if field == "thread_count":
        return f"{int(val)} Thread Count"
    if field == "material" and isinstance(val, str) and val.startswith("{"):
        blend = json.loads(val)
        top = sorted(blend.items(), key=lambda kv: -(kv[1] or 0))
        return "-".join(k for k, _ in top[:2]) + (" Blend" if len(top) > 1 else "")
    if field == "dimensions" and isinstance(val, str) and val.startswith("{"):
        dims = json.loads(val)
        return f"{round(dims['width_cm'])}x{round(dims['length_cm'])}cm"
    return str(val)


def compose(spec: TitleSpec, ctx: dict[str, str | None]) -> str:
    active = {k: v for k, v in ctx.items() if v}
    queue = list(spec.discard_order)
    while True:
        text = spec.pattern
        for f in spec.fields:
            text = text.replace(f"{{{f}}}", active.get(f, ""))
        text = smart_cap(dedup_tokens(clean_text(re.sub(r"\s+", " ", text))))
        if len(text) <= MAX_LEN or not queue:
            return text[:MAX_LEN].rstrip(" ,")
        active.pop(queue.pop(0), None)


def build_title(row: pd.Series, spec: TitleSpec) -> OutputRow:
    base = clean_text(row["title"])
    ctx: dict[str, str | None] = {}
    for f in spec.fields:
        conf = row.get(f"{f}_confidence")
        if conf is not None and not pd.isna(conf) and conf < CONF_THRESHOLD:
            ctx[f] = None
            continue
        ctx[f] = normalize_value(f, row.get(f))
    if row.get("share_1", 0) >= COLOR_MIN_SHARE:
        ctx["color"] = normalize_value("color", row.get("color_1"))
    else:
        ctx["color"] = None

    missing = [f for f in spec.must_have if not ctx.get(f)]
    if missing:
        return OutputRow(base, True, "missing:" + ",".join(missing))
    return OutputRow(compose(spec, ctx), False, "ok")


def fetch_data(conn) -> pd.DataFrame:
    q = f"""
        SELECT p.*, a.*, c.color_1, c.share_1
        FROM {SRC_PRODUCTS} p
        LEFT JOIN {SRC_ATTRS} a ON p.sku = a.sku
        LEFT JOIN {SRC_COLORS} c ON p.sku = c.sku
        WHERE p.is_active
    """
    return pd.read_sql(q, conn)


TEMPLATES = {
    "Sheets": TitleSpec(
        "{brand} {thread_count} {material} {product_type}, {size}, {color}",
        ("material", "product_type"),
        ("thread_count", "brand", "color", "size"),
    ),
    "Duvet Covers": TitleSpec(
        "{brand} {material} {pattern} {product_type}, {size}, {color}",
        ("material", "product_type", "size"),
        ("pattern", "brand", "color"),
    ),
    "Towels": TitleSpec(
        "{brand} {material} {pattern} {product_type}, {color}",
        ("product_type",),
        ("pattern", "brand", "material"),
    ),
    "Pillows": TitleSpec(
        "{brand} {material} {pattern} {product_type}, {dimensions}, {color}",
        ("product_type",),
        ("dimensions", "pattern", "brand", "material"),
    ),
    "Rugs": TitleSpec(
        "{brand} {material} {pattern} {product_type}, {dimensions}, {color}",
        ("product_type", "dimensions"),
        ("pattern", "brand", "color", "material"),
    ),
}
DEFAULT_SPEC = TitleSpec(
    "{brand} {material} {product_type}, {color}",
    ("product_type",),
    ("brand", "material", "color"),
)


def get_spec(cat: str | None) -> TitleSpec:
    return TEMPLATES.get(cat or "", DEFAULT_SPEC)


def main() -> None:
    parser = argparse.ArgumentParser(description="SEO title generator")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = create_engine(args.dsn)
    df = fetch_data(engine)
    log.info("Loaded %d records", len(df))

    specs = [get_spec(c) for c in df["category_l2"].fillna("default")]
    results = [build_title(r, s) for _, r in df.iterrows()]

    out = pd.DataFrame({
        "sku": df["sku"],
        "original_title": df["title"],
        "seo_title": [r.seo for r in results],
        "template": [s.pattern.split("{")[0].strip() or "default" for s in specs],
        "kept_original": [r.fallback for r in results],
        "reason": [r.note for r in results],
    })
    log.info("Generated %d new titles (%.1f%% override)", 
             (~out["kept_original"]).sum(), 100 * (~out["kept_original"]).mean())

    if args.dry_run:
        print(out.head(25).to_string(index=False))
        return

    schema, tbl = DEST_TABLE.split(".")
    out.to_sql(tbl, engine, schema=schema, if_exists="replace", index=False)
    log.info("Persisted %d rows to %s", len(out), DEST_TABLE)


if __name__ == "__main__":
    main()
