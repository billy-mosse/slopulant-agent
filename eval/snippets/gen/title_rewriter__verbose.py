"""
Product Title Optimization Module

This module enhances product discoverability by transforming marketing-facing titles
into search-engine-optimized versions using category-specific structural templates.
It integrates core product metadata—including materials, dimensions, colors, and attributes
with confidence scores—to produce concise, readable titles (≤70 characters) while
preserving original content when required fields are missing or low-confidence.

The process ensures:
- Removal of promotional boilerplate (e.g., "SALE", "free shipping", excessive punctuation)
- Deduplication of recurring words
- Smart title casing that respects units and minor words
- Graceful truncation by deprioritizing optional template slots

Designed for alignment with SEO guidelines and consistent user experience.
"""
from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine

from title_templates import TemplateCatalog, MAX_TITLE_LENGTH


logger = logging.getLogger(__name__)


@dataclass
class TitleGenerationOutcome:
    """Result of a single title transformation attempt."""
    final_title: str
    fallback_to_original: bool
    decision_reason: str


def sanitize_product_text(raw: str | None) -> str:
    """Cleans input by stripping promotional clutter and normalizing whitespace."""
    noise_pattern = re.compile(
        r"(!{2,}|\bfree\s+shipping\b|\bsale\b|\bclearance\b|\bbest\s+seller\b|\bhot\s+deal\b|\blimited\s+time\b|\bnew!?\b)",
        re.IGNORECASE
    )
    cleaned = noise_pattern.sub(" ", raw or "")
    cleaned = re.sub(r"\s*([,|-])\s*(?=[,|-]|$)", "", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" ,-|")


def resolve_duplicate_tokens(text: str) -> str:
    """Eliminates consecutive duplicate tokens (case-insensitive), preserving punctuation."""
    seen = set()
    tokens = []
    for token in text.split():
        key = re.sub(r"\W", "", token).lower()
        if key and key in seen:
            continue
        seen.add(key)
        tokens.append(token)
    return " ".join(tokens)


def apply_stylistic_title_casing(text: str) -> str:
    """Formats text in title case while preserving units (e.g., '400TC') and minor words."""
    words = text.split()
    exceptions = {"a", "an", "and", "the", "of", "in", "with", "for", "x", "by"}
    unit_indicator = re.compile(r"^\d+(\.\d+)?(cm|in|tc)?$", re.IGNORECASE)
    for idx, word in enumerate(words):
        core = word.strip(",")
        if unit_indicator.match(core) or re.search(r"\d", core):
            continue
        if idx > 0 and core.lower() in exceptions:
            words[idx] = word.lower()
        else:
            words[idx] = word.capitalize()
    return " ".join(words)


def materialize_attribute(name: str, value: object) -> str | None:
    """Maps raw attribute values to user-friendly strings; returns None if absent/unreliable."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    if name == "thread_count":
        return f"{int(value)} Thread Count"
    if name == "material" and isinstance(value, str) and value.startswith("{"):
        blend_data = pd.Series(json.loads(value))
        top_materials = blend_data.sort_values(ascending=False).head(2).index.tolist()
        return "-".join(top_materials) + (" Blend" if len(top_materials) > 1 else "")
    if name == "dimensions" and isinstance(value, str) and value.startswith("{"):
        dims = pd.Series(json.loads(value))
        return f"{round(dims['width_cm'])}x{round(dims['length_cm'])}cm"
    return str(value)


def compile_title(template: TemplateCatalog.Template, attributes: dict[str, str | None]) -> str:
    """Constructs a title using the template and attribute map, trimming optional elements as needed."""
    present = {k: v for k, v in attributes.items() if v}
    removal_priority = list(template.drop_order)
    while True:
        text = template.pattern
        for slot in template.slot_names:
            text = text.replace(f"{{{slot}}}", present.get(slot, ""))
        text = apply_stylistic_title_casing(
            resolve_duplicate_tokens(
                sanitize_product_text(re.sub(r"\s+", " ", text))
            )
        )
        if len(text) <= MAX_TITLE_LENGTH or not removal_priority:
            return text[:MAX_TITLE_LENGTH].rstrip(" ,")
        present.pop(removal_priority.pop(0), None)


def process_single_product(product: pd.Series) -> TitleGenerationOutcome:
    """Applies title optimization logic to a single product row."""
    base_title = sanitize_product_text(product["product_name"])
    template = TemplateCatalog.get_template(product.get("secondary_category") or "")
    attribute_map: dict[str, str | None] = {}

    for slot_name in template.slot_names:
        confidence = product.get(f"{slot_name}_reliability")
        if confidence is not None and not pd.isna(confidence) and confidence < 0.75:
            attribute_map[slot_name] = None
            continue
        attribute_map[slot_name] = materialize_attribute(slot_name, product.get(slot_name))

    # Incorporate dominant color if reliable
    primary_color_share = product.get("dominant_color_share", 0.0)
    if primary_color_share >= 0.30:
        attribute_map["hue"] = materialize_attribute("hue", product.get("primary_color"))
    else:
        attribute_map["hue"] = None

    missing_required = [
        field for field in template.mandatory_slots
        if not attribute_map.get(field)
    ]
    if missing_required:
        return TitleGenerationOutcome(base_title, True, "insufficient_required:" + ",".join(missing_required))
    return TitleGenerationOutcome(compile_title(template, attribute_map), False, "success")


def fetch_catalog_data(connection_engine) -> pd.DataFrame:
    """Retrieves active product records joined with material and color metadata."""
    return pd.read_sql("""
        SELECT * 
        FROM product_master
        LEFT JOIN product_specs USING (sku)
        LEFT JOIN product_color_summary USING (sku)
        WHERE product_status = 'active'
    """, connection_engine)


def execute_title_enhancement(dsn: str, simulate: bool = False) -> None:
    """Main pipeline: load, transform, and persist optimized titles."""
    logger.info("Initializing SEO title pipeline...")
    engine = create_engine(dsn)
    dataset = fetch_catalog_data(engine)
    logger.info("Loaded %d active products", len(dataset))

    outcomes = [process_single_product(row) for _, row in dataset.iterrows()]
    output_frame = pd.DataFrame({
        "sku": dataset["sku"],
        "original_title": dataset["product_name"],
        "seo_title": [o.final_title for o in outcomes],
        "template_category": dataset["secondary_category"].fillna("standard"),
        "used_original": [o.fallback_to_original for o in outcomes],
        "reason": [o.decision_reason for o in outcomes],
    })
    optimization_rate = 100 * (1 - output_frame["used_original"].mean())
    logger.info("Optimized %.1f%% of titles", optimization_rate)

    if simulate:
        logger.info("Dry run — previewing first 25 rows:")
        print(output_frame.head(25).to_string(index=False))
        return

    output_frame.to_sql("enhanced_titles", engine, schema="catalog", if_exists="replace", index=False)
    logger.info("Published %d optimized records to catalog.enhanced_titles", len(output_frame))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEO title enhancement engine for product catalog")
    parser.add_argument("--db", required=True, help="Database connection string")
    parser.add_argument("--dry-run", action="store_true", help="Preview results without persistence")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    execute_title_enhancement(dsn=args.db, simulate=args.dry_run)
