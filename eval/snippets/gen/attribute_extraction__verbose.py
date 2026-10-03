"""Module for identifying and extracting product attribute signals from unstructured content.

This component processes product metadata to surface structured attribute values—such as material composition,
bed size, dimensions, thread count, care instructions, and pattern type—by applying pattern-matching rules and
domain-specific gazetteers. Each extracted value is accompanied by a confidence score and source field, and only
values meeting minimum confidence thresholds are persisted to the output dataset.

The module is designed for batch processing of product records, supporting incremental ingestion and robust
handling of noisy input such as HTML fragments, inconsistent whitespace, and ambiguous phrasing.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sqlalchemy import create_engine

from gazetteers import (
    MaterialGazetteer,
    SizeGazetteer,
    PatternGazetteer,
    CareGazetteer,
)

log = logging.getLogger(__name__)

INCH_TO_CENTIMETRE = 2.54


@dataclass
class AttributeSignal:
    """Represents a candidate attribute value with associated confidence and provenance."""
    value: Any
    confidence: float
    source_field: str


def _normalize_text(raw: str | None) -> str:
    """Sanitize input by collapsing whitespace and stripping embedded HTML."""
    if raw is None:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_material_composition(title: str, description: str) -> AttributeSignal | None:
    """Parse fabric blend percentages and resolve to canonical material names."""
    for field, text in (("title", title), ("description", description)):
        blend_map: dict[str, int] = {}
        for match in re.finditer(
            r"(?P<percentage>\d{1,3})\s*%\s*(?P<material>[a-z][a-z \-]{1,24}?)(?=\s*(?:\d{1,3}\s*%|[,/;&]|\band\b|$))",
            text,
            re.IGNORECASE,
        ):
            canonical = MaterialGazetteer.canonicalize(match.group("material"))
            if canonical:
                blend_map[canonical] = blend_map.get(canonical, 0) + int(match.group("percentage"))
        if blend_map:
            confidence = 0.95 if sum(blend_map.values()) == 100 else 0.70
            return AttributeSignal(blend_map, confidence, field)
    for field, text, base_conf in (("title", title, 0.85), ("description", description, 0.65)):
        candidates = MaterialGazetteer.extract_all(text)
        if candidates:
            unique = {c[0] for c in candidates}
            confidence = base_conf if len(unique) == 1 else base_conf - 0.15
            return AttributeSignal({m: None for m in sorted(unique)}, confidence, field)
    return None


def _extract_dimensional_data(title: str, description: str) -> AttributeSignal | None:
    """Detect rectangular dimensions and convert to centimetres."""
    for field, text in (("title", title), ("description", description)):
        match = re.search(
            r"(?P<width>\d{1,3}(?:\.\d+)?)\s*(?:\"|in\.?|inches)?\s*[x×]\s*"
            r"(?P<length>\d{1,3}(?:\.\d+)?)\s*(?P<unit>\"|in\.?|inches|cm)\b",
            text,
            re.IGNORECASE,
        )
        if match:
            width = float(match.group("width"))
            length = float(match.group("length"))
            factor = 1.0 if match.group("unit").lower() == "cm" else INCH_TO_CENTIMETRE
            return AttributeSignal(
                {"width_cm": round(width * factor, 1), "length_cm": round(length * factor, 1)},
                0.90,
                field,
            )
    return None


def _extract_single_value_attribute(
    title: str,
    description: str,
    gazetteer: SizeGazetteer | PatternGazetteer | CareGazetteer,
    title_confidence: float,
    description_confidence: float,
) -> AttributeSignal | None:
    """General-purpose extractor for single-value attributes using gazetteer lookup."""
    for field, text, conf in (
        ("title", title, title_confidence),
        ("description", description, description_confidence),
    ):
        candidates = gazetteer.extract_all(text)
        if candidates:
            unique = {c[0] for c in candidates}
            confidence = conf if len(unique) == 1 else conf / 2
            return AttributeSignal(candidates[0][0], confidence, field)
    return None


def _extract_bed_size(title: str, description: str) -> AttributeSignal | None:
    return _extract_single_value_attribute(title, description, SizeGazetteer, 0.90, 0.70)


def _extract_pattern_type(title: str, description: str) -> AttributeSignal | None:
    return _extract_single_value_attribute(title, description, PatternGazetteer, 0.85, 0.60)


def _extract_thread_count(title: str, description: str) -> AttributeSignal | None:
    """Extract numeric thread count and apply domain-specific validity checks."""
    for field, text in (("title", title), ("description", description)):
        match = re.search(r"\b(?P<count>[1-9]\d{1,3})\s*(?:-|\s)?(?:thread[\s\-]?count|tc)\b", text, re.IGNORECASE)
        if match:
            count = int(match.group("count"))
            confidence = 0.95 if 150 <= count <= 1500 else 0.40
            return AttributeSignal(count, confidence, field)
    return None


def _extract_care_instructions(_, description: str) -> AttributeSignal | None:
    """Extract care labels from description only, flagging conflicting instructions."""
    candidates = sorted({c[0] for c in CareGazetteer.extract_all(description)})
    if not candidates:
        return None
    conflict = {"dry_clean_only", "machine_wash_cold"} <= set(candidates)
    return AttributeSignal(candidates, 0.40 if conflict else 0.80, "description")


EXTRACTION_PIPELINE = {
    "material": _extract_material_composition,
    "size": _extract_bed_size,
    "dimensions": _extract_dimensional_data,
    "thread_count": _extract_thread_count,
    "care": _extract_care_instructions,
    "pattern": _extract_pattern_type,
}


def _process_single_record(sku: str, title: str, description: str, thresholds: dict[str, float]) -> dict[str, Any]:
    """Apply all extractors to a product record and enforce per-attribute confidence thresholds."""
    title = _normalize_text(title)
    description = _normalize_text(description)
    output_row = {"sku": sku}
    for attribute_name, extractor in EXTRACTION_PIPELINE.items():
        signal = extractor(title, description)
        meets_threshold = signal is not None and signal.confidence >= thresholds.get(attribute_name, 0.5)
        output_row[attribute_name] = (
            json.dumps(signal.value) if isinstance(signal.value, (dict, list)) else signal.value
        ) if meets_threshold else None
        output_row[f"{attribute_name}_confidence"] = round(signal.confidence, 2) if signal else None
        output_row[f"{attribute_name}_source"] = signal.source_field if signal else None
    return output_row


def run_pipeline(config_path: Path, database_url: str) -> None:
    """Execute the full extraction workflow over product data."""
    config = yaml.safe_load(config_path.read_text())
    query = (
        f"SELECT {', '.join(config['input']['columns'])} "
        f"FROM {config['input']['table']} "
        f"WHERE {config['input']['where']}"
    )
    engine = create_engine(database_url)

    batch_results = []
    for batch in pd.read_sql(query, engine, chunksize=config["batch_size"]):
        batch_records = [
            _process_single_record(row.sku, row.title, row.description, config["thresholds"])
            for row in batch.itertuples()
        ]
        batch_results.append(pd.DataFrame(batch_records))
        log.info("Processed %d records so far", sum(len(df) for df in batch_results))

    final_dataframe = pd.concat(batch_results, ignore_index=True)
    final_dataframe["processing_date"] = pd.Timestamp.utcnow().date()

    schema_name, table_name = config["output"]["table"].split(".")
    final_dataframe.to_sql(
        table_name,
        engine,
        schema=schema_name,
        if_exists="replace",
        index=False,
    )
    log.info("Extracted attributes for %d products; coverage: %s", len(final_dataframe), {
        attr: f"{final_dataframe[attr].notna().mean():.1%}"
        for attr in EXTRACTION_PIPELINE
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured product attributes from unstructured content.")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--dsn", required=True, help="Database connection string")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_pipeline(args.config, args.dsn)


if __name__ == "__main__":
    main()
