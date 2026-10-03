from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from sqlalchemy import create_engine

log = logging.getLogger(__name__)

# —— utilities ——
def _normalize_whitespace(raw: str | None) -> str:
    """Sanitize input: strip HTML, collapse whitespace."""
    if not raw:
        return ""
    cleaned = re.sub(r"<[^>]*>", " ", raw)
    return re.sub(r"\s+", " ", cleaned).strip()


# —— regex patterns ——
BLEND_PATTERN = re.compile(
    r"(\d{1,3})\s*%\s*([a-z][a-z\s\-]{1,24}?)(?=\s*(?:\d{1,3}\s*%|[,/;&]|\band\b|$))",
    re.IGNORECASE,
)
DIM_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:\"|in(?:ches?)?)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(in(?:ches?)?|cm)\b",
    re.IGNORECASE,
)
TC_PATTERN = re.compile(r"\b(\d{2,4})\s*(?:thread[\s\-]?count|tc)\b", re.IGNORECASE)

INCH_TO_CM = 2.54


# —— gazetteer engine ——
@dataclass
class TermLookup:
    """Maps canonical labels to normalized surface forms."""
    label: str
    mapping: dict[str, list[str]]

    def __post_init__(self):
        self._surface_to_canonical = {}
        for canon, variants in self.mapping.items():
            for v in variants:
                norm = re.sub(r"[\s\-]+", " ", v.lower())
                self._surface_to_canonical[norm] = canon
        patterns = sorted(self._surface_to_canonical, key=len, reverse=True)
        regex_body = "|".join(re.escape(p).replace(r"\ ", r"[\s\-]+") for p in patterns)
        self._matcher = re.compile(rf"\b(?:{regex_body})\b", re.IGNORECASE)

    def scan(self, text: str) -> list[tuple[str, str]]:
        """Return list of (canonical, original_match) for non-overlapping hits."""
        results = []
        for m in self._matcher.finditer(text):
            surface = re.sub(r"[\s\-]+", " ", m.group(0).lower())
            canon = self._surface_to_canonical.get(surface, surface)
            results.append((canon, m.group(0)))
        return results


# —— gazetteer instances ——
MATERIALS = {
    "linen": ["linen", "flax", "french linen", "belgian linen"],
    "cotton": ["cotton", "organic cotton", "supima", "pima", "egyptian cotton"],
    "velvet": ["velvet", "velour", "crushed velvet"],
    "wool": ["wool", "merino", "lambswool"],
}

BED_SIZES = {
    "Twin": ["twin", "single"],
    "Twin XL": ["twin xl", "twin extra long"],
    "Full": ["full", "double"],
    "Queen": ["queen", "qn"],
    "King": ["king", "eastern king"],
    "Cal King": ["cal king", "california king"],
}

PATTERNS = {
    "solid": ["solid", "plain"],
    "striped": ["stripe", "striped", "pinstripe"],
    "gingham": ["gingham", "plaid", "tartan"],
    "floral": ["floral", "botanical"],
    "geometric": ["geometric", "diamond", "chevron"],
}

CARE = {
    "machine_wash_cold": ["machine wash cold", "wash cold"],
    "tumble_dry_low": ["tumble dry low", "dry low"],
    "dry_clean_only": ["dry clean only"],
}

MATERIALS_GAZ = TermLookup("material", MATERIALS)
SIZE_GAZ = TermLookup("size", BED_SIZES)
PATTERN_GAZ = TermLookup("pattern", PATTERNS)
CARE_GAZ = TermLookup("care", CARE)


# —— extraction functions ——
def _blend_parser(text: str) -> Optional[dict[str, int]]:
    """Parse fabric blend percentages."""
    blend = {}
    for m in BLEND_PATTERN.finditer(text):
        pct = int(m.group(1))
        mat = MATERIALS_GAZ._surface_to_canonical.get(re.sub(r"[\s\-]+", " ", m.group(2).lower()))
        if mat:
            blend[mat] = blend.get(mat, 0) + pct
    return blend if blend else None


def _first_match(gaz: TermLookup, title: str, desc: str, confs: tuple[float, float]) -> Optional[tuple]:
    """Return first gazetteer hit, preferring title."""
    for src, txt, conf in (("title", title, confs[0]), ("description", desc, confs[1])):
        hits = gaz.scan(txt)
        if hits:
            uniq = {h[0] for h in hits}
            return (hits[0][0], conf if len(uniq) == 1 else conf * 0.75, src)
    return None


def _extract_material(title: str, desc: str) -> Optional[tuple]:
    for src, txt in (("title", title), ("description", desc)):
        blend = _blend_parser(txt)
        if blend:
            conf = 0.92 if sum(blend.values()) == 100 else 0.72
            return (blend, conf, src)
    for src, txt, conf in (("title", title, 0.82), ("description", desc, 0.62)):
        hits = MATERIALS_GAZ.scan(txt)
        if hits:
            uniq = {h[0] for h in hits}
            return ({m: None for m in sorted(uniq)}, conf - (0.15 if len(uniq) > 1 else 0), src)
    return None


def _extract_size(title: str, desc: str) -> Optional[tuple]:
    return _first_match(SIZE_GAZ, title, desc, (0.88, 0.68))


def _extract_dimensions(title: str, desc: str) -> Optional[tuple]:
    for src, txt in (("title", title), ("description", desc)):
        m = DIM_PATTERN.search(txt)
        if m:
            w, l, unit = float(m.group(1)), float(m.group(2)), m.group(3).lower()
            factor = 1.0 if "cm" in unit else INCH_TO_CM
            dims = {"width_cm": round(w * factor, 1), "length_cm": round(l * factor, 1)}
            return (dims, 0.89, src)
    return None


def _extract_thread_count(title: str, desc: str) -> Optional[tuple]:
    for src, txt in (("title", title), ("description", desc)):
        m = TC_PATTERN.search(txt)
        if m:
            tc = int(m.group(1))
            conf = 0.93 if 150 <= tc <= 1500 else 0.38
            return (tc, conf, src)
    return None


def _extract_care(title: str, desc: str) -> Optional[tuple]:
    hits = CARE_GAZ.scan(desc)
    if not hits:
        return None
    uniq = {h[0] for h in hits}
    conflict = {"dry_clean_only", "machine_wash_cold"} <= uniq
    return (sorted(uniq), 0.38 if conflict else 0.78, "description")


def _extract_pattern(title: str, desc: str) -> Optional[tuple]:
    return _first_match(PATTERN_GAZ, title, desc, (0.83, 0.58))


EXTRACTORS = {
    "material": _extract_material,
    "size": _extract_size,
    "dimensions": _extract_dimensions,
    "thread_count": _extract_thread_count,
    "care": _extract_care,
    "pattern": _extract_pattern,
}


def _process_row(sku: str, title: str, desc: str, limits: dict[str, float]) -> dict:
    title, desc = _normalize_whitespace(title), _normalize_whitespace(desc)
    result = {"sku": sku}
    for attr, extractor in EXTRACTORS.items():
        raw = extractor(title, desc)
        if raw:
            value, conf, src = raw
            keep = conf >= limits.get(attr, 0.5)
            result[attr] = value if keep else None
            result[f"{attr}_confidence"] = round(conf, 2)
            result[f"{attr}_source"] = src
        else:
            result[attr] = None
            result[f"{attr}_confidence"] = None
            result[f"{attr}_source"] = None
    return result


def run_pipeline(dsn: str, cfg_path: Path) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    query = (
        f"SELECT {', '.join(cfg['input']['columns'])} "
        f"FROM {cfg['input']['table']} WHERE {cfg['input']['where']}"
    )
    engine = create_engine(dsn)

    chunks = pd.read_sql(query, engine, chunksize=cfg["batch_size"])
    frames = []
    for batch in chunks:
        rows = [_process_row(r.sku, r.title, r.description, cfg["thresholds"]) for r in batch.itertuples()]
        frames.append(pd.DataFrame(rows))
        log.info("processed %d rows", sum(len(f) for f in frames))

    output = pd.concat(frames, ignore_index=True)
    output["processed_at"] = pd.Timestamp.utcnow().date()

    schema, table = cfg["output"]["table"].split(".")
    output.to_sql(table, engine, schema=schema, if_exists="replace", index=False)
    log.info("wrote %d records to %s.%s", len(output), schema, table)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dsn", required=True)
    run_pipeline(parser.parse_args().dsn, parser.parse_args().config)
