from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

log = logging.getLogger("analytics.features")

CONFIG_FILE = Path(__file__).parent / "definitions.yaml"
VALID_AGG_FUNCS = {"nunique", "sum", "mean", "size"}
NULL_THRESHOLD = 0.05


@dataclass(frozen=True)
class AggSpec:
    field: str
    op: str
    name: str


@dataclass(frozen=True)
class InputTable:
    name: str
    ts_col: str
    filters: str | None
    metrics: list[AggSpec]


@dataclass(frozen=True)
class FeatureLayer:
    label: str
    key: str
    output: str
    lookbacks: list[int]
    retention: int
    inputs: list[InputTable]
    metadata: tuple[str, ...] = field(default_factory=tuple)
    meta_table: str | None = None


class ConfigError(Exception):
    pass


def parse_config(path: Path) -> list[FeatureLayer]:
    cfg = yaml.safe_load(path.read_text())
    base = cfg.get("defaults", {})
    layers: list[FeatureLayer] = []
    for entry in cfg["feature_views"]:
        inputs = []
        for tbl in entry["sources"]:
            specs = [AggSpec(**a) for a in tbl["aggregations"]]
            invalid = {s.op for s in specs if s.op not in VALID_AGG_FUNCS}
            if invalid:
                raise ConfigError(f"{entry['name']}: unsupported ops {invalid}")
            inputs.append(InputTable(
                name=tbl["table"],
                ts_col=tbl["timestamp_column"],
                filters=tbl.get("filter"),
                metrics=specs,
            ))
        meta = entry.get("static", {})
        layers.append(FeatureLayer(
            label=entry["name"],
            key=entry["entity"],
            output=entry["sink"],
            lookbacks=entry.get("windows", base["windows"]),
            retention=entry.get("ttl_days", base["ttl_days"]),
            inputs=inputs,
            meta_table=meta.get("table"),
            metadata=tuple(meta.get("columns", [])),
        ))
    log.info("parsed %d feature layers", len(layers))
    return layers


def pull_events(db: Any, table: InputTable, until: date, span: int) -> pd.DataFrame:
    start = until - timedelta(days=span)
    sql = f"SELECT * FROM {table.name} WHERE {table.ts_col} >= '{start}' AND {table.ts_col} < '{until}'"
    df = db.execute(sql).fetch_pandas()
    if table.filters:
        df = df.query(table.filters)
    return df


def build_windows(df: pd.DataFrame, table: InputTable, key: str, ref: date, window: int) -> pd.DataFrame:
    limit = pd.Timestamp(ref - timedelta(days=window))
    ts = pd.to_datetime(df[table.ts_col])
    subset = df[(ts >= limit) & (ts < pd.Timestamp(ref))]
    grp = subset.groupby(key, sort=False)
    res: dict[str, pd.Series] = {}
    for spec in table.metrics:
        col = spec.field
        if spec.op == "size":
            res[f"{spec.name}_{window}d"] = grp.size()
        else:
            res[f"{spec.name}_{window}d"] = getattr(grp[col], spec.op)()
    return pd.DataFrame(res).fillna(0)


def compile_layer(db: Any, layer: FeatureLayer, ref: date) -> pd.DataFrame:
    span = max(layer.lookbacks)
    parts = []
    for tbl in layer.inputs:
        raw = pull_events(db, tbl, ref, span)
        log.info("%s: fetched %d rows from %s", layer.label, len(raw), tbl.name)
        for w in layer.lookbacks:
            parts.append(build_windows(raw, tbl, layer.key, ref, w))
    features = pd.concat(parts, axis=1).reset_index()
    features.columns = [layer.key if c == "index" else c for c in features.columns]
    if layer.meta_table:
        meta = db.execute(f"SELECT {layer.key}, {', '.join(layer.metadata)} FROM {layer.meta_table}").fetch_pandas()
        features = features.merge(meta, on=layer.key, how="left")
    features["snapshot_date"] = ref
    features["valid_until"] = ref + timedelta(days=layer.retention)
    return features


def sanity_check(layer: FeatureLayer, df: pd.DataFrame) -> None:
    if df[layer.key].isnull().any():
        raise ValueError(f"{layer.label}: null keys found")
    if df[layer.key].duplicated().any():
        raise ValueError(f"{layer.label}: duplicate keys")
    expected = {f"{s.name}_{w}d" for s in layer.inputs for s in [s] for w in layer.lookbacks}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{layer.label}: missing {sorted(missing)}")
    nulls = df.drop(columns=[layer.key, "snapshot_date", "valid_until"]).isna().mean()
    bad = nulls[nulls > NULL_THRESHOLD]
    if not bad.empty:
        raise ValueError(f"{layer.label}: high nulls {bad.to_dict()}")


def execute(db: Any, ref: date, targets: list[str] | None = None, simulate: bool = False) -> None:
    for layer in parse_config(CONFIG_FILE):
        if targets and layer.label not in targets:
            continue
        batch = compile_layer(db, layer, ref)
        sanity_check(layer, batch)
        log.info("%s: %d rows, %d cols", layer.label, len(batch), batch.shape[1])
        if not simulate:
            db.insert(layer.output, batch, partition=ref.isoformat())


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily feature snapshot generation")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--layer", action="append", help="restrict to specific layers")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S", format="%(asctime)s %(message)s")
    from platform.db import Client
    execute(Client(), args.date, args.layer, args.dry)


if __name__ == "__main__":
    main()
