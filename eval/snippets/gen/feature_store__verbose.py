"""Daily feature engineering pipeline.

This module orchestrates the computation of time-bounded aggregate features for
product and customer entities, ensuring point-in-time correctness and data
integrity before persisting snapshots to the serving layer. It reads feature
specifications from a central configuration file, executes windowed aggregations
across multiple event streams, enriches results with static attributes, and
validates output quality prior to storage.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd
import yaml

log = logging.getLogger("feature_engineering.pipeline")

CONFIGURATION_FILE = Path(__file__).parent / "definitions.yaml"
VALID_AGGREGATION_TYPES = {"count", "count_distinct", "sum", "avg"}
NULLABILITY_THRESHOLD = 0.05


@dataclass(frozen=True)
class AggregationSpecification:
    input_column: str
    operation: str
    output_name: str


@dataclass(frozen=True)
class EventStream:
    source_table: str
    timestamp_field: str
    filters: str | None
    aggregation_rules: list[AggregationSpecification]


@dataclass(frozen=True)
class FeatureBlueprint:
    identifier: str
    entity_key: str
    destination: str
    aggregation_windows: list[int]
    retention_period_days: int
    event_streams: list[EventStream]
    static_attributes: dict[str, str] = field(default_factory=dict)


class ConfigurationError(Exception):
    pass


def parse_configuration(path: Path) -> list[FeatureBlueprint]:
    raw = yaml.safe_load(path.read_text())
    defaults = raw.get("defaults", {})
    blueprints: list[FeatureBlueprint] = []
    for spec in raw["feature_views"]:
        streams = []
        for entry in spec["sources"]:
            rules = [AggregationSpecification(**a) for a in entry["aggregations"]]
            invalid = {r.operation for r in rules if r.operation not in VALID_AGGREGATION_TYPES}
            if invalid:
                raise ConfigurationError(f"{spec['name']}: unsupported operations {invalid}")
            streams.append(EventStream(
                source_table=entry["table"],
                timestamp_field=entry["timestamp_column"],
                filters=entry.get("filter"),
                aggregation_rules=rules,
            ))
        static = spec.get("static", {})
        blueprints.append(FeatureBlueprint(
            identifier=spec["name"],
            entity_key=spec["entity"],
            destination=spec["sink"],
            aggregation_windows=spec.get("windows", defaults.get("windows", [7, 30, 90])),
            retention_period_days=spec.get("ttl_days", defaults.get("ttl_days", 2)),
            event_streams=streams,
            static_attributes={"table": static.get("table"), "columns": static.get("columns", [])},
        ))
    log.info("Parsed %d feature blueprints from %s", len(blueprints), path)
    return blueprints


def retrieve_events(
    warehouse: Warehouse, stream: EventStream, entity: str, reference_date: date, lookback: int
) -> pd.DataFrame:
    """Fetch event records strictly prior to the reference date within the lookback window."""
    start = reference_date - timedelta(days=lookback)
    end = reference_date
    sql = (
        f"SELECT * FROM {stream.source_table} "
        f"WHERE {stream.timestamp_field} >= %(start)s AND {stream.timestamp_field} < %(end)s"
    )
    df = warehouse.query(sql, {"start": start, "end": end})
    if stream.filters:
        df = df.query(stream.filters)
    return df


def compute_window_aggregates(
    events: pd.DataFrame, stream: EventStream, entity: str, reference_date: date, window: int
) -> pd.DataFrame:
    """Generate windowed aggregations for a single event stream and time window."""
    cutoff = pd.Timestamp(reference_date - timedelta(days=window))
    ts = pd.to_datetime(events[stream.timestamp_field])
    windowed = events[(ts >= cutoff) & (ts < pd.Timestamp(reference_date))]
    grouped = windowed.groupby(entity)
    results = {}
    for rule in stream.aggregation_rules:
        key = f"{rule.output_name}_{window}d"
        aggregator = {
            "count": lambda g: g.count(),
            "count_distinct": lambda g: g.nunique(),
            "sum": lambda g: g.sum(),
            "avg": lambda g: g.mean(),
        }[rule.operation]
        results[key] = aggregator(grouped[rule.input_column])
    return pd.DataFrame(results)


def materialize_blueprint(
    warehouse: Warehouse, blueprint: FeatureBlueprint, reference_date: date
) -> pd.DataFrame:
    """Construct feature matrix for a feature blueprint as of a given reference date."""
    lookback = max(blueprint.aggregation_windows)
    feature_chunks = []
    for stream in blueprint.event_streams:
        events = retrieve_events(warehouse, stream, blueprint.entity_key, reference_date, lookback)
        log.info("%s: retrieved %d rows from %s", blueprint.identifier, len(events), stream.source_table)
        for window in blueprint.aggregation_windows:
            chunk = compute_window_aggregates(events, stream, blueprint.entity_key, reference_date, window)
            feature_chunks.append(chunk)
    features = pd.concat(feature_chunks, axis=1).fillna(0)
    if blueprint.static_attributes["table"]:
        static_df = warehouse.query(
            f"SELECT {blueprint.entity_key}, {', '.join(blueprint.static_attributes['columns'])} "
            f"FROM {blueprint.static_attributes['table']}"
        )
        features = features.join(static_df.set_index(blueprint.entity_key), how="left")
    features = features.reset_index().rename(columns={"index": blueprint.entity_key})
    features["as_of_date"] = reference_date
    features["valid_until"] = reference_date + timedelta(days=blueprint.retention_period_days)
    return features


def validate_blueprint(blueprint: FeatureBlueprint, features: pd.DataFrame) -> None:
    """Ensure feature matrix meets quality standards before persistence."""
    if features[blueprint.entity_key].isna().any():
        raise ValueError(f"{blueprint.identifier}: null entity keys detected")
    if features[blueprint.entity_key].duplicated().any():
        raise ValueError(f"{blueprint.identifier}: duplicate entity keys detected")
    expected_columns = {
        f"{rule.output_name}_{window}d"
        for stream in blueprint.event_streams
        for rule in stream.aggregation_rules
        for window in blueprint.aggregation_windows
    }
    missing = expected_columns - set(features.columns)
    if missing:
        raise ValueError(f"{blueprint.identifier}: missing columns {sorted(missing)}")
    null_ratios = features.isna().mean()
    violations = null_ratios[null_ratios > NULLABILITY_THRESHOLD]
    if not violations.empty:
        raise ValueError(f"{blueprint.identifier}: null rate exceeds {NULLABILITY_THRESHOLD}: {violations.to_dict()}")


def execute_pipeline(
    warehouse: Warehouse, reference_date: date, selected: list[str] | None = None, dry: bool = False
) -> None:
    """Run the full feature engineering pipeline for specified blueprints."""
    for blueprint in parse_configuration(CONFIGURATION_FILE):
        if selected and blueprint.identifier not in selected:
            continue
        features = materialize_blueprint(warehouse, blueprint, reference_date)
        validate_blueprint(blueprint, features)
        log.info(
            "%s: generated %d rows with %d columns",
            blueprint.identifier, len(features), features.shape[1]
        )
        if not dry:
            warehouse.write(blueprint.destination, features, partition=reference_date.isoformat())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--blueprint", action="append", dest="selected", help="restrict to specific blueprints")
    parser.add_argument("--dry-run", action="store_true", help="skip persistence step")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from warehouse_client import connect
    execute_pipeline(connect(), args.as_of, args.selected, args.dry_run)


if __name__ == "__main__":
    main()
