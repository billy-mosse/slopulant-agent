class DriftAlertProcessor:
    """
    Detects statistical shifts in production model behavior by comparing recent
    prediction logs against a historical reference period. For each monitored
    model and feature, computes multiple drift indicators—including PSI, KS
    p-values, null rate changes, and prediction mean shifts—then evaluates them
    against configurable severity rules. Findings are deduplicated within a
    rolling 24-hour window, persisted for auditability, and optionally posted
    to Slack for rapid awareness.
    """

    def __init__(self, warehouse_client, webhook_url: str | None = None, enable_slack: bool = True) -> None:
        self._warehouse = warehouse_client
        self._webhook = webhook_url
        self._enable_slack = enable_slack
        self._buffer: list[DriftObservation] = []
        self._report_date: date | None = None

    def ingest(self, observations: list["DriftObservation"], on_date: date) -> None:
        self._report_date = on_date
        for obs in observations:
            obs.classify_using(self._severity_rules())
            if obs.severity:
                self._buffer.append(obs)

    def _severity_rules(self) -> list[tuple[str, str, float, str]]:
        return [
            ("psi", ">", 0.2, "critical"),
            ("psi", ">", 0.1, "elevated"),
            ("ks_pvalue", "<", 0.001, "critical"),
            ("null_rate_delta", ">", 0.05, "critical"),
            ("mean_shift_sd", "abs>", 0.5, "elevated"),
        ]

    def _active_alert_keys(self) -> set[tuple[str, str, str]]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        df = self._warehouse.execute(
            "SELECT model_id, feature_name, metric_type FROM drift_alerts WHERE created_at >= :cutoff",
            {"cutoff": cutoff},
        )
        return {(row["model_id"], row["feature_name"], row["metric_type"]) for row in df.to_dict("records")}

    def dispatch(self) -> list["DriftObservation"]:
        active_keys = self._active_alert_keys()
        unique_alerts: dict[tuple[str, str, str], "DriftObservation"] = {}
        for obs in self._buffer:
            key = (obs.model_id, obs.feature_name, obs.metric_type)
            if key not in active_keys and key not in unique_alerts:
                unique_alerts[key] = obs
        self._buffer.clear()
        if not unique_alerts:
            return []

        records = [asdict(a) for a in unique_alerts.values()]
        self._warehouse.insert("drift_alerts", pd.DataFrame(records), partition=self._report_date.isoformat())
        if self._enable_slack and self._webhook:
            payload = json.dumps(self._compose_slack_payload(list(unique_alerts.values()), self._report_date)).encode()
            req = Request(self._webhook, data=payload, headers={"Content-Type": "application/json"})
            urlopen(req, timeout=10)
        return list(unique_alerts.values())

    def _compose_slack_payload(self, alerts: list["DriftObservation"], report_date: date) -> dict:
        header = f"*Model drift summary {report_date.isoformat()}* ({len(alerts)} new alerts)"
        lines = [header]
        for item in sorted(alerts, key=lambda x: (x.severity != "critical", x.model_id, x.feature_name)):
            icon = ":red_circle:" if item.severity == "critical" else ":large_yellow_circle:"
            detail = f" {item.context}" if item.context else ""
            lines.append(f"{icon} `{item.model_id}` / `{item.feature_name}` {item.metric_type}={item.value:.4g}{detail}")
        return {"text": "\n".join(lines)}


@dataclass
class DriftObservation:
    model_id: str
    feature_name: str
    metric_type: str
    value: float
    context: str = ""
    severity: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def classify_using(self, rules: list[tuple[str, str, float, str]]) -> None:
        for metric, op, threshold, level in rules:
            if metric != self.metric_type:
                continue
            match op:
                case ">":
                    if self.value > threshold:
                        self.severity = level
                        return
                case "<":
                    if self.value < threshold:
                        self.severity = level
                        return
                case "abs>":
                    if abs(self.value) > threshold:
                        self.severity = level
                        return
        self.severity = None
