"""Weekly demand forecast per SKU and warehouse (Holt's linear trend)."""


def holt(series, alpha=0.4, beta=0.2, horizon=4):
    level, trend = series[0], series[1] - series[0]
    for y in series[1:]:
        prev = level
        level = alpha * y + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
    return [level + (h + 1) * trend for h in range(horizon)]


def forecast_all(weekly_units):
    return {key: holt(series) for key, series in weekly_units.items() if len(series) >= 3}
