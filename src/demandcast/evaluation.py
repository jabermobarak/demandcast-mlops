import math

import numpy as np
import pandas as pd


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Metrics need non-empty arrays of equal shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite values")
    error = np.abs(actual - predicted)
    return {"mae": float(error.mean()), "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
            "wape": float(error.sum() / np.abs(actual).sum()) if np.abs(actual).sum() else None,
            "rows": int(actual.size)}


def calibration_radius(actual, predicted, alpha=0.10):
    residuals = np.abs(np.asarray(actual, float) - np.asarray(predicted, float))
    if len(residuals) < 10 or not np.isfinite(residuals).all() or not 0 < alpha < 1:
        raise ValueError("Calibration needs at least ten finite residuals and 0 < alpha < 1")
    rank = min(math.ceil((len(residuals) + 1) * (1 - alpha)), len(residuals))
    return float(np.sort(residuals)[rank - 1])


def interval_bounds(predicted, radius):
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)
    return np.maximum(predicted - radius, 0), predicted + radius


def interval_metrics(actual, lower, upper):
    actual, lower, upper = map(lambda x: np.asarray(x, float), (actual, lower, upper))
    return {"coverage": float(np.mean((actual >= lower) & (actual <= upper))),
            "mean_width": float(np.mean(upper - lower)),
            "nominal_coverage": 0.90,
            "warning": "Empirical coverage only; temporal dependence prevents an exchangeability guarantee."}


def grouped_errors(predictions, group):
    frame = predictions.copy()
    ts = pd.to_datetime(frame["ts"])
    frame["group"] = ts.dt.hour if group == "hour" else ts.dt.strftime("%Y-%m")
    result = []
    for name, rows in frame.groupby("group", sort=True):
        result.append({"group": int(name) if group == "hour" else str(name),
                       **regression_metrics(rows.actual, rows.predicted)})
    return result


def paired_block_bootstrap(actual, predicted, baseline, timestamps, repetitions=1000, block_days=7, seed=42):
    """Descriptive paired moving-day-block MAE-difference interval.

    Hourly pairing and short within-block dependence are preserved. This does
    not establish stationarity, population significance, or a causal effect.
    Positive differences mean the selected forecast has lower MAE.
    """
    actual, predicted, baseline = map(lambda x: np.asarray(x, float), (actual, predicted, baseline))
    frame = pd.DataFrame({"day": pd.to_datetime(pd.Series(timestamps)).dt.normalize(),
                          "difference": np.abs(actual - baseline) - np.abs(actual - predicted)})
    daily = frame.groupby("day").difference.agg(["sum", "count"])
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0)
    length = len(daily)
    if length < block_days or repetitions < 10:
        raise ValueError("Need at least one full block and ten bootstrap repetitions")
    generator = np.random.default_rng(seed)
    starts = generator.integers(0, length - block_days + 1,
                                size=(repetitions, math.ceil(length / block_days)))
    indices = (starts[..., None] + np.arange(block_days)).reshape(repetitions, -1)[:, :length]
    sums, counts = daily["sum"].to_numpy(), daily["count"].to_numpy()
    denominator = counts[indices].sum(axis=1)
    differences = sums[indices].sum(axis=1) / denominator
    return {"mean_mae_reduction": float(sums.sum() / counts.sum()),
            "percentile_interval_95": np.quantile(differences, [.025, .975]).tolist(),
            "block_days": block_days, "repetitions": repetitions, "seed": seed,
            "caveat": "Descriptive paired moving-block bootstrap on one time period; not a formal population-significance or causal claim."}
