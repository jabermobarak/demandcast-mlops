"""Feature contract shared by SQL training and online inference.

The latest target-derived input is the count for hour t-24h, assumed complete
one hour after its label. Every such input is available by the day-start 00:00
forecast origin. Ingestion delay beyond that hour boundary is not modeled.
"""

from importlib.resources import files

import duckdb
import numpy as np
import pandas as pd

from .data import validate_observations

HISTORY_FEATURES = ["lag_24", "lag_48", "lag_168", "rolling_mean_24", "rolling_mean_168"]
CALENDAR_FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin",
                     "month_cos", "is_weekend", "day_index"]
FEATURES = HISTORY_FEATURES + CALENDAR_FEATURES
WARMUP_HOURS = 191


def add_calendar(frame):
    result = frame.copy()
    ts = pd.to_datetime(result["ts"])
    if ts.dt.tz is not None:
        raise ValueError("Use timezone-naive timestamps matching the UCI feature contract")
    for name, values, period in (("hour", ts.dt.hour, 24), ("weekday", ts.dt.dayofweek, 7),
                                  ("month", ts.dt.month - 1, 12)):
        result[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        result[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    result["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)
    result["day_index"] = (ts.dt.normalize() - pd.Timestamp("2011-01-01")).dt.days.astype(float)
    return result


def build_features(observations):
    observations = validate_observations(observations)
    grid = pd.DataFrame({"ts": pd.date_range(observations.ts.min(), observations.ts.max(), freq="h")})
    query = files("demandcast").joinpath("sql/features.sql").read_text()
    with duckdb.connect() as connection:
        connection.register("observations", observations)
        connection.register("hourly_grid", grid)
        result = connection.execute(query).fetchdf()
    return add_calendar(result)


def split_frame(frame):
    """A fixed, auditable four-way chronological split; no random shuffling."""
    ts = pd.to_datetime(frame["ts"])
    observed = frame["y"].notna()
    eligible = ts >= ts.min() + pd.Timedelta(hours=WARMUP_HOURS)
    masks = {
        "train": ts < pd.Timestamp("2012-07-01"),
        "validation": (ts >= pd.Timestamp("2012-07-01")) & (ts < pd.Timestamp("2012-08-01")),
        "calibration": (ts >= pd.Timestamp("2012-08-01")) & (ts < pd.Timestamp("2012-09-01")),
        "test": ts >= pd.Timestamp("2012-09-01"),
    }
    return {name: frame.loc[mask & observed & eligible].copy() for name, mask in masks.items()}


def feature_matrix(frame):
    return frame[FEATURES].astype(float)


def feature_availability(target_timestamp):
    target = pd.Timestamp(target_timestamp)
    origin = target.normalize()
    return {"forecast_origin": origin, "latest_target_derived_input": target - pd.Timedelta(hours=24),
            "latest_input_available_at": target - pd.Timedelta(hours=23)}
