import numpy as np
import pandas as pd
import pytest

from demandcast.data import validate_observations
from demandcast.evaluation import calibration_radius, interval_bounds, regression_metrics
from demandcast.features import FEATURES, build_features, feature_availability, split_frame
from demandcast.monitoring import build_reference, monitor_batch


@pytest.fixture
def observations():
    return pd.DataFrame({"ts": pd.date_range("2012-01-01", periods=550, freq="h"),
                         "y": np.arange(550, dtype=float)})


def test_clock_lags_preserve_missing_hours(observations):
    # A row-offset lag on a non-dense table would accidentally use hour 41 instead of 42.
    missing = observations.drop(index=42)
    result = build_features(missing).set_index("ts")
    assert len(result) == len(observations)
    assert np.isnan(result.loc[observations.ts.iloc[66], "lag_24"])
    assert np.isnan(result.loc[observations.ts.iloc[42], "y"])


def test_rolling_window_stops_at_t_minus_24(observations):
    result = build_features(observations).iloc[240]
    assert result.lag_24 == 216
    assert result.rolling_mean_24 == np.mean(np.arange(193, 217))
    assert result.rolling_mean_168 == np.mean(np.arange(49, 217))


def test_future_target_changes_cannot_change_next_day_features(observations):
    origin = pd.Timestamp("2012-01-16 00:00")
    changed = observations.copy()
    changed.loc[changed.ts >= origin, "y"] += 50000
    before = build_features(observations)
    after = build_features(changed)
    day = (before.ts >= origin) & (before.ts < origin + pd.Timedelta(hours=24))
    pd.testing.assert_frame_equal(before.loc[day, FEATURES], after.loc[day, FEATURES])


@pytest.mark.parametrize("hour", [0, 6, 12, 23])
def test_input_availability_never_exceeds_forecast_origin(hour):
    availability = feature_availability(f"2012-12-19 {hour:02d}:00")
    assert availability["latest_input_available_at"] <= availability["forecast_origin"]
    assert availability["forecast_origin"] == pd.Timestamp("2012-12-19 00:00")


def test_observed_weather_and_target_components_are_not_features():
    assert not ({"cnt", "y", "casual", "registered", "temp", "atemp", "hum", "windspeed", "weathersit"} & set(FEATURES))


def test_four_way_split_is_chronological_and_disjoint():
    ts = pd.date_range("2012-06-01", "2012-10-01", freq="h")
    frame = pd.DataFrame({"ts": ts, "y": np.ones(len(ts))})
    parts = split_frame(frame)
    previous = None
    seen = set()
    for name in ("train", "validation", "calibration", "test"):
        current = parts[name]
        assert not current.empty
        if previous is not None:
            assert previous < current.ts.min()
        assert not (seen & set(current.ts))
        seen.update(current.ts)
        previous = current.ts.max()
    assert parts["validation"].ts.min() == pd.Timestamp("2012-07-01")
    assert parts["calibration"].ts.min() == pd.Timestamp("2012-08-01")
    assert parts["test"].ts.min() == pd.Timestamp("2012-09-01")


@pytest.mark.parametrize("defect", ["duplicate", "negative", "infinite", "subhour", "timezone"])
def test_invalid_observations_rejected(observations, defect):
    frame = observations.copy()
    if defect == "duplicate":
        frame.loc[1, "ts"] = frame.loc[0, "ts"]
    elif defect == "negative":
        frame.loc[1, "y"] = -1
    elif defect == "infinite":
        frame.loc[1, "y"] = np.inf
    elif defect == "subhour":
        frame.loc[1, "ts"] += pd.Timedelta(minutes=1)
    else:
        frame["ts"] = frame.ts.dt.tz_localize("UTC")
    with pytest.raises(ValueError):
        validate_observations(frame)


def test_metrics_handle_zero_demand_without_mape_division():
    metrics = regression_metrics([0, 10], [2, 8])
    assert metrics["mae"] == 2
    assert metrics["rmse"] == 2
    assert metrics["wape"] == .4
    assert regression_metrics([0, 0], [1, 1])["wape"] is None


def test_finite_sample_calibration_rank_and_nonnegative_intervals():
    radius = calibration_radius(np.arange(10), np.zeros(10), alpha=.1)
    assert radius == 9
    lower, upper = interval_bounds([2, 20], radius)
    np.testing.assert_equal(lower, [0, 11])
    np.testing.assert_equal(upper, [11, 29])


def test_psi_identical_data_is_zero_and_large_shift_is_visible():
    values = np.linspace(1, 100, 500)
    from demandcast.features import HISTORY_FEATURES
    frame = pd.DataFrame({name: values for name in HISTORY_FEATURES})
    reference = build_reference(frame)
    unchanged = monitor_batch(frame, reference)
    assert all(abs(item["psi"]) < 1e-12 for item in unchanged["features"])
    shifted = monitor_batch(frame * 20, reference)
    assert shifted["status"] == "review"


def test_drift_does_not_issue_confident_verdict_on_tiny_batch():
    from demandcast.features import HISTORY_FEATURES
    frame = pd.DataFrame({name: np.arange(100) for name in HISTORY_FEATURES})
    report = monitor_batch(frame.head(5), build_reference(frame))
    assert report["status"] == "insufficient_sample"


def test_paired_block_interval_preserves_constant_hourly_improvement():
    from demandcast.evaluation import paired_block_bootstrap
    ts = pd.Series(pd.date_range("2012-09-01", periods=24 * 21, freq="h"))
    result = paired_block_bootstrap(np.full(len(ts), 100), np.full(len(ts), 80),
                                    np.full(len(ts), 50), ts, repetitions=100)
    assert result["mean_mae_reduction"] == 30
    assert result["percentile_interval_95"] == [30, 30]
