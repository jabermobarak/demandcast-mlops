"""A reproducible CPU experiment with a chronological holdout and local MLflow."""

import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import download_dataset, read_observations
from .evaluation import (
    calibration_radius,
    grouped_errors,
    interval_bounds,
    interval_metrics,
    paired_block_bootstrap,
    regression_metrics,
)
from .features import FEATURES, WARMUP_HOURS, build_features, feature_matrix, split_frame
from .monitoring import build_reference, monitor_batch

CANDIDATES = [
    {"name": "catboost_depth4", "depth": 4, "iterations": 650, "learning_rate": 0.07},
    {"name": "catboost_depth6", "depth": 6, "iterations": 650, "learning_rate": 0.06},
    {"name": "catboost_depth8", "depth": 8, "iterations": 450, "learning_rate": 0.06},
]


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def train(root, track=True):
    root = Path(root).resolve()
    artifacts = root / "artifacts"
    reports = root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source, provenance = download_dataset(root / "data/raw")
    observations = read_observations(source)
    frame = build_features(observations)
    parts = split_frame(frame)
    if any(len(part) < 100 for part in parts.values()):
        raise ValueError("The fixed benchmark needs at least 100 observed rows in every split")
    x = {name: feature_matrix(part) for name, part in parts.items()}
    y = {name: part["y"].to_numpy() for name, part in parts.items()}
    fallback = float(np.median(y["train"]))
    split_summary = {name: {"rows": len(part), "start": str(part.ts.min()), "end": str(part.ts.max())}
                     for name, part in parts.items()}

    mlflow = None
    experiment_id = None
    if track:
        os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
        import mlflow
        import mlflow.catboost
        tracking = root / "data/mlflow"
        tracking.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///" + str(tracking / "tracking.db"))
        experiment = mlflow.get_experiment_by_name("DemandCast chronological benchmark")
        experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(
            "DemandCast chronological benchmark", artifact_location=(tracking / "runs").as_uri())

    def track_metrics(name, params, metrics):
        if not track:
            return None
        with mlflow.start_run(experiment_id=experiment_id, run_name=name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics({"validation_" + key: value for key, value in metrics.items()
                                if value is not None})
            mlflow.set_tags({"dataset": "UCI Bike Sharing", "dataset_sha256": provenance["archive_sha256"],
                             "forecast_protocol": "daily 00:00 origin, next 24 hours; complete hourly counts assumed available at the next boundary",
                             "test_used_for_selection": "false"})
            return run.info.run_id

    validation_rows = []
    fitted = {}
    for label, lag in (("seasonal_24h", "lag_24"), ("seasonal_168h", "lag_168")):
        predictions = parts["validation"][lag].fillna(fallback).to_numpy()
        metrics = regression_metrics(y["validation"], predictions)
        run_id = track_metrics(label, {"lag_hours": lag.removeprefix("lag_")}, metrics)
        validation_rows.append({"name": label, "family": "baseline", "metrics": metrics, "run_id": run_id})

    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=20.0))
    ridge.fit(x["train"], y["train"])
    ridge_metrics = regression_metrics(y["validation"], np.maximum(ridge.predict(x["validation"]), 0))
    validation_rows.append({"name": "ridge", "family": "baseline", "metrics": ridge_metrics,
                            "run_id": track_metrics("ridge", {"alpha": 20.0, "imputation": "train-median"}, ridge_metrics)})
    for spec in CANDIDATES:
        params = {k: v for k, v in spec.items() if k != "name"}
        model = CatBoostRegressor(**params, loss_function="RMSE", eval_metric="MAE",
                                  random_seed=42, thread_count=2, verbose=False,
                                  allow_writing_files=False)
        model.fit(x["train"], y["train"], eval_set=(x["validation"], y["validation"]),
                  early_stopping_rounds=80, use_best_model=True)
        predicted = np.maximum(model.predict(x["validation"]), 0)
        metrics = regression_metrics(y["validation"], predicted)
        run_id = track_metrics(spec["name"], params | {"seed": 42, "trees_retained": model.tree_count_}, metrics)
        fitted[spec["name"]] = model
        validation_rows.append({"name": spec["name"], "family": "catboost", "params": params,
                                "trees_retained": model.tree_count_, "metrics": metrics, "run_id": run_id})

    winner = min((r for r in validation_rows if r["family"] == "catboost"), key=lambda r: r["metrics"]["mae"])
    model = fitted[winner["name"]]
    baseline_val_mae = min(r["metrics"]["mae"] for r in validation_rows if r["family"] == "baseline")
    selection = {"selected_candidate": winner["name"], "criterion": "Lowest July validation MAE among CatBoost candidates",
                 "beats_all_validation_baselines": winner["metrics"]["mae"] < baseline_val_mae,
                 "test_used_for_selection": False, "refit_on_calibration": False,
                 "candidates": validation_rows}
    model.save_model(str(artifacts / "model.cbm"))

    calibration_predicted = np.maximum(model.predict(x["calibration"]), 0)
    radius = calibration_radius(y["calibration"], calibration_predicted)
    calibration = {"method": "Absolute-residual split calibration", "alpha": 0.10,
                   "radius": radius, "rows": len(y["calibration"]),
                   "period": split_summary["calibration"],
                   "guarantee": "No distribution-free temporal coverage guarantee; inspect measured holdout coverage."}

    # Only after selecting the model and calibrating intervals do we score the final test period.
    test_predictions = np.maximum(model.predict(x["test"]), 0)
    lower, upper = interval_bounds(test_predictions, radius)
    holdout = parts["test"][["ts", "forecast_origin"]].copy()
    holdout["actual"] = y["test"]
    holdout["predicted"] = test_predictions
    holdout["lower"] = lower
    holdout["upper"] = upper
    holdout["seasonal_24h"] = parts["test"].lag_24.fillna(fallback).to_numpy()
    holdout["seasonal_168h"] = parts["test"].lag_168.fillna(fallback).to_numpy()
    holdout["ridge"] = np.maximum(ridge.predict(x["test"]), 0)
    holdout["absolute_error"] = np.abs(holdout.actual - holdout.predicted)
    holdout.to_csv(reports / "holdout_predictions.csv", index=False, float_format="%.8f")
    leaderboard = [{"name": name, **regression_metrics(holdout.actual, holdout[column])}
                   for name, column in ((winner["name"], "predicted"), ("seasonal_24h", "seasonal_24h"),
                                        ("seasonal_168h", "seasonal_168h"), ("ridge", "ridge"))]
    leaderboard.sort(key=lambda r: r["mae"])
    cat_mae = regression_metrics(holdout.actual, holdout.predicted)["mae"]
    weekly_mae = regression_metrics(holdout.actual, holdout.seasonal_168h)["mae"]
    daily_errors = holdout.assign(day=holdout.ts.dt.strftime("%Y-%m-%d")).groupby("day").absolute_error.mean().nlargest(8)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(), "python": platform.python_version(),
        "versions": {p: version(p) for p in ("catboost", "duckdb", "pandas", "numpy", "scikit-learn")},
        "source": provenance, "observed_rows": len(observations), "clock_hours": len(frame),
        "missing_clock_hours": int(frame.y.isna().sum()), "warmup_hours": WARMUP_HOURS,
        "splits": split_summary, "selection": selection,
        "leaderboard": leaderboard, "selected_test_metrics": regression_metrics(holdout.actual, holdout.predicted),
        "mae_improvement_vs_weekly_baseline_percent": (weekly_mae - cat_mae) / weekly_mae * 100,
        "intervals": interval_metrics(y["test"], lower, upper), "calibration": calibration,
        "by_hour": grouped_errors(holdout, "hour"), "by_month": grouped_errors(holdout, "month"),
        "paired_uncertainty": {name: paired_block_bootstrap(holdout.actual, holdout.predicted, holdout[name], holdout.ts)
                               for name in ("seasonal_168h", "ridge")},
        "worst_days": [{"date": date, "mae": float(error)} for date, error in daily_errors.items()],
        "feature_importance": sorted([{"feature": feature, "importance": float(value)}
                                      for feature, value in zip(FEATURES, model.feature_importances_, strict=True)],
                                     key=lambda r: -r["importance"]),
        "protocol": "Rolling day-ahead: at each day-start 00:00 origin, predict that day's 24 hourly intervals. A count labeled h is assumed available at h+1 hour, with no extra ingestion delay. Actual past test observations become available at later origins.",
        "limitations": ["One public, historical, aggregate demand series; not a live operating system.",
                        "Measured rentals are observed demand, not unconstrained demand or inventory requirements.",
                        "No same-day weather, casual or registered counts are used as features.",
                        "Past-count features are conservative: recent observations after t-24h are omitted even when available.",
                        "No random split. Final test was excluded from model and threshold selection.",
                        "Temporal dependence and seasonal shift can reduce interval coverage.",
                        "SHAP explains model associations, not causal effects.",
                        "No production traffic, distributed scale, cloud deployment, or business savings are claimed."]}

    registry = {"enabled": False}
    if track:
        from mlflow.models import infer_signature
        with mlflow.start_run(run_id=winner["run_id"]):
            example = x["train"].head(4)
            info = mlflow.catboost.log_model(
                model, name="model", signature=infer_signature(example, model.predict(example)),
                input_example=example,
                pip_requirements=[f"catboost=={version('catboost')}", f"pandas=={version('pandas')}",
                                  f"numpy=={version('numpy')}"])
            mlflow.log_metrics({"test_" + k: v for k, v in report["selected_test_metrics"].items() if v is not None})
        registered = mlflow.register_model(info.model_uri, "DemandCast")
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias("DemandCast", "candidate", registered.version)
        registry = {"enabled": True, "tracking_backend": "local SQLite", "name": "DemandCast",
                    "version": registered.version, "alias": "candidate", "run_id": winner["run_id"],
                    "mlflow_version": version("mlflow")}
        reloaded = mlflow.catboost.load_model("models:/DemandCast@candidate")
        sample = x["test"].head(10)
        difference = float(np.max(np.abs(reloaded.predict(sample) - model.predict(sample))))
        if difference > 1e-9:
            raise RuntimeError("Registered model predictions differ from the selected model")
        _write_json(reports / "mlflow-check.json", {
            **registry, "loaded_registered_model": True, "sample_rows": len(sample),
            "maximum_prediction_difference": difference,
            "runs_this_training": len(validation_rows),
        })
    else:
        _write_json(reports / "mlflow-check.json", {"enabled": False, "loaded_registered_model": False})
    report["registry"] = registry
    report["training_seconds"] = round(time.perf_counter() - started, 2)

    # A compact, attributed public-data sample makes the dashboard runnable without retraining.
    demo = frame.loc[frame.ts >= frame.ts.max().normalize() - pd.Timedelta(days=27)].copy()
    demo.to_csv(artifacts / "demo_features.csv", index=False, float_format="%.10f")
    reference = build_reference(parts["calibration"])
    monitor = monitor_batch(demo.tail(168), reference)
    monitor["reference_period"] = split_summary["calibration"]
    monitor["current_period"] = {"start": str(demo.tail(168).ts.min()), "end": str(demo.ts.max())}
    _write_json(artifacts / "reference.json", reference)
    _write_json(artifacts / "calibration.json", calibration)
    _write_json(artifacts / "model_card.json", {"name": "DemandCast", "version": "0.1.0",
                "candidate": winner["name"], "features": FEATURES, "seed": 42,
                "prediction_units": "rentals per hour", "forecast_origin": "Day-start 00:00, dataset-local naive time; counts complete at the next hour boundary",
                "train_period": split_summary["train"], "source": provenance, "registry": registry,
                "limitations": report["limitations"]})
    _write_json(reports / "benchmark.json", report)
    _write_json(reports / "monitoring.json", monitor)
    _write_json(reports / "experiment_runs.json", selection)
    write_markdown_report(reports, report)
    make_figures(reports, holdout, report)
    names = ["model.cbm", "model_card.json", "reference.json", "calibration.json", "demo_features.csv"]
    manifest = {"schema_version": 1, "files": {name: hashlib.sha256((artifacts / name).read_bytes()).hexdigest()
                                             for name in names}}
    _write_json(artifacts / "manifest.json", manifest)
    return report


def write_markdown_report(directory, report):
    rows = ["# Measured chronological holdout", "", "UCI Bike Sharing · day-ahead hourly rentals · final test: September–December 2012.", "",
            "| Model | MAE (rentals/hour) | RMSE | WAPE |", "| --- | ---: | ---: | ---: |"]
    for item in report["leaderboard"]:
        rows.append(f"| {item['name']} | {item['mae']:.2f} | {item['rmse']:.2f} | {item['wape']:.1%} |")
    rows += ["", f"Selected CatBoost candidate: `{report['selection']['selected_candidate']}`.",
             f"MAE change relative to the weekly seasonal baseline: {report['mae_improvement_vs_weekly_baseline_percent']:.2f}% improvement.",
             f"Nominal 90% interval: measured test coverage {report['intervals']['coverage']:.1%}, mean width {report['intervals']['mean_width']:.2f} rentals/hour.",
             "", "## Split and protocol", "", "| Split | Rows | First target hour | Last target hour |", "| --- | ---: | --- | --- |"]
    for name, part in report["splits"].items():
        rows.append(f"| {name} | {part['rows']} | {part['start']} | {part['end']} |")
    rows += ["", report["protocol"], "", "## Paired uncertainty diagnostic", "",
             "Seven-day moving-block bootstrap, 1,000 resamples. Positive MAE reduction favors CatBoost.", "",
             "| Comparator | MAE reduction | 95% descriptive interval |", "| --- | ---: | --- |"]
    for comparator, item in report["paired_uncertainty"].items():
        low, high = item["percentile_interval_95"]
        rows.append(f"| {comparator} | {item['mean_mae_reduction']:.2f} | [{low:.2f}, {high:.2f}] |")
    rows += ["", "The bootstrap intervals are descriptive diagnostics of this fixed holdout; no formal significance or future-performance claim is made.",
             "", "## Limits", "", *[f"- {x}" for x in report["limitations"]],
             "", "These are actual executed results. No metric is a production or commercial-experience claim.",
             "Read `benchmark.json` for versions, provenance, worst days, hourly errors and candidate runs."]
    (directory / "benchmark.md").write_text("\n".join(rows) + "\n")


def make_figures(directory, predictions, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#f8fafc", "axes.facecolor": "#f8fafc"})
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]})
    sample = predictions.loc[(predictions.ts >= "2012-12-17") & (predictions.ts < "2012-12-20")]
    axes[0].fill_between(sample.ts, sample.lower, sample.upper, color="#bfdbfe", alpha=.65, label="90% calibrated interval")
    axes[0].plot(sample.ts, sample.actual, color="#111827", linewidth=1.6, label="Observed")
    axes[0].plot(sample.ts, sample.predicted, color="#2563eb", linewidth=1.6, label="CatBoost forecast")
    axes[0].set(title="A rolling day-ahead forecast on held-out data", ylabel="Rentals / hour")
    axes[0].legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
    labels = [r["name"] for r in report["leaderboard"]]
    values = [r["mae"] for r in report["leaderboard"]]
    axes[1].barh(labels, values, color=["#2563eb" if "catboost" in name else "#94a3b8" for name in labels])
    axes[1].set(xlabel="Final-test mean absolute error · rentals / hour")
    figure.tight_layout(pad=2)
    figure.savefig(directory / "holdout_results.png", dpi=160)
    plt.close(figure)
