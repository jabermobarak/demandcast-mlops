import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from .evaluation import interval_bounds, regression_metrics
from .features import FEATURES, HISTORY_FEATURES, add_calendar, feature_matrix
from .monitoring import monitor_batch

REQUIRED_BUNDLE_FILES = {"model.cbm", "model_card.json", "reference.json", "calibration.json", "demo_features.csv"}


class ForecastService:
    def __init__(self, settings):
        self.settings = settings
        bundle = Path(settings.bundle_dir)
        if not (bundle / "manifest.json").exists():
            raise ValueError("Model bundle missing. Run from the repository root or set DC_BUNDLE_DIR.")
        manifest = json.loads((bundle / "manifest.json").read_text())
        if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != REQUIRED_BUNDLE_FILES:
            raise ValueError("Unexpected model bundle manifest")
        for name, checksum in manifest["files"].items():
            if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Bundle integrity check failed for {name}")
        self.model = CatBoostRegressor()
        self.model.load_model(str(bundle / "model.cbm"))
        if self.model.feature_names_ != FEATURES:
            raise ValueError("Model feature contract does not match the service")
        self.card = json.loads((bundle / "model_card.json").read_text())
        self.calibration = json.loads((bundle / "calibration.json").read_text())
        self.reference = json.loads((bundle / "reference.json").read_text())
        self.demo = pd.read_csv(bundle / "demo_features.csv", parse_dates=["ts", "forecast_origin"])
        self.report = json.loads((Path(settings.reports_dir) / "benchmark.json").read_text())
        self.model_hash = manifest["files"]["model.cbm"]

    def config(self):
        dates = sorted(self.demo.ts.dt.strftime("%Y-%m-%d").unique().tolist())
        return {"dates": dates, "default_date": "2012-12-19" if "2012-12-19" in dates else dates[-1],
                "model": self.card["candidate"], "model_version": self.card["version"],
                "mode": "historical_replay", "units": "rentals per hour",
                "source": "UCI Bike Sharing, 2011–2012", "model_hash": self.model_hash}

    def _predict_frame(self, frame):
        predicted = np.maximum(self.model.predict(feature_matrix(frame), thread_count=2), 0)
        lower, upper = interval_bounds(predicted, self.calibration["radius"])
        return predicted, lower, upper

    def forecast(self, date):
        start = time.perf_counter()
        frame = self.demo.loc[self.demo.ts.dt.strftime("%Y-%m-%d") == str(date)].copy()
        if frame.empty:
            raise KeyError("This historical demo date is not in the bundle")
        predicted, lower, upper = self._predict_frame(frame)
        points = []
        for i, (_, row) in enumerate(frame.iterrows()):
            points.append({"timestamp": row.ts.isoformat(), "predicted": float(predicted[i]),
                           "lower": float(lower[i]), "upper": float(upper[i]),
                           "actual": float(row.y) if pd.notna(row.y) else None,
                           "hour": int(row.ts.hour)})
        valid = frame.y.notna().to_numpy()
        metrics = regression_metrics(frame.loc[valid, "y"], predicted[valid]) if valid.any() else None
        return {"date": str(date), "forecast_origin": frame.forecast_origin.iloc[0].isoformat(),
                "mode": "historical_replay", "model_version": self.card["version"],
                "points": points, "total_predicted_rentals": float(predicted.sum()),
                "peak_hour": points[int(np.argmax(predicted))]["hour"], "observed_metrics": metrics,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "interval_note": "90% nominal intervals apply per hour, not to the daily total."}

    def predict(self, request):
        records = [{"ts": row.timestamp, **{name: getattr(row, name) for name in HISTORY_FEATURES}}
                   for row in request.rows]
        frame = add_calendar(pd.DataFrame(records))
        predicted, lower, upper = self._predict_frame(frame)
        points = [{"timestamp": row.timestamp.isoformat(), "predicted": float(predicted[i]),
                   "lower": float(lower[i]), "upper": float(upper[i])}
                  for i, row in enumerate(request.rows)]
        outside_training = any(row.timestamp.year > 2012 for row in request.rows)
        return {"model_version": self.card["version"], "points": points,
                "warnings": ["Caller must supply historical features computed with the documented t-24h cutoff."]
                + (["This date is outside the historical evaluation period; forecast quality is unvalidated."] if outside_training else [])}

    def explain(self, timestamp):
        ts = pd.Timestamp(timestamp)
        frame = self.demo.loc[self.demo.ts == ts]
        if frame.empty:
            raise KeyError("This historical hour is not available")
        matrix = feature_matrix(frame)
        shap = self.model.get_feature_importance(Pool(matrix), type="ShapValues", thread_count=2)[0]
        raw = float(self.model.predict(matrix, thread_count=2)[0])
        contributions = [{"feature": name, "value": float(matrix.iloc[0][name]) if pd.notna(matrix.iloc[0][name]) else None,
                          "contribution": float(value)} for name, value in zip(FEATURES, shap[:-1], strict=True)]
        contributions.sort(key=lambda row: -abs(row["contribution"]))
        return {"timestamp": ts.isoformat(), "base_value": float(shap[-1]), "raw_prediction": raw,
                "display_prediction": max(raw, 0), "contributions": contributions,
                "additivity_error": float(abs(shap.sum() - raw)),
                "caveat": "Native CatBoost SHAP contributions explain the raw model output, not causal effects. The displayed demand is clipped at zero."}

    def monitor(self, factor=1.0):
        frame = self.demo.tail(168).copy()
        if factor != 1:
            frame[HISTORY_FEATURES] *= factor
        report = monitor_batch(frame, self.reference)
        report.update({"simulated": factor != 1, "history_scale_factor": factor,
                       "reference_period": self.calibration["period"],
                       "current_period": {"start": frame.ts.min().isoformat(), "end": frame.ts.max().isoformat()},
                       "automatic_retraining": False})
        return report
