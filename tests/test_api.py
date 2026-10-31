import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from demandcast.api import create_app
from demandcast.config import Settings
from demandcast.features import HISTORY_FEATURES
from demandcast.schemas import PredictionBatch, PredictionRow
from demandcast.service import ForecastService

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def settings():
    return Settings(bundle_dir=ROOT / "artifacts", reports_dir=ROOT / "reports")


@pytest.fixture(scope="module")
def service(settings):
    return ForecastService(settings)


@pytest.fixture(scope="module")
def client(settings):
    with TestClient(create_app(settings)) as value:
        yield value


def test_dashboard_and_bundle_are_available(client):
    assert client.get("/health").json()["bundle_verified"] is True
    result = client.get("/")
    assert "DemandCast" in result.text
    assert "frame-ancestors 'none'" in result.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_forecast_returns_24_clock_hours_with_intervals(client):
    result = client.get("/api/forecast", params={"date": "2012-12-19"}).json()
    assert result["forecast_origin"] == "2012-12-19T00:00:00"
    assert result["mode"] == "historical_replay"
    assert [point["hour"] for point in result["points"]] == list(range(24))
    for point in result["points"]:
        assert 0 <= point["lower"] <= point["predicted"] <= point["upper"]


def test_native_shap_values_add_up_to_the_raw_model_output(client):
    result = client.get("/api/explain", params={"timestamp": "2012-12-19T08:00:00"}).json()
    total = result["base_value"] + sum(row["contribution"] for row in result["contributions"])
    assert abs(total - result["raw_prediction"]) < 1e-6
    assert result["additivity_error"] < 1e-6


def test_training_and_online_feature_contract_agree(service):
    row = service.demo.iloc[20]
    request_row = PredictionRow(timestamp=row.ts.to_pydatetime(), **{name: float(row[name]) for name in HISTORY_FEATURES})
    online = service.predict(PredictionBatch(rows=[request_row]))["points"][0]["predicted"]
    offline = max(float(service.model.predict(row[service.model.feature_names_].astype(float).to_numpy())), 0)
    assert abs(online - offline) < 1e-6


def test_json_prediction_api_accepts_valid_features_and_reports_out_of_period(client):
    row = {"timestamp": "2013-01-01T08:00:00", "lag_24": 100, "lag_48": 120, "lag_168": 140,
           "rolling_mean_24": 110, "rolling_mean_168": 130}
    result = client.post("/api/predict", json={"rows": [row]})
    assert result.status_code == 200
    assert result.json()["points"][0]["predicted"] >= 0
    assert any("unvalidated" in warning for warning in result.json()["warnings"])


@pytest.mark.parametrize("defect", ["negative", "all_missing", "timezone", "subhour", "leaky_field", "infinite"])
def test_prediction_contract_rejects_bad_inputs(client, defect):
    row = {"timestamp": "2012-12-19T08:00:00", **{name: 100 for name in HISTORY_FEATURES}}
    if defect == "negative":
        row["lag_24"] = -5
    elif defect == "all_missing":
        row.update(dict.fromkeys(HISTORY_FEATURES))
    elif defect == "timezone":
        row["timestamp"] += "Z"
    elif defect == "subhour":
        row["timestamp"] = "2012-12-19T08:15:00"
    elif defect == "leaky_field":
        row["registered"] = 120
    else:
        row["lag_24"] = "Infinity"
    assert client.post("/api/predict", json={"rows": [row]}).status_code == 422


def test_prediction_batch_size_is_bounded(client):
    assert client.post("/api/predict", json={"rows": []}).status_code == 422


def test_auth_protects_model_endpoints_and_metrics(settings):
    with TestClient(create_app(replace(settings, api_token="test-workspace-token"))) as app:
        assert app.get("/api/benchmark").status_code == 401
        assert app.get("/metrics").status_code == 401
        assert app.get("/health").status_code == 200
        assert app.get("/api/config", headers={"Authorization": "Bearer test-workspace-token"}).status_code == 200


def test_colab_framing_is_opt_in(settings):
    with TestClient(create_app(replace(settings, allow_colab_embed=True))) as app:
        assert "https://colab.research.google.com" in app.get("/").headers["content-security-policy"]


def test_bundle_tampering_fails_before_model_load(settings, tmp_path):
    copied = tmp_path / "bundle"
    shutil.copytree(settings.bundle_dir, copied)
    with (copied / "model.cbm").open("ab") as output:
        output.write(b"modified")
    with pytest.raises(ValueError, match="integrity"):
        ForecastService(replace(settings, bundle_dir=copied))


def test_drift_simulation_is_explicit_and_does_not_retrain(client):
    baseline = client.get("/api/monitor").json()
    shifted = client.post("/api/monitor/simulate", json={"factor": 1.4}).json()
    assert baseline["simulated"] is False
    assert shifted["simulated"] is True
    assert shifted["automatic_retraining"] is False
    assert baseline["features"] != shifted["features"]
    assert client.post("/api/monitor/simulate", json={"factor": 100}).status_code == 422


def test_prometheus_metrics_are_real_counters(client):
    client.get("/api/forecast", params={"date": "2012-12-19"})
    result = client.get("/metrics")
    assert "demandcast_prediction_requests_total" in result.text
    assert "demandcast_prediction_seconds_count" in result.text


def test_unknown_demo_date_and_hour_fail_clearly(client):
    assert client.get("/api/forecast", params={"date": "2030-01-01"}).status_code == 404
    assert client.get("/api/explain", params={"timestamp": "2030-01-01T00:00:00"}).status_code == 404


def test_exported_holdout_results_are_internally_consistent(service):
    import pandas as pd

    from demandcast.evaluation import regression_metrics
    predictions = pd.read_csv(ROOT / "reports/holdout_predictions.csv")
    measured = regression_metrics(predictions.actual, predictions.predicted)
    assert np.isclose(measured["mae"], service.report["selected_test_metrics"]["mae"], atol=1e-6)
    assert measured["rows"] == 2888
    # Source counts are independent data checks, not a model-score threshold.
    assert service.report["source"]["archive_sha256"] == "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
    assert len(json.loads((ROOT / "artifacts/manifest.json").read_text())["files"]) == 5
