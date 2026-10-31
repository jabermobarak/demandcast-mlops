# Validation record

Date: 31 October 2026. This document distinguishes executed local checks from supplied configurations.

## Executed

- Downloaded the official UCI archive, pinned its SHA-256, and parsed 17,379 observed rental hours.
- Aligned the complete 17,544-hour clock grid and retained 165 unknown hourly targets as missing.
- Fit three CatBoost candidates and Ridge; measured two seasonal baselines; selected the CatBoost candidate on July validation MAE.
- Calibrated interval width on August data, then scored 2,888 observed September–December target hours.
- Recorded real local MLflow experiments and a registered candidate model. Reloaded the registered model and checked equality with bundled-model predictions. See `reports/mlflow-check.json`.
- Measured point-level errors, hourly/monthly summaries, worst days, interval coverage and a descriptive paired seven-day-block uncertainty diagnostic.
- All 38 behavior and integration tests passed. They cover clock-hour lags, future-invariance, completed-hour availability, split boundaries, invalid inputs, model integrity, SHAP additivity, online/offline parity, drift simulation and Prometheus metrics. Ruff passed. The environment emits an upstream FastAPI/Starlette TestClient deprecation warning; no test failed.
- Replayed all seven notebook code cells in a fresh local project directory with `RETRAIN=True`: source restoration, dependency installation, official-data download, fitting, MLflow registration/read-back, tests, predictions, SHAP, authenticated dashboard API and export all passed. See `reports/notebook-check.json`. Colab's hosted iframe proxy was not exercised by this local replay.
- Desktop Chromium at 1512×1120 and mobile viewport at 390×844: initial forecast, 24-hour chart, native explanations, date/capacity controls, leaderboard, drift simulation/reset, JSON export and mobile layout. No JavaScript page errors or horizontal overflow. See `reports/browser-check.json`.

## Results and practical limits

The selected model's final-test MAE is about 65.67 rentals/hour, compared with 69.50 for weekly seasonal copying and 65.90 for Ridge. The 5.51% point-estimate gain over the weekly baseline is modest, and descriptive block intervals include zero. Nominal 90% intervals cover only 82.86% of test observations. The project reports these limitations instead of selecting a new model on the test set.

Tests and figures establish behavior on the pinned historical dataset and included artifact. They do not establish real-time ingestion correctness, a source timezone, daylight-saving handling, production load, business impact, or future commercial demand quality. Raw training observations are public source data, not private customer records.

## Supplied or pending environment checks

- Docker and Compose configuration is included; Docker is unavailable in the build environment, so the image was not built here.
- A GitHub Actions matrix for Python 3.11 and 3.12 is supplied. Its hosted results must be read after repository publication; local validation used Python 3.12.
- The Colab notebook is self-contained. Local code-cell replay checks are recorded separately when run; a hosted Colab runtime and its port-proxy display are not claimed as tested here.
- There is no hosted MLflow UI, Kubernetes, cloud infrastructure, production traffic or autonomous model promotion.

`requirements-tested.txt` records the release-build package versions. Broad supported dependency ranges live in `pyproject.toml`; exact byte-for-byte model-file reproducibility is not promised because serialized metadata may vary.
