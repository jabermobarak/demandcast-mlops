# Measured chronological holdout

UCI Bike Sharing · day-ahead hourly rentals · final test: September–December 2012.

| Model | MAE (rentals/hour) | RMSE | WAPE |
| --- | ---: | ---: | ---: |
| catboost_depth6 | 65.67 | 100.16 | 27.3% |
| ridge | 65.90 | 100.70 | 27.4% |
| seasonal_168h | 69.50 | 117.78 | 28.9% |
| seasonal_24h | 79.77 | 133.15 | 33.2% |

Selected CatBoost candidate: `catboost_depth6`.
MAE change relative to the weekly seasonal baseline: 5.51% improvement.
Nominal 90% interval: measured test coverage 82.9%, mean width 223.79 rentals/hour.

## Split and protocol

| Split | Rows | First target hour | Last target hour |
| --- | ---: | --- | --- |
| train | 12819 | 2011-01-08 23:00:00 | 2012-06-30 23:00:00 |
| validation | 744 | 2012-07-01 00:00:00 | 2012-07-31 23:00:00 |
| calibration | 744 | 2012-08-01 00:00:00 | 2012-08-31 23:00:00 |
| test | 2888 | 2012-09-01 00:00:00 | 2012-12-31 23:00:00 |

Rolling day-ahead: at each day-start 00:00 origin, predict that day's 24 hourly intervals. A count labeled h is assumed available at h+1 hour, with no extra ingestion delay. Actual past test observations become available at later origins.

## Paired uncertainty diagnostic

Seven-day moving-block bootstrap, 1,000 resamples. Positive MAE reduction favors CatBoost.

| Comparator | MAE reduction | 95% descriptive interval |
| --- | ---: | --- |
| seasonal_168h | 3.83 | [-6.13, 14.68] |
| ridge | 0.23 | [-5.52, 4.81] |

The bootstrap intervals are descriptive diagnostics of this fixed holdout; no formal significance or future-performance claim is made.

## Limits

- One public, historical, aggregate demand series; not a live operating system.
- Measured rentals are observed demand, not unconstrained demand or inventory requirements.
- No same-day weather, casual or registered counts are used as features.
- Past-count features are conservative: recent observations after t-24h are omitted even when available.
- No random split. Final test was excluded from model and threshold selection.
- Temporal dependence and seasonal shift can reduce interval coverage.
- SHAP explains model associations, not causal effects.
- No production traffic, distributed scale, cloud deployment, or business savings are claimed.

These are actual executed results. No metric is a production or commercial-experience claim.
Read `benchmark.json` for versions, provenance, worst days, hourly errors and candidate runs.
