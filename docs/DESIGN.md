# Design and evaluation decisions

## One clear forecast contract

At a day-start 00:00 origin, predict all 24 hourly intervals of that calendar day. Timestamps follow the source's naive local clock labels. We assume an hourly count becomes available at the next hour boundary, without additional ingestion delay. Real deployments need a source timezone, daylight-saving policy, completeness watermark, and delay-aware cutoff.

For target hour `t`, lag 24 is the count labeled `t−24h`; lag 48 and lag 168 are analogous. The 24-hour mean covers labels `t−47h` through `t−24h`. The 168-hour mean covers `t−191h` through `t−24h`. For the latest target hour 23:00, the most recent input is the preceding day's 23:00 count, assumed complete at the origin. This convention avoids treating an unfinished hour's total as known.

Earlier target hours intentionally omit some more recent history that would be available. This conservative shared feature rule is easy to audit but can reduce accuracy. More elaborate horizon-dependent features would require equally explicit availability tests.

## Missing hours and target leakage

The source has 17,379 observations but its complete hourly grid contains 17,544 positions. SQL row-window operations only correspond to clock-hour lags after constructing that full grid. Missing rental counts remain unknown; they are not zero rentals. Window means ignore unknown past values. History-count diagnostics record how many observations contributed.

Feature windows are computed on the full grid before dropping unknown target labels. The first 191 clock hours are warmup. CatBoost handles missing numeric inputs natively; the Ridge comparator imputes using medians fitted only on training data.

Actual same-day weather is unavailable at forecast time without a weather forecast product. `casual` and `registered` directly compose the target. Neither class of variables is used. Calendar features are deterministic target-hour sine/cosine encodings, weekend status and a time index. The time index and calendar range do not make long-range future predictions validated.

## Four chronological partitions

| Partition | Target period | Purpose |
| --- | --- | --- |
| Training | Jan 2011–Jun 2012 after warmup | Fit weights and Ridge preprocessing |
| Validation | July 2012 | Early stopping and CatBoost-candidate selection |
| Calibration | August 2012 | Absolute-error interval radius |
| Final test | Sep–Dec 2012 | Final scoring and descriptive failure analysis |

There is no shuffled split and no refit on calibration data. Each test day uses earlier actual test observations when they would be available at its origin. This is rolling daily-origin evaluation, not an open-loop four-month forecast. Test results were not used to tune candidates or interval thresholds. Added diagnostic summaries do not change the selected model.

## Models and comparisons

The two seasonal baselines copy the same hour yesterday or last week; missing lag values use the training median. Ridge is a fitted linear baseline with train-only imputation and scaling. Three CatBoost parameter sets vary depth, iteration budget and learning rate, using a fixed seed and validation early stopping. The CatBoost candidate with lowest July MAE becomes the bundled candidate. The report also states whether it beat every validation baseline.

The final-test CatBoost and Ridge errors are close. A small point-estimate gain is not strong evidence that a more complex model is needed. The included paired moving-block bootstrap resamples contiguous seven-day blocks to preserve some short temporal dependence and hourly pairing. Its percentile intervals include zero in the release run. It is a descriptive uncertainty diagnostic, not a randomized experiment, population-level significance proof or causal estimate.

## Intervals, explanations and monitoring

The interval radius is the finite-sample-ranked 90th-percentile absolute residual on the separate calibration month. Lower bounds are clipped at zero. Temporal dependence and distribution change violate the simple exchangeability premise, so empirical test coverage is reported directly: about 82.9%, below the nominal 90%. Summing per-hour bounds does not produce a calibrated daily interval.

Native CatBoost SHAP values sum to the raw prediction and model baseline. They explain the model's associations; they do not identify causal levers. Demand is subsequently clipped at zero for display and error metrics.

PSI uses quantile bins from the calibration reference, an explicit missing-value bin, and 0.5 additive smoothing. The 0.20 review threshold is heuristic. Batches under 100 rows receive an insufficient-sample result. The shipped December replay differs substantially from August, so the drift page may already suggest review. Seasonality can explain such shifts; no automatic retraining or causal claim follows.

## MLflow and bundle lifecycle

Each training invocation records two seasonal baselines, Ridge and three CatBoost candidates in a local SQLite-backed experiment. The chosen native model is logged in MLflow's CatBoost flavor, registered as `DemandCast`, and given the `candidate` alias. A reload check compares registered-model predictions with the locally bundled model. This is a real local registry demonstration, not a cloud registry or production promotion.

The serving bundle includes native CatBoost bytes, a model card, interval parameters, drift reference and a small attributed replay table. A SHA-256 manifest detects accidental changes. It is not an authenticity signature or defense against an adversary who can replace both the files and manifest. There is no untrusted model-upload or pickle-loading API.

## Operational boundary

The API serves one fixed bundle, validates input shapes and values, derives calendar features with the same function used in training, and emits request counters and latency histograms. It does not store user prediction payloads. A single optional bearer token protects a localhost demonstration; per-user permissions, TLS termination, ingress limits, rate limiting, distributed feature storage and deployment orchestration are future engineering work.

`POST /api/predict` cannot verify that a caller's numeric lag values were actually computed from correctly timed observations. The source feature pipeline and tests demonstrate the contract; a real system would enforce it in an upstream feature service.
