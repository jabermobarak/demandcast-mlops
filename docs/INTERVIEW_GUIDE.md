# Explain the project in an interview

## Simple explanation

DemandCast predicts how many bike rentals to expect in each hour of a day. It learns from past rental counts and calendar patterns. The dashboard shows the forecast, a range around it, the model's reasons, and checks for changing input data.

The distinctive work is the pipeline around the model: making features available at the right time, comparing simple alternatives, measuring on later data, recording the experiment, serving predictions, and reporting limitations.

## A three-minute demonstration

1. Open the historical forecast for 19 December 2012. Point out the replay label, forecast origin and 24-hour horizon. This is not today's live demand.
2. Compare the forecast curve with observed rentals. Explain the interval's nominal 90% target and actual 82.9% test coverage.
3. Select 08:00 and show native SHAP contributions. Explain that they add to the model output but do not prove causality.
4. Open Model benchmark. Compare the selected model with yesterday/weekly seasonal forecasts and Ridge. Explain the modest point-estimate gain and uncertainty.
5. Open Drift monitor. Distinguish observed historical seasonality from the synthetic multiplier stress test. Point out that the system does not retrain automatically.
6. Open the feature SQL or tests to show a future-data mutation test and a missing-hour test.

## Questions and defensible answers

| Question | What to explain |
| --- | --- |
| What did you train? | A CatBoost regressor on a public hourly rental series, comparing three candidate settings and three simple baselines. |
| Why not a language model or a neural network? | The task is structured regression on a small tabular series. Simple baselines and boosting are appropriate comparisons; complexity needs measured justification. |
| How do you prevent leakage? | Define the day-start origin, use completed counts no later than t−24h, exclude actual same-day weather and target components, and test future invariance. |
| Why fill missing clock positions? | Otherwise SQL LAG means previous rows rather than exactly 24 or 168 clock hours. Unknown targets remain missing. |
| Why four splits? | Fit parameters, choose/early-stop candidates, calibrate intervals, then score on a later untouched period. |
| Did you use earlier test-day actuals? | Yes, when available at subsequent daily origins. This is rolling day-ahead evaluation, not one forecast across four months. |
| Which metrics? | MAE in rentals/hour, RMSE for larger errors, and WAPE for aggregate relative error without dividing by each zero-demand hour. |
| Is the gain significant? | The point estimate is 5.51% versus weekly seasonal copying, but the descriptive seven-day-block intervals include zero. No decisive superiority claim. |
| Are the 90% intervals reliable? | Empirical test coverage is only 82.86%. Temporal dependence/shift make a simple exchangeability guarantee inappropriate. |
| What does SHAP prove? | How this model's raw prediction decomposes across inputs. It is not a causal explanation or an action recommendation. |
| Is PSI a model-failure test? | No. It measures marginal feature-distribution changes with a heuristic threshold. Seasonality and small samples matter. |
| What does MLflow do here? | Records real comparison runs, model artifacts and metrics; registers the selected model and a candidate alias; reload validation checks the saved model. |
| What is production-ready? | This is a local demonstration with tests, validation, packaging and metrics. It has not been run as a real commercial service. |
| What would you improve? | Proper source timezone/delays, richer known calendar features, rolling-origin selection, better calibrated intervals, external feature storage and controlled deployment. Improvements need a fresh evaluation period. |

## A learning exercise before applying

Trace one prediction back to the exact lag values and SQL windows. Run the tests. Reproduce the training command and locate the resulting MLflow runs. Inspect one bad day, propose a hypothesis, and design a new evaluation rather than choosing a feature solely because it improves the current holdout.

The project was developed with AI assistance. Explain that honestly and demonstrate your understanding through code walkthroughs, measurements and your own modifications.

## CV bullet after you reproduce it

Built a day-ahead demand forecasting service with Python, DuckDB SQL, CatBoost, chronological model selection, separately calibrated prediction intervals, MLflow tracking/registry, native SHAP explanations, FastAPI, drift diagnostics, Docker and automated tests.

If adding metrics, include the dataset and split. For example, state the historical 2,888-hour test MAE and the baseline comparison; do not claim proven business savings, deployment scale or years of commercial experience.
