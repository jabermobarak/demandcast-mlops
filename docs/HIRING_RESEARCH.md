# Hiring research and portfolio choice

Research date: 31 October 2026. This is a targeted sample of postings, not a representative labor-market survey. Availability varies and archived roles are labeled below.

| Source | Requirements visible in the posting | Evidence in DemandCast |
| --- | --- | --- |
| [EPAM — AI/Machine Learning Engineer with Databricks](https://careers.epam.com/en/vacancy/ai-machine-learning-engineer-with-databricks-blt830q3zomkdrlbhdm_en) | Python/SQL, feature engineering, training and inference pipelines; 3+ years, Spark/Databricks/streaming; MLflow advantageous | SQL feature pipeline, trained models, API and MLflow. No Spark/Databricks or commercial-experience claim |
| [EPAM — MLOps Engineer](https://careers.epam.com/en/vacancy/mlops-engineer-blt2phqv31rqvszgtj6_en) | Serving, CI/CD, experiment/artifact management, registries, monitoring, Docker/Kubernetes, orchestration, cloud | Local experiment lifecycle, registry, bundle, monitoring, Docker and CI. Kubernetes, cloud and production operation remain gaps |
| [Джетлин — Machine learning Engineer](https://career.habr.com/vacancies/1000168444) | Employer-posted Python/SQL, CatBoost or comparable ML libraries, testing, leakage-aware validation, interpretation and monitoring; MLOps tools | Directly inspectable training, SQL, tests, temporal validation, SHAP, MLflow and metrics. Broader agent/LLM/GPU requirements are outside this repo |
| [HH Moscow junior search results](https://hh.ru/vacancies/machine-learning-engineer) | Sampled junior roles mention data preparation, model development and evaluation; other sampled junior DS postings request Python/SQL and 1–3 years | A reproducible practical project provides evidence of skills, not years of employment |
| [Yandex Market demand-forecasting Data Scientist](https://yandex.ru/jobs/vacancies/data-scientist-v-komandu-prognozirovaniya-sprosa-yandeksmarketa-5275) | Python/SQL, reliable data pipelines, model implementation; experience from one year | Relevant problem framing. **Archived** on direct inspection; included only as historical employer context |
| [Yandex Market forecasting analyst/developer](https://yandex.ru/jobs/vacancies/analitikrazrabotchik-algoritmov-prognozirovaniya-v-market-14754) | Time series, regression, ETL, Python/SQL; CatBoost listed as a plus | Relevant techniques. **Archived** on direct inspection; not advertised here as an open vacancy |

## Why a forecasting/MLOps project

EvidenceDesk demonstrates retrieval, language-model integration and cited evidence. DemandCast adds a different capability: learning a predictive model from observed data, validating it over time, and operating a small model service with monitored inputs and reproducible artifacts. Together they show more breadth than two similar chatbots.

The project uses a well-known public dataset on purpose: it is small enough to reproduce on CPU and clear enough to audit. Its strength is the complete engineering and evaluation workflow, not dataset novelty or a new algorithm.

## Experience employers are asking for

The sampled postings often seek more than a notebook: reliable Python, SQL, source control, tests, APIs, deployment, model evaluation and communication. Many roles also request prior production experience. A portfolio should explicitly distinguish hands-on project work from employment and cloud production ownership. Be ready to explain failures, assumptions and tradeoffs in an interview.

## Documentation checked

- [UCI source and attribution](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- [MLflow CatBoost logging/loading](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.catboost.html)
- [CatBoost native SHAP documentation](https://github.com/catboost/catboost/blob/master/catboost/docs/en/concepts/shap-values.md)
- [DuckDB window functions](https://duckdb.org/docs/stable/sql/functions/window_functions.html)

No placement guarantee, salary claim or production-scale result is made.
