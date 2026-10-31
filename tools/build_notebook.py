"""Build an offline-source, self-contained Colab notebook; model data is public."""

import base64
import io
import textwrap
import zipfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"data", "notebooks", ".git", ".pytest_cache", ".ruff_cache", "__pycache__", ".venv"}


def source_archive():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            relative = path.relative_to(ROOT)
            if not path.is_file() or any(p in EXCLUDED or p.endswith(".egg-info") for p in relative.parts):
                continue
            if path.name == ".env" or path.suffix in {".pyc", ".log"}:
                continue
            archive.writestr(relative.as_posix(), path.read_bytes())
    return base64.b64encode(stream.getvalue()).decode()


def main():
    def md(text):
        return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip())

    def code(text):
        return nbformat.v4.new_code_cell(textwrap.dedent(text).strip())

    cells = [md("""
        # DemandCast · train, evaluate, explain and serve

        A day-ahead bike-rental forecasting portfolio project. It demonstrates Python,
        SQL features, chronological evaluation, CatBoost, MLflow, FastAPI and monitoring.

        **Run:** choose **Runtime → Run all** on a CPU runtime. No API key or GPU is needed.
        By default this downloads the public, hash-pinned UCI dataset and trains real models.
        Set `RETRAIN=False` below to use the already-trained model instead.

        This notebook contains the complete source and model, so it does not depend on a
        GitHub checkout. MIT source; transformed public data is CC BY 4.0, attributed in
        `DATA_LICENSE.md`. Historical data is from 2011–2012; this is not a live demand feed.

        The release model has holdout MAE 65.67 rentals/hour, versus 69.50 for weekly copying
        and 65.90 for Ridge. Its 90% nominal intervals cover 82.86% of test observations.
        These are modest historical results, not proof of future or commercial performance.
    """), code("""
        import os
        from pathlib import Path

        RETRAIN = True  # @param {type:"boolean"}
        PROJECT = Path(os.environ.get("DC_NOTEBOOK_ROOT", "/content/demandcast_portfolio")).resolve()
        PROJECT.mkdir(parents=True, exist_ok=True)
        print("Project:", PROJECT)
    """), code('''
        # @title Restore the complete project and install dependencies
        import base64
        import io
        import subprocess
        import sys
        import zipfile

        SOURCE_ZIP = "__PAYLOAD__"
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ZIP))) as archive:
            for member in archive.infolist():
                target = (PROJECT / member.filename).resolve()
                if not target.is_relative_to(PROJECT):
                    raise ValueError("Unexpected source-archive path")
            archive.extractall(PROJECT)
        os.chdir(PROJECT)
        extras = ".[dev,train]" if RETRAIN else ".[dev]"
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", extras], check=True)
        sys.path.insert(0, str(PROJECT / "src"))
        print("Source restored. The raw dataset and MLflow database stay in this runtime.")
    '''), md("""
        ## Fit and evaluate without future-data leakage

        Training ends June 2012; July selects a CatBoost candidate; August calibrates
        intervals; September–December is the final test. No weather or target-component
        columns enter the features. Missing clock hours remain missing.

        At each day-start 00:00 we predict that day's 24 hours. Counts labeled hour h are
        assumed available at h+1h with no additional ingestion delay. All count features
        stop at target time minus 24 hours. Earlier test observations can become history
        for later daily origins. This is rolling evaluation, not a four-month open-loop forecast.
    """), code("""
        import json

        if RETRAIN:
            from demandcast.training import train
            benchmark = train(PROJECT, track=True)
        else:
            benchmark = json.loads((PROJECT / "reports/benchmark.json").read_text())
        import pandas as pd

        display(pd.DataFrame(benchmark["leaderboard"]).round(3))
        print("Interval coverage:", round(benchmark["intervals"]["coverage"] * 100, 2), "%")
        print("Registered model:", benchmark["registry"])
        print("Read-back check:", (PROJECT / "reports/mlflow-check.json").read_text())
        print("Block-bootstrap uncertainty:", json.dumps(benchmark["paired_uncertainty"], indent=2))
    """), code("""
        # Verify the model bundle and run behavior/integration checks.
        subprocess.run([sys.executable, "-m", "demandcast.cli", "verify"], check=True)
        subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
        subprocess.run([sys.executable, "-m", "ruff", "check", "src", "tests", "tools"], check=True)
    """), code("""
        from demandcast.config import Settings
        from demandcast.service import ForecastService

        service = ForecastService(Settings(bundle_dir=PROJECT / "artifacts", reports_dir=PROJECT / "reports"))
        forecast = service.forecast("2012-12-19")
        display(pd.DataFrame(forecast["points"]).round(2))
        explanation = service.explain("2012-12-19T08:00:00")
        display(pd.DataFrame(explanation["contributions"]).round(3))
        print("SHAP additivity error:", explanation["additivity_error"])
        print("SHAP describes model associations, not causes. Intervals apply per hour, not to the daily sum.")
    """), md("""
        ## Open the interactive dashboard

        Run the next cell. Copy its temporary workspace token into the dashboard's token
        box. Forecast dates, explanations, model comparison and drift simulation are interactive.
        The Colab server and link stop when the runtime disconnects.

        **Before sharing a copy of this notebook, clear all outputs:** the next cell prints
        a temporary access token. Source exports below exclude this token and runtime logs.
        The token is a demo access control, not a production deployment setup.
    """), code("""
        import secrets
        import time

        import httpx

        previous = globals().get("dashboard_process")
        if previous is not None and previous.poll() is None:
            previous.terminate()
            previous.wait(timeout=10)
        workspace_token = secrets.token_urlsafe(24)
        server_env = os.environ.copy()
        server_env.update({"DC_BUNDLE_DIR": str(PROJECT / "artifacts"),
                           "DC_REPORTS_DIR": str(PROJECT / "reports"),
                           "DC_API_TOKEN": workspace_token, "DC_ALLOW_COLAB_EMBED": "true"})
        dashboard_log = open(PROJECT / "dashboard.log", "w")
        dashboard_process = subprocess.Popen(
            [sys.executable, "-m", "demandcast.cli", "serve", "--host", "0.0.0.0", "--port", "8010"],
            cwd=PROJECT, env=server_env, stdout=dashboard_log, stderr=subprocess.STDOUT)
        with httpx.Client(trust_env=False) as client:
            for _attempt in range(60):
                if dashboard_process.poll() is not None:
                    raise RuntimeError((PROJECT / "dashboard.log").read_text())
                try:
                    client.get("http://127.0.0.1:8010/health").raise_for_status()
                    break
                except httpx.HTTPError:
                    time.sleep(0.5)
            else:
                raise TimeoutError("Dashboard did not become ready")
            response = client.get("http://127.0.0.1:8010/api/forecast?date=2012-12-19",
                                  headers={"Authorization": "Bearer " + workspace_token})
            response.raise_for_status()
            assert len(response.json()["points"]) == 24
        print("Temporary workspace token (clear outputs before sharing):", workspace_token)
        try:
            from google.colab import output
            output.serve_kernel_port_as_iframe(8010, height=1000)
        except ImportError:
            print("Local runtime: open http://127.0.0.1:8010")
    """), code("""
        # Export source, models and reports; no raw dataset, tracking database or token.
        export_path = PROJECT.parent / "DemandCast_Trained_Project.zip"
        ignored = {"data", ".git", ".pytest_cache", ".ruff_cache", "__pycache__", ".venv", "notebooks"}
        with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(PROJECT.rglob("*")):
                relative = path.relative_to(PROJECT)
                if not path.is_file() or any(p in ignored or p.endswith(".egg-info") for p in relative.parts):
                    continue
                if path.name == ".env" or path.suffix in {".log", ".pyc"}:
                    continue
                archive.write(path, "demandcast/" + relative.as_posix())
        print("Export:", export_path)
        try:
            from google.colab import files
            files.download(str(export_path))
        except ImportError:
            pass
    """), md("""
        ## Talk through it in an interview

        Explain the forecast origin, missing-hour handling, baseline comparison, and why
        interval coverage is below its nominal level. Explain why the measured CatBoost gain
        is modest and not decisive. Demonstrate native SHAP and a clearly labeled synthetic
        drift simulation. PSI signals an input-distribution change; it does not establish
        model failure and never triggers automatic retraining here.

        Read `docs/INTERVIEW_GUIDE.md`, `docs/DESIGN.md` and `docs/HIRING_RESEARCH.md`.
        Be clear that this is an AI-assisted portfolio project with local experiments;
        it is not employment history or a production cloud deployment.
    """)]
    cells[2].source = cells[2].source.replace("__PAYLOAD__", source_archive())
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": "DemandCast_Complete_Portfolio.ipynb"},
    })
    nbformat.validate(notebook)
    destination = ROOT / "notebooks/DemandCast_Complete_Portfolio.ipynb"
    destination.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, destination)
    print(destination, destination.stat().st_size)


if __name__ == "__main__":
    main()
