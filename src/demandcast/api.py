import hmac
import time
from datetime import date, datetime
from importlib.resources import files

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from .config import Settings
from .schemas import PredictionBatch, ShiftRequest
from .service import ForecastService


def create_app(settings=None):
    settings = settings or Settings.from_env()
    service = ForecastService(settings)
    app = FastAPI(title="DemandCast", version="0.1.0", description="Day-ahead demand forecasting and model diagnostics")
    app.state.service = service
    registry = CollectorRegistry()
    counter = Counter("demandcast_prediction_requests_total", "Successful prediction requests", registry=registry)
    latency = Histogram("demandcast_prediction_seconds", "Prediction request latency", registry=registry)

    def authorize(request: Request):
        if settings.api_token:
            supplied = request.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied.encode(), f"Bearer {settings.api_token}".encode()):
                raise HTTPException(status_code=401, detail="A valid workspace bearer token is required")

    auth = [Depends(authorize)]

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path == "/":
            ancestors = ("https://colab.research.google.com https://*.googleusercontent.com"
                         if settings.allow_colab_embed else "'none'")
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                f"img-src 'self' data:; connect-src 'self'; frame-ancestors {ancestors}")
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": service.card["version"], "bundle_verified": True}

    @app.get("/api/config", dependencies=auth)
    def config():
        return service.config()

    @app.get("/api/forecast", dependencies=auth)
    def forecast(date: date):
        started = time.perf_counter()
        try:
            result = service.forecast(date)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
        counter.inc()
        latency.observe(time.perf_counter() - started)
        return result

    @app.post("/api/predict", dependencies=auth)
    def predict(batch: PredictionBatch):
        started = time.perf_counter()
        result = service.predict(batch)
        counter.inc()
        latency.observe(time.perf_counter() - started)
        return result

    @app.get("/api/explain", dependencies=auth)
    def explain(timestamp: datetime):
        if timestamp.tzinfo is not None:
            raise HTTPException(status_code=400, detail="Use dataset-local timezone-naive timestamps")
        try:
            return service.explain(timestamp)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc

    @app.get("/api/benchmark", dependencies=auth)
    def benchmark():
        return service.report

    @app.get("/api/monitor", dependencies=auth)
    def monitor():
        return service.monitor()

    @app.post("/api/monitor/simulate", dependencies=auth)
    def simulate(request: ShiftRequest):
        return service.monitor(request.factor)

    @app.get("/metrics", dependencies=auth)
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    static = files("demandcast").joinpath("static")
    app.mount("/static", StaticFiles(directory=str(static)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(static.joinpath("index.html")))

    return app
