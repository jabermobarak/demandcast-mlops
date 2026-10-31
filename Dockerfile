FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir . && useradd --uid 10001 --create-home appuser
COPY artifacts ./artifacts
COPY reports ./reports
USER appuser
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8010/health', timeout=2)"
CMD ["python", "-m", "demandcast.cli", "serve", "--host", "0.0.0.0"]
