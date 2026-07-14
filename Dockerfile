# Dockerfile — FastAPI-агент (без GPU: только оркестрация, инференс в Ollama).
# Многоэтапная сборка на uv.

FROM python:3.12-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY pyproject.toml ./
# Зависимости в изолированный venv (а не pip --user в /root/.local, который
# недоступен non-root пользователю и ломает импорты в рантайме)
RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache -r pyproject.toml

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"
WORKDIR /app
# curl нужен для HEALTHCHECK — ставим в финальном образе
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data/docs /app/storage \
    && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=20s \
    CMD curl -fsS http://localhost:8000/health || exit 1
# --app-dir src: модули (main, retriever, ...) импортируют друг друга по коротким
# именам (from retriever import ...), поэтому src кладём на sys.path.
# 1 воркер: воркеры uvicorn не делят состояние; масштабирование — репликами.
CMD ["uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
