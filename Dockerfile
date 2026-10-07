FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VLLM_EVAL_CONFIG=/app/config.toml

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./
RUN python -m pip install --no-cache-dir .

COPY config.example.toml ./config.toml
RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "vllm_evaluation.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
