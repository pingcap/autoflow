FROM python:3.11.18-slim@sha256:db20d8506e2f15b6db39b40b6015ca46bae855dec6dd326d462cfe79ae972893

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/opt/model-cache HF_HUB_DISABLE_PROGRESS_BARS=1
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /app/requirements.lock
COPY model_manifest.py cache_models.py /app/
ARG PRELOAD_MODELS=true
RUN if [ "$PRELOAD_MODELS" = "true" ]; then python cache_models.py; fi
COPY main.py /app/main.py
RUN groupadd --gid 10001 model && useradd --uid 10001 --gid model --create-home model && mkdir -p /opt/model-cache && chown -R model:model /opt/model-cache
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
USER model
EXPOSE 5001
CMD ["uvicorn", "main:app", "--port", "5001", "--host", "0.0.0.0", "--workers", "1", "--limit-concurrency", "16", "--timeout-keep-alive", "5"]
