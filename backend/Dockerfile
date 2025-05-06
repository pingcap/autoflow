FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app/

# Setup supervisord.
RUN apt-get update && apt-get install -y supervisor gcc
COPY supervisord.conf /usr/etc/supervisord.conf

# Install dependencies.
COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml

RUN PYTHONDONTWRITEBYTECODE=1 uv sync --frozen
ENV PATH="/app/.venv/bin:$PATH"

# Pre-download playwright dependencies.
RUN playwright install --with-deps chromium

# Pre-download nltk data.
RUN python -c 'import nltk; \
download_dir = "/usr/local/lib/python3.11/site-packages/llama_index/core/_static/nltk_cache";\
nltk.download("stopwords", download_dir=download_dir);\
nltk.download("punkt", download_dir=download_dir);'

ENV PYTHONPATH=/app

COPY . /app/

# Default number of workers
ENV WEB_CONCURRENCY=4

CMD ["sh", "-c", "fastapi run app/api_server.py --host 0.0.0.0 --port 80 --workers ${WEB_CONCURRENCY}"]
