.PHONY: test

makemigrations:
	@echo "Creating migrations..."
	@if [ -z "$(NAME)" ]; then \
		uv run alembic revision --autogenerate; \
	else \
		uv run alembic revision --autogenerate -m "$(NAME)"; \
	fi

migrate:
	@echo "Migrating database..."
	@uv run alembic upgrade head

lint:
	@echo "Linting code..."
	@uv run ruff check .

format:
	@echo "Formatting code..."
	@uv run ruff format .

test:
	@echo "Running tests..."
	@uv run pytest -v tests/

dev_backend:
	@echo "Running development backend server..."
	@uv run fastapi run app/api_server.py --host 0.0.0.0 --port 5001

dev_background_worker:
	@echo "Running background worker..."
	@uv run celery -A app.celery worker -Q default

dev_eval_worker:
	@echo "Running evaluation worker..."
	@uv run celery -A app.celery worker -Q evaluation --loglevel=debug --pool=solo
