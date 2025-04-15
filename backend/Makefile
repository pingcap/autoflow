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

run_dev_server:
	@echo "Running server..."
	@uv run python main.py runserver --host 0.0.0.0 --port 5001

run_dev_celery_worker:
	@echo "Running celery..."
	@uv run celery -A app.celery worker -Q default

run_eval_dev_celery_worker:
	@echo "Running evaluation celery..."
	@uv run celery -A app.celery worker -Q evaluation --loglevel=debug --pool=solo
