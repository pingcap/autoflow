# Backend of tidb.ai


## Development

### Install dependencies

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Use `uv` to install dependencies

```bash
uv sync
```

### Prepare environment

```
cp .env.example .env
```

Edit `.env` to set environment variables.


Configure independent `SECRET_KEY` and base64 `DATA_ENCRYPTION_KEY` values in the
private environment file. Existing deployments must follow
[the security migration instructions](SECURITY_MIGRATION.md) before upgrading.

### Run migrations

```bash
make migrate
```

### Run development server

```bash
uv run python main.py runserver
```