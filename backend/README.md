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


### Run migrations

```bash
make migrate
```

### Run development server

```bash
uv run python main.py runserver
```