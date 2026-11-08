# Autoflow

A unified storage layer for AI agents and RAG applications.

## Isolated SDK compatibility checks

The core SDK now uses the same LiteLLM 1.108.3 release as the backend, including
the fix for GHSA-g5ff-637f-6q2m. The advisory concerns LiteLLM Proxy routes; the
core SDK does not start a Proxy service. The lockfile also records the compatible
transitive dependencies and platform wheels. Both projects require filelock
4.2.4 or later; LiteLLM 1.108.3 supports that patched version without a dependency
override. Earlier LiteLLM releases constrained filelock below 4 and would restore
known vulnerabilities during resolution.

Run the checks from this directory with Python 3.11 and uv 0.13.6:

```sh
uv sync --frozen --no-dev --python 3.11
uv run --frozen --no-dev --python 3.11 python -m unittest discover -s security_tests -v
```

These checks exercise chat, streaming, single/batch embeddings and DSPy conversion
through a loopback HTTP fixture, with explicit dummy credentials. They block
external DNS/connections and do not load the integration-test `.env`, connect to
TiDB or call a paid provider. The existing `tests/` integration suite needs its
own explicitly provisioned test database and provider credentials.
