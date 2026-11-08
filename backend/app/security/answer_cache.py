"""Versioned cache context; authorization is enforced separately in SQL."""

import hashlib
import hmac
import json

from app.core.config import settings


def _digest(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hmac.new(
        settings.SECRET_KEY.get_secret_value().encode(),
        b"autoflow.answer-cache.v1\0" + payload.encode(),
        hashlib.sha256,
    ).hexdigest()


def answer_cache_context(config, knowledge_bases, models) -> str:
    def snapshot(record):
        return (
            record.model_dump(mode="json", exclude={"credentials"}) if record else None
        )

    # Include private configuration in the keyed digest, never in public snapshots.
    # Model/KB revisions invalidate reuse when their configuration changes. Content
    # freshness still follows the cache's age limit; this is not a document index.
    return _digest(
        {
            "config": config.model_dump(mode="json"),
            "engine": snapshot(config.get_db_chat_engine()),
            "knowledge_bases": sorted(
                [snapshot(kb) for kb in knowledge_bases], key=lambda kb: kb["id"]
            ),
            "models": [snapshot(model) for model in models],
        }
    )


def answer_goal_key(goal: str, response_format: dict) -> str:
    return _digest({"goal": goal, "response_format": response_format})
