"""Sentry diagnostics contain code locations and timing, not application data."""

import math
import re
from datetime import datetime
from pathlib import PurePath


def _identifier(value, length=128):
    return (
        isinstance(value, str)
        and len(value) <= length
        and re.fullmatch(r"[A-Za-z0-9_.<>/-]+", value)
    )


def _time(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError:
            pass
    return None


def _trace(value):
    result = {}
    for key, size in [("trace_id", 32), ("span_id", 16), ("parent_span_id", 16)]:
        field = value.get(key)
        if isinstance(field, str) and re.fullmatch(r"[0-9a-f]{%d}" % size, field):
            result[key] = field
    # Span operation is a category, never a URL, SQL statement or model prompt.
    operation = value.get("op", "")
    if operation in {
        "http.server",
        "http.client",
        "db",
        "db.sql.query",
        "queue.task",
        "function",
    }:
        result["op"] = operation
    else:
        result["op"] = "app"
    for key in ["start_timestamp", "timestamp"]:
        field = _time(value.get(key))
        if field is not None:
            result[key] = field
    return result


def _stacktrace(value):
    frames = []
    for frame in value.get("frames", [])[-100:]:
        safe = {}
        for key in ["module", "function"]:
            if _identifier(frame.get(key)):
                safe[key] = frame[key]
        filename = frame.get("filename")
        if isinstance(filename, str) and _identifier(PurePath(filename).name):
            safe["filename"] = PurePath(filename).name
        if type(frame.get("lineno")) is int:
            safe["lineno"] = frame["lineno"]
        if type(frame.get("in_app")) is bool:
            safe["in_app"] = frame["in_app"]
        frames.append(safe)
    return {"frames": frames}


def privacy_options(routes=()):
    # Only names from registered application routes may identify transactions.
    # Raw 404 paths, provider URLs and caller-set names are never forwarded.
    route_names = {
        route.path for route in routes if isinstance(getattr(route, "path", None), str)
    }

    def project(event, hint):
        # Attachments are separate envelope items, outside the event JSON.
        hint["attachments"] = []
        safe = {"platform": "python"}
        if event.get("environment") in {"local", "staging", "production"}:
            safe["environment"] = event["environment"]
        release = event.get("release")
        if isinstance(release, str) and re.fullmatch(
            r"[0-9a-f]{7,40}|v?\d+\.\d+\.\d+", release
        ):
            safe["release"] = release
        event_id = event.get("event_id")
        if isinstance(event_id, str) and re.fullmatch(r"[0-9a-f]{32}", event_id):
            safe["event_id"] = event_id
        for key in ["timestamp", "start_timestamp"]:
            value = _time(event.get(key))
            if value is not None:
                safe[key] = value
        if event.get("level") in {"debug", "info", "warning", "error", "fatal"}:
            safe["level"] = event["level"]
        if _identifier(event.get("logger")):
            safe["logger"] = event["logger"]
        safe["transaction"] = (
            event.get("transaction")
            if event.get("transaction") in route_names
            else "application"
        )
        safe["transaction_info"] = {"source": "custom"}
        trace = event.get("contexts", {}).get("trace")
        if isinstance(trace, dict):
            safe["contexts"] = {"trace": _trace(trace)}
        method = event.get("request", {}).get("method")
        if method in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}:
            safe["request"] = {"method": method}
        exceptions = []
        for error in event.get("exception", {}).get("values", [])[-10:]:
            clean = {"value": "[redacted]"}
            for key in ["type", "module"]:
                if _identifier(error.get(key)):
                    clean[key] = error[key]
            if isinstance(error.get("stacktrace"), dict):
                clean["stacktrace"] = _stacktrace(error["stacktrace"])
            mechanism = error.get("mechanism", {})
            if type(mechanism.get("handled")) is bool:
                clean["mechanism"] = {
                    "type": "generic",
                    "handled": mechanism["handled"],
                }
            exceptions.append(clean)
        if exceptions:
            safe["exception"] = {"values": exceptions}
        else:
            safe["message"] = "Application diagnostic; message content omitted"
        if isinstance(event.get("stacktrace"), dict):
            safe["stacktrace"] = _stacktrace(event["stacktrace"])
        if event.get("type") == "transaction":
            safe["type"] = "transaction"
            safe["spans"] = [
                _trace(span)
                for span in event.get("spans", [])[:1000]
                if isinstance(span, dict)
            ]
        return safe

    def scrub(event, hint):
        # Fail closed if an SDK update supplies an unexpected event structure.
        try:
            return project(event, hint)
        except Exception:
            return None

    return {
        "send_default_pii": False,
        "include_local_variables": False,
        "include_source_context": False,
        "max_request_body_size": "never",
        "before_send": scrub,
        "before_send_transaction": scrub,
        "before_breadcrumb": lambda breadcrumb, hint: None,
        # SDK 2.26.1 profiles and sessions bypass these event projection hooks.
        "profiles_sample_rate": 0.0,
        "profile_session_sample_rate": 0.0,
        "auto_session_tracking": False,
        "_experiments": {"enable_logs": False, "enable_metrics": False},
        "spotlight": False,
    }
