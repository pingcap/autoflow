"""Attributed, value-free audit records and a separate security event channel."""

from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, UTC
import hashlib
import hmac
import json
import logging
import re
from uuid import uuid4

from sqlalchemy import event, inspect
from sqlalchemy.orm import Session
from sqlalchemy.sql import visitors, operators
from sqlalchemy.sql.elements import BinaryExpression, BindParameter

audit_context = ContextVar("audit_context", default=None)
security_logger = logging.getLogger("security.audit")
SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")


@dataclass
class AuditContext:
    request_id: str = field(default_factory=lambda: str(uuid4()))
    actor_id: str | None = None
    actor_type: str = "anonymous"
    api_key_id: int | None = None
    privileged: bool = False
    method: str = ""
    source_hash: str | None = None
    token_flow: str | None = None


@contextmanager
def token_authentication_flow(purpose):
    context = audit_context.get()
    previous = context.token_flow if context else None
    if context:
        context.token_flow = purpose
    try:
        yield
    finally:
        if context:
            context.token_flow = previous


def bind_actor(user, request=None):
    context = audit_context.get()
    if context is not None and user is not None:
        context.actor_id = str(user.id)
        context.api_key_id = (
            getattr(request.state, "api_key_id", None) if request else None
        )
        context.actor_type = "api_key" if context.api_key_id is not None else "user"
        context.privileged = bool(user.is_superuser)


def security_event(name, *, status=None, route=None):
    context = audit_context.get() or AuditContext(actor_type="system")
    payload = {
        "event": name,
        "time": datetime.now(UTC).isoformat(),
        "request_id": context.request_id,
        "actor_type": context.actor_type,
        "actor_id": context.actor_id,
        "api_key_id": context.api_key_id,
        "method": context.method,
        "source_hash": context.source_hash,
    }
    if status is not None:
        payload["status"] = int(status)
    if route:
        payload["route"] = route
    security_logger.info("security_event", extra={"security_event": payload})


class SecurityEventFormatter(logging.Formatter):
    def format(self, record):
        payload = getattr(record, "security_event", {})
        keys = (
            "event",
            "time",
            "request_id",
            "actor_type",
            "actor_id",
            "api_key_id",
            "method",
            "source_hash",
            "status",
            "route",
        )
        return json.dumps(
            {key: payload[key] for key in keys if key in payload}, separators=(",", ":")
        )


class AuditMiddleware:
    def __init__(self, app, prefix="/api/v1"):
        self.app, self.prefix = app, prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        from app.core.config import settings
        from app.security.generation import client_ip
        from starlette.requests import Request

        context = AuditContext(method=scope.get("method", ""))
        peer = client_ip(Request(scope), settings.GENERATION_TRUSTED_PROXY_CIDRS)
        context.source_hash = hmac.new(
            settings.SECRET_KEY.get_secret_value().encode(),
            b"autoflow-audit-ip-v1\0" + peer.encode(),
            hashlib.sha256,
        ).hexdigest()[:32]
        token = audit_context.set(context)
        status = 500

        async def audited_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = dict(message)
                message["headers"] = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ] + [(b"x-request-id", context.request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, audited_send)
        finally:
            route = getattr(scope.get("route"), "path", None)
            # Use the router template, never the raw path, query, headers or body.
            if status in (401, 403):
                security_event("access_denied", status=status, route=route)
            elif route and "/auth/" in route and context.method == "POST":
                security_event("authentication_request", status=status, route=route)
            elif context.privileged and context.method in {
                "POST",
                "PUT",
                "PATCH",
                "DELETE",
            }:
                security_event("privileged_request", status=status, route=route)
            audit_context.reset(token)


def make_audit_record(action, target_type, target_id, fields, knowledge_base_id=None):
    from app.models.staff_action_log import StaffActionLog

    context = audit_context.get() or AuditContext(actor_type="system")
    identifier = str(target_id)
    if len(identifier) > 128 or not (
        identifier.isdecimal()
        or SAFE_NAME.fullmatch(identifier)
        or re.fullmatch(r"[0-9a-fA-F-]{36}", identifier)
    ):
        raise ValueError("Invalid audit resource identifier")
    names = sorted(
        {name for name in fields if isinstance(name, str) and SAFE_NAME.fullmatch(name)}
    )[:64]
    return StaffActionLog(
        action=action,
        target_type=target_type,
        target_id=target_id if isinstance(target_id, int) else 0,
        target_identifier=identifier,
        changed_fields=names,
        actor_id=context.actor_id,
        actor_type=context.actor_type,
        api_key_id=context.api_key_id,
        request_id=context.request_id,
        knowledge_base_id=knowledge_base_id,
        before={},
        after={},
    )


AUDITED_TABLES = {
    "users",
    "llms",
    "embedding_models",
    "reranker_models",
    "chat_engines",
    "site_settings",
    "knowledge_bases",
    "data_sources",
    "api_keys",
    "evaluation_datasets",
    "evaluation_tasks",
    "evaluation_dataset_items",
}


def _before_flush(session, flush_context, instances):
    session.info.pop("audit_pending_flush", None)
    context = audit_context.get()
    if (
        context is None
        or context.actor_id is None
        or context.method not in {"POST", "PUT", "PATCH", "DELETE"}
    ):
        return
    pending = []
    for objects, action in (
        (session.new, "create"),
        (session.dirty, "update"),
        (session.deleted, "delete"),
    ):
        for obj in list(objects):
            state = inspect(obj)
            table = state.mapper.local_table.name
            if table not in AUDITED_TABLES:
                continue
            fields = [
                attribute.key
                for attribute in state.mapper.column_attrs
                if action == "create"
                or state.attrs[attribute.key].history.has_changes()
            ]
            if action == "update" and not fields:
                continue
            pending.append((obj, action, table, fields))
    session.info["audit_pending_flush"] = pending


def _after_flush(session, flush_context):
    for obj, action, table, fields in session.info.pop("audit_pending_flush", []):
        state = inspect(obj)
        identifier = state.identity[0]
        if table == "site_settings":
            identifier = obj.name
        kb_id = obj.__dict__.get("knowledge_base_id")
        if table == "knowledge_bases":
            kb_id = identifier
        session.add(make_audit_record(action, table, identifier, fields, kb_id))


def _audit_bulk_write(execution):
    context = audit_context.get()
    if (
        context is None
        or context.actor_id is None
        or context.method not in {"POST", "PUT", "PATCH", "DELETE"}
    ):
        return
    if not (execution.is_update or execution.is_delete):
        return
    statement = execution.statement
    table = getattr(statement, "table", None)
    if table is None or table.name not in AUDITED_TABLES:
        return
    identifiers = []
    for clause in visitors.iterate(statement.whereclause):
        if (
            not isinstance(clause, BinaryExpression)
            or clause.operator != operators.eq
            or not isinstance(clause.right, BindParameter)
        ):
            continue
        column = clause.left
        if getattr(
            getattr(column, "table", None), "name", None
        ) == table.name and getattr(column, "primary_key", False):
            value = clause.right.value
            if value is None and isinstance(execution.parameters, dict):
                value = execution.parameters.get(clause.right.key)
            if value is not None:
                identifiers.append(value)
    identifier = identifiers[0] if len(identifiers) == 1 else "bulk"
    fields = [
        getattr(key, "name", key) for key in (getattr(statement, "_values", None) or {})
    ]
    result = execution.invoke_statement()
    if getattr(result, "rowcount", None) != 0:
        execution.session.add(
            make_audit_record(
                "update" if execution.is_update else "delete",
                table.name,
                identifier,
                fields,
                identifier
                if table.name == "knowledge_bases" and isinstance(identifier, int)
                else None,
            )
        )
    return result


def install_audit_hooks():
    if not event.contains(Session, "before_flush", _before_flush):
        event.listen(Session, "before_flush", _before_flush)
        event.listen(Session, "after_flush_postexec", _after_flush)
        event.listen(Session, "do_orm_execute", _audit_bulk_write)
