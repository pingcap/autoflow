import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from redis import Redis
from app.security.generation_limits import (
    GenerationLimiter,
    GenerationLimits,
    GenerationRateLimited,
    GenerationUnavailable,
    RELEASE,
    ip_bucket,
)

from types import SimpleNamespace
from app.security import generation


def request(peer="192.0.2.1", forwarded=""):
    return SimpleNamespace(
        client=SimpleNamespace(host=peer),
        headers={"x-forwarded-for": forwarded},
        state=SimpleNamespace(browser_id="signed-guest"),
    )


def admitted_scope(redis_server, monkeypatch, **limits):
    service = limiter(redis_server, **limits)
    monkeypatch.setattr(generation, "get_generation_limiter", lambda: service)
    scope = generation.GenerationScope(request(), None)
    scope.admit()
    return scope


@pytest.fixture(scope="module")
def redis_server(tmp_path_factory):
    binary = os.environ.get("SECURITY_REDIS_SERVER") or shutil.which("redis-server")
    if not binary:
        pytest.fail(
            "Install redis-server or set SECURITY_REDIS_SERVER to run real quota tests"
        )
    directory = tmp_path_factory.mktemp("quota-redis")
    # macOS Unix socket paths have a small length limit; an ephemeral loopback
    # port avoids coupling tests to the caller's checkout or temp directory.
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [
            binary,
            "--bind",
            "127.0.0.1",
            "--port",
            str(port),
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(directory),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = Redis(
        host="127.0.0.1",
        port=port,
        socket_timeout=1,
        socket_connect_timeout=1,
        decode_responses=True,
    )
    try:
        for _ in range(100):
            if process.poll() is not None:
                pytest.fail("Isolated Redis exited before becoming ready")
            try:
                if client.ping():
                    break
            except Exception:
                time.sleep(0.02)
        else:
            pytest.fail("Isolated Redis did not start")
        yield client
    finally:
        client.close()
        process.terminate()
        process.wait(timeout=5)


def limiter(redis_server, **overrides):
    return GenerationLimiter(
        redis_server,
        "test-secret-" * 4,
        GenerationLimits(**overrides),
        namespace=uuid4().hex,
    )


def acquire(
    service, actor="user:1", peer="192.0.2.1", resource=None, authenticated=True
):
    return service.acquire(
        actor=actor, client_ip=peer, resource=resource, authenticated=authenticated
    )


def test_atomic_admission_across_workers_never_oversubscribes(redis_server):
    service = limiter(redis_server, global_concurrent=4)
    other_worker = GenerationLimiter(
        redis_server,
        "test-secret-" * 4,
        service.limits,
        namespace=service.prefix.split("{")[1].split("}")[0],
    )

    def attempt(index):
        try:
            return acquire(
                service if index % 2 else other_worker, actor=f"user:{index}"
            )
        except GenerationRateLimited:
            return None

    with ThreadPoolExecutor(max_workers=32) as pool:
        admitted = [lease for lease in pool.map(attempt, range(64)) if lease]
    assert len(admitted) == 4
    assert redis_server.zcard(service.prefix + "global:active") == 4
    assert redis_server.hget(service.prefix + "global:day", "count") == "4"
    for lease in admitted:
        lease.close()
    assert redis_server.zcard(service.prefix + "global:active") == 0
    acquire(service).close()


def test_account_budget_cannot_reset_by_changing_ip_or_browser(redis_server):
    service = limiter(redis_server, user_per_day=2)
    acquire(service).close()
    acquire(service, peer="192.0.2.2").close()
    with pytest.raises(GenerationRateLimited) as error:
        acquire(service, peer="192.0.2.3")
    assert error.value.retry_after > 0
    acquire(service, actor="user:2").close()


def test_guest_reset_still_hits_ip_budget_and_global_budget(redis_server):
    service = limiter(redis_server, ip_per_day=2, global_per_day=3)
    for index in range(2):
        acquire(service, actor=f"guest:{index}", authenticated=False).close()
    with pytest.raises(GenerationRateLimited):
        acquire(service, actor="guest:new", authenticated=False)
    acquire(service, actor="guest:next", peer="192.0.2.2", authenticated=False).close()
    with pytest.raises(GenerationRateLimited):
        acquire(service, actor="guest:last", peer="192.0.2.3", authenticated=False)


def test_resource_lock_and_expired_owner_cannot_release_successor(redis_server):
    service = limiter(redis_server)
    old = acquire(service, resource="chat:123")
    with pytest.raises(GenerationRateLimited):
        acquire(service, resource="chat:123")
    # Simulate expiration while the first worker is suspended.
    for key in old.keys[6:9]:
        redis_server.zadd(key, {old.token: 0})
    redis_server.delete(old.keys[9])
    new = acquire(service, resource="chat:123")
    redis_server.eval(RELEASE, 4, *old.keys[6:], old.token)
    assert redis_server.get(new.keys[9]) == new.token
    assert redis_server.zscore(new.keys[6], new.token) is not None
    assert all(redis_server.pttl(key) > 0 for key in new.keys)
    new.close()


def test_window_rollover_and_rejected_admission_do_not_charge_other_budgets(
    redis_server,
):
    service = limiter(redis_server, user_per_minute=1)
    lease = acquire(service)
    lease.close()
    with pytest.raises(GenerationRateLimited):
        acquire(service)
    assert redis_server.hget(lease.keys[1], "count") == "1"
    redis_server.hset(lease.keys[2], "bucket", -1)
    acquire(service).close()
    assert redis_server.hget(lease.keys[2], "count") == "1"
    assert redis_server.hget(lease.keys[1], "count") == "2"
    assert all("192.0.2.1" not in key and "user:1" not in key for key in lease.keys)


def test_unavailable_redis_never_grants_work():
    client = Redis(
        unix_socket_path="/does-not-exist-autoflow-security.sock", socket_timeout=0.1
    )
    with pytest.raises(GenerationUnavailable, match="temporarily unavailable"):
        acquire(limiter(client))
    client.close()


def test_ipv6_privacy_addresses_share_network_budget():
    assert ip_bucket("2001:db8:1234:5678::1") == ip_bucket("2001:db8:1234:5678::ffff")
    assert ip_bucket("::ffff:192.0.2.1") == ip_bucket("192.0.2.1")
    assert ip_bucket("bad-address") == "unknown"


def test_only_explicit_proxy_chain_can_supply_ip():
    assert generation.client_ip(request(forwarded="198.51.100.2"), []) == "192.0.2.1"
    assert (
        generation.client_ip(
            request("127.0.0.1", "198.51.100.1, 192.0.2.2"), ["127.0.0.1/32"]
        )
        == "192.0.2.2"
    )
    assert (
        generation.client_ip(
            request("127.0.0.1", "198.51.100.1, 192.0.2.2"),
            ["127.0.0.1/32", "192.0.2.0/24"],
        )
        == "198.51.100.1"
    )
    assert (
        generation.client_ip(request("127.0.0.1", "invalid"), ["127.0.0.1/32"])
        == "127.0.0.1"
    )


def test_api_keys_share_owner_budget(redis_server, monkeypatch):
    from fastapi import HTTPException

    service = limiter(redis_server, user_per_day=1)
    monkeypatch.setattr(generation, "get_generation_limiter", lambda: service)
    first = request()
    first.state.api_key_id = 1
    scope = generation.GenerationScope(first, SimpleNamespace(id="owner"))
    scope.admit()
    scope.close()
    second = request(peer="198.51.100.2")
    second.state.api_key_id = 2
    with pytest.raises(HTTPException) as error:
        generation.GenerationScope(second, SimpleNamespace(id="owner")).admit()
    assert error.value.status_code == 429


def test_denied_chat_does_not_charge_and_denied_quota_does_not_create_chat(
    redis_server, monkeypatch
):
    from unittest.mock import Mock
    from fastapi import HTTPException
    from app.api.routes import chat as routes
    from app.rag.chat.config import ChatEngineConfig

    service = limiter(redis_server, global_per_day=1)
    monkeypatch.setattr(generation, "get_generation_limiter", lambda: service)
    monkeypatch.setattr(routes.chat_repo, "get", lambda *args: None)
    create = Mock()
    monkeypatch.setattr(routes.chat_repo, "create", create)
    with pytest.raises(HTTPException) as denied:
        routes.chats(
            request(),
            Mock(),
            None,
            routes.ChatRequest(
                messages=[{"role": "user", "content": "hello"}], chat_id=uuid4()
            ),
        )
    assert denied.value.status_code == 404
    assert redis_server.hget(service.prefix + "global:day", "count") is None
    acquire(service).close()
    config = Mock()
    config.get_db_chat_engine.return_value.id = 1
    monkeypatch.setattr(
        ChatEngineConfig, "load_from_db", lambda *args, **kwargs: config
    )
    with pytest.raises(HTTPException) as limited:
        routes.chats(
            request(),
            Mock(),
            None,
            routes.ChatRequest(messages=[{"role": "user", "content": "hello"}]),
        )
    assert limited.value.status_code == 429
    create.assert_not_called()
    config.get_llama_llm.assert_not_called()


def test_generator_context_survives_thread_changes_and_closes_on_disconnect(
    redis_server, monkeypatch
):
    import anyio
    from contextvars import ContextVar
    from starlette.requests import ClientDisconnect

    scope = admitted_scope(redis_server, monkeypatch)
    local = ContextVar("test-generation-context", default="outside")
    finished = []

    def chunks():
        token = local.set("inside")
        try:
            assert generation.current_generation.get() is scope
            yield "first"
            assert local.get() == "inside"
            assert generation.current_generation.get() is scope
            yield "second"
        finally:
            local.reset(token)
            finished.append(True)

    iterator = generation.GenerationIterator(chunks(), scope)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(next, iterator).result() == "first"
    assert local.get() == "outside"

    async def exercise():
        response = generation.GenerationStreamingResponse(iterator)

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("client disconnected")

        async def receive():
            await anyio.sleep_forever()

        with pytest.raises(ClientDisconnect):
            await response(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
            )

    anyio.run(exercise)
    assert finished == [True]
    assert redis_server.zcard(scope.lease.keys[6]) == 0
    assert generation.current_generation.get() is None


def test_disconnect_keeps_lease_until_running_model_work_returns(
    redis_server, monkeypatch
):
    import anyio
    from threading import Event, Timer

    scope = admitted_scope(redis_server, monkeypatch)
    started, finish = Event(), Event()

    def chunks():
        started.set()
        finish.wait(3)
        yield "done"

    async def exercise():
        response = generation.GenerationStreamingResponse(
            generation.GenerationIterator(chunks(), scope)
        )

        async def send(message):
            pass

        async def receive():
            await anyio.to_thread.run_sync(started.wait)
            return {"type": "http.disconnect"}

        async def verify_held():
            await anyio.to_thread.run_sync(started.wait)
            await anyio.sleep(0.05)
            try:
                assert redis_server.zcard(scope.lease.keys[6]) == 1
                assert not scope.lease.closed
            finally:
                finish.set()

        async with anyio.create_task_group() as group:
            group.start_soon(verify_held)
            await response(
                {"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send
            )

    fallback = Timer(3, finish.set)
    fallback.start()
    try:
        anyio.run(exercise)
    finally:
        fallback.cancel()
    assert scope.lease.closed
    assert redis_server.zcard(scope.lease.keys[6]) == 0


def test_real_model_transports_bound_response_size_and_deadline(
    redis_server, monkeypatch
):
    import asyncio
    import httpx
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from app.security.provider_http import ProviderSession, openai_http_clients
    from app.security.generation_limits import GenerationExpired
    from app.security.generation_http import MAX_RESPONSE_BYTES

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            if self.path == "/compressed":
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            try:
                if self.path == "/slow":
                    self.wfile.write(b"first")
                    self.wfile.flush()
                    time.sleep(0.15)
                    self.wfile.write(b"last")
                elif self.path == "/large":
                    self.wfile.write(b"x" * (MAX_RESPONSE_BYTES + 1))
                else:
                    self.wfile.write(b"ok")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    scope = admitted_scope(redis_server, monkeypatch)
    clients = openai_http_clients()
    try:
        with (
            scope.activate(),
            clients["http_client"] as client,
            ProviderSession(origin) as session,
        ):
            assert client.get(origin).text == "ok"
            assert session.get(origin).text == "ok"
            with pytest.raises(GenerationExpired, match="response budget"):
                client.get(origin + "/large")
            with pytest.raises(GenerationExpired, match="response budget"):
                session.get(origin + "/large")
            with pytest.raises(GenerationExpired, match="Compressed"):
                client.get(origin + "/compressed")

            async def asynchronous():
                async with clients["async_http_client"] as asynchronous_client:
                    assert (await asynchronous_client.get(origin)).text == "ok"
                    with pytest.raises(GenerationExpired, match="response budget"):
                        await asynchronous_client.get(origin + "/large")

            asyncio.run(asynchronous())
            scope.lease.deadline = time.monotonic() + 0.04
            with pytest.raises((httpx.ReadTimeout, GenerationExpired)):
                client.get(origin + "/slow")
            # Leave activate() while the scope is valid; the next assertion
            # independently proves expired requests never reach the transport.
            scope.lease.deadline = time.monotonic() + 1
        assert scope.provider_calls == 8
    finally:
        scope.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_recommendation_refresh_preserves_cache_on_failure_and_replaces_atomically(
    monkeypatch,
):
    from unittest.mock import Mock
    from sqlmodel import Session, create_engine, select
    from app.models.recommend_question import RecommendQuestion
    from app.rag.chat import chat_service

    database = create_engine("sqlite://")
    RecommendQuestion.__table__.create(database)
    model = Mock()
    config = SimpleNamespace(
        llm=SimpleNamespace(further_questions_prompt="{{ chat_message_content }}"),
        get_llama_llm=lambda session: model,
    )
    monkeypatch.setattr(
        chat_service.ChatEngineConfig, "load_from_db", lambda *args, **kwargs: config
    )
    message = SimpleNamespace(id=1, content="answer")
    admitted = Mock()
    with Session(database) as session:
        session.add(RecommendQuestion(chat_message_id=1, questions=["old question"]))
        session.commit()
        model.predict.side_effect = ValueError("model failure")
        with pytest.raises(ValueError):
            chat_service.get_chat_message_recommend_questions(
                session,
                message,
                allow_generate=True,
                force_refresh=True,
                on_authorized=admitted,
            )
        assert session.exec(select(RecommendQuestion.questions)).all() == [
            ["old question"]
        ]
        model.predict.side_effect = [
            "**incorrect format**",
            "First question?\nSecond question?",
        ]
        assert chat_service.get_chat_message_recommend_questions(
            session,
            message,
            allow_generate=True,
            force_refresh=True,
            on_authorized=admitted,
        ) == ["First question?", "Second question?"]
        assert session.exec(select(RecommendQuestion.questions)).all() == [
            ["First question?", "Second question?"]
        ]
        admitted.reset_mock()
        assert chat_service.get_chat_message_recommend_questions(
            session, message, allow_generate=True, on_authorized=admitted
        )
        admitted.assert_not_called()
    database.dispose()


def test_generation_body_limit_applies_before_json_parsing():
    import anyio

    reached, sent = [], []

    async def application(scope, receive, send):
        reached.append((await receive())["body"])

    async def exercise(parts):
        messages = iter(
            [
                {
                    "type": "http.request",
                    "body": part,
                    "more_body": index < len(parts) - 1,
                }
                for index, part in enumerate(parts)
            ]
        )

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        await generation.GenerationBodyLimitMiddleware(application, max_bytes=16)(
            {"type": "http", "method": "POST", "path": "/api/v1/chats"}, receive, send
        )

    anyio.run(exercise, [b"12345678", b"123456789"])
    assert not reached and sent[0]["status"] == 413
    anyio.run(exercise, [b"1234", b"5678"])
    assert reached == [b"12345678"]


def test_deadline_and_model_request_budget_stop_additional_calls(
    redis_server, monkeypatch
):
    from app.security.generation_limits import GenerationExpired
    from app.security.generation_http import prepare
    import httpx

    scope = admitted_scope(redis_server, monkeypatch, provider_calls_per_request=1)
    with scope.activate():
        outbound = httpx.Request("POST", "https://model.example/chat")
        prepare(outbound)
        assert all(0 < value <= 60 for value in outbound.extensions["timeout"].values())
        with pytest.raises(GenerationExpired, match="call budget"):
            prepare(outbound)
    scope.lease.deadline = time.monotonic() - 1
    iterator = generation.GenerationIterator(iter(["must not start"]), scope)
    assert "timed out" in next(iterator).payload
    with pytest.raises(StopIteration):
        next(iterator)
    assert scope.lease.closed
