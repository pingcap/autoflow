"""Real signed HTTP and AWS binary events, confined to owned loopback sockets."""

import asyncio
import base64
import json
import socket
import struct
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from llama_index.core.base.llms.types import (
    ChatMessage,
    ThinkingBlock,
    ToolCallBlock,
    ImageBlock,
    DocumentBlock,
)

from app.rag.llms.provider import LLMProvider
from app.rag.llms.resolver import resolve_llm
from app.security import bedrock_http
from app.security.generation import GenerationScope, current_generation
from app.security.generation_limits import GenerationExpired
from app.security.provider_config import ProviderConfigurationError

MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


def frame(kind, data, message_type="event"):
    headers = b""
    for key, value in {
        ":message-type": message_type,
        ":event-type": kind,
        ":content-type": "application/json",
    }.items():
        name, value = key.encode(), value.encode()
        headers += (
            bytes([len(name)]) + name + b"\x07" + struct.pack("!H", len(value)) + value
        )
    payload = json.dumps(data).encode()
    prelude = struct.pack("!II", 16 + len(headers) + len(payload), len(headers))
    body = prelude + struct.pack("!I", zlib.crc32(prelude)) + headers + payload
    return body + struct.pack("!I", zlib.crc32(body))


@pytest.fixture
def provider(monkeypatch, tmp_path):
    state = SimpleNamespace(
        calls=[], mode="normal", disconnected=Event(), started=Event()
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            state.calls.append((self.path, dict(self.headers), json.loads(raw), raw))
            streaming = self.path.endswith("/converse-stream")
            content = (
                "x" * (4 * 1024 * 1024 + 1)
                if state.mode == "oversized"
                else "fixture answer"
            )
            if state.mode == "adapter":
                content = "[[ ## answer ## ]]\nfixture answer\n\n[[ ## completed ## ]]"
            usage = {
                "inputTokens": 1,
                "outputTokens": 2,
                "totalTokens": 3,
                "cacheReadInputTokens": 4,
                "cacheWriteInputTokens": 5,
            }
            blocks = [{"text": content}]
            if state.mode == "schema_tool":
                blocks = [
                    {
                        "toolUse": {
                            "toolUseId": "fixture-json",
                            "name": "json_tool_call",
                            "input": {"answer": "fixture answer"},
                        }
                    }
                ]
            if state.mode == "tools":
                blocks = [
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": "reason", "signature": "sig"}
                        }
                    },
                    {
                        "toolUse": {
                            "toolUseId": "fixture-tool",
                            "name": "lookup",
                            "input": {"q": "fixture"},
                        }
                    },
                ]
            body = json.dumps(
                {
                    "output": {"message": {"role": "assistant", "content": blocks}},
                    "stopReason": "tool_use"
                    if state.mode in {"tools", "schema_tool"}
                    else "end_turn",
                    "usage": usage,
                    "metrics": {"latencyMs": 1},
                }
            ).encode()
            if streaming:
                chunks = [frame("messageStart", {"role": "assistant"})]
                if state.mode == "tools":
                    chunks += [
                        frame(
                            "contentBlockDelta",
                            {
                                "delta": {"reasoningContent": {"text": "reason"}},
                                "contentBlockIndex": 0,
                            },
                        ),
                        frame(
                            "contentBlockDelta",
                            {
                                "delta": {"reasoningContent": {"signature": "sig"}},
                                "contentBlockIndex": 0,
                            },
                        ),
                        frame(
                            "contentBlockStart",
                            {
                                "start": {
                                    "toolUse": {
                                        "toolUseId": "fixture-tool",
                                        "name": "lookup",
                                    }
                                },
                                "contentBlockIndex": 1,
                            },
                        ),
                        frame(
                            "contentBlockDelta",
                            {
                                "delta": {"toolUse": {"input": '{"q":'}},
                                "contentBlockIndex": 1,
                            },
                        ),
                        frame(
                            "contentBlockDelta",
                            {
                                "delta": {"toolUse": {"input": '"fixture"}'}},
                                "contentBlockIndex": 1,
                            },
                        ),
                    ]
                else:
                    pieces = (
                        [
                            "[[ ## answer ## ]]\n",
                            "fixture ",
                            "answer",
                            "\n\n[[ ## completed ## ]]",
                        ]
                        if state.mode == "adapter"
                        else [content]
                    )
                    chunks += [
                        frame(
                            "contentBlockDelta",
                            {"delta": {"text": piece}, "contentBlockIndex": 0},
                        )
                        for piece in pieces
                    ]
                chunks += [
                    frame(
                        "contentBlockStop",
                        {"contentBlockIndex": 1 if state.mode == "tools" else 0},
                    ),
                    frame(
                        "messageStop",
                        {
                            "stopReason": "tool_use"
                            if state.mode == "tools"
                            else "end_turn"
                        },
                    ),
                    frame("metadata", {"usage": usage, "metrics": {"latencyMs": 1}}),
                ]
                body = b"".join(chunks)
                if state.mode == "truncated":
                    body = body[:-4]
                if state.mode == "bad_crc":
                    body = body[:-1] + bytes([body[-1] ^ 1])
                if state.mode == "exception":
                    body = frame(
                        "modelStreamErrorException",
                        {"message": "fixture-secret-echo"},
                        "exception",
                    )
            status = {"redirect": 302, "auth": 403}.get(state.mode, 200)
            if state.mode == "auth":
                body = b'{"message":"fixture-secret-echo"}'
            self.send_response(status)
            self.send_header(
                "Content-Type",
                "application/vnd.amazon.eventstream"
                if streaming
                else "application/json",
            )
            self.send_header("Content-Length", str(len(body)))
            if state.mode == "compressed":
                self.send_header("Content-Encoding", "gzip")
            if state.mode == "redirect":
                self.send_header("Location", state.origin + "/unexpected")
            self.end_headers()
            state.started.set()
            try:
                if state.mode == "slow":
                    for char in body:
                        self.wfile.write(bytes([char]))
                        self.wfile.flush()
                        time.sleep(0.01)
                elif state.mode in {"held_stream", "held_body"}:
                    if state.mode == "held_stream":
                        self.wfile.write(b"".join(chunks[:2]))
                        self.wfile.flush()
                    self.connection.settimeout(3)
                    if self.connection.recv(1) == b"":
                        state.disconnected.set()
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                state.disconnected.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.origin = f"http://127.0.0.1:{server.server_port}"
    connect, dns = socket.socket.connect, socket.getaddrinfo

    def owned_connect(sock, address):
        assert address == ("127.0.0.1", server.server_port), (
            "Unexpected network destination"
        )
        return connect(sock, address)

    def owned_dns(host, port, *args, **kwargs):
        assert host == "127.0.0.1" and port == server.server_port, (
            "Unexpected DNS destination"
        )
        return dns(host, port, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", owned_connect)
    monkeypatch.setattr(socket, "getaddrinfo", owned_dns)
    monkeypatch.setattr(bedrock_http, "bedrock_endpoint", lambda region: state.origin)
    for name, value in {
        "AWS_ENDPOINT_URL": "http://127.0.0.1:1",
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": "http://127.0.0.1:1",
        "AWS_BEDROCK_RUNTIME_ENDPOINT": "http://127.0.0.1:1",
        "AWS_ACCESS_KEY_ID": "ambient-id",
        "AWS_SECRET_ACCESS_KEY": "ambient-secret",
        "AWS_SESSION_TOKEN": "ambient-token",
        "AWS_PROFILE": "nonexistent",
        "AWS_DEFAULT_PROFILE": "nonexistent",
        "AWS_DEFAULT_REGION": "eu-west-1",
        "AWS_CONFIG_FILE": str(tmp_path / "config"),
        "AWS_SHARED_CREDENTIALS_FILE": str(tmp_path / "credentials"),
        "AWS_DATA_PATH": str(tmp_path),
        "AWS_CA_BUNDLE": str(tmp_path / "missing-ca"),
        "HTTP_PROXY": "http://127.0.0.1:1",
        "HTTPS_PROXY": "http://127.0.0.1:1",
        "ALL_PROXY": "http://127.0.0.1:1",
    }.items():
        monkeypatch.setenv(name, value)
    (tmp_path / "config").write_text("invalid AWS profile contents")
    (tmp_path / "credentials").write_text("invalid AWS credential contents")
    state.llm = resolve_llm(
        LLMProvider.BEDROCK,
        MODEL,
        {"timeout": 2, "max_tokens": 128},
        {
            "aws_access_key_id": "explicit-fixture-id",
            "aws_secret_access_key": "explicit-fixture-secret",
            "aws_region_name": "us-west-2",
        },
    )
    try:
        yield state
    finally:
        state.llm._client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def scope(seconds=10, attempts=1):
    started = time.monotonic()
    result = GenerationScope(None, None)
    result.lease = SimpleNamespace(
        remaining=lambda: seconds - (time.monotonic() - started),
        limiter=SimpleNamespace(
            limits=SimpleNamespace(provider_calls_per_request=attempts)
        ),
    )
    return result


def invoke(llm, asynchronous, streaming, **kwargs):
    messages = [
        ChatMessage(role="system", content="system fixture"),
        ChatMessage(role="user", content="question"),
    ]
    if asynchronous:

        async def run():
            if not streaming:
                return await llm.achat(messages, **kwargs)
            responses = await llm.astream_chat(messages, **kwargs)
            try:
                return [r async for r in responses][-1]
            finally:
                await responses.aclose()

        return asyncio.run(run())
    if not streaming:
        return llm.chat(messages, **kwargs)
    responses = llm.stream_chat(messages, **kwargs)
    try:
        return list(responses)[-1]
    finally:
        responses.close()


@pytest.mark.parametrize(
    "region,hostname",
    [
        ("us-west-2", "bedrock-runtime.us-west-2.amazonaws.com"),
        ("cn-north-1", "bedrock-runtime.cn-north-1.amazonaws.com.cn"),
        ("us-gov-west-1", "bedrock-runtime.us-gov-west-1.amazonaws.com"),
    ],
)
def test_official_partition_endpoints(region, hostname):
    assert bedrock_http.bedrock_endpoint(region) == "https://" + hostname


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_normal_signing_protocol_identity_and_budget(provider, asynchronous, streaming):
    budget = scope()
    token = current_generation.set(budget)
    try:
        response = invoke(provider.llm, asynchronous, streaming)
    finally:
        current_generation.reset(token)
    assert response.message.content == "fixture answer"
    assert response.additional_kwargs["total_tokens"] == 3
    assert response.additional_kwargs["cache_read_input_tokens"] == 4
    assert budget.provider_calls == 1
    path, headers, body, raw = provider.calls[0]
    assert path == "/model/anthropic.claude-3-haiku-20240307-v1%3A0/" + (
        "converse-stream" if streaming else "converse"
    )
    assert body["system"] == [{"text": "system fixture"}]
    assert body["inferenceConfig"]["maxTokens"] == 128
    assert "explicit-fixture-id/" in headers["Authorization"]
    assert "/us-west-2/bedrock/" in headers["Authorization"]
    assert "X-Amz-Security-Token" not in headers
    signed_headers = (
        headers["Authorization"].split("SignedHeaders=", 1)[1].split(",")[0].split(";")
    )
    normalized = {key.lower(): value for key, value in headers.items()}
    request = AWSRequest(
        method="POST",
        url=provider.origin + path,
        data=raw,
        headers={key: normalized[key] for key in signed_headers},
    )
    request.context["timestamp"] = headers["X-Amz-Date"]
    signer = SigV4Auth(
        Credentials("explicit-fixture-id", "explicit-fixture-secret"),
        "bedrock",
        "us-west-2",
    )
    signature = signer.signature(
        signer.string_to_sign(request, signer.canonical_request(request)), request
    )
    assert headers["Authorization"].endswith("Signature=" + signature)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mode", ["oversized", "compressed", "redirect", "auth"])
def test_unsafe_responses_rejected(provider, asynchronous, streaming, mode):
    provider.mode = mode
    with pytest.raises(Exception) as caught:
        invoke(provider.llm, asynchronous, streaming)
    assert type(caught.value).__name__ in {
        "GenerationExpired",
        "ProviderConfigurationError",
        "AccessDeniedException",
    }
    assert "fixture-secret" not in str(caught.value)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_runtime_caps_and_tools(provider, asynchronous, streaming):
    provider.mode = "tools"
    for cap in [8192, 64]:
        response = invoke(provider.llm, asynchronous, streaming, max_tokens=cap)
        tools = [b for b in response.message.blocks if isinstance(b, ToolCallBlock)]
        thoughts = [b for b in response.message.blocks if isinstance(b, ThinkingBlock)]
        assert tools[0].tool_kwargs == {"q": "fixture"}
        assert tools[0].tool_name == "lookup"
        assert thoughts[0].content == "reason"
        assert thoughts[0].additional_information["signature"] == "sig"
    assert [c[2]["inferenceConfig"]["maxTokens"] for c in provider.calls] == [128, 64]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mode", ["expired", "exhausted", "slow"])
def test_generation_deadlines_and_attempts(provider, asynchronous, streaming, mode):
    provider.mode = "slow" if mode == "slow" else "normal"
    budget = scope(
        seconds=-1 if mode == "expired" else 0.15 if mode == "slow" else 10,
        attempts=0 if mode == "exhausted" else 1,
    )
    token = current_generation.set(budget)
    try:
        with pytest.raises(GenerationExpired):
            invoke(provider.llm, asynchronous, streaming)
    finally:
        current_generation.reset(token)
    assert len(provider.calls) == (1 if mode == "slow" else 0)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["truncated", "bad_crc", "exception"])
def test_invalid_stream_is_not_success(provider, asynchronous, mode):
    provider.mode = mode
    with pytest.raises(ProviderConfigurationError) as caught:
        invoke(provider.llm, asynchronous, True)
    assert "fixture-secret" not in str(caught.value)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_stream_close_releases_connection(provider, asynchronous):
    provider.mode = "held_stream"
    if asynchronous:

        async def run():
            stream = await provider.llm.astream_complete("fixture")
            assert (await anext(stream)).text == "fixture answer"
            await stream.aclose()
            assert await asyncio.to_thread(provider.disconnected.wait, 1)

        asyncio.run(run())
    else:
        stream = provider.llm.stream_complete("fixture")
        assert next(stream).text == "fixture answer"
        stream.close()
        assert provider.disconnected.wait(1)


@pytest.mark.parametrize("streaming", [False, True])
def test_async_cancel_releases_pending_connection(provider, streaming):
    provider.mode = "held_body"

    async def run():
        async def consume():
            if not streaming:
                return await provider.llm.acomplete("fixture")
            responses = await provider.llm.astream_complete("fixture")
            try:
                return await anext(responses)
            finally:
                await responses.aclose()

        task = asyncio.create_task(consume())
        assert await asyncio.to_thread(provider.started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await asyncio.to_thread(provider.disconnected.wait, 1)

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_inline_media_and_tool_history_preserve_wire_shape(provider, asynchronous):
    image = b"fixture-inline-image"
    document = b"fixture-inline-document"
    messages = [
        ChatMessage(
            role="user",
            blocks=[
                ImageBlock(image=image, image_mimetype="image/png"),
                DocumentBlock(
                    data=document,
                    document_mimetype="application/pdf",
                    title="fixture document",
                ),
            ],
        ),
        ChatMessage(
            role="assistant",
            blocks=[
                ToolCallBlock(
                    tool_name="lookup",
                    tool_call_id="fixture-id",
                    tool_kwargs={"q": "fixture"},
                )
            ],
        ),
        ChatMessage(
            role="tool",
            content="fixture result",
            additional_kwargs={"tool_call_id": "fixture-id"},
        ),
        ChatMessage(role="user", content="follow up"),
    ]
    tools = {
        "tools": [
            {
                "toolSpec": {
                    "name": "lookup",
                    "description": "fixture",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        }
                    },
                }
            }
        ],
        "toolChoice": {"auto": {}},
    }
    if asynchronous:
        asyncio.run(provider.llm.achat(messages, tools=tools))
    else:
        provider.llm.chat(messages, tools=tools)
    body = provider.calls[0][2]
    assert (
        body["messages"][0]["content"][0]["image"]["source"]["bytes"]
        == base64.b64encode(image).decode()
    )
    assert (
        body["messages"][0]["content"][1]["document"]["source"]["bytes"]
        == base64.b64encode(document).decode()
    )
    assert body["messages"][1]["content"][0]["toolUse"]["input"] == {"q": "fixture"}
    assert body["messages"][2]["content"][0]["toolResult"]["toolUseId"] == "fixture-id"
    assert body["toolConfig"] == tools


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("location", ["url", "path"])
@pytest.mark.parametrize("block_type", [ImageBlock, DocumentBlock])
def test_media_cannot_fetch_remote_or_local_resources(
    provider, asynchronous, location, block_type, tmp_path
):
    value = (
        "http://127.0.0.1:1/private"
        if location == "url"
        else tmp_path / "owned-fixture"
    )
    if location == "path":
        value.write_bytes(b"owned fixture must not be read by model conversion")
    blocks = [block_type(**{location: value})]
    messages = [ChatMessage(role="user", blocks=blocks)]
    with pytest.raises(ProviderConfigurationError, match="inline"):
        if asynchronous:
            asyncio.run(provider.llm.achat(messages))
        else:
            provider.llm.chat(messages)
    assert provider.calls == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_oversized_request_rejected_before_network(provider, asynchronous, streaming):
    provider.llm.additional_kwargs = {"system": [{"text": "x" * (2 * 1024 * 1024 + 1)}]}
    with pytest.raises(GenerationExpired, match="request body"):
        invoke(provider.llm, asynchronous, streaming)
    assert provider.calls == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_model_copy_cannot_raise_cap_or_change_destination(provider, asynchronous):
    copied = provider.llm.model_copy(
        update={"max_tokens": 8192, "endpoint_url": "http://127.0.0.1:1"}
    )
    invoke(copied, asynchronous, False)
    assert provider.calls[0][2]["inferenceConfig"]["maxTokens"] == 128
    with pytest.raises(ProviderConfigurationError, match="model"):
        invoke(copied, asynchronous, False, model="different-model")
    assert len(provider.calls) == 1
