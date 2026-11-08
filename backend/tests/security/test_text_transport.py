"""Legacy text completions keep their wire contract and generation boundaries."""

import asyncio
import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest
import dspy
from dspy.utils.exceptions import LMUnexpectedError, LMAuthError, LMProviderError

from app.core.config import settings
from app.rag.llms.dspy import get_dspy_lm_by_llama_llm
from app.rag.llms.provider import LLMProvider
from app.rag.llms.resolver import resolve_llm
from app.security.generation import current_generation, GenerationScope
from app.security.generation_limits import GenerationExpired


@pytest.fixture
def provider(monkeypatch):
    state = SimpleNamespace(calls=[], mode="normal")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.calls.append((self.path, dict(self.headers), data))
            content = (
                "x" * (4 * 1024 * 1024 + 1)
                if state.mode == "oversized"
                else "fixture answer"
            )
            if state.mode == "structured":
                content = "[[ ## answer ## ]]\nfixture answer\n\n[[ ## completed ## ]]"
            choice = {"index": 0, "text": content, "finish_reason": "stop"}
            if data.get("logprobs"):
                choice["logprobs"] = {
                    "tokens": ["fixture"],
                    "token_logprobs": [-0.5],
                    "top_logprobs": [{"fixture": -0.5}],
                    "text_offset": [0],
                }
            body = json.dumps(
                {
                    "id": "fixture",
                    "object": "text_completion",
                    "created": 0,
                    "model": "fixture-text-model",
                    "choices": [choice],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "total_tokens": 3,
                    },
                }
            ).encode()
            status = {"redirect": 302, "error": 400, "auth": 401}.get(state.mode, 200)
            if state.mode in {"error", "auth"}:
                body = json.dumps(
                    {
                        "error": {
                            "message": "forbidden-upstream-secret",
                            "type": "fixture",
                        }
                    }
                ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if state.mode == "compressed":
                self.send_header("Content-Encoding", "gzip")
            if state.mode == "redirect":
                self.send_header("Location", state.origin + "/must-not-follow")
            self.end_headers()
            try:
                if state.mode == "slow":
                    for char in body:
                        self.wfile.write(bytes([char]))
                        self.wfile.flush()
                        time.sleep(0.01)
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    state.origin = f"http://127.0.0.1:{server.server_port}"
    original = socket.socket.connect

    def owned_only(sock, address):
        assert address == ("127.0.0.1", server.server_port)
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", owned_only)
    monkeypatch.setattr(settings, "PROVIDER_ALLOWED_ORIGINS", [state.origin])
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(key, "http://127.0.0.1:1")
        monkeypatch.setenv(key.lower(), "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    for key in ("OPENAI_API_BASE", "OPENAI_BASE_URL"):
        monkeypatch.setenv(key, "http://127.0.0.1:1/forbidden")
    monkeypatch.setenv("OPENAI_API_KEY", "forbidden-ambient-key")
    monkeypatch.setenv("OPENAI_ORG_ID", "forbidden-organization")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "forbidden-project")
    llama = resolve_llm(
        LLMProvider.OPENAI_LIKE,
        "fixture-text-model",
        {
            "api_base": state.origin + "/gateway/v1",
            "is_chat_model": False,
            "max_tokens": 128,
            "timeout": 2,
        },
        "explicit-fixture-key",
    )
    state.llama = llama
    state.lm = get_dspy_lm_by_llama_llm(llama)
    try:
        yield state
    finally:
        state.lm.close()
        asyncio.run(state.lm.aclose())
        llama._http_client.close()
        asyncio.run(llama._async_http_client.aclose())
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def invoke(lm, asynchronous, **kwargs):
    kwargs.setdefault("prompt", "fixture")
    return asyncio.run(lm.acall(**kwargs)) if asynchronous else lm(**kwargs)


def scope(seconds, attempts=16):
    deadline = time.monotonic() + seconds

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise GenerationExpired("Generation deadline exceeded")
        return value

    result = GenerationScope(None, None)
    result.lease = SimpleNamespace(
        remaining=remaining,
        limiter=SimpleNamespace(
            limits=SimpleNamespace(provider_calls_per_request=attempts)
        ),
    )
    return result


@pytest.mark.parametrize("asynchronous", [False, True])
def test_text_completion_preserves_prompt_history_options_and_logprobs(
    provider, asynchronous
):
    budget = scope(10)
    token = current_generation.set(budget)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "previous"},
        {"role": "user", "content": "followup"},
    ]
    try:
        result = invoke(
            provider.lm,
            asynchronous,
            messages=messages,
            logprobs=1,
            suffix="ending",
            echo=True,
            stop=["END"],
            top_p=0.6,
            temperature=0.3,
            seed=7,
        )
    finally:
        current_generation.reset(token)
    assert result[0]["text"] == "fixture answer"
    assert result[0]["logprobs"]["token_logprobs"] == [-0.5]
    assert result[0]["logprobs"]["text_offset"] == [0]
    path, headers, body = provider.calls[0]
    assert path == "/gateway/v1/completions"
    assert headers["Authorization"] == "Bearer explicit-fixture-key"
    assert "OpenAI-Organization" not in headers
    assert "OpenAI-Project" not in headers
    assert (
        body["prompt"]
        == "system\n\nquestion\n\nprevious\n\nfollowup\n\nBEGIN RESPONSE:"
    )
    assert body["model"] == "fixture-text-model"
    for name, value in {
        "logprobs": 1,
        "suffix": "ending",
        "echo": True,
        "stop": ["END"],
        "top_p": 0.6,
        "temperature": 0.3,
        "seed": 7,
        "max_tokens": 128,
    }.items():
        assert body[name] == value
    assert budget.provider_calls == 1
    assert provider.lm.history[-1]["usage"]["completion_tokens"] == 2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_text_output_caps_survive_runtime_overrides_and_lm_copy(provider, asynchronous):
    assert invoke(provider.lm, asynchronous, max_tokens=8192) == ["fixture answer"]
    assert invoke(provider.lm, asynchronous, max_tokens=64) == ["fixture answer"]
    copied = provider.lm.copy(max_tokens=8192)
    try:
        assert invoke(copied, asynchronous) == ["fixture answer"]
    finally:
        copied.close()
        asyncio.run(copied.aclose())
    assert [body["max_tokens"] for _, _, body in provider.calls] == [128, 64, 128]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["oversized", "compressed", "redirect"])
def test_text_responses_are_bounded(provider, asynchronous, mode):
    provider.mode = mode
    with pytest.raises(
        LMProviderError if mode == "redirect" else LMUnexpectedError
    ) as failure:
        invoke(provider.lm, asynchronous)
    if mode != "redirect":
        assert isinstance(failure.value.__cause__, GenerationExpired)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_text_request_budget_checked_before_network(provider, asynchronous):
    with pytest.raises(LMUnexpectedError) as failure:
        invoke(provider.lm, asynchronous, prompt="x" * (2 * 1024 * 1024 + 1))
    assert isinstance(failure.value.__cause__, GenerationExpired)
    assert not provider.calls


@pytest.mark.parametrize("asynchronous", [False, True])
def test_text_generation_deadline_and_attempts(provider, asynchronous):
    token = current_generation.set(scope(-1))
    try:
        with pytest.raises(LMUnexpectedError):
            invoke(provider.lm, asynchronous)
        assert not provider.calls
    finally:
        current_generation.reset(token)
    token = current_generation.set(scope(10, attempts=1))
    try:
        assert invoke(provider.lm, asynchronous) == ["fixture answer"]
        with pytest.raises(LMUnexpectedError):
            invoke(provider.lm, asynchronous)
        assert len(provider.calls) == 1
    finally:
        current_generation.reset(token)
    provider.mode = "slow"
    token = current_generation.set(scope(0.15))
    started = time.monotonic()
    try:
        with pytest.raises(LMUnexpectedError) as failure:
            invoke(provider.lm, asynchronous)
        assert isinstance(failure.value.__cause__, GenerationExpired)
        assert time.monotonic() - started < 1
        assert len(provider.calls) == 2
    finally:
        current_generation.reset(token)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_text_provider_error_details_do_not_escape(provider, asynchronous):
    provider.mode = "auth"
    with pytest.raises(LMAuthError) as failure:
        invoke(provider.lm, asynchronous)
    assert "forbidden-upstream-secret" not in str(failure.value)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_text_chat_adapter_still_returns_structured_prediction(provider, asynchronous):
    provider.mode = "structured"
    predictor = dspy.Predict("question -> answer")
    with dspy.context(lm=provider.lm, adapter=dspy.ChatAdapter()):
        result = (
            asyncio.run(predictor.acall(question="fixture question"))
            if asynchronous
            else predictor(question="fixture question")
        )
    assert result.answer == "fixture answer"
    assert len(provider.calls) == 1
    prompt = provider.calls[0][2]["prompt"]
    assert "fixture question" in prompt
    assert prompt.endswith("BEGIN RESPONSE:")
