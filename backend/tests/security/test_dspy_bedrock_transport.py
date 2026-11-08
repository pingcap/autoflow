"""DSPy Converse contract and HTTP bounds against the owned AWS event fixture."""

import asyncio
import base64

import dspy
import pytest
from dspy.lm15 import Message, Request, TextDelta, ToolCallPart
from dspy.utils.exceptions import LMUnexpectedError, LMAuthError

from app.rag.llms.dspy import get_dspy_lm_by_llama_llm
from app.security.generation import current_generation
from app.security.generation_limits import GenerationExpired
from app.security.provider_config import ProviderConfigurationError
from tests.security.test_bedrock_transport import (
    MODEL,
    scope,
    provider as http_provider,  # noqa: F401 - imported pytest fixture
)


@pytest.fixture
def provider(http_provider):
    http_provider.lm = get_dspy_lm_by_llama_llm(http_provider.llm)
    try:
        yield http_provider
    finally:
        http_provider.lm.close()
        asyncio.run(http_provider.lm.aclose())


def invoke(lm, asynchronous, **kwargs):
    return (
        asyncio.run(lm.acall(prompt="fixture", **kwargs))
        if asynchronous
        else lm(prompt="fixture", **kwargs)
    )


def native_request():
    return Request(
        model=f"bedrock/converse/{MODEL}", messages=[Message.user("fixture")]
    )


def stream(lm, asynchronous, early=False):
    if asynchronous:

        async def run():
            responses = lm._async_engine_spec.stream(native_request())
            result = []
            try:
                async for event in responses:
                    result.append(event)
                    if early and event.type == "delta":
                        break
            finally:
                await responses.aclose()
            return result

        return asyncio.run(run())
    responses = lm.engine.stream(native_request())
    result = []
    try:
        for event in responses:
            result.append(event)
            if early and event.type == "delta":
                break
    finally:
        responses.close()
    return result


@pytest.mark.parametrize("asynchronous", [False, True])
def test_normal_identity_budget_output_caps_and_copies(provider, asynchronous):
    budget = scope(attempts=3)
    token = current_generation.set(budget)
    copied = provider.lm.copy(max_tokens=8192)
    try:
        assert invoke(provider.lm, asynchronous) == ["fixture answer"]
        assert invoke(copied, asynchronous) == ["fixture answer"]
        assert invoke(copied, asynchronous, max_tokens=64) == ["fixture answer"]
    finally:
        current_generation.reset(token)
        copied.close()
        asyncio.run(copied.aclose())
    assert budget.provider_calls == 3
    assert [call[2]["inferenceConfig"]["maxTokens"] for call in provider.calls] == [
        128,
        128,
        64,
    ]
    for path, headers, _, _ in provider.calls:
        assert path.endswith("/converse")
        assert "Credential=explicit-fixture-id/" in headers["Authorization"]
        assert "X-Amz-Security-Token" not in headers
        assert "autoflow-codec-only" not in str(headers)
    assert provider.lm.history[-1]["usage"]["total_tokens"] >= 3


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("adapter", ["chat", "json"])
def test_actual_dspy_adapters_preserve_converse_output(provider, asynchronous, adapter):
    provider.mode = "adapter" if adapter == "chat" else "schema_tool"
    predictor = dspy.Predict("question -> answer")
    with dspy.context(
        lm=provider.lm,
        adapter=dspy.ChatAdapter() if adapter == "chat" else dspy.JSONAdapter(),
    ):
        result = (
            asyncio.run(predictor.acall(question="fixture"))
            if asynchronous
            else predictor(question="fixture")
        )
    assert result.answer == "fixture answer"
    assert len(provider.calls) == 1
    if adapter == "json":
        tool = provider.calls[0][2]["toolConfig"]["tools"][0]["toolSpec"]
        assert tool["name"] == "json_tool_call"
        assert tool["inputSchema"]["json"]["properties"]["answer"]["type"] == "string"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_actual_dspy_streamify_keeps_prompt_listeners_and_budget(
    provider, asynchronous
):
    provider.mode = "adapter"
    program = dspy.streamify(
        dspy.Predict("question -> answer"),
        stream_listeners=[dspy.streaming.StreamListener("answer")],
        is_async_program=asynchronous,
    )

    async def run():
        with dspy.context(lm=provider.lm, adapter=dspy.ChatAdapter()):
            return [
                event async for event in program(question="unique fixture question")
            ]

    budget = scope(attempts=1)
    token = current_generation.set(budget)
    try:
        events = asyncio.run(run())
    finally:
        current_generation.reset(token)
    assert events[-1].answer == "fixture answer"
    assert "fixture answer" in "".join(
        event.chunk
        for event in events
        if isinstance(event, dspy.streaming.StreamResponse)
    )
    assert budget.provider_calls == 1
    assert len(provider.calls) == 1
    path, _, body, _ = provider.calls[0]
    assert path.endswith("/converse-stream")
    assert "unique fixture question" in str(body["messages"])
    assert body["inferenceConfig"]["maxTokens"] == 128


@pytest.mark.parametrize("asynchronous", [False, True])
def test_actual_dspy_streamify_cancel_closes_pending_connection(provider, asynchronous):
    provider.mode = "held_body"
    program = dspy.streamify(
        dspy.Predict("question -> answer"), is_async_program=asynchronous
    )

    async def run():
        async def consume():
            with dspy.context(lm=provider.lm, adapter=dspy.ChatAdapter()):
                async for event in program(question="fixture"):
                    pass

        task = asyncio.create_task(consume())
        assert await asyncio.to_thread(provider.started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert await asyncio.to_thread(provider.disconnected.wait, 1)

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["oversized", "compressed", "redirect"])
def test_response_bounds(provider, asynchronous, mode):
    provider.mode = mode
    with pytest.raises(LMUnexpectedError) as caught:
        invoke(provider.lm, asynchronous)
    assert isinstance(
        caught.value.__cause__,
        GenerationExpired if mode != "redirect" else ProviderConfigurationError,
    )
    assert len(provider.calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["expired", "exhausted", "slow"])
def test_deadlines_and_attempts(provider, asynchronous, mode):
    provider.mode = "slow" if mode == "slow" else "normal"
    budget = scope(
        seconds=-1 if mode == "expired" else 0.15 if mode == "slow" else 10,
        attempts=0 if mode == "exhausted" else 1,
    )
    token = current_generation.set(budget)
    try:
        with pytest.raises(LMUnexpectedError) as caught:
            invoke(provider.lm, asynchronous)
        assert isinstance(caught.value.__cause__, GenerationExpired)
    finally:
        current_generation.reset(token)
    assert len(provider.calls) == (1 if mode == "slow" else 0)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_stream_text_usage_and_early_close(provider, asynchronous):
    budget = scope(attempts=2)
    token = current_generation.set(budget)
    try:
        events = stream(provider.lm, asynchronous)
        assert (
            "".join(
                e.delta.text
                for e in events
                if e.type == "delta" and isinstance(e.delta, TextDelta)
            )
            == "fixture answer"
        )
        assert events[-1].type == "end"
        assert events[-1].usage.output_tokens == 2
        provider.mode = "held_stream"
        stream(provider.lm, asynchronous, early=True)
        assert provider.disconnected.wait(1)
    finally:
        current_generation.reset(token)
    assert budget.provider_calls == 2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_typed_tool_and_thinking_response(provider, asynchronous):
    provider.mode = "tools"
    response = (
        asyncio.run(provider.lm._async_engine_spec.complete(native_request()))
        if asynchronous
        else provider.lm.engine.complete(native_request())
    )
    tool = response.message.parts_of(ToolCallPart)[0]
    assert tool.name == "lookup" and tool.input == {"q": "fixture"}
    assert response.finish_reason == "tool_call"
    events = stream(provider.lm, asynchronous)
    assert events[-1].finish_reason == "tool_call"
    assert any(e.type == "delta" and e.delta.type == "thinking" for e in events)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "mode", ["oversized", "compressed", "slow", "truncated", "bad_crc", "exception"]
)
def test_stream_response_bounds(provider, asynchronous, mode):
    provider.mode = mode
    token = current_generation.set(scope(0.15 if mode == "slow" else 10))
    try:
        with pytest.raises(
            GenerationExpired
            if mode in {"oversized", "compressed", "slow"}
            else ProviderConfigurationError
        ):
            stream(provider.lm, asynchronous)
    finally:
        current_generation.reset(token)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_inline_image_and_tool_history(provider, asynchronous):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                }
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": "fixture-id",
                    "function": {"name": "lookup", "arguments": '{"q":"fixture"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "fixture-id", "content": "fixture result"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "fixture",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }
    ]
    result = (
        asyncio.run(provider.lm.acall(messages=messages, tools=tools))
        if asynchronous
        else provider.lm(messages=messages, tools=tools)
    )
    assert result == ["fixture answer"]
    body = provider.calls[0][2]
    assert (
        body["messages"][0]["content"][0]["image"]["source"]["bytes"]
        == base64.b64encode(b"image").decode()
    )
    assert body["messages"][1]["content"][0]["toolUse"]["input"] == {"q": "fixture"}
    assert body["messages"][2]["content"][0]["toolResult"]["toolUseId"] == "fixture-id"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_remote_image_cannot_start_extra_request(provider, asynchronous):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "http://127.0.0.1:1/private"},
                }
            ],
        }
    ]
    with pytest.raises(LMUnexpectedError) as caught:
        if asynchronous:
            asyncio.run(provider.lm.acall(messages=messages))
        else:
            provider.lm(messages=messages)
    assert isinstance(caught.value.__cause__, ProviderConfigurationError)
    assert provider.calls == []


@pytest.mark.parametrize("kind", ["image_url", "file"])
def test_data_prefix_cannot_disguise_remote_media(provider, kind, monkeypatch):
    from litellm.litellm_core_utils.prompt_templates.factory import (
        BedrockImageProcessor,
    )

    downloaded = []

    def forbidden(*args, **kwargs):
        downloaded.append(True)
        raise AssertionError("Media must not download before admission")

    monkeypatch.setattr(BedrockImageProcessor, "get_image_details", forbidden)
    source = "data:text/plain,http://127.0.0.1:1/private"
    part = {"type": "image_url", "image_url": {"url": source}}
    if kind == "file":
        part = {"type": "file", "file": {"file_data": source}}
    with pytest.raises(LMUnexpectedError) as caught:
        provider.lm(messages=[{"role": "user", "content": [part]}])
    assert isinstance(caught.value.__cause__, ProviderConfigurationError)
    assert not downloaded and not provider.calls


@pytest.mark.parametrize("asynchronous", [False, True])
def test_auth_errors_are_classified_without_echo(provider, asynchronous):
    provider.mode = "auth"
    with pytest.raises(LMAuthError) as caught:
        invoke(provider.lm, asynchronous)
    assert "fixture-secret-echo" not in str(caught.value)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_oversized_prompt_fails_before_http(provider, asynchronous):
    with pytest.raises(LMUnexpectedError) as caught:
        prompt = "x" * (2 * 1024 * 1024 + 1)
        if asynchronous:
            asyncio.run(provider.lm.acall(prompt=prompt))
        else:
            provider.lm(prompt=prompt)
    assert isinstance(caught.value.__cause__, GenerationExpired)
    assert provider.calls == []


@pytest.mark.parametrize("streaming", [False, True])
def test_async_cancel_closes_pending_model_connection(provider, streaming):
    provider.mode = "held_body"

    async def run():
        async def consume():
            if not streaming:
                return await provider.lm.acall(prompt="fixture")
            responses = provider.lm._async_engine_spec.stream(native_request())
            try:
                async for response in responses:
                    pass
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
def test_reasoning_usage_does_not_invoke_tokenizer_downloads(
    provider, asynchronous, monkeypatch
):
    from litellm.llms.bedrock.chat import converse_transformation

    def forbidden(*args, **kwargs):
        raise AssertionError("Response conversion must not invoke a lazy tokenizer")

    monkeypatch.setattr(converse_transformation, "token_counter", forbidden)
    provider.mode = "tools"
    response = (
        asyncio.run(provider.lm._async_engine_spec.complete(native_request()))
        if asynchronous
        else provider.lm.engine.complete(native_request())
    )
    assert response.usage.output_tokens == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["file", "document"])
def test_inline_documents_preserve_data(provider, asynchronous, kind):
    data = base64.b64encode(b"fixture PDF document").decode()
    part = {
        "type": "file",
        "file": {
            "file_data": "data:application/pdf;base64," + data,
            "filename": "fixture.pdf",
        },
    }
    if kind == "document":
        part = {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": data},
        }
    messages = [
        {"role": "user", "content": [part, {"type": "text", "text": "fixture"}]}
    ]
    result = (
        asyncio.run(provider.lm.acall(messages=messages))
        if asynchronous
        else provider.lm(messages=messages)
    )
    assert result == ["fixture answer"]
    assert (
        provider.calls[0][2]["messages"][0]["content"][0]["document"]["source"]["bytes"]
        == data
    )
