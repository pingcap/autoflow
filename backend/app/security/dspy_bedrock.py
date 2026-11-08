"""Retain DSPy's Converse protocol without LiteLLM's global network lifecycle."""

import base64
import json
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace
from types import SimpleNamespace

import httpx
import anyio
import dspy
from botocore.exceptions import ClientError
from dspy import lm15
from dspy.clients.engines import LM15Engine, AsyncLM15Engine
from dspy.clients.call_context import completed_legacy, stream_emitted
from dspy.clients.call_result import CallResult
from dspy.clients.engines.streaming import ListenerBridge
from dspy._vendor.lm15.result import StreamAccumulator
from dspy.lm15 import RouterConfig
from dspy._vendor.lm15.transports import TransportResponse, AsyncTransportResponse
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder
from litellm.types.utils import ModelResponse
from llama_index.llms.bedrock_converse.utils import is_reasoning
from pydantic import BaseModel

from app.security.bedrock_http import EventDecoder
from app.security.dspy_http import ModelTransport, bind_output_limit
from app.security.generation_limits import GenerationExpired
from app.security.provider_config import ProviderConfigurationError


class _Codec(AmazonConverseConfig):
    @classmethod
    def get_config(cls):
        return {}

    def transform_usage(
        self,
        usage,
        reasoning_content=None,
        thinking_ran=False,
        provider_reasoning_tokens=None,
    ):
        # Keep provider token totals. Estimating an unreported reasoning count
        # can lazily download a tokenizer outside the guarded model connection.
        return super().transform_usage(
            usage,
            reasoning_content=None,
            thinking_ran=thinking_ran,
            provider_reasoning_tokens=provider_reasoning_tokens,
        )


def _inline_media(messages):
    # These conversion helpers otherwise download image/file URLs themselves,
    # before the signed model request. Text and ordinary tool arguments are data.
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            kind = part.get("type")
            if kind == "image_url":
                image = part.get("image_url")
                source = image.get("url") if isinstance(image, dict) else image
                if not _base64_media(source):
                    raise ProviderConfigurationError(
                        "Bedrock media must be supplied inline"
                    )
            elif kind == "file":
                source = part.get("file", {})
                if source.get("file_id") or not _base64_media(source.get("file_data")):
                    raise ProviderConfigurationError(
                        "Bedrock media must be supplied inline"
                    )
            elif kind == "document":
                if part.get("source", {}).get("type") != "base64":
                    raise ProviderConfigurationError(
                        "Bedrock documents must be supplied inline"
                    )


def _base64_media(source):
    if not isinstance(source, str):
        return False
    header, separator, _ = source.partition(",")
    return bool(separator and header.startswith("data:") and header.endswith(";base64"))


def _binary_values(value, shape):
    # LiteLLM's JSON codec emits base64 strings; botocore expects raw bytes and
    # performs the base64 encoding itself. Keep binary media encoded exactly once.
    if shape.type_name == "blob" and isinstance(value, str):
        return base64.b64decode(value, validate=True)
    if shape.type_name == "structure" and isinstance(value, dict):
        return {
            key: _binary_values(item, shape.members[key])
            if key in shape.members
            else item
            for key, item in value.items()
        }
    if shape.type_name == "list" and isinstance(value, list):
        return [_binary_values(item, shape.member) for item in value]
    if shape.type_name == "map" and isinstance(value, dict):
        return {key: _binary_values(item, shape.value) for key, item in value.items()}
    return value


@contextmanager
def _errors():
    try:
        yield
    except ClientError as error:
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 500)
        kind = {
            400: lm15.InvalidRequestError,
            401: lm15.AuthError,
            403: lm15.AuthError,
            404: lm15.UnsupportedModelError,
            408: lm15.TimeoutError,
            429: lm15.RateLimitError,
        }.get(status, lm15.ServerError if status >= 500 else lm15.ProviderError)
        raise kind(
            "Bedrock request failed", provider="bedrock", status=status
        ) from error
    except httpx.TimeoutException as error:
        raise lm15.TimeoutError(
            "Bedrock connection timed out", provider="bedrock"
        ) from error
    except httpx.HTTPError as error:
        raise lm15.TransportError(
            "Bedrock connection failed", provider="bedrock"
        ) from error


class _Protocol:
    def __init__(self, client, request):
        self.client = client
        if request.method != "POST" or len(request.body) > 2 * 1024 * 1024:
            raise GenerationExpired("Generation request body budget exceeded")
        try:
            payload = json.loads(request.body)
            if payload.pop("model") != client.model:
                raise ProviderConfigurationError("Unexpected Bedrock model")
            self.messages = payload.pop("messages")
            _inline_media(self.messages)
            self.streaming = payload.get("stream", False)
            bind_output_limit(payload, "max_tokens", client.max_tokens)
            self.codec = _Codec()
            self.options = self.codec.map_openai_params(
                payload, {}, client.model, False
            )
            body = self.codec.transform_request(
                client.model, self.messages, self.options, {}, {}
            )
            self.operation = "ConverseStream" if self.streaming else "Converse"
            self.parameters = _binary_values(
                {"modelId": client.model, **body},
                client.service.operation_model(self.operation).input_shape,
            )
            self.events = EventDecoder(client.service)
            self.stream_codec = AWSEventStreamDecoder(
                client.model, json_mode=self.options.get("json_mode")
            )
        except (ProviderConfigurationError, GenerationExpired):
            raise
        except Exception:
            raise ProviderConfigurationError("Invalid Bedrock request") from None

    def result(self, body):
        try:
            raw = httpx.Response(200, content=body)
            result = self.codec._transform_response(
                model=self.client.model,
                response=raw,
                model_response=ModelResponse(),
                stream=False,
                logging_obj=None,
                optional_params=self.options,
                api_key=None,
                data={},
                messages=self.messages,
                encoding=None,
            )
            return result
        except Exception:
            raise ProviderConfigurationError("Invalid Bedrock response") from None

    def complete(self, body):
        return json.dumps(self.result(body).model_dump(exclude_none=True)).encode()

    def feed(self, chunk):
        for event in self.events.feed(chunk):
            try:
                if "messageStart" in event:
                    self.stream_codec._initialize_converse_response_id(event)
                    continue
                payload = next(iter(event.values()))
                result = self.stream_codec.converse_chunk_parser(payload)
                yield (
                    b"data: "
                    + json.dumps(result.model_dump(exclude_none=True)).encode()
                    + b"\n\n"
                )
            except Exception:
                raise ProviderConfigurationError(
                    "Invalid Bedrock stream response"
                ) from None

    def finish(self):
        self.events.finish()
        return b"data: [DONE]\n\n"

    def response_options(self, response):
        return dict(
            status=200,
            reason="OK",
            http_version=response.http_version,
            headers=[
                (
                    "Content-Type",
                    "text/event-stream" if self.streaming else "application/json",
                )
            ],
        )


class BedrockTransport(ModelTransport):
    def __init__(self, client, protocol=None):
        super().__init__(client.endpoint, client.timeout)
        self.client = client
        self.protocol = protocol

    @contextmanager
    def stream(self, request):
        self.options(request)
        with _errors():
            protocol = self.protocol or _Protocol(self.client, request)
            with self.client.stream(
                protocol.operation, protocol.parameters
            ) as response:

                def chunks():
                    if not protocol.streaming:
                        yield protocol.complete(response.read())
                        return
                    for chunk in response.iter_bytes():
                        yield from protocol.feed(chunk)
                    yield protocol.finish()

                yield TransportResponse(
                    **protocol.response_options(response),
                    chunks=chunks(),
                    release=lambda _consumed: response.close(),
                )


class AsyncBedrockTransport(BedrockTransport):
    @asynccontextmanager
    async def stream(self, request):
        self.options(request)
        with _errors():
            protocol = self.protocol or _Protocol(self.client, request)
            async with self.client.astream(
                protocol.operation, protocol.parameters
            ) as response:

                async def chunks():
                    if not protocol.streaming:
                        yield protocol.complete(await response.aread())
                        return
                    async for chunk in response.aiter_bytes():
                        for event in protocol.feed(chunk):
                            yield event
                    yield protocol.finish()

                async def release(_consumed):
                    await response.aclose()

                yield AsyncTransportResponse(
                    **protocol.response_options(response),
                    chunks=chunks(),
                    release=release,
                )


class _BedrockRouting:
    def __init__(self, config, model):
        self.model = model
        super().__init__(config)

    def resolve(self, model):
        if model != f"bedrock/converse/{self.model}":
            raise ProviderConfigurationError("Unexpected Bedrock model")
        # This is an in-memory protocol adapter. Only our transport does I/O,
        # always using the configured AWS Converse endpoint and SigV4 identity.
        return super().resolve(f"openai/{self.model}")

    def _target(self, request, *, streaming=False):
        if request.config.top_k is not None:
            request = replace(
                request,
                config=replace(
                    request.config,
                    top_k=None,
                    extensions={
                        **(request.config.extensions or {}),
                        "top_k": request.config.top_k,
                    },
                ),
            )
        return super()._target(request, streaming=streaming)

    def _legacy_protocol(self, request):
        # Keep existing provider-shaped documents, file names and options. The
        # native canonical OpenAI request cannot represent all of these fields.
        # This compatibility hook still uses only our owned Converse transport.
        if self._closed:
            raise ProviderConfigurationError("Model client is closed")
        self.resolve(request["model"])
        payload = {key: value for key, value in request.items() if value is not None}
        for name in (
            "rollout_id",
            "timeout",
            "api_key",
            "api_base",
            "base_url",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_region_name",
            "aws_bedrock_runtime_endpoint",
        ):
            payload.pop(name, None)
        payload["model"] = self.model
        format_ = payload.get("response_format")
        if isinstance(format_, type) and issubclass(format_, BaseModel):
            from dspy.clients.legacy_requests import _strict_json_schema

            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": format_.__name__,
                    "schema": _strict_json_schema(format_.model_json_schema()),
                    "strict": True,
                },
            }
        return _Protocol(
            self.config.transport.client,
            SimpleNamespace(method="POST", body=json.dumps(payload).encode()),
        )

    @staticmethod
    def _legacy_result(lm, raw, request):
        completed_legacy(raw)
        lm._check_truncation(raw)
        return CallResult.legacy(lm, raw, kwargs=request)

    async def _stream_legacy(self, lm, request):
        # Preserve provider-shaped inputs while using the native response
        # assembler/listeners. The minimal request selects only the decoder;
        # the owned transport sends the complete original Converse parameters.
        protocol = self._legacy_protocol({**request, "stream": True})
        client = protocol.client
        engine = AsyncBedrockEngine(
            RouterConfig(
                env={},
                api_keys={"openai-chat": "autoflow-codec-only"},
                base_urls={"openai-chat": client.endpoint},
                transport=AsyncBedrockTransport(client, protocol),
            ),
            self.model,
        )
        decoder_request = lm15.Request(model=lm.model, messages=[lm15.Message.user("")])
        accumulator = StreamAccumulator(decoder_request)
        caller = dspy.settings.caller_predict
        bridge = ListenerBridge(lm.model, id(caller) if caller else None)
        events = engine.stream(decoder_request)
        try:
            async for event in events:
                accumulator.push(event)
                if chunk := bridge.chunk(event):
                    stream_emitted()
                    await dspy.settings.send_stream.send(chunk)
            response = accumulator.response()
            result = CallResult.native(response)
            raw = result.provider_response()
            raw.choices[0].finish_reason = (
                "tool_calls"
                if response.finish_reason == "tool_call"
                else response.finish_reason
            )
            result.raw = raw
            completed_legacy(raw)
            lm._check_truncation(raw)
            return result
        finally:
            await events.aclose()
            await engine.aclose()


class BedrockEngine(_BedrockRouting, LM15Engine):
    def complete_legacy(self, lm, request, **context):
        if dspy.settings.send_stream is not None:
            return anyio.from_thread.run(self._stream_legacy, lm, request)
        with _errors():
            protocol = self._legacy_protocol(request)
            with protocol.client.stream("Converse", protocol.parameters) as response:
                return self._legacy_result(
                    lm, protocol.result(response.read()), request
                )


class AsyncBedrockEngine(_BedrockRouting, AsyncLM15Engine):
    async def complete_legacy(self, lm, request, **context):
        if dspy.settings.send_stream is not None:
            return await self._stream_legacy(lm, request)
        with _errors():
            protocol = self._legacy_protocol(request)
            async with protocol.client.astream(
                "Converse", protocol.parameters
            ) as response:
                return self._legacy_result(
                    lm, protocol.result(await response.aread()), request
                )


def bedrock_engines(llm):
    client = llm._client
    options = dict(
        env={},
        api_keys={"openai-chat": "autoflow-codec-only"},
        base_urls={"openai-chat": client.endpoint},
    )
    engines = {
        "engine": BedrockEngine(
            RouterConfig(**options, transport=BedrockTransport(client)), llm.model
        ),
        "async_engine": AsyncBedrockEngine(
            RouterConfig(**options, transport=AsyncBedrockTransport(client)), llm.model
        ),
    }
    params = frozenset(_Codec().get_supported_openai_params(llm.model))
    for engine in engines.values():
        engine.supports_function_calling = llm.metadata.is_function_calling_model
        engine.supports_reasoning = is_reasoning(llm.model)
        engine.supports_response_schema = "response_format" in params
        engine.supported_params = params
    return engines
