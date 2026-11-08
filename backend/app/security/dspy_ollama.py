"""Ollama protocol over owned, budgeted HTTP, without SDK metadata requests.

Use the locked SDK's pure protocol codecs only. Its completion lifecycle also
fetches /api/show using a process-global client, outside the inference budget.
DSPy's native engine retains its canonical response/stream and adapter behavior.
"""

import json
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace
from uuid import uuid4

import httpx
from dspy.clients.engines import LM15Engine, AsyncLM15Engine
from dspy.lm15 import RouterConfig
from dspy._vendor.lm15.transports import TransportResponse, AsyncTransportResponse
from litellm.llms.ollama.chat.transformation import (
    OllamaChatConfig,
    OllamaChatCompletionResponseIterator,
)

from app.security.dspy_http import ModelTransport
from app.security.generation_http import GenerationTransport, GenerationAsyncTransport
from app.security.generation_limits import GenerationExpired
from app.security.provider_config import ProviderConfigurationError


class _RequestCodec(OllamaChatConfig):
    @classmethod
    def get_config(cls):
        # A different chat must not change this call through global SDK defaults.
        return {}


class _ResponseCodec:
    def __init__(self):
        self.codec = OllamaChatCompletionResponseIterator(iter(()), sync_stream=True)
        self.identifier = "chatcmpl-" + uuid4().hex
        self.done = False

    def convert(self, body, streaming):
        try:
            chunk = json.loads(body)
            if not isinstance(chunk, dict) or "error" in chunk:
                raise ValueError
            self.done = chunk["done"] is True
            if not streaming and not self.done:
                raise ValueError
            converted = self.codec.chunk_parser(chunk).model_dump(exclude_none=True)
            converted["id"] = self.identifier
            if not streaming:
                converted["object"] = "chat.completion"
                for choice in converted["choices"]:
                    choice["message"] = choice.pop("delta")
                    choice["message"]["role"] = "assistant"
            data = json.dumps(converted).encode()
        except Exception:
            # SDK parse errors can include the complete upstream body.
            raise ProviderConfigurationError("Invalid Ollama response") from None
        return b"data: " + data + b"\n\n" if streaming else data

    def finish(self):
        if not self.done:
            raise ProviderConfigurationError("Incomplete Ollama response")
        return b"data: [DONE]\n\n"


class OllamaTransport(ModelTransport):
    def __init__(self, model, endpoint, timeout, defaults):
        super().__init__(endpoint, timeout)
        self.model = model
        self.defaults = dict(defaults)
        base = endpoint.rstrip("/")
        self.url = base if base.endswith("/api/chat") else base + "/api/chat"

    def options(self, request):
        options = super().options(request)
        if len(request.body) > 2 * 1024 * 1024:
            raise GenerationExpired("Generation request body budget exceeded")
        try:
            payload = json.loads(request.body)
            if request.method != "POST" or payload.pop("model") != self.model:
                raise ValueError
            messages = payload.pop("messages")
            codec = _RequestCodec()
            optional = codec.map_openai_params(
                payload, dict(self.defaults), self.model, False
            )
            # Native/provider-specific options do not pass through OpenAI's map.
            for key in ("top_k", "num_ctx", "num_predict", "keep_alive", "think"):
                if key in payload:
                    optional[key] = payload[key]
            body = codec.transform_request(self.model, messages, optional, {}, {})
        except Exception:
            raise ProviderConfigurationError("Invalid Ollama request") from None
        options.update(
            url=self.url,
            content=json.dumps(body).encode(),
            # The router supplies explicit auth. Do not forward stale framing
            # headers from the OpenAI-shaped body after protocol conversion.
            headers=[
                (key, value)
                for key, value in request.headers
                if key.lower() not in {"content-length", "content-encoding"}
            ],
        )
        return options, body["stream"]

    @staticmethod
    def response_options(response, streaming):
        if 300 <= response.status_code < 400:
            raise ProviderConfigurationError("Provider redirects are disabled")
        if not 200 <= response.status_code < 300:
            # A provider can echo credentials or prompts in an error body.
            raise ProviderConfigurationError("Ollama request failed")
        return dict(
            status=response.status_code,
            reason=response.reason_phrase,
            headers=[
                (
                    "Content-Type",
                    "text/event-stream" if streaming else "application/json",
                )
            ],
            http_version=response.http_version,
        )

    @contextmanager
    def stream(self, request):
        options, streaming = self.options(request)
        with httpx.Client(
            transport=GenerationTransport(trust_env=False),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            with client.stream(**options) as response:
                metadata = self.response_options(response, streaming)

                def chunks():
                    codec = _ResponseCodec()
                    if not streaming:
                        yield codec.convert(response.read(), False)
                        return
                    for line in response.iter_lines():
                        if line.strip():
                            yield codec.convert(line, True)
                            if codec.done:
                                break
                    yield codec.finish()

                yield TransportResponse(
                    **metadata,
                    chunks=chunks(),
                    release=lambda _consumed: response.close(),
                )


class AsyncOllamaTransport(OllamaTransport):
    @asynccontextmanager
    async def stream(self, request):
        options, streaming = self.options(request)
        async with httpx.AsyncClient(
            transport=GenerationAsyncTransport(trust_env=False),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            async with client.stream(**options) as response:
                metadata = self.response_options(response, streaming)

                async def chunks():
                    codec = _ResponseCodec()
                    if not streaming:
                        yield codec.convert(await response.aread(), False)
                        return
                    async for line in response.aiter_lines():
                        if line.strip():
                            yield codec.convert(line, True)
                            if codec.done:
                                break
                    yield codec.finish()

                async def release(_consumed):
                    await response.aclose()

                yield AsyncTransportResponse(
                    **metadata,
                    chunks=chunks(),
                    release=release,
                )


class _OllamaOptions:
    def _target(self, request, *, streaming=False):
        # Native OpenAI encoding has no top_k slot. Carry the explicit canonical
        # setting through its extension field to our Ollama protocol codec.
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


class OllamaEngine(_OllamaOptions, LM15Engine):
    pass


class AsyncOllamaEngine(_OllamaOptions, AsyncLM15Engine):
    pass


def ollama_engines(
    model,
    endpoint,
    credential,
    timeout,
    *,
    function_calling,
    thinking,
    generation_options,
    json_mode,
):
    options = dict(
        env={},
        api_keys={"ollama": credential},
        base_urls={"ollama": endpoint},
    )
    defaults = dict(generation_options)
    if thinking is not None:
        defaults["think"] = thinking
    if json_mode:
        defaults["format"] = "json"
    engines = {
        "engine": OllamaEngine(
            RouterConfig(
                **options, transport=OllamaTransport(model, endpoint, timeout, defaults)
            )
        ),
        "async_engine": AsyncOllamaEngine(
            RouterConfig(
                **options,
                transport=AsyncOllamaTransport(model, endpoint, timeout, defaults),
            )
        ),
    }
    for engine in engines.values():
        # Custom engines must declare adapter hints explicitly. Otherwise
        # JSONAdapter silently omits response_format despite the API's support.
        # Tool/reasoning support follows the operator's model configuration.
        engine.supports_function_calling = function_calling
        engine.supports_reasoning = thinking is not None
        engine.supports_response_schema = True
        engine.supported_params = frozenset(
            _RequestCodec().get_supported_openai_params(model)
        ) | {"top_k"}
    return engines
