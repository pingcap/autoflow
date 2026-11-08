"""Keep DSPy's text-completion contract on owned, bounded OpenAI SDK clients.

The pinned DSPy native engine supports chat/Responses, not text completions.
Use its text compatibility contract without invoking LiteLLM's global clients or
callbacks. Preserve text-only options and the original completion/logprob shape.
"""

from contextlib import contextmanager

import httpx
import openai
from openai import OpenAI, AsyncOpenAI
from dspy import lm15
from dspy.clients.call_context import completed_legacy
from dspy.clients.call_result import CallResult, attributes
from dspy.clients.engines.litellm_engine import LiteLLMEngine, AsyncLiteLLMEngine

from app.security.dspy_http import bind_output_limit
from app.security.generation_http import GenerationTransport, GenerationAsyncTransport
from app.security.generation_limits import GenerationExpired
from app.security.provider_config import ProviderConfigurationError


class _TextOptions:
    def __init__(self, endpoint, credential, timeout, max_tokens):
        super().__init__(model_type="text")
        self.endpoint = endpoint
        self.credential = credential
        self.timeout = min(timeout, 60)
        self.max_tokens = max_tokens

    def arguments(self, request):
        if self._closed:
            raise ProviderConfigurationError("Model client is closed")
        options = {key: value for key, value in request.items() if value is not None}
        options.pop("rollout_id", None)
        model = options.pop("model")
        if not model.startswith("openai/"):
            raise ProviderConfigurationError("Unexpected text model provider")
        messages = options.pop("messages")
        # Match the pinned DSPy text formatter, including system/history order
        # and its final delimiter. A chat endpoint would change model behavior.
        prompt = "\n\n".join(
            [item["content"] for item in messages] + ["BEGIN RESPONSE:"]
        )
        bind_output_limit(options, "max_tokens", self.max_tokens)
        return dict(model=model.removeprefix("openai/"), prompt=prompt, **options)

    def sdk_options(self):
        return dict(
            api_key=self.credential,
            base_url=self.endpoint,
            max_retries=0,
            timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, 10)),
        )

    @staticmethod
    def result(lm, raw, request):
        if not raw.choices or any(
            not isinstance(choice.text, str) for choice in raw.choices
        ):
            raise ProviderConfigurationError("Invalid text model response")
        # LiteLLM's legacy result supports both mapping and attribute access.
        # Keep that public shape for logprobs/history with the official SDK too.
        raw = attributes(raw.model_dump())
        completed_legacy(raw)
        lm._check_truncation(raw)
        return CallResult.legacy(lm, raw, kwargs=request)

    @contextmanager
    def errors(self):
        try:
            yield
        except openai.APIConnectionError as error:
            if isinstance(error.__cause__, GenerationExpired):
                raise error.__cause__
            kind = (
                lm15.TimeoutError
                if isinstance(error, openai.APITimeoutError)
                else lm15.TransportError
            )
            raise kind("Text model connection failed", provider="openai") from error
        except openai.APIStatusError as error:
            status = error.status_code
            kind = {
                400: lm15.InvalidRequestError,
                401: lm15.AuthError,
                403: lm15.AuthError,
                404: lm15.UnsupportedModelError,
                408: lm15.TimeoutError,
                429: lm15.RateLimitError,
            }.get(status, lm15.ServerError if status >= 500 else lm15.ProviderError)
            raise kind(
                "Text model request failed", provider="openai", status=status
            ) from error
        except openai.APIError as error:
            raise lm15.ProviderError(
                "Invalid text model response", provider="openai"
            ) from error


class TextEngine(_TextOptions, LiteLLMEngine):
    def complete_legacy(self, lm, request, **context):
        arguments = self.arguments(request)
        with httpx.Client(
            transport=GenerationTransport(trust_env=False),
            trust_env=False,
            follow_redirects=False,
        ) as transport:
            with (
                OpenAI(http_client=transport, **self.sdk_options()) as client,
                self.errors(),
            ):
                # Organization/project have no configuration fields in this
                # provider. Never attach unrelated process-wide billing hints.
                client.organization = None
                client.project = None
                raw = client.completions.create(**arguments)
        return self.result(lm, raw, request)


class AsyncTextEngine(_TextOptions, AsyncLiteLLMEngine):
    async def complete_legacy(self, lm, request, **context):
        arguments = self.arguments(request)
        async with httpx.AsyncClient(
            transport=GenerationAsyncTransport(trust_env=False),
            trust_env=False,
            follow_redirects=False,
        ) as transport:
            async with AsyncOpenAI(
                http_client=transport, **self.sdk_options()
            ) as client:
                client.organization = None
                client.project = None
                with self.errors():
                    raw = await client.completions.create(**arguments)
        return self.result(lm, raw, request)


def text_engines(llm):
    options = (llm.api_base, llm.api_key, llm.timeout, llm.max_tokens or 4096)
    return dict(engine=TextEngine(*options), async_engine=AsyncTextEngine(*options))
