"""Use bounded native DSPy HTTP while retaining the configured Azure REST API."""

import json
from urllib.parse import quote, urlencode

from dspy.clients.engines import LM15Engine, AsyncLM15Engine
from dspy.lm15 import RouterConfig

from app.security.dspy_http import (
    ModelTransport,
    AsyncModelTransport,
    bind_native_capabilities,
    bind_output_limit,
)
from app.security.generation_limits import GenerationExpired
from app.security.provider_config import ProviderConfigurationError


class _AzureRequest:
    def __init__(
        self,
        endpoint,
        timeout,
        credential,
        deployment,
        api_version,
        token_parameter,
        max_tokens,
    ):
        super().__init__(endpoint, timeout)
        self.credential = credential
        self.deployment = deployment
        self.token_parameter = token_parameter
        self.max_tokens = max_tokens
        # Native DSPy Azure uses /openai/v1. Existing Autoflow configurations use
        # the deployment API; changing its version implicitly breaks gateways
        # and model features. Bind the exact legacy route and version instead.
        self.url = (
            endpoint.rstrip("/")
            + "/openai/deployments/"
            + quote(deployment, safe="")
            + "/chat/completions?"
            + urlencode({"api-version": api_version})
        )

    def options(self, request):
        options = super().options(request)
        if request.method != "POST":
            raise ProviderConfigurationError("Unexpected Azure operation")
        if len(request.body) > 2 * 1024 * 1024:
            raise GenerationExpired("Generation request body budget exceeded")
        payload = json.loads(request.body)
        payload["model"] = self.deployment
        # Older Azure versions expect max_tokens for ordinary chat models;
        # reasoning models use max_completion_tokens. Match the primary SDK's
        # model-specific spelling, not the native engine's newer v1 default.
        bind_output_limit(payload, self.token_parameter, self.max_tokens)
        options.update(
            url=self.url,
            content=json.dumps(payload).encode(),
            headers=[
                (name, value)
                for name, value in request.headers
                if name.lower()
                not in {
                    "authorization",
                    "api-key",
                    "content-length",
                    "content-encoding",
                }
            ]
            + [("api-key", self.credential)],
        )
        return options


class AzureTransport(_AzureRequest, ModelTransport):
    pass


class AsyncAzureTransport(_AzureRequest, AsyncModelTransport):
    pass


def azure_engines(
    endpoint,
    credential,
    timeout,
    *,
    model,
    deployment,
    api_version,
    token_parameter,
    max_tokens,
):
    options = dict(
        env={},
        api_keys={"azure-chat": credential},
        base_urls={"azure-chat": endpoint},
    )
    arguments = (
        endpoint,
        timeout,
        credential,
        deployment,
        api_version,
        token_parameter,
        max_tokens,
    )
    engines = {
        "engine": LM15Engine(
            RouterConfig(**options, transport=AzureTransport(*arguments))
        ),
        "async_engine": AsyncLM15Engine(
            RouterConfig(**options, transport=AsyncAzureTransport(*arguments))
        ),
    }
    return bind_native_capabilities(engines, f"azure/{model}")
