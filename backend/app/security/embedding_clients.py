"""Embedding codecs over explicit, owned and budgeted HTTP connections."""

import json

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.embeddings.bedrock.base import (
    BedrockEmbedding,
    PROVIDER_SPECIFIC_IDENTIFIERS,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from ollama import EmbedResponse, ResponseError
from ollama._types import EmbedRequest

from app.security.bedrock_http import BedrockHTTP, ExplicitSession
from app.security.generation_http import GenerationTransport, GenerationAsyncTransport
from app.security.provider_config import ProviderConfigurationError


class BoundBedrockEmbedding(BedrockEmbedding):
    def __init__(
        self,
        model_name,
        aws_access_key_id,
        aws_secret_access_key,
        region_name,
        **kwargs,
    ):
        client = BedrockHTTP(
            model_name, region_name, aws_access_key_id, aws_secret_access_key, 60, 1
        )
        # Retain the pinned SDK's pure Titan/Cohere request and response mapping.
        # Its session constructor receives no profile or ambient credentials;
        # neither that session nor the optional aioboto3 session performs I/O.
        super().__init__(
            model_name=model_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            botocore_session=ExplicitSession(),
            client=client,
            max_retries=1,
            timeout=60,
            **kwargs,
        )
        self._asession = None

    async def _aget_embedding(self, payload, type):
        provider = self._get_provider()
        response = await self._client.ainvoke_model(
            body=self._get_request_body(provider, payload, type),
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
        )
        identifiers = PROVIDER_SPECIFIC_IDENTIFIERS.get(provider)
        if identifiers is None:
            raise ProviderConfigurationError("Unsupported embedding provider")
        return identifiers["get_embeddings_func"](
            json.loads(response["body"].read()), isinstance(payload, list)
        )


class BoundOllamaEmbedding(OllamaEmbedding):
    _api_key: str = PrivateAttr()

    def __init__(self, model_name, base_url, api_key, **kwargs):
        # Initialize the pinned SDK's data/formatting fields without its shared
        # HTTP pools. Every sync/async call below owns and closes its connection.
        BaseEmbedding.__init__(self, model_name=model_name, base_url=base_url, **kwargs)
        self._api_key = api_key or "autoflow-local-no-auth"

    def _request_options(self, text):
        return dict(
            url=self.base_url.rstrip("/") + "/api/embed",
            json=EmbedRequest(
                model=self.model_name,
                input=text,
                options=self.ollama_additional_kwargs,
            ).model_dump(exclude_none=True),
            headers={"Authorization": "Bearer " + self._api_key},
        )

    @staticmethod
    def _parse(response):
        if not 200 <= response.status_code < 300:
            raise ResponseError(
                "Embedding provider request failed", response.status_code
            )
        try:
            return list(
                EmbedResponse.model_validate_json(response.content).embeddings[0]
            )
        except Exception:
            raise ProviderConfigurationError("Invalid embedding response") from None

    def get_general_text_embedding(self, texts):
        with httpx.Client(
            transport=GenerationTransport(trust_env=False),
            trust_env=False,
            follow_redirects=False,
            timeout=60,
        ) as client:
            return self._parse(client.post(**self._request_options(texts)))

    async def aget_general_text_embedding(self, prompt):
        async with httpx.AsyncClient(
            transport=GenerationAsyncTransport(trust_env=False),
            trust_env=False,
            follow_redirects=False,
            timeout=60,
        ) as client:
            return self._parse(await client.post(**self._request_options(prompt)))
