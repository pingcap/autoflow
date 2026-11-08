"""Azure's configured API-key mode must not inherit a server Entra token."""

from openai import AzureOpenAI as AzureClient, AsyncAzureOpenAI as AsyncAzureClient
from llama_index.llms.azure_openai import AzureOpenAI

from app.security.provider_config import ProviderConfigurationError


class _ExplicitAPIKey:
    def __init__(self, *args, **kwargs):
        if not isinstance(kwargs.get("api_key"), str) or not kwargs["api_key"]:
            raise ProviderConfigurationError("Azure requires an explicit API key")
        super().__init__(*args, **kwargs)
        # The pinned SDK reads AZURE_OPENAI_AD_TOKEN even when api_key is given,
        # then prefers that identity. Clear it on this owned instance before any
        # request. SDK with_options()/copy() constructs this same subclass, so
        # copying cannot restore ambient identity. The application's provider
        # configuration exposes API-key authentication, not Entra providers.
        self._azure_ad_token = None
        self._azure_ad_token_provider = None


class ExplicitAzureClient(_ExplicitAPIKey, AzureClient):
    pass


class ExplicitAsyncAzureClient(_ExplicitAPIKey, AsyncAzureClient):
    pass


class BoundAzureOpenAI(AzureOpenAI):
    def _get_client(self):
        if not self.reuse_client:
            return ExplicitAzureClient(**self._get_credential_kwargs())
        if self._client is None:
            self._client = ExplicitAzureClient(**self._get_credential_kwargs())
        return self._client

    def _get_aclient(self):
        if not self.reuse_client:
            return ExplicitAsyncAzureClient(
                **self._get_credential_kwargs(is_async=True)
            )
        if self._aclient is None:
            self._aclient = ExplicitAsyncAzureClient(
                **self._get_credential_kwargs(is_async=True)
            )
        return self._aclient
