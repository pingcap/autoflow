"""AWS Converse codecs and signing over owned, budgeted HTTP connections.

Only packaged botocore models are loaded. No AWS profile, metadata credential,
endpoint override, proxy, or SDK background producer participates in a call.
"""

from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from functools import lru_cache

import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest, prepare_request_dict
from botocore.credentials import Credentials
from botocore.errorfactory import ClientExceptionsFactory
from botocore.eventstream import EventStreamBuffer
from botocore.loaders import Loader
from botocore.model import ServiceModel
from botocore.parsers import EventStreamJSONParser, create_parser
from botocore.regions import EndpointResolver
from botocore.serialize import create_serializer
from botocore.session import Session

from app.security.dspy_http import bind_output_limit
from app.security.generation_http import GenerationTransport, GenerationAsyncTransport
from app.security.generation_limits import GenerationExpired
from app.security.provider_config import ProviderConfigurationError


@lru_cache(maxsize=1)
def _models():
    loader = Loader(
        extra_search_paths=[Loader.BUILTIN_DATA_PATH],
        include_default_search_paths=False,
    )
    return (
        ServiceModel(loader.load_service_model("bedrock-runtime", "service-2")),
        EndpointResolver(loader.load_data("endpoints")),
    )


def bedrock_endpoint(region):
    endpoint = _models()[1].construct_endpoint("bedrock-runtime", region)
    if not endpoint or "https" not in endpoint.get("protocols", []):
        raise ProviderConfigurationError("Unsupported Bedrock region")
    return "https://" + endpoint["hostname"]


class ExplicitSession(Session):
    """The LlamaIndex constructor still creates a session around our client."""

    def get_scoped_config(self):
        return {}

    @property
    def full_config(self):
        return {"profiles": {}}


class EventDecoder:
    def __init__(self, service):
        self.buffer = EventStreamBuffer()
        self.parser = EventStreamJSONParser()
        self.shape = service.operation_model("ConverseStream").output_shape.members[
            "stream"
        ]
        self.pending = 0
        self.stopped = False

    def feed(self, chunk):
        self.pending += len(chunk)
        self.buffer.add_data(chunk)
        try:
            for event in self.buffer:
                self.pending -= event.prelude.total_length
                raw = event.to_response_dict()
                if raw["status_code"] != 200:
                    raise ProviderConfigurationError("Bedrock stream failed")
                parsed = self.parser.parse(raw, self.shape)
                self.stopped |= "messageStop" in parsed
                if parsed:
                    yield parsed
        except ProviderConfigurationError:
            raise
        except Exception:
            raise ProviderConfigurationError("Invalid Bedrock event stream") from None

    def finish(self):
        if self.pending or not self.stopped:
            raise ProviderConfigurationError("Incomplete Bedrock event stream")


class BedrockHTTP:
    def __init__(self, model, region, access_key, secret_key, timeout, max_tokens):
        self.model = model
        self.region = region
        self.endpoint = bedrock_endpoint(region)
        self.credentials = Credentials(access_key, secret_key)
        self.timeout = min(timeout, 60)
        self.max_tokens = max_tokens
        self.service = _models()[0]
        self.exceptions = ClientExceptionsFactory().create_client_exceptions(
            self.service
        )

    def request(self, operation, parameters):
        parameters = deepcopy(parameters)
        if parameters.get("modelId") != self.model:
            raise ProviderConfigurationError("Unexpected Bedrock model")
        inference = parameters.setdefault("inferenceConfig", {})
        bind_output_limit(inference, "maxTokens", self.max_tokens)
        model = self.service.operation_model(operation)
        try:
            request = create_serializer("rest-json").serialize_to_request(
                parameters, model
            )
        except Exception:
            raise ProviderConfigurationError("Invalid Bedrock request") from None
        if len(request["body"]) > 2 * 1024 * 1024:
            raise GenerationExpired("Generation request body budget exceeded")
        prepare_request_dict(request, endpoint_url=self.endpoint, context={})
        aws = AWSRequest(
            method=request["method"],
            url=request["url"],
            data=request["body"],
            headers=request["headers"],
        )
        # Set before signing; HTTP transport must not change signed values.
        aws.headers["Accept-Encoding"] = "identity"
        SigV4Auth(self.credentials, "bedrock", self.region).add_auth(aws)
        signed = aws.prepare()
        return dict(
            method=signed.method,
            url=signed.url,
            content=signed.body,
            headers=dict(signed.headers),
            timeout=httpx.Timeout(self.timeout),
        )

    def check_response(self, response, operation):
        if 300 <= response.status_code < 400:
            raise ProviderConfigurationError("Provider redirects are disabled")
        if not 200 <= response.status_code < 300:
            # Keep retry/error classification, without upstream messages/headers.
            code = {
                400: "ValidationException",
                401: "AccessDeniedException",
                403: "AccessDeniedException",
                404: "ResourceNotFoundException",
                408: "ModelTimeoutException",
                429: "ThrottlingException",
                500: "InternalServerException",
                503: "ServiceUnavailableException",
            }.get(response.status_code, "BedrockRequestFailed")
            error = self.exceptions.from_code(code)
            raise error(
                {
                    "Error": {"Code": code, "Message": "Bedrock request failed"},
                    "ResponseMetadata": {"HTTPStatusCode": response.status_code},
                },
                operation,
            )

    def parse(self, response, body):
        try:
            return create_parser("rest-json").parse(
                {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": body,
                },
                self.service.operation_model("Converse").output_shape,
            )
        except Exception:
            raise ProviderConfigurationError("Invalid Bedrock response") from None

    @contextmanager
    def stream(self, operation, parameters):
        options = self.request(operation, parameters)
        with httpx.Client(
            transport=GenerationTransport(trust_env=False),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            with client.stream(**options) as response:
                self.check_response(response, operation)
                yield response

    @asynccontextmanager
    async def astream(self, operation, parameters):
        options = self.request(operation, parameters)
        async with httpx.AsyncClient(
            transport=GenerationAsyncTransport(trust_env=False),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            async with client.stream(**options) as response:
                self.check_response(response, operation)
                yield response

    def converse(self, **parameters):
        with self.stream("Converse", parameters) as response:
            return self.parse(response, response.read())

    async def aconverse(self, **parameters):
        async with self.astream("Converse", parameters) as response:
            return self.parse(response, await response.aread())

    def events(self, parameters):
        with self.stream("ConverseStream", parameters) as response:
            decoder = EventDecoder(self.service)
            for chunk in response.iter_bytes():
                yield from decoder.feed(chunk)
            decoder.finish()

    async def aevents(self, parameters):
        async with self.astream("ConverseStream", parameters) as response:
            decoder = EventDecoder(self.service)
            async for chunk in response.aiter_bytes():
                for event in decoder.feed(chunk):
                    yield event
            decoder.finish()

    def close(self):
        # Connections are owned by each call, including each streamed response.
        pass
