"""Converse integration with native async I/O and explicit stream ownership."""

from copy import deepcopy

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ImageBlock,
    DocumentBlock,
)
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.llms.bedrock_converse.base import _parse_tool_input
from llama_index.llms.bedrock_converse.utils import (
    converse_with_retry,
    messages_to_converse_messages,
)

from app.security.bedrock_http import BedrockHTTP, ExplicitSession
from app.security.provider_config import ProviderConfigurationError


class _Parameters:
    def __init__(self, exceptions):
        self.exceptions = exceptions

    def converse(self, **kwargs):
        return kwargs

    converse_stream = converse


class _StreamContent:
    """Retain LlamaIndex's cumulative text, thinking, tools and usage shape."""

    def __init__(self, llm):
        self.llm = llm
        self.text = ""
        self.thinking = ""
        self.signature = ""
        self.tools = {}

    def convert(self, event):
        extra = {}
        delta_text = ""
        if delta := event.get("contentBlockDelta"):
            content = delta["delta"]
            delta_text = content.get("text", "")
            self.text += delta_text
            if reasoning := content.get("reasoningContent"):
                self.thinking += reasoning.get("text", "")
                self.signature += reasoning.get("signature", "")
                extra["thinking_delta"] = reasoning.get("text", "")
            if tool := content.get("toolUse"):
                current = self.tools[delta["contentBlockIndex"]]
                current["input"] = current.get("input", "") + tool.get("input", "")
        elif start := event.get("contentBlockStart"):
            if tool := start["start"].get("toolUse"):
                self.tools[start["contentBlockIndex"]] = dict(tool)
        elif metadata := event.get("metadata"):
            extra = self.llm._get_response_token_counts(metadata)
        else:
            return None
        blocks = [TextBlock(text=self.text)]
        if self.thinking:
            blocks.insert(
                0,
                ThinkingBlock(
                    content=self.thinking,
                    additional_information={"signature": self.signature},
                ),
            )
        for tool in self.tools.values():
            blocks.append(
                ToolCallBlock(
                    tool_name=tool.get("name", ""),
                    tool_call_id=tool.get("toolUseId"),
                    tool_kwargs=_parse_tool_input(tool.get("input", {})),
                )
            )
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                blocks=blocks,
                additional_kwargs={
                    "tool_call_id": [
                        t.get("toolUseId", "") for t in self.tools.values()
                    ],
                    "status": [],
                },
            ),
            delta=delta_text,
            raw=event,
            additional_kwargs=extra,
        )


class BoundBedrockConverse(BedrockConverse):
    def __init__(
        self,
        *,
        model,
        region_name,
        aws_access_key_id,
        aws_secret_access_key,
        timeout=60,
        max_tokens=4096,
        **kwargs,
    ):
        client = BedrockHTTP(
            model,
            region_name,
            aws_access_key_id,
            aws_secret_access_key,
            timeout,
            max_tokens,
        )
        session = ExplicitSession()
        session.set_credentials(aws_access_key_id, aws_secret_access_key)
        session.set_config_variable("region", region_name)
        super().__init__(
            model=model,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            timeout=timeout,
            max_tokens=max_tokens,
            client=client,
            botocore_session=session,
            endpoint_url=client.endpoint,
            **kwargs,
        )

    def _parameters(self, messages, stream, kwargs):
        # SDK message conversion resolves paths and remote URLs eagerly, outside
        # the model connection. Inline media remains supported and body bounded.
        for message in messages:
            for block in message.blocks:
                if isinstance(block, (ImageBlock, DocumentBlock)) and (
                    getattr(block, "url", None) or getattr(block, "path", None)
                ):
                    raise ProviderConfigurationError(
                        "Bedrock media must be supplied inline"
                    )
        converted, system = messages_to_converse_messages(messages, self.model)
        options = deepcopy(self._get_all_kwargs(**kwargs))
        if self.thinking is not None:
            options["thinking"] = self.thinking
        # Reuse the installed integration's pure request conversion, including
        # tools, system caching and guardrails. The collector cannot do I/O.
        return converse_with_retry(
            client=_Parameters(self._client.exceptions),
            messages=converted,
            system_prompt=system,
            system_prompt_caching=self.system_prompt_caching,
            tool_caching=self.tool_caching,
            max_retries=1,
            stream=stream,
            guardrail_identifier=self.guardrail_identifier,
            guardrail_version=self.guardrail_version,
            guardrail_stream_processing_mode=self.guardrail_stream_processing_mode,
            trace=self.trace,
            **options,
        )

    def _response(self, response):
        blocks, tool_ids, status = self._get_content_and_tool_calls(response)
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                blocks=blocks,
                additional_kwargs={"tool_call_id": tool_ids, "status": status},
            ),
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    @llm_chat_callback()
    def chat(self, messages, **kwargs):
        return self._response(
            self._client.converse(**self._parameters(messages, False, kwargs))
        )

    @llm_chat_callback()
    async def achat(self, messages, **kwargs):
        return self._response(
            await self._client.aconverse(**self._parameters(messages, False, kwargs))
        )

    def _stream_responses(self, messages, events):
        content = _StreamContent(self)
        for event in events:
            if (response := content.convert(event)) is not None:
                yield response

    async def _astream_responses(self, messages, events):
        async def responses():
            content = _StreamContent(self)
            async for event in events:
                if (response := content.convert(event)) is not None:
                    yield response

        return responses()

    def stream_chat(self, messages, **kwargs):
        events = self._client.events(self._parameters(messages, True, kwargs))

        @llm_chat_callback()
        def observed(_self, messages):
            return self._stream_responses(messages, events)

        responses = observed(self, messages)
        try:
            yield from responses
        finally:
            responses.close()
            events.close()

    async def astream_chat(self, messages, **kwargs):
        events = self._client.aevents(self._parameters(messages, True, kwargs))

        @llm_chat_callback()
        async def observed(_self, messages):
            return await self._astream_responses(messages, events)

        responses = await observed(self, messages)

        async def owned():
            try:
                async for response in responses:
                    yield response
            finally:
                await responses.aclose()
                await events.aclose()

        return owned()

    @staticmethod
    def _completion(response):
        return CompletionResponse(
            text=response.message.content or "",
            delta=response.delta,
            raw=response.raw,
            additional_kwargs={
                **response.message.additional_kwargs,
                **response.additional_kwargs,
            },
        )

    def stream_complete(self, prompt, formatted=False, **kwargs):
        responses = self.stream_chat(
            [ChatMessage(role=MessageRole.USER, content=prompt)], **kwargs
        )
        try:
            for response in responses:
                yield self._completion(response)
        finally:
            responses.close()

    async def astream_complete(self, prompt, formatted=False, **kwargs):
        responses = await self.astream_chat(
            [ChatMessage(role=MessageRole.USER, content=prompt)], **kwargs
        )

        async def owned():
            try:
                async for response in responses:
                    yield self._completion(response)
            finally:
                await responses.aclose()

        return owned()
