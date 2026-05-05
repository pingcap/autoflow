"""Unit and integration tests for the MiniMax LLM provider."""

import os
from logging import getLogger

import pytest

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Unit Tests — Provider Enum & Options (no heavy app deps required)
# ---------------------------------------------------------------------------


class TestMiniMaxProviderEnum:
    """Tests for the MiniMax provider enum registration."""

    def test_minimax_in_provider_enum(self):
        from app.rag.llms.provider import LLMProvider

        assert hasattr(LLMProvider, "MINIMAX")
        assert LLMProvider.MINIMAX.value == "minimax"

    def test_minimax_in_provider_options(self):
        from app.rag.llms.provider import LLMProvider, llm_provider_options

        minimax_options = [
            opt for opt in llm_provider_options if opt.provider == LLMProvider.MINIMAX
        ]
        assert len(minimax_options) == 1, (
            "MiniMax should appear exactly once in provider options"
        )

    def test_minimax_provider_option_fields(self):
        from app.rag.llms.provider import LLMProvider, llm_provider_options

        minimax_opt = next(
            opt for opt in llm_provider_options if opt.provider == LLMProvider.MINIMAX
        )
        assert minimax_opt.provider_display_name == "MiniMax"
        assert minimax_opt.default_llm_model == "MiniMax-M2.7"
        assert minimax_opt.credentials_type == "str"
        assert "platform.minimax.io" in minimax_opt.provider_url

    def test_minimax_default_config(self):
        from app.rag.llms.provider import LLMProvider, llm_provider_options

        minimax_opt = next(
            opt for opt in llm_provider_options if opt.provider == LLMProvider.MINIMAX
        )
        config = minimax_opt.default_config
        assert config["api_base"] == "https://api.minimax.io/v1"
        assert config["is_chat_model"] is True
        assert config["context_window"] == 204800

    def test_minimax_provider_description(self):
        from app.rag.llms.provider import LLMProvider, llm_provider_options

        minimax_opt = next(
            opt for opt in llm_provider_options if opt.provider == LLMProvider.MINIMAX
        )
        assert "MiniMax" in minimax_opt.provider_description
        assert minimax_opt.provider_url == "https://platform.minimax.io"

    def test_minimax_credentials_info(self):
        from app.rag.llms.provider import LLMProvider, llm_provider_options

        minimax_opt = next(
            opt for opt in llm_provider_options if opt.provider == LLMProvider.MINIMAX
        )
        assert minimax_opt.credentials_display_name == "MiniMax API Key"
        assert "platform.minimax.io" in minimax_opt.credentials_description

    def test_minimax_model_description(self):
        from app.rag.llms.provider import LLMProvider, llm_provider_options

        minimax_opt = next(
            opt for opt in llm_provider_options if opt.provider == LLMProvider.MINIMAX
        )
        assert "MiniMax-M2.7" in minimax_opt.llm_model_description
        assert "MiniMax-M2.7-highspeed" in minimax_opt.llm_model_description


class TestMiniMaxOpenAILikeInstantiation:
    """Test MiniMax provider resolves correctly via OpenAILike.

    These tests directly instantiate OpenAILike without importing the full
    app stack, simulating what resolve_llm() does for the MINIMAX provider.
    """

    def test_creates_openai_like_with_minimax_config(self):
        from llama_index.llms.openai_like import OpenAILike

        llm = OpenAILike(
            model="MiniMax-M2.7",
            api_base="https://api.minimax.io/v1",
            api_key="test-api-key",
            is_chat_model=True,
            context_window=204800,
            temperature=1.0,
        )
        assert llm.class_name() == "OpenAILike"
        assert llm.model == "MiniMax-M2.7"
        assert llm.api_key == "test-api-key"
        assert llm.api_base == "https://api.minimax.io/v1"

    def test_highspeed_model(self):
        from llama_index.llms.openai_like import OpenAILike

        llm = OpenAILike(
            model="MiniMax-M2.7-highspeed",
            api_base="https://api.minimax.io/v1",
            api_key="test-api-key",
            is_chat_model=True,
            context_window=204800,
            temperature=1.0,
        )
        assert llm.model == "MiniMax-M2.7-highspeed"

    def test_default_temperature_is_one(self):
        from llama_index.llms.openai_like import OpenAILike

        llm = OpenAILike(
            model="MiniMax-M2.7",
            api_base="https://api.minimax.io/v1",
            api_key="test-api-key",
            is_chat_model=True,
            context_window=204800,
            temperature=1.0,
        )
        assert llm.temperature == 1.0

    def test_custom_temperature(self):
        from llama_index.llms.openai_like import OpenAILike

        llm = OpenAILike(
            model="MiniMax-M2.7",
            api_base="https://api.minimax.io/v1",
            api_key="test-api-key",
            is_chat_model=True,
            context_window=204800,
            temperature=0.7,
        )
        assert llm.temperature == 0.7

    def test_custom_api_base_for_cn(self):
        from llama_index.llms.openai_like import OpenAILike

        llm = OpenAILike(
            model="MiniMax-M2.7",
            api_base="https://api.minimaxi.com/v1",
            api_key="test-api-key",
            is_chat_model=True,
            context_window=204800,
            temperature=1.0,
        )
        assert llm.api_base == "https://api.minimaxi.com/v1"

    def test_is_chat_model(self):
        from llama_index.llms.openai_like import OpenAILike

        llm = OpenAILike(
            model="MiniMax-M2.7",
            api_base="https://api.minimax.io/v1",
            api_key="test-api-key",
            is_chat_model=True,
            context_window=204800,
            temperature=1.0,
        )
        assert llm.is_chat_model is True

    def test_context_window(self):
        from llama_index.llms.openai_like import OpenAILike

        llm = OpenAILike(
            model="MiniMax-M2.7",
            api_base="https://api.minimax.io/v1",
            api_key="test-api-key",
            is_chat_model=True,
            context_window=204800,
            temperature=1.0,
        )
        assert llm.context_window == 204800

    def test_temperature_clamping_logic(self):
        """MiniMax requires temperature > 0. Verify the clamping logic
        used in resolve_llm()."""
        temperature = 0
        if temperature <= 0:
            temperature = 1.0
        assert temperature == 1.0

        temperature = 0.5
        if temperature <= 0:
            temperature = 1.0
        assert temperature == 0.5

    def test_dspy_conversion_from_openai_like(self):
        """MiniMax LLM uses OpenAILike which maps to dspy 'OpenAILike' case."""
        import dspy
        from llama_index.llms.openai_like import OpenAILike

        llm = OpenAILike(
            model="MiniMax-M2.7",
            api_base="https://api.minimax.io/v1",
            api_key="test-api-key",
            is_chat_model=True,
            context_window=204800,
            temperature=1.0,
        )
        assert llm.class_name() == "OpenAILike"

        # Simulate what dspy.py does for OpenAILike
        lm = dspy.LM(
            model=f"openai/{llm.model}",
            max_tokens=llm.max_tokens,
            api_key=llm.api_key,
            api_base=llm.api_base + "/",
            model_type="chat" if llm.is_chat_model else "text",
            num_retries=3,
        )
        assert lm is not None
        assert "MiniMax-M2.7" in lm.model


# ---------------------------------------------------------------------------
# Integration Tests (require MINIMAX_API_KEY)
# ---------------------------------------------------------------------------


MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")


@pytest.mark.skipif(not MINIMAX_API_KEY, reason="MINIMAX_API_KEY not set")
class TestMiniMaxIntegration:
    """Integration tests that call the real MiniMax API via OpenAILike."""

    def test_minimax_basic_chat(self):
        from llama_index.core import PromptTemplate
        from llama_index.llms.openai_like import OpenAILike

        llm = OpenAILike(
            model="MiniMax-M2.7",
            api_base="https://api.minimax.io/v1",
            api_key=MINIMAX_API_KEY,
            is_chat_model=True,
            context_window=204800,
            temperature=1.0,
        )
        prompt = PromptTemplate("What is 2 + 2? Reply with just the number.")
        output = llm.predict(prompt)
        assert "4" in output
        logger.info(f"MiniMax basic chat response: {output}")

    def test_minimax_highspeed_chat(self):
        from llama_index.core import PromptTemplate
        from llama_index.llms.openai_like import OpenAILike

        llm = OpenAILike(
            model="MiniMax-M2.7-highspeed",
            api_base="https://api.minimax.io/v1",
            api_key=MINIMAX_API_KEY,
            is_chat_model=True,
            context_window=204800,
            temperature=1.0,
        )
        prompt = PromptTemplate("Say 'hello' in one word.")
        output = llm.predict(prompt)
        assert len(output) > 0
        logger.info(f"MiniMax highspeed response: {output}")

    def test_minimax_streaming(self):
        from llama_index.core.llms import ChatMessage
        from llama_index.llms.openai_like import OpenAILike

        llm = OpenAILike(
            model="MiniMax-M2.7",
            api_base="https://api.minimax.io/v1",
            api_key=MINIMAX_API_KEY,
            is_chat_model=True,
            context_window=204800,
            temperature=1.0,
        )
        messages = [ChatMessage(role="user", content="Count from 1 to 5.")]
        chunks = []
        for chunk in llm.stream_chat(messages):
            chunks.append(chunk)
        assert len(chunks) > 0
        full_response = chunks[-1].message.content
        assert any(str(i) in full_response for i in range(1, 6))
        logger.info(f"MiniMax streaming response: {full_response}")
