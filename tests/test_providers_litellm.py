"""Tests for LitellmProvider.

Verifies:
1. Provider instantiation and info
2. is_available() behavior
3. reason() with and without system message
4. reason() with schema (structured output)
5. reason() with model override and extra kwargs
6. API key resolution
7. Registry integration
8. Protocol compliance
9. No eager import
10. JSON parsing with markdown code blocks
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claudetube.providers.base import Reasoner
from claudetube.providers.capabilities import Capability
from claudetube.providers.litellm.client import (
    LitellmProvider,
    _get_schema_json,
)


class TestInstantiation:
    """Tests for provider instantiation."""

    def test_default_args(self):
        provider = LitellmProvider()
        assert provider._model == "gpt-4o"
        assert provider._api_key is None
        assert provider._api_base is None
        assert provider._max_tokens == 1024

    def test_custom_args(self):
        provider = LitellmProvider(
            model="anthropic/claude-sonnet-4-20250514",
            api_key="sk-test",
            api_base="https://my-proxy.com",
            max_tokens=2048,
        )
        assert provider._model == "anthropic/claude-sonnet-4-20250514"
        assert provider._api_key == "sk-test"
        assert provider._api_base == "https://my-proxy.com"
        assert provider._max_tokens == 2048


class TestInfoAndCapabilities:
    """Tests for provider info and capabilities."""

    def test_info_name(self):
        provider = LitellmProvider()
        assert provider.info.name == "litellm"

    def test_has_reason(self):
        provider = LitellmProvider()
        assert provider.info.can(Capability.REASON)

    def test_no_transcribe(self):
        provider = LitellmProvider()
        assert not provider.info.can(Capability.TRANSCRIBE)

    def test_no_vision(self):
        provider = LitellmProvider()
        assert not provider.info.can(Capability.VISION)

    def test_no_video(self):
        provider = LitellmProvider()
        assert not provider.info.can(Capability.VIDEO)

    def test_no_embed(self):
        provider = LitellmProvider()
        assert not provider.info.can(Capability.EMBED)

    def test_supports_structured_output(self):
        provider = LitellmProvider()
        assert provider.info.supports_structured_output is True

    def test_supports_streaming(self):
        provider = LitellmProvider()
        assert provider.info.supports_streaming is True


class TestIsAvailable:
    """Tests for is_available() behavior."""

    def test_available_with_sdk(self):
        with patch.dict("sys.modules", {"litellm": MagicMock()}):
            provider = LitellmProvider()
            assert provider.is_available() is True

    def test_not_available_without_sdk(self):
        with patch.dict("sys.modules", {"litellm": None}):
            provider = LitellmProvider()
            assert provider.is_available() is False


class TestApiKeyResolution:
    """Tests for API key resolution."""

    def test_explicit_key(self):
        provider = LitellmProvider(api_key="sk-explicit")
        assert provider._resolve_api_key() == "sk-explicit"

    def test_env_key(self, monkeypatch):
        monkeypatch.setenv("LITELLM_API_KEY", "sk-from-env")
        provider = LitellmProvider()
        assert provider._resolve_api_key() == "sk-from-env"

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LITELLM_API_KEY", "sk-from-env")
        provider = LitellmProvider(api_key="sk-explicit")
        assert provider._resolve_api_key() == "sk-explicit"

    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        provider = LitellmProvider()
        assert provider._resolve_api_key() is None


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_implements_reasoner_protocol(self):
        provider = LitellmProvider()
        assert isinstance(provider, Reasoner)


class TestNoEagerImport:
    """Tests that provider doesn't eagerly import litellm."""

    def test_no_eager_import(self):
        """LitellmProvider should not import litellm at init time."""
        # If litellm is not installed, instantiation should still work
        provider = LitellmProvider()
        assert provider._model == "gpt-4o"


class TestSchemaJson:
    """Tests for schema JSON extraction."""

    def test_with_pydantic_model(self):
        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        result = _get_schema_json(mock_schema)
        parsed = json.loads(result)
        assert parsed["type"] == "object"
        assert "name" in parsed["properties"]

    def test_without_pydantic(self):
        result = _get_schema_json(dict)
        assert "dict" in result


def _make_mock_litellm(response_content="result"):
    """Create a mock litellm module with acompletion returning given content."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=response_content))]
    mock_module = MagicMock()
    mock_module.acompletion = AsyncMock(return_value=mock_response)
    return mock_module


class TestReason:
    """Tests for reason()."""

    @pytest.mark.asyncio
    async def test_simple_messages(self):
        mock_litellm = _make_mock_litellm("Python is a programming language.")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            provider = LitellmProvider()
            messages = [{"role": "user", "content": "What is Python?"}]
            result = await provider.reason(messages)

        assert result == "Python is a programming language."
        mock_litellm.acompletion.assert_called_once()
        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["max_tokens"] == 1024
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_system_message_passed_through(self):
        mock_litellm = _make_mock_litellm("response")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            provider = LitellmProvider()
            messages = [
                {"role": "system", "content": "You are a video analyst."},
                {"role": "user", "content": "What happens in this video?"},
            ]
            await provider.reason(messages)

        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == "You are a video analyst."
        assert call_kwargs["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_with_schema(self):
        mock_litellm = _make_mock_litellm(
            '{"summary": "A tutorial video", "topics": ["Python"]}'
        )

        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "topics": {"type": "array", "items": {"type": "string"}},
            },
        }

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            provider = LitellmProvider()
            messages = [{"role": "user", "content": "Summarize this transcript"}]
            result = await provider.reason(messages, schema=mock_schema)

        assert isinstance(result, dict)
        assert result["summary"] == "A tutorial video"
        assert result["topics"] == ["Python"]
        # Verify schema was appended to user message
        call_kwargs = mock_litellm.acompletion.call_args[1]
        user_msg = call_kwargs["messages"][0]
        assert "Respond with JSON matching this schema" in user_msg["content"]

    @pytest.mark.asyncio
    async def test_schema_with_markdown_code_block(self):
        """Model returns JSON wrapped in markdown code block."""
        mock_litellm = _make_mock_litellm('```json\n{"summary": "test"}\n```')

        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {"type": "object"}

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            provider = LitellmProvider()
            messages = [{"role": "user", "content": "Summarize"}]
            result = await provider.reason(messages, schema=mock_schema)

        assert isinstance(result, dict)
        assert result["summary"] == "test"

    @pytest.mark.asyncio
    async def test_schema_unparseable_returns_text(self):
        """If the model doesn't return valid JSON, return raw text."""
        mock_litellm = _make_mock_litellm("I cannot produce JSON for this")

        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {"type": "object"}

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            provider = LitellmProvider()
            messages = [{"role": "user", "content": "Summarize"}]
            result = await provider.reason(messages, schema=mock_schema)

        assert result == "I cannot produce JSON for this"

    @pytest.mark.asyncio
    async def test_model_override(self):
        mock_litellm = _make_mock_litellm("result")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            provider = LitellmProvider()
            messages = [{"role": "user", "content": "Hello"}]
            await provider.reason(messages, model="anthropic/claude-sonnet-4-20250514")

        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_extra_kwargs_passed_through(self):
        """Extra kwargs should be passed to litellm.acompletion."""
        mock_litellm = _make_mock_litellm("result")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            provider = LitellmProvider()
            messages = [{"role": "user", "content": "Hello"}]
            await provider.reason(messages, temperature=0.7, top_p=0.9)

        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_api_key_passed_to_completion(self):
        mock_litellm = _make_mock_litellm("result")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            provider = LitellmProvider(api_key="sk-test-key")
            messages = [{"role": "user", "content": "Hello"}]
            await provider.reason(messages)

        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert call_kwargs["api_key"] == "sk-test-key"

    @pytest.mark.asyncio
    async def test_api_base_passed_to_completion(self):
        mock_litellm = _make_mock_litellm("result")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            provider = LitellmProvider(api_base="https://my-proxy.com")
            messages = [{"role": "user", "content": "Hello"}]
            await provider.reason(messages)

        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert call_kwargs["api_base"] == "https://my-proxy.com"

    @pytest.mark.asyncio
    async def test_no_api_key_no_kwarg(self, monkeypatch):
        """When no API key is set, api_key should not be in kwargs."""
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        mock_litellm = _make_mock_litellm("result")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            provider = LitellmProvider()
            messages = [{"role": "user", "content": "Hello"}]
            await provider.reason(messages)

        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert "api_key" not in call_kwargs


class TestTryParseJson:
    """Tests for _try_parse_json static method."""

    def test_valid_json(self):
        result = LitellmProvider._try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json_returns_text(self):
        result = LitellmProvider._try_parse_json("not json at all")
        assert result == "not json at all"

    def test_json_in_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = LitellmProvider._try_parse_json(text)
        assert result == {"key": "value"}

    def test_json_in_plain_code_block(self):
        text = '```\n{"key": "value"}\n```'
        result = LitellmProvider._try_parse_json(text)
        assert result == {"key": "value"}

    def test_invalid_json_in_code_block(self):
        text = "```\nnot json\n```"
        result = LitellmProvider._try_parse_json(text)
        assert result == "```\nnot json\n```"


class TestRegistryIntegration:
    """Tests for provider registry integration."""

    def test_get_provider_by_name(self):
        from claudetube.providers.registry import clear_cache, get_provider

        clear_cache()
        provider = get_provider("litellm")
        assert isinstance(provider, LitellmProvider)

    def test_get_provider_with_kwargs(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider(
            "litellm",
            model="anthropic/claude-sonnet-4-20250514",
            api_key="sk-test",
        )
        assert isinstance(provider, LitellmProvider)
        assert provider._model == "anthropic/claude-sonnet-4-20250514"
        assert provider._api_key == "sk-test"

    def test_alias_lite_llm(self):
        from claudetube.providers.registry import clear_cache, get_provider

        clear_cache()
        provider = get_provider("lite-llm")
        assert isinstance(provider, LitellmProvider)

    def test_alias_lite_llm_underscore(self):
        from claudetube.providers.registry import clear_cache, get_provider

        clear_cache()
        provider = get_provider("lite_llm")
        assert isinstance(provider, LitellmProvider)

    def test_package_level_get_provider(self):
        from claudetube.providers import get_provider
        from claudetube.providers.registry import clear_cache

        clear_cache()
        provider = get_provider("litellm")
        assert isinstance(provider, LitellmProvider)
