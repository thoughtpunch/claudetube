"""Tests for AnthropicProvider.

Verifies:
1. Provider instantiation and info
2. is_available() behavior
3. Image loading and media type detection
4. analyze_images() with and without schema
5. reason() with system messages and schema
6. Registry integration
7. Protocol compliance
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claudetube.providers.anthropic.client import (
    AnthropicProvider,
    _detect_media_type,
    _load_image_base64,
    _schema_to_tool,
)
from claudetube.providers.base import Reasoner, VisionAnalyzer
from claudetube.providers.capabilities import Capability


class TestMediaTypeDetection:
    """Tests for image media type detection."""

    def test_jpeg(self):
        assert _detect_media_type(Path("photo.jpg")) == "image/jpeg"

    def test_jpeg_long(self):
        assert _detect_media_type(Path("photo.jpeg")) == "image/jpeg"

    def test_png(self):
        assert _detect_media_type(Path("diagram.png")) == "image/png"

    def test_gif(self):
        assert _detect_media_type(Path("anim.gif")) == "image/gif"

    def test_webp(self):
        assert _detect_media_type(Path("modern.webp")) == "image/webp"

    def test_unknown_defaults_jpeg(self):
        assert _detect_media_type(Path("file.bmp")) == "image/jpeg"

    def test_case_insensitive(self):
        assert _detect_media_type(Path("PHOTO.JPG")) == "image/jpeg"
        assert _detect_media_type(Path("image.PNG")) == "image/png"


class TestImageLoading:
    """Tests for base64 image loading."""

    def test_loads_file(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake jpeg data")
        result = _load_image_base64(img)
        assert isinstance(result, str)
        # Verify it's valid base64
        import base64

        decoded = base64.standard_b64decode(result)
        assert decoded == b"\xff\xd8\xff\xe0fake jpeg data"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _load_image_base64(Path("/nonexistent/image.jpg"))


class TestSchemaToTool:
    """Tests for Pydantic schema to Anthropic tool conversion."""

    def test_with_pydantic_model(self):
        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        tool = _schema_to_tool(mock_schema)
        assert tool["name"] == "structured_output"
        assert tool["input_schema"]["type"] == "object"
        assert "name" in tool["input_schema"]["properties"]

    def test_without_pydantic(self):
        tool = _schema_to_tool(dict)
        assert tool["name"] == "structured_output"
        assert tool["input_schema"] == {"type": "object"}


class TestAnthropicProvider:
    """Tests for the AnthropicProvider class."""

    def test_instantiation_default(self):
        provider = AnthropicProvider()
        assert provider._model == "claude-sonnet-4-20250514"
        assert provider._max_tokens == 1024
        assert provider._client is None

    def test_instantiation_custom(self):
        provider = AnthropicProvider(
            model="claude-3-haiku-20240307",
            api_key="test-key",
            max_tokens=500,
        )
        assert provider._model == "claude-3-haiku-20240307"
        assert provider._api_key == "test-key"
        assert provider._max_tokens == 500

    def test_info(self):
        provider = AnthropicProvider()
        info = provider.info
        assert info.name == "anthropic"
        assert info.can(Capability.VISION)
        assert info.can(Capability.REASON)
        assert not info.can(Capability.TRANSCRIBE)

    def test_no_eager_import(self):
        """Provider instantiation does NOT import anthropic."""
        provider = AnthropicProvider()
        assert provider._client is None

    def test_implements_vision_protocol(self):
        provider = AnthropicProvider()
        assert isinstance(provider, VisionAnalyzer)

    def test_implements_reasoner_protocol(self):
        provider = AnthropicProvider()
        assert isinstance(provider, Reasoner)


class TestIsAvailable:
    """Tests for is_available() behavior."""

    def test_available_with_api_key_arg(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            provider = AnthropicProvider(api_key="test-key")
            assert provider.is_available() is True

    def test_available_with_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            provider = AnthropicProvider()
            assert provider.is_available() is True

    def test_not_available_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            provider = AnthropicProvider()
            assert provider.is_available() is False

    def test_not_available_without_sdk(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            provider = AnthropicProvider(api_key="test-key")
            assert provider.is_available() is False


class TestAnalyzeImages:
    """Tests for analyze_images()."""

    @pytest.mark.asyncio
    async def test_single_image(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake jpeg")

        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.text = "A person standing at a whiteboard"
        mock_response.content = [mock_text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider()
        provider._client = mock_client

        result = await provider.analyze_images([img], "Describe this frame")

        assert result == "A person standing at a whiteboard"
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        # Should have image block + text block
        content = call_kwargs["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "image"
        assert content[1]["type"] == "text"

    @pytest.mark.asyncio
    async def test_multiple_images(self, tmp_path):
        imgs = []
        for i in range(3):
            img = tmp_path / f"frame{i}.png"
            img.write_bytes(b"fake png")
            imgs.append(img)

        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.text = "Three frames showing progression"
        mock_response.content = [mock_text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider()
        provider._client = mock_client

        result = await provider.analyze_images(imgs, "Describe the progression")

        assert result == "Three frames showing progression"
        content = mock_client.messages.create.call_args[1]["messages"][0]["content"]
        # 3 images + 1 text = 4 blocks
        assert len(content) == 4
        for i in range(3):
            assert content[i]["type"] == "image"
            assert content[i]["source"]["media_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_image_not_found(self, tmp_path):
        provider = AnthropicProvider()
        provider._client = AsyncMock()

        with pytest.raises(FileNotFoundError, match="not found"):
            await provider.analyze_images(
                [tmp_path / "nonexistent.jpg"], "Describe"
            )

    @pytest.mark.asyncio
    async def test_with_schema(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake jpeg")

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.input = {"description": "A whiteboard", "objects": ["marker"]}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "objects": {"type": "array", "items": {"type": "string"}},
            },
        }

        provider = AnthropicProvider()
        provider._client = mock_client

        result = await provider.analyze_images([img], "Describe", schema=mock_schema)

        assert isinstance(result, dict)
        assert result["description"] == "A whiteboard"
        # Verify tool_choice was set
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tool_choice"] == {
            "type": "tool",
            "name": "structured_output",
        }

    @pytest.mark.asyncio
    async def test_model_override(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake jpeg")

        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.text = "result"
        mock_response.content = [mock_text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider()
        provider._client = mock_client

        await provider.analyze_images(
            [img], "Describe", model="claude-3-haiku-20240307"
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-haiku-20240307"


class TestReason:
    """Tests for reason()."""

    @pytest.mark.asyncio
    async def test_simple_messages(self):
        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.text = "Python is a programming language."
        mock_response.content = [mock_text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider()
        provider._client = mock_client

        messages = [{"role": "user", "content": "What is Python?"}]
        result = await provider.reason(messages)

        assert result == "Python is a programming language."

    @pytest.mark.asyncio
    async def test_system_message_extracted(self):
        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.text = "response"
        mock_response.content = [mock_text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider()
        provider._client = mock_client

        messages = [
            {"role": "system", "content": "You are a video analyst."},
            {"role": "user", "content": "What happens in this video?"},
        ]
        await provider.reason(messages)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a video analyst."
        # System message should NOT be in messages list
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_no_system_message(self):
        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.text = "response"
        mock_response.content = [mock_text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider()
        provider._client = mock_client

        messages = [{"role": "user", "content": "Hello"}]
        await provider.reason(messages)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    @pytest.mark.asyncio
    async def test_with_schema(self):
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.input = {"summary": "A tutorial video", "topics": ["Python"]}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "topics": {"type": "array", "items": {"type": "string"}},
            },
        }

        provider = AnthropicProvider()
        provider._client = mock_client

        messages = [{"role": "user", "content": "Summarize this transcript"}]
        result = await provider.reason(messages, schema=mock_schema)

        assert isinstance(result, dict)
        assert result["summary"] == "A tutorial video"
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == {
            "type": "tool",
            "name": "structured_output",
        }


class TestExtractToolResult:
    """Tests for _extract_tool_result()."""

    def test_extracts_from_tool_use_block(self):
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.input = {"key": "value"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_block]

        result = AnthropicProvider._extract_tool_result(mock_response)
        assert result == {"key": "value"}

    def test_falls_back_to_json_text(self):
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = '{"key": "value"}'

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        result = AnthropicProvider._extract_tool_result(mock_response)
        assert result == {"key": "value"}

    def test_raises_on_no_structured_output(self):
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Just a regular response"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        with pytest.raises(ValueError, match="No structured output"):
            AnthropicProvider._extract_tool_result(mock_response)

    def test_prefers_tool_use_over_text(self):
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = '{"wrong": true}'

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.input = {"correct": True}

        mock_response = MagicMock()
        mock_response.content = [mock_text_block, mock_tool_block]

        result = AnthropicProvider._extract_tool_result(mock_response)
        assert result == {"correct": True}


class TestRegistryIntegration:
    """Tests for provider registry integration."""

    def test_get_provider_by_name(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)

    def test_get_provider_by_alias_claude(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("claude")
        assert isinstance(provider, AnthropicProvider)

    def test_get_provider_by_alias_claude_sonnet(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("claude-sonnet")
        assert isinstance(provider, AnthropicProvider)

    def test_get_provider_with_kwargs(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider(
            "anthropic", model="claude-3-haiku-20240307", max_tokens=500
        )
        assert isinstance(provider, AnthropicProvider)
        assert provider._model == "claude-3-haiku-20240307"
        assert provider._max_tokens == 500

    def test_package_level_get_provider(self):
        from claudetube.providers import get_provider

        provider = get_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)
