"""Tests for OllamaProvider.

Verifies:
1. Provider instantiation and info
2. is_available() behavior
3. analyze_images() with mocked client (single/multiple images, LLaVA limitation)
4. analyze_images() with schema (structured output)
5. reason() with and without system message
6. reason() with schema
7. Registry integration
8. Protocol compliance
9. No eager import
10. File not found for images
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claudetube.providers.base import Reasoner, VisionAnalyzer
from claudetube.providers.capabilities import Capability
from claudetube.providers.ollama.client import (
    OllamaProvider,
    _get_schema_json,
    _load_image_base64,
)


class TestInstantiation:
    """Tests for provider instantiation."""

    def test_default_args(self):
        provider = OllamaProvider()
        assert provider._vision_model == "llava:13b"
        assert provider._reason_model == "llama3.2"
        assert provider._host == "http://localhost:11434"
        assert provider._client is None

    def test_custom_args(self):
        provider = OllamaProvider(
            vision_model="llava:7b",
            reason_model="mistral",
            host="http://myserver:11434",
        )
        assert provider._vision_model == "llava:7b"
        assert provider._reason_model == "mistral"
        assert provider._host == "http://myserver:11434"

    def test_host_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://env-host:11434")
        provider = OllamaProvider()
        assert provider._host == "http://env-host:11434"

    def test_host_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://env-host:11434")
        provider = OllamaProvider(host="http://explicit:11434")
        assert provider._host == "http://explicit:11434"


class TestInfoAndCapabilities:
    """Tests for provider info and capabilities."""

    def test_info_name(self):
        provider = OllamaProvider()
        assert provider.info.name == "ollama"

    def test_has_vision(self):
        provider = OllamaProvider()
        assert provider.info.can(Capability.VISION)

    def test_has_reason(self):
        provider = OllamaProvider()
        assert provider.info.can(Capability.REASON)

    def test_no_transcribe(self):
        provider = OllamaProvider()
        assert not provider.info.can(Capability.TRANSCRIBE)

    def test_no_video(self):
        provider = OllamaProvider()
        assert not provider.info.can(Capability.VIDEO)

    def test_no_embed(self):
        provider = OllamaProvider()
        assert not provider.info.can(Capability.EMBED)

    def test_max_images_is_one(self):
        provider = OllamaProvider()
        assert provider.info.max_images_per_request == 1

    def test_free_cost(self):
        provider = OllamaProvider()
        assert provider.info.cost_per_1m_input_tokens == 0
        assert provider.info.cost_per_1m_output_tokens == 0


class TestIsAvailable:
    """Tests for is_available() behavior."""

    def test_available_when_sdk_and_server_up(self):
        with patch.dict("sys.modules", {"ollama": MagicMock()}):
            provider = OllamaProvider()
            with patch.object(provider, "_ping_server", return_value=True):
                assert provider.is_available() is True

    def test_not_available_without_sdk(self):
        with patch.dict("sys.modules", {"ollama": None}):
            provider = OllamaProvider()
            assert provider.is_available() is False

    def test_not_available_when_server_down(self):
        with patch.dict("sys.modules", {"ollama": MagicMock()}):
            provider = OllamaProvider()
            with patch.object(provider, "_ping_server", return_value=False):
                assert provider.is_available() is False

    def test_caches_health_check_result(self):
        with patch.dict("sys.modules", {"ollama": MagicMock()}):
            provider = OllamaProvider()
            with patch.object(provider, "_ping_server", return_value=True) as mock_ping:
                assert provider.is_available() is True
                assert provider.is_available() is True
                # Should only ping once due to caching
                mock_ping.assert_called_once()

    def test_cache_expires(self):
        with patch.dict("sys.modules", {"ollama": MagicMock()}):
            provider = OllamaProvider()
            call_count = 0

            def counting_ping():
                nonlocal call_count
                call_count += 1
                return True

            monotonic_values = iter([0.0, 10.0, 50.0])

            with (
                patch.object(provider, "_ping_server", side_effect=counting_ping),
                patch(
                    "claudetube.providers.ollama.client.time.monotonic",
                    side_effect=monotonic_values,
                ),
            ):
                provider.is_available()  # now=0.0, no cache → ping
                assert call_count == 1
                provider.is_available()  # now=10.0, 10-0=10 < 30 → cached
                assert call_count == 1
                provider.is_available()  # now=50.0, 50-0=50 >= 30 → ping again
                assert call_count == 2


class TestPingServer:
    """Tests for _ping_server() HTTP health check."""

    def test_returns_true_on_200(self):
        provider = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch(
            "claudetube.providers.ollama.client.urllib.request.urlopen",
            return_value=mock_resp,
        ):
            assert provider._ping_server() is True

    def test_returns_false_on_connection_error(self):
        import urllib.error

        provider = OllamaProvider()
        with patch(
            "claudetube.providers.ollama.client.urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            assert provider._ping_server() is False

    def test_returns_false_on_timeout(self):
        provider = OllamaProvider()
        with patch(
            "claudetube.providers.ollama.client.urllib.request.urlopen",
            side_effect=OSError("timed out"),
        ):
            assert provider._ping_server() is False

    def test_uses_configured_host(self):
        provider = OllamaProvider(host="http://myserver:9999")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch(
            "claudetube.providers.ollama.client.urllib.request.urlopen",
            return_value=mock_resp,
        ) as mock_open:
            provider._ping_server()
            # Verify the URL includes the custom host
            req = mock_open.call_args[0][0]
            assert req.full_url == "http://myserver:9999/api/tags"


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_implements_vision_protocol(self):
        provider = OllamaProvider()
        assert isinstance(provider, VisionAnalyzer)

    def test_implements_reasoner_protocol(self):
        provider = OllamaProvider()
        assert isinstance(provider, Reasoner)


class TestNoEagerImport:
    """Tests that provider doesn't eagerly import ollama."""

    def test_no_client_on_init(self):
        provider = OllamaProvider()
        assert provider._client is None


class TestImageLoading:
    """Tests for base64 image loading."""

    def test_loads_file(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake jpeg data")
        result = _load_image_base64(img)
        assert isinstance(result, str)
        import base64

        decoded = base64.b64decode(result)
        assert decoded == b"\xff\xd8\xff\xe0fake jpeg data"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _load_image_base64(Path("/nonexistent/image.jpg"))


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


class TestAnalyzeImages:
    """Tests for analyze_images()."""

    @pytest.mark.asyncio
    async def test_single_image(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake jpeg")

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"message": {"content": "A person at a whiteboard"}}
        )

        provider = OllamaProvider()
        provider._client = mock_client

        result = await provider.analyze_images([img], "Describe this frame")

        assert result == "A person at a whiteboard"
        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["model"] == "llava:13b"
        assert len(call_kwargs["messages"]) == 1
        msg = call_kwargs["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == "Describe this frame"
        assert len(msg["images"]) == 1

    @pytest.mark.asyncio
    async def test_multiple_images_llava_limitation(self, tmp_path):
        """LLaVA only supports 1 image - prompt should mention extra images."""
        imgs = []
        for i in range(3):
            img = tmp_path / f"frame{i}.png"
            img.write_bytes(b"fake png")
            imgs.append(img)

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"message": {"content": "Analysis of first frame"}}
        )

        provider = OllamaProvider()
        provider._client = mock_client

        result = await provider.analyze_images(imgs, "Describe the frames")

        assert result == "Analysis of first frame"
        call_kwargs = mock_client.chat.call_args[1]
        msg = call_kwargs["messages"][0]
        # Only 1 image should be sent
        assert len(msg["images"]) == 1
        # Prompt should mention additional images
        assert "2 additional image(s)" in msg["content"]
        assert "frame1.png" in msg["content"]
        assert "frame2.png" in msg["content"]

    @pytest.mark.asyncio
    async def test_image_not_found(self, tmp_path):
        provider = OllamaProvider()
        provider._client = AsyncMock()

        with pytest.raises(FileNotFoundError, match="not found"):
            await provider.analyze_images([tmp_path / "nonexistent.jpg"], "Describe")

    @pytest.mark.asyncio
    async def test_empty_images_raises(self):
        provider = OllamaProvider()
        provider._client = AsyncMock()

        with pytest.raises(ValueError, match="At least one image"):
            await provider.analyze_images([], "Describe")

    @pytest.mark.asyncio
    async def test_with_schema(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake jpeg")

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={
                "message": {
                    "content": '{"description": "A whiteboard", "objects": ["marker"]}'
                }
            }
        )

        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "objects": {"type": "array", "items": {"type": "string"}},
            },
        }

        provider = OllamaProvider()
        provider._client = mock_client

        result = await provider.analyze_images([img], "Describe", schema=mock_schema)

        assert isinstance(result, dict)
        assert result["description"] == "A whiteboard"
        assert result["objects"] == ["marker"]
        # Verify schema was appended to prompt
        call_kwargs = mock_client.chat.call_args[1]
        msg = call_kwargs["messages"][0]
        assert "Respond with JSON matching this schema" in msg["content"]

    @pytest.mark.asyncio
    async def test_schema_unparseable_returns_text(self, tmp_path):
        """If the model doesn't return valid JSON, return raw text."""
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake jpeg")

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"message": {"content": "I cannot produce JSON for this"}}
        )

        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {"type": "object"}

        provider = OllamaProvider()
        provider._client = mock_client

        result = await provider.analyze_images([img], "Describe", schema=mock_schema)

        # Should return raw text since JSON parsing failed
        assert result == "I cannot produce JSON for this"

    @pytest.mark.asyncio
    async def test_model_override(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake jpeg")

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={"message": {"content": "result"}})

        provider = OllamaProvider()
        provider._client = mock_client

        await provider.analyze_images([img], "Describe", model="llava:7b")

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["model"] == "llava:7b"


class TestReason:
    """Tests for reason()."""

    @pytest.mark.asyncio
    async def test_simple_messages(self):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"message": {"content": "Python is a programming language."}}
        )

        provider = OllamaProvider()
        provider._client = mock_client

        messages = [{"role": "user", "content": "What is Python?"}]
        result = await provider.reason(messages)

        assert result == "Python is a programming language."
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["model"] == "llama3.2"
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_system_message_passed_through(self):
        """Ollama supports system messages directly."""
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={"message": {"content": "response"}})

        provider = OllamaProvider()
        provider._client = mock_client

        messages = [
            {"role": "system", "content": "You are a video analyst."},
            {"role": "user", "content": "What happens in this video?"},
        ]
        await provider.reason(messages)

        call_kwargs = mock_client.chat.call_args[1]
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == "You are a video analyst."
        assert call_kwargs["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_no_system_message(self):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={"message": {"content": "response"}})

        provider = OllamaProvider()
        provider._client = mock_client

        messages = [{"role": "user", "content": "Hello"}]
        await provider.reason(messages)

        call_kwargs = mock_client.chat.call_args[1]
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_with_schema(self):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={
                "message": {
                    "content": '{"summary": "A tutorial video", "topics": ["Python"]}'
                }
            }
        )

        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "topics": {"type": "array", "items": {"type": "string"}},
            },
        }

        provider = OllamaProvider()
        provider._client = mock_client

        messages = [{"role": "user", "content": "Summarize this transcript"}]
        result = await provider.reason(messages, schema=mock_schema)

        assert isinstance(result, dict)
        assert result["summary"] == "A tutorial video"
        assert result["topics"] == ["Python"]
        # Verify schema was appended to user message
        call_kwargs = mock_client.chat.call_args[1]
        user_msg = call_kwargs["messages"][0]
        assert "Respond with JSON matching this schema" in user_msg["content"]

    @pytest.mark.asyncio
    async def test_schema_with_markdown_code_block(self):
        """Model returns JSON wrapped in markdown code block."""
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"message": {"content": '```json\n{"summary": "test"}\n```'}}
        )

        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {"type": "object"}

        provider = OllamaProvider()
        provider._client = mock_client

        messages = [{"role": "user", "content": "Summarize"}]
        result = await provider.reason(messages, schema=mock_schema)

        assert isinstance(result, dict)
        assert result["summary"] == "test"

    @pytest.mark.asyncio
    async def test_model_override(self):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={"message": {"content": "result"}})

        provider = OllamaProvider()
        provider._client = mock_client

        messages = [{"role": "user", "content": "Hello"}]
        await provider.reason(messages, model="mistral")

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["model"] == "mistral"


class TestTryParseJson:
    """Tests for _try_parse_json static method."""

    def test_valid_json(self):
        result = OllamaProvider._try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json_returns_text(self):
        result = OllamaProvider._try_parse_json("not json at all")
        assert result == "not json at all"

    def test_json_in_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = OllamaProvider._try_parse_json(text)
        assert result == {"key": "value"}

    def test_json_in_plain_code_block(self):
        text = '```\n{"key": "value"}\n```'
        result = OllamaProvider._try_parse_json(text)
        assert result == {"key": "value"}

    def test_invalid_json_in_code_block(self):
        text = "```\nnot json\n```"
        result = OllamaProvider._try_parse_json(text)
        assert result == "```\nnot json\n```"


class TestRegistryIntegration:
    """Tests for provider registry integration."""

    def test_get_provider_by_name(self):
        from claudetube.providers.registry import clear_cache, get_provider

        clear_cache()
        provider = get_provider("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_get_provider_with_kwargs(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider(
            "ollama",
            vision_model="llava:7b",
            reason_model="mistral",
        )
        assert isinstance(provider, OllamaProvider)
        assert provider._vision_model == "llava:7b"
        assert provider._reason_model == "mistral"

    def test_package_level_get_provider(self):
        from claudetube.providers import get_provider
        from claudetube.providers.registry import clear_cache

        clear_cache()
        provider = get_provider("ollama")
        assert isinstance(provider, OllamaProvider)
