"""Tests for GoogleProvider.

Verifies:
1. Provider instantiation and info
2. is_available() behavior
3. analyze_images() with PIL images and with/without schema
4. analyze_video() with File API upload and time ranges
5. reason() with system messages and multi-turn chat
6. Structured output via response_schema
7. Registry integration
8. Protocol compliance
9. Time formatting
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from claudetube.providers.base import Reasoner, VideoAnalyzer, VisionAnalyzer
from claudetube.providers.capabilities import Capability
from claudetube.providers.google.client import GoogleProvider

if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Provider Instantiation and Info
# =============================================================================


class TestGoogleProvider:
    """Tests for the GoogleProvider class."""

    def test_instantiation_default(self):
        provider = GoogleProvider()
        assert provider._model == "gemini-2.0-flash"
        assert provider._max_tokens == 1024
        assert provider._genai is None

    def test_instantiation_custom(self):
        provider = GoogleProvider(
            model="gemini-2.0-flash-lite",
            api_key="test-key",
            max_tokens=500,
        )
        assert provider._model == "gemini-2.0-flash-lite"
        assert provider._api_key == "test-key"
        assert provider._max_tokens == 500

    def test_info(self):
        provider = GoogleProvider()
        info = provider.info
        assert info.name == "google"
        assert info.can(Capability.VISION)
        assert info.can(Capability.VIDEO)
        assert info.can(Capability.REASON)
        assert not info.can(Capability.TRANSCRIBE)

    def test_no_eager_import(self):
        """Provider instantiation does NOT import google.generativeai."""
        provider = GoogleProvider()
        assert provider._genai is None

    def test_implements_vision_protocol(self):
        provider = GoogleProvider()
        assert isinstance(provider, VisionAnalyzer)

    def test_implements_video_protocol(self):
        provider = GoogleProvider()
        assert isinstance(provider, VideoAnalyzer)

    def test_implements_reasoner_protocol(self):
        provider = GoogleProvider()
        assert isinstance(provider, Reasoner)


# =============================================================================
# is_available()
# =============================================================================


class TestIsAvailable:
    """Tests for is_available() behavior."""

    def test_available_with_api_key_arg(self):
        mock_genai = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
            provider = GoogleProvider(api_key="test-key")
            assert provider.is_available() is True

    def test_available_with_env_var(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        mock_genai = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
            provider = GoogleProvider()
            assert provider.is_available() is True

    def test_not_available_without_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        mock_genai = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
            provider = GoogleProvider()
            assert provider.is_available() is False

    def test_not_available_without_sdk(self):
        with patch.dict("sys.modules", {"google": None, "google.generativeai": None}):
            provider = GoogleProvider(api_key="test-key")
            assert provider.is_available() is False


# =============================================================================
# Time Formatting
# =============================================================================


class TestFormatTime:
    """Tests for _format_time()."""

    def test_zero(self):
        assert GoogleProvider._format_time(0) == "00:00"

    def test_seconds_only(self):
        assert GoogleProvider._format_time(45) == "00:45"

    def test_minutes_and_seconds(self):
        assert GoogleProvider._format_time(150) == "02:30"

    def test_large_value(self):
        assert GoogleProvider._format_time(3661) == "61:01"

    def test_float_truncation(self):
        assert GoogleProvider._format_time(90.7) == "01:30"


# =============================================================================
# analyze_images()
# =============================================================================


class TestAnalyzeImages:
    """Tests for analyze_images()."""

    @pytest.mark.asyncio
    async def test_single_image(self, tmp_path):
        # Create a minimal valid PNG (1x1 pixel)
        img = tmp_path / "frame.png"
        _write_minimal_png(img)

        mock_response = MagicMock()
        mock_response.text = "A person at a whiteboard"

        mock_model = MagicMock()
        mock_model.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        result = await provider.analyze_images([img], "Describe this frame")

        assert result == "A person at a whiteboard"
        mock_model.generate_content.assert_called_once()
        call_args = mock_model.generate_content.call_args[0][0]
        # Should have PIL image + prompt string
        assert len(call_args) == 2
        assert call_args[1] == "Describe this frame"

    @pytest.mark.asyncio
    async def test_multiple_images(self, tmp_path):
        imgs = []
        for i in range(3):
            img = tmp_path / f"frame{i}.png"
            _write_minimal_png(img)
            imgs.append(img)

        mock_response = MagicMock()
        mock_response.text = "Three frames showing progression"

        mock_model = MagicMock()
        mock_model.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        result = await provider.analyze_images(imgs, "Describe the progression")

        assert result == "Three frames showing progression"
        call_args = mock_model.generate_content.call_args[0][0]
        # 3 images + 1 prompt = 4 items
        assert len(call_args) == 4
        assert call_args[3] == "Describe the progression"

    @pytest.mark.asyncio
    async def test_image_not_found(self, tmp_path):
        mock_genai = MagicMock()
        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        with pytest.raises(FileNotFoundError, match="not found"):
            await provider.analyze_images(
                [tmp_path / "nonexistent.jpg"], "Describe"
            )

    @pytest.mark.asyncio
    async def test_with_schema(self, tmp_path):
        img = tmp_path / "frame.png"
        _write_minimal_png(img)

        structured_data = {"description": "A whiteboard", "objects": ["marker"]}
        mock_response = MagicMock()
        mock_response.text = json.dumps(structured_data)

        mock_model = MagicMock()
        mock_model.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        mock_schema = MagicMock()

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        result = await provider.analyze_images([img], "Describe", schema=mock_schema)

        assert isinstance(result, dict)
        assert result["description"] == "A whiteboard"
        # Verify structured output config was applied
        gen_config_call = mock_genai.GenerationConfig.call_args
        assert gen_config_call[1]["response_mime_type"] == "application/json"
        assert gen_config_call[1]["response_schema"] is mock_schema

    @pytest.mark.asyncio
    async def test_model_override(self, tmp_path):
        img = tmp_path / "frame.png"
        _write_minimal_png(img)

        mock_response = MagicMock()
        mock_response.text = "result"

        mock_model = MagicMock()
        mock_model.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        await provider.analyze_images(
            [img], "Describe", model="gemini-2.0-flash-lite"
        )

        # Verify the model name was passed to GenerativeModel
        model_call_args = mock_genai.GenerativeModel.call_args
        assert model_call_args[0][0] == "gemini-2.0-flash-lite"


# =============================================================================
# analyze_video()
# =============================================================================


class TestAnalyzeVideo:
    """Tests for analyze_video()."""

    @pytest.mark.asyncio
    async def test_basic_video(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake video data")

        mock_video_file = MagicMock()
        mock_video_file.state.name = "ACTIVE"
        mock_video_file.name = "files/abc123"

        mock_response = MagicMock()
        mock_response.text = "A tutorial about Python"

        mock_model = MagicMock()
        mock_model.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.upload_file.return_value = mock_video_file

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        result = await provider.analyze_video(video, "What is this video about?")

        assert result == "A tutorial about Python"
        mock_genai.upload_file.assert_called_once_with(str(video))
        mock_model.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_video_with_time_range(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake video data")

        mock_video_file = MagicMock()
        mock_video_file.state.name = "ACTIVE"
        mock_video_file.name = "files/abc123"

        mock_response = MagicMock()
        mock_response.text = "Code example on screen"

        mock_model = MagicMock()
        mock_model.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.upload_file.return_value = mock_video_file

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        result = await provider.analyze_video(
            video,
            "What code is shown?",
            start_time=120.0,
            end_time=180.0,
        )

        assert result == "Code example on screen"
        # Verify time spec is included in the prompt
        call_content = mock_model.generate_content.call_args[0][0]
        prompt_text = call_content[1]
        assert "02:00" in prompt_text
        assert "03:00" in prompt_text

    @pytest.mark.asyncio
    async def test_video_start_time_only(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake video data")

        mock_video_file = MagicMock()
        mock_video_file.state.name = "ACTIVE"

        mock_response = MagicMock()
        mock_response.text = "result"

        mock_model = MagicMock()
        mock_model.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.upload_file.return_value = mock_video_file

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        await provider.analyze_video(video, "What happens?", start_time=60.0)

        call_content = mock_model.generate_content.call_args[0][0]
        prompt_text = call_content[1]
        assert "Starting from 01:00" in prompt_text
        assert "Until" not in prompt_text

    @pytest.mark.asyncio
    async def test_video_not_found(self, tmp_path):
        mock_genai = MagicMock()
        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        with pytest.raises(FileNotFoundError, match="not found"):
            await provider.analyze_video(
                tmp_path / "nonexistent.mp4", "Describe"
            )

    @pytest.mark.asyncio
    async def test_video_processing_failure(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake video data")

        mock_video_file = MagicMock()
        mock_video_file.state.name = "FAILED"
        mock_video_file.name = "files/abc123"

        mock_genai = MagicMock()
        mock_genai.upload_file.return_value = mock_video_file

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        with pytest.raises(RuntimeError, match="FAILED"):
            await provider.analyze_video(video, "Describe")

    @pytest.mark.asyncio
    async def test_video_waits_for_active(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake video data")

        # First call returns PROCESSING, second returns ACTIVE
        processing_file = MagicMock()
        processing_file.state.name = "PROCESSING"
        processing_file.name = "files/abc123"

        active_file = MagicMock()
        active_file.state.name = "ACTIVE"
        active_file.name = "files/abc123"

        mock_response = MagicMock()
        mock_response.text = "result"

        mock_model = MagicMock()
        mock_model.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.upload_file.return_value = processing_file
        mock_genai.get_file.return_value = active_file

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        result = await provider.analyze_video(video, "Describe")

        assert result == "result"
        mock_genai.get_file.assert_called_once_with("files/abc123")

    @pytest.mark.asyncio
    async def test_video_with_schema(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake video data")

        mock_video_file = MagicMock()
        mock_video_file.state.name = "ACTIVE"

        structured_data = {"summary": "Tutorial", "topics": ["Python"]}
        mock_response = MagicMock()
        mock_response.text = json.dumps(structured_data)

        mock_model = MagicMock()
        mock_model.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.upload_file.return_value = mock_video_file

        mock_schema = MagicMock()

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        result = await provider.analyze_video(
            video, "Summarize", schema=mock_schema
        )

        assert isinstance(result, dict)
        assert result["summary"] == "Tutorial"
        gen_config_call = mock_genai.GenerationConfig.call_args
        assert gen_config_call[1]["response_schema"] is mock_schema

    @pytest.mark.asyncio
    async def test_invalid_time_range(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake video data")

        provider = GoogleProvider(api_key="test-key")
        provider._genai = MagicMock()

        with pytest.raises(ValueError, match="start_time.*end_time"):
            await provider.analyze_video(
                video, "Describe", start_time=200.0, end_time=100.0
            )


# =============================================================================
# reason()
# =============================================================================


class TestReason:
    """Tests for reason()."""

    @pytest.mark.asyncio
    async def test_single_message(self):
        mock_response = MagicMock()
        mock_response.text = "Python is a programming language."

        mock_model = MagicMock()
        mock_model.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        messages = [{"role": "user", "content": "What is Python?"}]
        result = await provider.reason(messages)

        assert result == "Python is a programming language."
        # Single message should use generate_content, not chat
        mock_model.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_turn_chat(self):
        mock_response = MagicMock()
        mock_response.text = "Variables store data values."

        mock_chat = MagicMock()
        mock_chat.send_message = MagicMock(return_value=mock_response)

        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_chat

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
            {"role": "user", "content": "What are variables?"},
        ]
        result = await provider.reason(messages)

        assert result == "Variables store data values."
        mock_model.start_chat.assert_called_once()
        # History should have first 2 messages (user + model)
        history = mock_model.start_chat.call_args[1]["history"]
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "model"
        # Last message sent via send_message
        mock_chat.send_message.assert_called_once_with("What are variables?")

    @pytest.mark.asyncio
    async def test_system_message_prepended(self):
        mock_response = MagicMock()
        mock_response.text = "response"

        mock_model = MagicMock()
        mock_model.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        messages = [
            {"role": "system", "content": "You are a video analyst."},
            {"role": "user", "content": "What happens in this video?"},
        ]
        await provider.reason(messages)

        # System message should be prepended to the first user message
        call_args = mock_model.generate_content.call_args[0][0]
        assert "You are a video analyst." in call_args[0]
        assert "What happens in this video?" in call_args[0]

    @pytest.mark.asyncio
    async def test_with_schema(self):
        structured_data = {"summary": "A tutorial", "topics": ["Python"]}
        mock_response = MagicMock()
        mock_response.text = json.dumps(structured_data)

        mock_model = MagicMock()
        mock_model.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        mock_schema = MagicMock()

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        messages = [{"role": "user", "content": "Summarize this transcript"}]
        result = await provider.reason(messages, schema=mock_schema)

        assert isinstance(result, dict)
        assert result["summary"] == "A tutorial"
        gen_config_call = mock_genai.GenerationConfig.call_args
        assert gen_config_call[1]["response_mime_type"] == "application/json"

    @pytest.mark.asyncio
    async def test_model_override(self):
        mock_response = MagicMock()
        mock_response.text = "result"

        mock_model = MagicMock()
        mock_model.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GoogleProvider(api_key="test-key")
        provider._genai = mock_genai

        messages = [{"role": "user", "content": "Hello"}]
        await provider.reason(messages, model="gemini-pro")

        model_call_args = mock_genai.GenerativeModel.call_args
        assert model_call_args[0][0] == "gemini-pro"


# =============================================================================
# Registry Integration
# =============================================================================


class TestRegistryIntegration:
    """Tests for provider registry integration."""

    def test_get_provider_by_name(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("google")
        assert isinstance(provider, GoogleProvider)

    def test_get_provider_by_alias_gemini(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("gemini")
        assert isinstance(provider, GoogleProvider)

    def test_get_provider_by_alias_gemini_flash(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("gemini-flash")
        assert isinstance(provider, GoogleProvider)

    def test_get_provider_by_alias_gemini_2_0_flash(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("gemini-2.0-flash")
        assert isinstance(provider, GoogleProvider)

    def test_get_provider_with_kwargs(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider(
            "google", model="gemini-pro", max_tokens=500
        )
        assert isinstance(provider, GoogleProvider)
        assert provider._model == "gemini-pro"
        assert provider._max_tokens == 500

    def test_package_level_get_provider(self):
        from claudetube.providers import get_provider

        provider = get_provider("google")
        assert isinstance(provider, GoogleProvider)


# =============================================================================
# Helpers
# =============================================================================


def _write_minimal_png(path: Path) -> None:
    """Write a minimal valid 1x1 pixel PNG file.

    This avoids needing PIL installed in the test environment just for
    file creation. PIL.Image.open() can read this.
    """
    import struct
    import zlib

    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"
    # IHDR chunk: 1x1 pixel, 8-bit RGB
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    # IDAT chunk: single row, filter byte 0, then RGB (0,0,0)
    raw = b"\x00\x00\x00\x00"
    compressed = zlib.compress(raw)
    idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat = (
        struct.pack(">I", len(compressed))
        + b"IDAT"
        + compressed
        + struct.pack(">I", idat_crc)
    )
    # IEND chunk
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

    path.write_bytes(signature + ihdr + idat + iend)
