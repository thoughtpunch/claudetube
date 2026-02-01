"""Tests for WhisperLocalProvider.

Verifies:
1. Provider instantiation and info
2. SRT parsing (unit tests)
3. is_available() behavior
4. Registry integration
5. Transcriber protocol compliance
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claudetube.providers.base import Transcriber
from claudetube.providers.capabilities import Capability
from claudetube.providers.types import TranscriptionResult
from claudetube.providers.whisper_local import (
    WhisperLocalProvider,
    _parse_srt,
    _parse_srt_time,
)


class TestSrtTimeParsing:
    """Tests for SRT timestamp parsing."""

    def test_zero(self):
        assert _parse_srt_time("00:00:00,000") == 0.0

    def test_seconds_only(self):
        assert _parse_srt_time("00:00:05,000") == 5.0

    def test_minutes_and_seconds(self):
        assert _parse_srt_time("00:02:30,000") == 150.0

    def test_hours(self):
        assert _parse_srt_time("01:00:00,000") == 3600.0

    def test_milliseconds(self):
        result = _parse_srt_time("00:00:01,500")
        assert abs(result - 1.5) < 0.001

    def test_complex_time(self):
        result = _parse_srt_time("01:23:45,678")
        expected = 3600 + 23 * 60 + 45 + 0.678
        assert abs(result - expected) < 0.001

    def test_vtt_format_dot(self):
        """Also handles VTT-style dot separator."""
        result = _parse_srt_time("00:00:01.500")
        assert abs(result - 1.5) < 0.001

    def test_invalid_format(self):
        assert _parse_srt_time("invalid") == 0.0

    def test_whitespace(self):
        result = _parse_srt_time("  00:00:01,500  ")
        assert abs(result - 1.5) < 0.001


class TestSrtParsing:
    """Tests for SRT text parsing into segments."""

    def test_empty_string(self):
        assert _parse_srt("") == []

    def test_single_segment(self):
        srt = "1\n00:00:00,000 --> 00:00:02,500\nHello world\n"
        segments = _parse_srt(srt)
        assert len(segments) == 1
        assert segments[0].text == "Hello world"
        assert segments[0].start == 0.0
        assert abs(segments[0].end - 2.5) < 0.001

    def test_multiple_segments(self):
        srt = (
            "1\n00:00:00,000 --> 00:00:02,500\nHello\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\nWorld\n"
        )
        segments = _parse_srt(srt)
        assert len(segments) == 2
        assert segments[0].text == "Hello"
        assert segments[1].text == "World"

    def test_multiline_text(self):
        srt = "1\n00:00:00,000 --> 00:00:02,500\nLine one\nLine two\n"
        segments = _parse_srt(srt)
        assert len(segments) == 1
        assert segments[0].text == "Line one\nLine two"

    def test_extra_blank_lines(self):
        srt = (
            "\n\n1\n00:00:00,000 --> 00:00:02,500\nHello\n\n\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\nWorld\n\n\n"
        )
        segments = _parse_srt(srt)
        assert len(segments) == 2

    def test_skips_malformed_blocks(self):
        srt = "1\nno timestamp here\nHello\n\n2\n00:00:03,000 --> 00:00:05,000\nWorld\n"
        segments = _parse_srt(srt)
        assert len(segments) == 1
        assert segments[0].text == "World"

    def test_realistic_whisper_output(self):
        srt = (
            "1\n00:00:00,000 --> 00:00:04,200\n"
            "Welcome to this tutorial on Python programming.\n\n"
            "2\n00:00:04,500 --> 00:00:08,100\n"
            "Today we'll be covering functions and classes.\n\n"
            "3\n00:00:08,500 --> 00:00:12,300\n"
            "Let's start with a simple example.\n"
        )
        segments = _parse_srt(srt)
        assert len(segments) == 3
        assert segments[0].start == 0.0
        assert abs(segments[2].end - 12.3) < 0.001


class TestWhisperLocalProvider:
    """Tests for the WhisperLocalProvider class."""

    def test_instantiation_default(self):
        provider = WhisperLocalProvider()
        assert provider._model_size == "small"
        assert provider._language == "en"
        assert provider._use_batched is True

    def test_instantiation_custom(self):
        provider = WhisperLocalProvider(
            model_size="large", language="es", use_batched=False
        )
        assert provider._model_size == "large"
        assert provider._language == "es"
        assert provider._use_batched is False

    def test_info(self):
        provider = WhisperLocalProvider()
        info = provider.info
        assert info.name == "whisper-local"
        assert info.can(Capability.TRANSCRIBE)
        assert not info.can(Capability.VISION)

    def test_no_eager_import(self):
        """Provider instantiation does NOT import faster-whisper."""
        provider = WhisperLocalProvider()
        assert provider._tool is None

    def test_implements_transcriber_protocol(self):
        provider = WhisperLocalProvider()
        assert isinstance(provider, Transcriber)

    def test_is_available_when_installed(self):
        with patch.dict("sys.modules", {"faster_whisper": MagicMock()}):
            provider = WhisperLocalProvider()
            assert provider.is_available() is True

    def test_is_available_when_not_installed(self):
        with patch.dict("sys.modules", {"faster_whisper": None}):
            provider = WhisperLocalProvider()
            assert provider.is_available() is False

    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self):
        provider = WhisperLocalProvider()
        with pytest.raises(FileNotFoundError, match="not found"):
            await provider.transcribe(Path("/nonexistent/audio.mp3"))

    @pytest.mark.asyncio
    async def test_transcribe_wraps_whisper_tool(self, tmp_path):
        """Transcribe delegates to WhisperTool and returns TranscriptionResult."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        mock_srt = (
            "1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\nGoodbye world\n"
        )
        mock_txt = "Hello world\nGoodbye world"
        mock_result = {"srt": mock_srt, "txt": mock_txt}

        mock_tool = MagicMock()
        mock_tool.transcribe.return_value = mock_result

        provider = WhisperLocalProvider(model_size="tiny")
        provider._tool = mock_tool

        result = await provider.transcribe(audio_file)

        assert isinstance(result, TranscriptionResult)
        assert result.text == mock_txt
        assert result.provider == "whisper-local"
        assert result.language == "en"
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello world"
        assert result.segments[1].text == "Goodbye world"
        assert abs(result.segments[0].start - 0.0) < 0.001
        assert abs(result.segments[1].end - 5.0) < 0.001
        assert abs(result.duration - 5.0) < 0.001

        mock_tool.transcribe.assert_called_once_with(
            audio_file, language="en", use_batched=True
        )

    @pytest.mark.asyncio
    async def test_transcribe_custom_language(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_tool = MagicMock()
        mock_tool.transcribe.return_value = {
            "srt": "1\n00:00:00,000 --> 00:00:01,000\nHola\n",
            "txt": "Hola",
        }

        provider = WhisperLocalProvider()
        provider._tool = mock_tool

        result = await provider.transcribe(audio_file, language="es")

        assert result.language == "es"
        mock_tool.transcribe.assert_called_once_with(
            audio_file, language="es", use_batched=True
        )

    @pytest.mark.asyncio
    async def test_transcribe_empty_result(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_tool = MagicMock()
        mock_tool.transcribe.return_value = {"srt": "", "txt": ""}

        provider = WhisperLocalProvider()
        provider._tool = mock_tool

        result = await provider.transcribe(audio_file)

        assert result.text == ""
        assert result.segments == []
        assert result.duration is None


class TestRegistryIntegration:
    """Tests for provider registry integration."""

    def test_get_provider_by_name(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("whisper-local")
        assert isinstance(provider, WhisperLocalProvider)
        assert provider.info.name == "whisper-local"

    def test_get_provider_by_alias(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("whisper")
        assert isinstance(provider, WhisperLocalProvider)

    def test_get_provider_with_kwargs(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("whisper-local", model_size="large")
        assert isinstance(provider, WhisperLocalProvider)
        assert provider._model_size == "large"

    def test_get_provider_caching(self):
        from claudetube.providers.registry import clear_cache, get_provider

        clear_cache()
        p1 = get_provider("whisper-local")
        p2 = get_provider("whisper-local")
        assert p1 is p2

    def test_get_provider_no_cache_with_kwargs(self):
        from claudetube.providers.registry import clear_cache, get_provider

        clear_cache()
        p1 = get_provider("whisper-local")
        p2 = get_provider("whisper-local", model_size="large")
        assert p1 is not p2

    def test_package_level_get_provider(self):
        from claudetube.providers import get_provider

        provider = get_provider("whisper-local")
        assert isinstance(provider, WhisperLocalProvider)
