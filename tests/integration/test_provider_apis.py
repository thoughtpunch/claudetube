"""Integration tests for real provider API calls.

These tests make actual API calls to verify that providers work against
real services. They require API keys to be set as environment variables
and the corresponding SDKs to be installed.

Run with: pytest tests/integration/test_provider_apis.py --run-integration

Each test class checks for its required API key and SDK, skipping
gracefully when either is missing.
"""

from __future__ import annotations

import os
import struct
import wave
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_sdk(module_name: str) -> bool:
    """Check if a Python SDK is importable."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _generate_sine_wav(path: Path, duration_s: float = 2.0, freq: int = 440) -> Path:
    """Generate a short sine-wave WAV file for transcription tests.

    Creates a minimal valid audio file without requiring external tools.
    The audio is silence-like (very short) so transcription results may
    vary, but it validates the API round-trip.
    """
    sample_rate = 16000
    n_samples = int(sample_rate * duration_s)

    import math

    samples = [
        int(32767 * 0.3 * math.sin(2 * math.pi * freq * t / sample_rate))
        for t in range(n_samples)
    ]
    raw = struct.pack(f"<{n_samples}h", *samples)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)

    return path


def _generate_test_image(path: Path, width: int = 100, height: int = 100) -> Path:
    """Generate a minimal PNG test image.

    Creates a simple solid-color PNG without requiring PIL/Pillow,
    using the minimal valid PNG structure.
    """
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"

    # IHDR: width, height, bit_depth=8, color_type=2 (RGB)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    # IDAT: image data (blue pixels)
    raw_rows = b""
    for _ in range(height):
        raw_rows += b"\x00"  # filter byte
        raw_rows += b"\x00\x00\xff" * width  # blue pixels

    compressed = zlib.compress(raw_rows)
    idat = _chunk(b"IDAT", compressed)

    # IEND
    iend = _chunk(b"IEND", b"")

    path.write_bytes(sig + ihdr + idat + iend)
    return path


# ---------------------------------------------------------------------------
# OpenAI integration tests
# ---------------------------------------------------------------------------

_skip_openai = pytest.mark.skipif(
    not _has_sdk("openai") or not os.environ.get("OPENAI_API_KEY"),
    reason="openai SDK not installed or OPENAI_API_KEY not set",
)


@pytest.mark.integration
@_skip_openai
class TestOpenAIIntegration:
    """Integration tests for OpenAI provider (Whisper + GPT-4o)."""

    @pytest.mark.asyncio
    async def test_transcribe_audio(self, tmp_path):
        """OpenAI Whisper API transcribes a short audio file."""
        from claudetube.providers.openai.client import OpenaiProvider
        from claudetube.providers.types import TranscriptionResult

        audio_file = _generate_sine_wav(tmp_path / "test.wav")
        provider = OpenaiProvider()

        assert provider.is_available()

        result = await provider.transcribe(audio_file, language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.provider == "openai"
        # A sine tone may produce empty or minimal text, but the
        # API call should succeed and return a valid result object
        assert isinstance(result.text, str)

    @pytest.mark.asyncio
    async def test_reason_simple_prompt(self):
        """OpenAI chat completions returns a text response."""
        from claudetube.providers.openai.client import OpenaiProvider

        provider = OpenaiProvider()
        messages = [
            {"role": "user", "content": "Reply with exactly: HELLO"},
        ]

        result = await provider.reason(messages)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "HELLO" in result.upper()

    @pytest.mark.asyncio
    async def test_analyze_image(self, tmp_path):
        """OpenAI GPT-4o vision analyzes an image."""
        from claudetube.providers.openai.client import OpenaiProvider

        image_file = _generate_test_image(tmp_path / "test.png")
        provider = OpenaiProvider()

        result = await provider.analyze_images(
            [image_file],
            prompt="What color is this image? Reply with just the color name.",
        )

        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Anthropic integration tests
# ---------------------------------------------------------------------------

_skip_anthropic = pytest.mark.skipif(
    not _has_sdk("anthropic") or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="anthropic SDK not installed or ANTHROPIC_API_KEY not set",
)


@pytest.mark.integration
@_skip_anthropic
class TestAnthropicIntegration:
    """Integration tests for Anthropic provider (Claude vision + reasoning)."""

    @pytest.mark.asyncio
    async def test_reason_simple_prompt(self):
        """Anthropic Claude returns a text response."""
        from claudetube.providers.anthropic.client import AnthropicProvider

        provider = AnthropicProvider()

        assert provider.is_available()

        messages = [
            {"role": "user", "content": "Reply with exactly: HELLO"},
        ]

        result = await provider.reason(messages)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "HELLO" in result.upper()

    @pytest.mark.asyncio
    async def test_analyze_image(self, tmp_path):
        """Anthropic Claude vision analyzes an image."""
        from claudetube.providers.anthropic.client import AnthropicProvider

        image_file = _generate_test_image(tmp_path / "test.png")
        provider = AnthropicProvider()

        result = await provider.analyze_images(
            [image_file],
            prompt="What color is this image? Reply with just the color name.",
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_reason_with_system_prompt(self):
        """Anthropic correctly handles system messages."""
        from claudetube.providers.anthropic.client import AnthropicProvider

        provider = AnthropicProvider()

        messages = [
            {
                "role": "system",
                "content": "You are a calculator. Only output numbers.",
            },
            {"role": "user", "content": "What is 2 + 2?"},
        ]

        result = await provider.reason(messages)

        assert isinstance(result, str)
        assert "4" in result


# ---------------------------------------------------------------------------
# Voyage integration tests
# ---------------------------------------------------------------------------

_skip_voyage = pytest.mark.skipif(
    not _has_sdk("voyageai") or not os.environ.get("VOYAGE_API_KEY"),
    reason="voyageai SDK not installed or VOYAGE_API_KEY not set",
)


@pytest.mark.integration
@_skip_voyage
class TestVoyageIntegration:
    """Integration tests for Voyage AI embedding provider."""

    @pytest.mark.asyncio
    async def test_embed_text(self):
        """Voyage embeds text and returns a vector."""
        from claudetube.providers.voyage.client import VoyageProvider

        provider = VoyageProvider()

        assert provider.is_available()

        vector = await provider.embed("A person walking through a park")

        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.asyncio
    async def test_embed_returns_consistent_dimensions(self):
        """Voyage returns same-dimensioned vectors for different inputs."""
        from claudetube.providers.voyage.client import VoyageProvider

        provider = VoyageProvider()

        vector_a = await provider.embed("The cat sat on the mat")
        vector_b = await provider.embed("Machine learning and neural networks")

        assert len(vector_a) == len(vector_b)
        # voyage-multimodal-3 returns 1024-d vectors
        assert len(vector_a) == 1024

    @pytest.mark.asyncio
    async def test_different_texts_produce_different_embeddings(self):
        """Voyage produces distinct embeddings for semantically different texts."""
        from claudetube.providers.voyage.client import VoyageProvider

        provider = VoyageProvider()

        vector_a = await provider.embed("The weather is sunny today")
        vector_b = await provider.embed("Quantum physics and string theory")

        # Vectors should not be identical
        assert vector_a != vector_b


# ---------------------------------------------------------------------------
# Deepgram integration tests
# ---------------------------------------------------------------------------

_skip_deepgram = pytest.mark.skipif(
    not _has_sdk("deepgram") or not os.environ.get("DEEPGRAM_API_KEY"),
    reason="deepgram SDK not installed or DEEPGRAM_API_KEY not set",
)


@pytest.mark.integration
@_skip_deepgram
class TestDeepgramIntegration:
    """Integration tests for Deepgram transcription provider."""

    @pytest.mark.asyncio
    async def test_transcribe_audio(self, tmp_path):
        """Deepgram API transcribes a short audio file."""
        from claudetube.providers.deepgram.client import DeepgramProvider
        from claudetube.providers.types import TranscriptionResult

        audio_file = _generate_sine_wav(tmp_path / "test.wav")
        provider = DeepgramProvider()

        assert provider.is_available()

        result = await provider.transcribe(audio_file, language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.provider == "deepgram"
        assert isinstance(result.text, str)


# ---------------------------------------------------------------------------
# Cross-provider validation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestProviderAvailability:
    """Validate provider availability detection matches reality."""

    @pytest.mark.skipif(
        not _has_sdk("openai"),
        reason="openai SDK not installed",
    )
    def test_openai_availability_reflects_api_key(self, monkeypatch):
        """OpenAI provider availability matches API key presence."""
        from claudetube.providers.openai.client import OpenaiProvider

        # With key
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert OpenaiProvider().is_available() is True

        # Without key
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert OpenaiProvider().is_available() is False

    @pytest.mark.skipif(
        not _has_sdk("anthropic"),
        reason="anthropic SDK not installed",
    )
    def test_anthropic_availability_reflects_api_key(self, monkeypatch):
        """Anthropic provider availability matches API key presence."""
        from claudetube.providers.anthropic.client import AnthropicProvider

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert AnthropicProvider().is_available() is True

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert AnthropicProvider().is_available() is False

    @pytest.mark.skipif(
        not _has_sdk("voyageai"),
        reason="voyageai SDK not installed",
    )
    def test_voyage_availability_reflects_api_key(self, monkeypatch):
        """Voyage provider availability matches API key presence."""
        from claudetube.providers.voyage.client import VoyageProvider

        monkeypatch.setenv("VOYAGE_API_KEY", "voy-test")
        assert VoyageProvider().is_available() is True

        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        assert VoyageProvider().is_available() is False

    @pytest.mark.skipif(
        not _has_sdk("deepgram"),
        reason="deepgram SDK not installed",
    )
    def test_deepgram_availability_reflects_api_key(self, monkeypatch):
        """Deepgram provider availability matches API key presence."""
        from claudetube.providers.deepgram.client import DeepgramProvider

        monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
        assert DeepgramProvider().is_available() is True

        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        assert DeepgramProvider().is_available() is False
