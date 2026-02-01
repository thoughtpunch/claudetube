"""Tests for OpenaiProvider.

Verifies:
1. Provider instantiation and info
2. is_available() behavior
3. Image encoding and media type detection
4. transcribe() with single file and chunked files
5. analyze_images() with and without schema
6. reason() with and without schema
7. Audio chunking logic
8. Registry integration
9. Protocol compliance
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claudetube.providers.base import Reasoner, Transcriber, VisionAnalyzer
from claudetube.providers.capabilities import Capability
from claudetube.providers.openai.client import (
    OpenaiProvider,
    _build_response_format,
    _detect_media_type,
    _encode_image,
)

# =============================================================================
# Media Type Detection
# =============================================================================


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


# =============================================================================
# Image Encoding
# =============================================================================


class TestImageEncoding:
    """Tests for base64 image encoding."""

    def test_encodes_file(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake jpeg data")
        result = _encode_image(img)
        assert isinstance(result, str)
        import base64

        decoded = base64.b64decode(result)
        assert decoded == b"\xff\xd8\xff\xe0fake jpeg data"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _encode_image(Path("/nonexistent/image.jpg"))


# =============================================================================
# Response Format Builder
# =============================================================================


class TestBuildResponseFormat:
    """Tests for response_format construction."""

    def test_with_pydantic_model(self):
        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        result = _build_response_format(mock_schema)
        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "response"
        assert result["json_schema"]["schema"]["type"] == "object"

    def test_without_pydantic(self):
        result = _build_response_format(dict)
        assert result["type"] == "json_schema"
        assert result["json_schema"]["schema"] == {"type": "object"}


# =============================================================================
# Provider Instantiation
# =============================================================================


class TestOpenaiProvider:
    """Tests for the OpenaiProvider class."""

    def test_instantiation_default(self):
        provider = OpenaiProvider()
        assert provider._model == "gpt-4o"
        assert provider._whisper_model == "whisper-1"
        assert provider._max_tokens == 1024
        assert provider._client is None

    def test_instantiation_custom(self):
        provider = OpenaiProvider(
            model="gpt-4o-mini",
            whisper_model="whisper-1",
            api_key="test-key",
            max_tokens=500,
        )
        assert provider._model == "gpt-4o-mini"
        assert provider._api_key == "test-key"
        assert provider._max_tokens == 500

    def test_info(self):
        provider = OpenaiProvider()
        info = provider.info
        assert info.name == "openai"
        assert info.can(Capability.TRANSCRIBE)
        assert info.can(Capability.VISION)
        assert info.can(Capability.REASON)
        assert not info.can(Capability.VIDEO)

    def test_no_eager_import(self):
        """Provider instantiation does NOT import openai."""
        provider = OpenaiProvider()
        assert provider._client is None

    def test_implements_transcriber_protocol(self):
        provider = OpenaiProvider()
        assert isinstance(provider, Transcriber)

    def test_implements_vision_protocol(self):
        provider = OpenaiProvider()
        assert isinstance(provider, VisionAnalyzer)

    def test_implements_reasoner_protocol(self):
        provider = OpenaiProvider()
        assert isinstance(provider, Reasoner)


# =============================================================================
# is_available()
# =============================================================================


class TestIsAvailable:
    """Tests for is_available() behavior."""

    def test_available_with_api_key_arg(self):
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            provider = OpenaiProvider(api_key="test-key")
            assert provider.is_available() is True

    def test_available_with_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            provider = OpenaiProvider()
            assert provider.is_available() is True

    def test_not_available_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            provider = OpenaiProvider()
            assert provider.is_available() is False

    def test_not_available_without_sdk(self):
        with patch.dict("sys.modules", {"openai": None}):
            provider = OpenaiProvider(api_key="test-key")
            assert provider.is_available() is False


# =============================================================================
# transcribe()
# =============================================================================


class TestTranscribe:
    """Tests for transcribe()."""

    @pytest.mark.asyncio
    async def test_single_file_under_limit(self, tmp_path):
        """Transcribe a file under 25MB (no chunking)."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        # Mock the Whisper API response
        mock_segment = {"start": 0.0, "end": 2.5, "text": " Hello world."}
        mock_response = MagicMock()
        mock_response.text = "Hello world."
        mock_response.segments = [mock_segment]
        mock_response.language = "en"
        mock_response.duration = 2.5

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        provider = OpenaiProvider()
        provider._client = mock_client

        result = await provider.transcribe(audio)

        assert result.text == "Hello world."
        assert result.provider == "openai"
        assert result.language == "en"
        assert result.duration == 2.5
        assert len(result.segments) == 1
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 2.5
        assert result.segments[0].text == "Hello world."

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        provider = OpenaiProvider()
        provider._client = AsyncMock()

        with pytest.raises(FileNotFoundError, match="not found"):
            await provider.transcribe(Path("/nonexistent/audio.mp3"))

    @pytest.mark.asyncio
    async def test_language_passed_to_api(self, tmp_path):
        """Language parameter is forwarded to the Whisper API."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        mock_response = MagicMock()
        mock_response.text = "Hola mundo."
        mock_response.segments = []
        mock_response.language = "es"
        mock_response.duration = 1.0

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        provider = OpenaiProvider()
        provider._client = mock_client

        await provider.transcribe(audio, language="es")

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["language"] == "es"

    @pytest.mark.asyncio
    async def test_whisper_model_override(self, tmp_path):
        """Custom whisper model is forwarded to API."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        mock_response = MagicMock()
        mock_response.text = "Test."
        mock_response.segments = []
        mock_response.language = "en"
        mock_response.duration = 1.0

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        provider = OpenaiProvider()
        provider._client = mock_client

        await provider.transcribe(audio, whisper_model="whisper-1")

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["model"] == "whisper-1"

    @pytest.mark.asyncio
    async def test_chunked_transcription(self, tmp_path):
        """Files over 25MB are chunked and segments have correct offsets."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        # Mock chunker to return two chunks
        from claudetube.providers.openai.chunker import AudioChunk

        mock_chunks = [
            AudioChunk(path=audio, offset=0.0),
            AudioChunk(path=audio, offset=600.0),
        ]

        mock_response_1 = MagicMock()
        mock_response_1.text = "Part one."
        mock_response_1.segments = [
            {"start": 0.0, "end": 5.0, "text": " Part one."},
        ]
        mock_response_1.language = "en"
        mock_response_1.duration = 600.0

        mock_response_2 = MagicMock()
        mock_response_2.text = "Part two."
        mock_response_2.segments = [
            {"start": 0.0, "end": 3.0, "text": " Part two."},
        ]
        mock_response_2.language = "en"
        mock_response_2.duration = 300.0

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            side_effect=[mock_response_1, mock_response_2]
        )

        provider = OpenaiProvider()
        provider._client = mock_client

        with patch(
            "claudetube.providers.openai.chunker.chunk_audio_if_needed",
            new_callable=AsyncMock,
            return_value=mock_chunks,
        ):
            result = await provider.transcribe(audio)

        assert result.text == "Part one. Part two."
        assert len(result.segments) == 2
        # First chunk: no offset
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 5.0
        # Second chunk: 600s offset applied
        assert result.segments[1].start == 600.0
        assert result.segments[1].end == 603.0
        # Total duration = last chunk offset + its duration
        assert result.duration == 900.0


# =============================================================================
# analyze_images()
# =============================================================================


class TestAnalyzeImages:
    """Tests for analyze_images()."""

    @pytest.mark.asyncio
    async def test_single_image(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake jpeg")

        mock_message = MagicMock()
        mock_message.content = "A person at a whiteboard"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenaiProvider()
        provider._client = mock_client

        result = await provider.analyze_images([img], "Describe this frame")

        assert result == "A person at a whiteboard"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        content = call_kwargs["messages"][0]["content"]
        assert len(content) == 2  # 1 image + 1 text
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "text"

    @pytest.mark.asyncio
    async def test_multiple_images(self, tmp_path):
        imgs = []
        for i in range(3):
            img = tmp_path / f"frame{i}.png"
            img.write_bytes(b"fake png")
            imgs.append(img)

        mock_message = MagicMock()
        mock_message.content = "Three frames showing progression"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenaiProvider()
        provider._client = mock_client

        result = await provider.analyze_images(imgs, "Describe the progression")

        assert result == "Three frames showing progression"
        content = mock_client.chat.completions.create.call_args[1]["messages"][0][
            "content"
        ]
        assert len(content) == 4  # 3 images + 1 text
        for i in range(3):
            assert content[i]["type"] == "image_url"
            assert "image/png" in content[i]["image_url"]["url"]

    @pytest.mark.asyncio
    async def test_image_not_found(self, tmp_path):
        provider = OpenaiProvider()
        provider._client = AsyncMock()

        with pytest.raises(FileNotFoundError, match="not found"):
            await provider.analyze_images(
                [tmp_path / "nonexistent.jpg"], "Describe"
            )

    @pytest.mark.asyncio
    async def test_max_images_capped(self, tmp_path):
        """Only first 10 images are sent."""
        imgs = []
        for i in range(15):
            img = tmp_path / f"frame{i}.jpg"
            img.write_bytes(b"fake jpeg")
            imgs.append(img)

        mock_message = MagicMock()
        mock_message.content = "result"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenaiProvider()
        provider._client = mock_client

        await provider.analyze_images(imgs, "Describe")

        content = mock_client.chat.completions.create.call_args[1]["messages"][0][
            "content"
        ]
        # 10 images + 1 text = 11 blocks
        assert len(content) == 11

    @pytest.mark.asyncio
    async def test_with_schema(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake jpeg")

        result_data = {"description": "A whiteboard", "objects": ["marker"]}
        mock_message = MagicMock()
        mock_message.content = json.dumps(result_data)
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "objects": {"type": "array", "items": {"type": "string"}},
            },
        }

        provider = OpenaiProvider()
        provider._client = mock_client

        result = await provider.analyze_images([img], "Describe", schema=mock_schema)

        assert isinstance(result, dict)
        assert result["description"] == "A whiteboard"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_model_override(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake jpeg")

        mock_message = MagicMock()
        mock_message.content = "result"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenaiProvider()
        provider._client = mock_client

        await provider.analyze_images([img], "Describe", model="gpt-4o-mini")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"


# =============================================================================
# reason()
# =============================================================================


class TestReason:
    """Tests for reason()."""

    @pytest.mark.asyncio
    async def test_simple_messages(self):
        mock_message = MagicMock()
        mock_message.content = "Python is a programming language."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenaiProvider()
        provider._client = mock_client

        messages = [{"role": "user", "content": "What is Python?"}]
        result = await provider.reason(messages)

        assert result == "Python is a programming language."

    @pytest.mark.asyncio
    async def test_messages_passed_directly(self):
        """OpenAI passes messages as-is (including system)."""
        mock_message = MagicMock()
        mock_message.content = "response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenaiProvider()
        provider._client = mock_client

        messages = [
            {"role": "system", "content": "You are a video analyst."},
            {"role": "user", "content": "Summarize this video."},
        ]
        await provider.reason(messages)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        # OpenAI accepts system messages in the messages list
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_with_schema(self):
        result_data = {"summary": "A tutorial video", "topics": ["Python"]}
        mock_message = MagicMock()
        mock_message.content = json.dumps(result_data)
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "topics": {"type": "array", "items": {"type": "string"}},
            },
        }

        provider = OpenaiProvider()
        provider._client = mock_client

        messages = [{"role": "user", "content": "Summarize this transcript"}]
        result = await provider.reason(messages, schema=mock_schema)

        assert isinstance(result, dict)
        assert result["summary"] == "A tutorial video"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_model_override(self):
        mock_message = MagicMock()
        mock_message.content = "result"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenaiProvider()
        provider._client = mock_client

        await provider.reason(
            [{"role": "user", "content": "Hello"}], model="gpt-4o-mini"
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"


# =============================================================================
# Audio Chunking
# =============================================================================


class TestAudioChunking:
    """Tests for audio chunking logic."""

    @pytest.mark.asyncio
    async def test_small_file_no_chunking(self, tmp_path):
        from claudetube.providers.openai.chunker import chunk_audio_if_needed

        audio = tmp_path / "small.mp3"
        # Write 1MB of data (under 25MB)
        audio.write_bytes(b"x" * (1024 * 1024))

        chunks = await chunk_audio_if_needed(audio)

        assert len(chunks) == 1
        assert chunks[0].path == audio
        assert chunks[0].offset == 0.0

    @pytest.mark.asyncio
    async def test_large_file_triggers_chunking(self, tmp_path):
        from claudetube.providers.openai.chunker import chunk_audio_if_needed

        audio = tmp_path / "large.mp3"
        # Write 30MB of data (over 25MB)
        audio.write_bytes(b"x" * (30 * 1024 * 1024))

        with patch(
            "claudetube.providers.openai.chunker.get_audio_duration",
            new_callable=AsyncMock,
            return_value=1200.0,  # 20 minutes
        ), patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_exec:
            # Mock the ffmpeg subprocess
            mock_process = AsyncMock()
            mock_process.wait = AsyncMock()
            mock_exec.return_value = mock_process

            # Create the expected chunk files so they "exist"
            chunk_dir = tmp_path / "chunks"
            chunk_dir.mkdir()
            for i in range(2):
                chunk_path = chunk_dir / f"chunk_{i:03d}.mp3"
                chunk_path.write_bytes(b"chunk data")

            chunks = await chunk_audio_if_needed(audio)

        # 1200s / 600s per chunk = 2 chunks
        assert len(chunks) == 2
        assert chunks[0].offset == 0.0
        assert chunks[1].offset == 600.0


# =============================================================================
# Registry Integration
# =============================================================================


class TestRegistryIntegration:
    """Tests for provider registry integration."""

    def test_get_provider_by_name(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("openai")
        assert isinstance(provider, OpenaiProvider)

    def test_get_provider_by_alias_gpt4o(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("gpt-4o")
        assert isinstance(provider, OpenaiProvider)

    def test_get_provider_by_alias_gpt(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("gpt")
        assert isinstance(provider, OpenaiProvider)

    def test_get_provider_by_alias_whisper_api(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("whisper-api")
        assert isinstance(provider, OpenaiProvider)

    def test_get_provider_with_kwargs(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider(
            "openai", model="gpt-4o-mini", max_tokens=500
        )
        assert isinstance(provider, OpenaiProvider)
        assert provider._model == "gpt-4o-mini"
        assert provider._max_tokens == 500

    def test_package_level_get_provider(self):
        from claudetube.providers import get_provider

        provider = get_provider("openai")
        assert isinstance(provider, OpenaiProvider)
