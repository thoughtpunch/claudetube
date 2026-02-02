"""Tests for DeepgramProvider.

Verifies:
1. Provider instantiation and info
2. is_available() behavior
3. transcribe() with mocked SDK response
4. Diarization support
5. Registry integration
6. Protocol compliance
7. No eager import
8. File not found error
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claudetube.providers.base import Transcriber
from claudetube.providers.capabilities import Capability
from claudetube.providers.deepgram.client import DeepgramProvider


def _mock_deepgram_module():
    """Create a mock deepgram module with PrerecordedOptions."""
    mock_module = MagicMock()
    mock_module.PrerecordedOptions = MagicMock()
    return mock_module


class TestDeepgramProvider:
    """Tests for the DeepgramProvider class."""

    def test_instantiation_default(self):
        provider = DeepgramProvider()
        assert provider._model == "nova-2"
        assert provider._api_key is None
        assert provider._client is None

    def test_instantiation_custom(self):
        provider = DeepgramProvider(
            model="nova-2-general",
            api_key="test-key",
        )
        assert provider._model == "nova-2-general"
        assert provider._api_key == "test-key"

    def test_info(self):
        provider = DeepgramProvider()
        info = provider.info
        assert info.name == "deepgram"
        assert info.can(Capability.TRANSCRIBE)
        assert not info.can(Capability.VISION)
        assert not info.can(Capability.REASON)
        assert info.supports_diarization is True

    def test_info_capabilities(self):
        provider = DeepgramProvider()
        info = provider.info
        assert info.capabilities == frozenset({Capability.TRANSCRIBE})
        assert info.supports_streaming is True
        assert info.supports_translation is False

    def test_no_eager_import(self):
        """Provider instantiation does NOT import deepgram."""
        provider = DeepgramProvider()
        assert provider._client is None

    def test_implements_transcriber_protocol(self):
        provider = DeepgramProvider()
        assert isinstance(provider, Transcriber)


class TestIsAvailable:
    """Tests for is_available() behavior."""

    def test_available_with_api_key_arg(self):
        with patch.dict("sys.modules", {"deepgram": MagicMock()}):
            provider = DeepgramProvider(api_key="test-key")
            assert provider.is_available() is True

    def test_available_with_env_var(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        with patch.dict("sys.modules", {"deepgram": MagicMock()}):
            provider = DeepgramProvider()
            assert provider.is_available() is True

    def test_not_available_without_key(self, monkeypatch):
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        with patch.dict("sys.modules", {"deepgram": MagicMock()}):
            provider = DeepgramProvider()
            assert provider.is_available() is False

    def test_not_available_without_sdk(self):
        with patch.dict("sys.modules", {"deepgram": None}):
            provider = DeepgramProvider(api_key="test-key")
            assert provider.is_available() is False


class TestTranscribe:
    """Tests for transcribe() with mocked SDK responses."""

    @staticmethod
    def _make_utterance(transcript, start, end, confidence=0.99, speaker=None):
        """Helper to create a mock utterance object."""
        utt = MagicMock()
        utt.transcript = transcript
        utt.start = start
        utt.end = end
        utt.confidence = confidence
        if speaker is not None:
            utt.speaker = speaker
        return utt

    @staticmethod
    def _make_word(word, start, end, confidence=0.99, speaker=None):
        """Helper to create a mock word object."""
        w = MagicMock()
        w.word = word
        w.start = start
        w.end = end
        w.confidence = confidence
        if speaker is not None:
            w.speaker = speaker
        return w

    @staticmethod
    def _make_response(transcript, utterances=None, words=None):
        """Helper to create a mock Deepgram API response."""
        alternative = MagicMock()
        alternative.transcript = transcript
        alternative.words = words or []

        channel = MagicMock()
        channel.alternatives = [alternative]

        results = MagicMock()
        results.channels = [channel]
        results.utterances = utterances

        response = MagicMock()
        response.results = results
        return response

    def _setup_provider(self, mock_response):
        """Set up a provider with a mocked client returning mock_response."""
        mock_transcribe = AsyncMock(return_value=mock_response)
        mock_client = MagicMock()
        mock_client.listen.asyncprerecorded.v.return_value.transcribe_file = (
            mock_transcribe
        )

        provider = DeepgramProvider()
        provider._client = mock_client
        return provider, mock_client

    @pytest.mark.asyncio
    async def test_basic_transcription(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        utterances = [
            self._make_utterance("Hello world.", 0.0, 1.5),
            self._make_utterance("How are you?", 2.0, 3.5),
        ]
        mock_response = self._make_response(
            "Hello world. How are you?",
            utterances=utterances,
        )

        provider, _ = self._setup_provider(mock_response)

        mock_dg = _mock_deepgram_module()
        with patch.dict("sys.modules", {"deepgram": mock_dg}):
            result = await provider.transcribe(audio)

        assert result.text == "Hello world. How are you?"
        assert result.provider == "deepgram"
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello world."
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 1.5
        assert result.segments[1].text == "How are you?"
        assert result.segments[1].start == 2.0
        assert result.segments[1].end == 3.5
        assert result.duration == 3.5
        # No diarization by default
        assert result.segments[0].speaker is None

    @pytest.mark.asyncio
    async def test_transcription_with_diarization(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        utterances = [
            self._make_utterance("Hello there.", 0.0, 1.5, speaker=0),
            self._make_utterance("Hi, how are you?", 2.0, 3.5, speaker=1),
            self._make_utterance("I'm doing well.", 4.0, 5.0, speaker=0),
        ]
        mock_response = self._make_response(
            "Hello there. Hi, how are you? I'm doing well.",
            utterances=utterances,
        )

        provider, _ = self._setup_provider(mock_response)

        mock_dg = _mock_deepgram_module()
        with patch.dict("sys.modules", {"deepgram": mock_dg}):
            result = await provider.transcribe(audio, diarize=True)

        assert len(result.segments) == 3
        assert result.segments[0].speaker == "SPEAKER_0"
        assert result.segments[0].text == "Hello there."
        assert result.segments[1].speaker == "SPEAKER_1"
        assert result.segments[1].text == "Hi, how are you?"
        assert result.segments[2].speaker == "SPEAKER_0"
        assert result.segments[2].text == "I'm doing well."

    @pytest.mark.asyncio
    async def test_transcription_with_language(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        utterances = [
            self._make_utterance("Hola mundo.", 0.0, 1.5),
        ]
        mock_response = self._make_response(
            "Hola mundo.",
            utterances=utterances,
        )

        provider, _ = self._setup_provider(mock_response)

        mock_dg = _mock_deepgram_module()
        with patch.dict("sys.modules", {"deepgram": mock_dg}):
            result = await provider.transcribe(audio, language="es")

        assert result.text == "Hola mundo."
        assert result.language == "es"

    @pytest.mark.asyncio
    async def test_transcription_fallback_to_words(self, tmp_path):
        """When utterances are not available, fall back to grouping words."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        words = [
            self._make_word("Hello", 0.0, 0.5),
            self._make_word("world.", 0.6, 1.0),
            # Gap > 1 second triggers new segment
            self._make_word("How", 2.5, 2.8),
            self._make_word("are", 2.9, 3.1),
            self._make_word("you?", 3.2, 3.5),
        ]
        mock_response = self._make_response(
            "Hello world. How are you?",
            utterances=None,
            words=words,
        )

        provider, _ = self._setup_provider(mock_response)

        mock_dg = _mock_deepgram_module()
        with patch.dict("sys.modules", {"deepgram": mock_dg}):
            result = await provider.transcribe(audio)

        assert result.text == "Hello world. How are you?"
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello world."
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 1.0
        assert result.segments[1].text == "How are you?"
        assert result.segments[1].start == 2.5
        assert result.segments[1].end == 3.5

    @pytest.mark.asyncio
    async def test_transcription_model_override(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        mock_response = self._make_response("test", utterances=[])

        provider, _ = self._setup_provider(mock_response)

        mock_dg = _mock_deepgram_module()
        with patch.dict("sys.modules", {"deepgram": mock_dg}):
            await provider.transcribe(audio, model="nova-2-general")

        # Verify PrerecordedOptions was called with the overridden model
        mock_dg.PrerecordedOptions.assert_called_once_with(
            model="nova-2-general",
            diarize=False,
            utterances=True,
        )

    @pytest.mark.asyncio
    async def test_confidence_passed_through(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        utterances = [
            self._make_utterance("Hello.", 0.0, 1.0, confidence=0.95),
        ]
        mock_response = self._make_response("Hello.", utterances=utterances)

        provider, _ = self._setup_provider(mock_response)

        mock_dg = _mock_deepgram_module()
        with patch.dict("sys.modules", {"deepgram": mock_dg}):
            result = await provider.transcribe(audio)

        assert result.segments[0].confidence == 0.95


class TestFileNotFound:
    """Tests for file not found error handling."""

    @pytest.mark.asyncio
    async def test_audio_file_not_found(self):
        provider = DeepgramProvider()
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            await provider.transcribe(Path("/nonexistent/audio.mp3"))


class TestRegistryIntegration:
    """Tests for provider registry integration."""

    def test_get_provider_by_name(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("deepgram")
        assert isinstance(provider, DeepgramProvider)

    def test_get_provider_with_kwargs(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider(
            "deepgram", model="nova-2-general", api_key="test-key"
        )
        assert isinstance(provider, DeepgramProvider)
        assert provider._model == "nova-2-general"
        assert provider._api_key == "test-key"

    def test_package_level_get_provider(self):
        from claudetube.providers import get_provider

        provider = get_provider("deepgram")
        assert isinstance(provider, DeepgramProvider)
