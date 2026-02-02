"""Tests for AssemblyAIProvider.

Verifies:
1. Provider instantiation and info
2. is_available() behavior
3. transcribe() with mocked SDK response
4. Diarization support
5. Auto-chapters support
6. Registry integration
7. Protocol compliance
8. No eager import
9. File not found error
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claudetube.providers.assemblyai.client import AssemblyAIProvider
from claudetube.providers.base import Transcriber
from claudetube.providers.capabilities import Capability


def _mock_assemblyai_module():
    """Create a mock assemblyai module with TranscriptionConfig and TranscriptStatus."""
    mock_module = MagicMock()
    mock_module.TranscriptionConfig = MagicMock()
    mock_module.TranscriptStatus.error = "error"
    mock_module.TranscriptStatus.completed = "completed"
    mock_module.settings = MagicMock()
    mock_module.Transcriber = MagicMock
    return mock_module


class TestAssemblyAIProvider:
    """Tests for the AssemblyAIProvider class."""

    def test_instantiation_default(self):
        provider = AssemblyAIProvider()
        assert provider._api_key is None
        assert provider._client is None

    def test_instantiation_with_api_key(self):
        provider = AssemblyAIProvider(api_key="test-key")
        assert provider._api_key == "test-key"

    def test_info(self):
        provider = AssemblyAIProvider()
        info = provider.info
        assert info.name == "assemblyai"
        assert info.can(Capability.TRANSCRIBE)
        assert not info.can(Capability.VISION)
        assert not info.can(Capability.REASON)
        assert info.supports_diarization is True

    def test_info_capabilities(self):
        provider = AssemblyAIProvider()
        info = provider.info
        assert info.capabilities == frozenset({Capability.TRANSCRIBE})
        assert info.supports_streaming is True
        assert info.supports_translation is False

    def test_no_eager_import(self):
        """Provider instantiation does NOT import assemblyai."""
        provider = AssemblyAIProvider()
        assert provider._client is None

    def test_implements_transcriber_protocol(self):
        provider = AssemblyAIProvider()
        assert isinstance(provider, Transcriber)


class TestIsAvailable:
    """Tests for is_available() behavior."""

    def test_available_with_api_key_arg(self):
        with patch.dict("sys.modules", {"assemblyai": MagicMock()}):
            provider = AssemblyAIProvider(api_key="test-key")
            assert provider.is_available() is True

    def test_available_with_env_var(self, monkeypatch):
        monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key")
        with patch.dict("sys.modules", {"assemblyai": MagicMock()}):
            provider = AssemblyAIProvider()
            assert provider.is_available() is True

    def test_not_available_without_key(self, monkeypatch):
        monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
        with patch.dict("sys.modules", {"assemblyai": MagicMock()}):
            provider = AssemblyAIProvider()
            assert provider.is_available() is False

    def test_not_available_without_sdk(self):
        with patch.dict("sys.modules", {"assemblyai": None}):
            provider = AssemblyAIProvider(api_key="test-key")
            assert provider.is_available() is False


class TestTranscribe:
    """Tests for transcribe() with mocked SDK responses."""

    @staticmethod
    def _make_word(text, start_ms, end_ms, confidence=0.99, speaker=None):
        """Helper to create a mock word object.

        Args:
            text: The word text.
            start_ms: Start time in milliseconds.
            end_ms: End time in milliseconds.
            confidence: Confidence score.
            speaker: Speaker identifier.
        """
        w = MagicMock()
        w.text = text
        w.start = start_ms
        w.end = end_ms
        w.confidence = confidence
        if speaker is not None:
            w.speaker = speaker
        return w

    @staticmethod
    def _make_utterance(text, start_ms, end_ms, confidence=0.99, speaker=None):
        """Helper to create a mock utterance object.

        Args:
            text: The utterance text.
            start_ms: Start time in milliseconds.
            end_ms: End time in milliseconds.
            confidence: Confidence score.
            speaker: Speaker identifier string (e.g., "A").
        """
        utt = MagicMock()
        utt.text = text
        utt.start = start_ms
        utt.end = end_ms
        utt.confidence = confidence
        if speaker is not None:
            utt.speaker = speaker
        return utt

    @staticmethod
    def _make_chapter(headline, summary, start_ms, end_ms):
        """Helper to create a mock chapter object."""
        ch = MagicMock()
        ch.headline = headline
        ch.summary = summary
        ch.start = start_ms
        ch.end = end_ms
        return ch

    @staticmethod
    def _make_transcript(
        text,
        words=None,
        utterances=None,
        chapters=None,
        sentiment_analysis=None,
        status="completed",
        error=None,
    ):
        """Helper to create a mock AssemblyAI transcript object."""
        transcript = MagicMock()
        transcript.text = text
        transcript.words = words or []
        transcript.utterances = utterances
        transcript.chapters = chapters
        transcript.sentiment_analysis = sentiment_analysis
        transcript.status = status
        transcript.error = error
        return transcript

    def _setup_provider(self, mock_transcript):
        """Set up a provider with a mocked client returning mock_transcript."""
        mock_client = MagicMock()
        mock_client.transcribe.return_value = mock_transcript

        provider = AssemblyAIProvider(api_key="test-key")
        provider._client = mock_client
        return provider, mock_client

    @pytest.mark.asyncio
    async def test_basic_transcription(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        words = [
            self._make_word("Hello", 0, 500),
            self._make_word("world.", 600, 1500),
            self._make_word("How", 2000, 2300),
            self._make_word("are", 2400, 2600),
            self._make_word("you?", 2700, 3500),
        ]
        mock_transcript = self._make_transcript(
            "Hello world. How are you?",
            words=words,
        )

        provider, _ = self._setup_provider(mock_transcript)

        mock_aai = _mock_assemblyai_module()
        with patch.dict("sys.modules", {"assemblyai": mock_aai}):
            result = await provider.transcribe(audio)

        assert result.text == "Hello world. How are you?"
        assert result.provider == "assemblyai"
        assert len(result.segments) >= 1
        # Words should be grouped into segments
        assert result.segments[0].start == 0.0

    @pytest.mark.asyncio
    async def test_transcription_with_language(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        words = [
            self._make_word("Hola", 0, 500),
            self._make_word("mundo.", 600, 1500),
        ]
        mock_transcript = self._make_transcript(
            "Hola mundo.",
            words=words,
        )

        provider, _ = self._setup_provider(mock_transcript)

        mock_aai = _mock_assemblyai_module()
        with patch.dict("sys.modules", {"assemblyai": mock_aai}):
            result = await provider.transcribe(audio, language="es")

        assert result.text == "Hola mundo."
        assert result.language == "es"
        # Verify TranscriptionConfig was called with language_code
        mock_aai.TranscriptionConfig.assert_called_once_with(
            speaker_labels=False,
            auto_chapters=False,
            sentiment_analysis=False,
            language_code="es",
        )

    @pytest.mark.asyncio
    async def test_transcription_with_diarization(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        utterances = [
            self._make_utterance("Hello there.", 0, 1500, speaker="A"),
            self._make_utterance("Hi, how are you?", 2000, 3500, speaker="B"),
            self._make_utterance("I'm doing well.", 4000, 5000, speaker="A"),
        ]
        mock_transcript = self._make_transcript(
            "Hello there. Hi, how are you? I'm doing well.",
            utterances=utterances,
        )

        provider, _ = self._setup_provider(mock_transcript)

        mock_aai = _mock_assemblyai_module()
        with patch.dict("sys.modules", {"assemblyai": mock_aai}):
            result = await provider.transcribe(audio, diarize=True)

        assert len(result.segments) == 3
        assert result.segments[0].speaker == "SPEAKER_A"
        assert result.segments[0].text == "Hello there."
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 1.5
        assert result.segments[1].speaker == "SPEAKER_B"
        assert result.segments[1].text == "Hi, how are you?"
        assert result.segments[2].speaker == "SPEAKER_A"
        assert result.segments[2].text == "I'm doing well."

    @pytest.mark.asyncio
    async def test_transcription_with_auto_chapters(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        words = [
            self._make_word("Hello", 0, 500),
            self._make_word("world.", 600, 1500),
        ]
        mock_transcript = self._make_transcript(
            "Hello world.",
            words=words,
            chapters=[
                self._make_chapter(
                    "Introduction",
                    "The speaker greets the audience.",
                    0,
                    1500,
                ),
            ],
        )

        provider, _ = self._setup_provider(mock_transcript)

        mock_aai = _mock_assemblyai_module()
        with patch.dict("sys.modules", {"assemblyai": mock_aai}):
            result = await provider.transcribe(audio, auto_chapters=True)

        assert result.text == "Hello world."
        # Verify TranscriptionConfig was called with auto_chapters
        mock_aai.TranscriptionConfig.assert_called_once_with(
            speaker_labels=False,
            auto_chapters=True,
            sentiment_analysis=False,
        )

    @pytest.mark.asyncio
    async def test_transcription_with_sentiment_analysis(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        words = [
            self._make_word("Great", 0, 500),
            self._make_word("day.", 600, 1000),
        ]
        mock_transcript = self._make_transcript(
            "Great day.",
            words=words,
            sentiment_analysis=[MagicMock(text="Great day.", sentiment="POSITIVE")],
        )

        provider, _ = self._setup_provider(mock_transcript)

        mock_aai = _mock_assemblyai_module()
        with patch.dict("sys.modules", {"assemblyai": mock_aai}):
            result = await provider.transcribe(audio, sentiment_analysis=True)

        assert result.text == "Great day."
        mock_aai.TranscriptionConfig.assert_called_once_with(
            speaker_labels=False,
            auto_chapters=False,
            sentiment_analysis=True,
        )

    @pytest.mark.asyncio
    async def test_word_grouping_by_gap(self, tmp_path):
        """Words with > 1 second gap are split into separate segments."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        words = [
            self._make_word("Hello", 0, 500),
            self._make_word("world.", 600, 1000),
            # Gap > 1 second (1000ms gap -> 1500ms to 2500ms)
            self._make_word("How", 2500, 2800),
            self._make_word("are", 2900, 3100),
            self._make_word("you?", 3200, 3500),
        ]
        mock_transcript = self._make_transcript(
            "Hello world. How are you?",
            words=words,
        )

        provider, _ = self._setup_provider(mock_transcript)

        mock_aai = _mock_assemblyai_module()
        with patch.dict("sys.modules", {"assemblyai": mock_aai}):
            result = await provider.transcribe(audio)

        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello world."
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 1.0
        assert result.segments[1].text == "How are you?"
        assert result.segments[1].start == 2.5
        assert result.segments[1].end == 3.5

    @pytest.mark.asyncio
    async def test_transcription_error(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        mock_transcript = self._make_transcript(
            None,
            status="error",
            error="Authentication failed",
        )

        provider, _ = self._setup_provider(mock_transcript)

        mock_aai = _mock_assemblyai_module()
        with (
            patch.dict("sys.modules", {"assemblyai": mock_aai}),
            pytest.raises(RuntimeError, match="Authentication failed"),
        ):
            await provider.transcribe(audio)

    @pytest.mark.asyncio
    async def test_duration_from_last_segment(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio data")

        words = [
            self._make_word("Hello", 0, 500),
            self._make_word("world.", 600, 5000),
        ]
        mock_transcript = self._make_transcript(
            "Hello world.",
            words=words,
        )

        provider, _ = self._setup_provider(mock_transcript)

        mock_aai = _mock_assemblyai_module()
        with patch.dict("sys.modules", {"assemblyai": mock_aai}):
            result = await provider.transcribe(audio)

        assert result.duration == 5.0


class TestFileNotFound:
    """Tests for file not found error handling."""

    @pytest.mark.asyncio
    async def test_audio_file_not_found(self):
        provider = AssemblyAIProvider()
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            await provider.transcribe(Path("/nonexistent/audio.mp3"))


class TestRegistryIntegration:
    """Tests for provider registry integration."""

    def test_get_provider_by_name(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("assemblyai")
        assert isinstance(provider, AssemblyAIProvider)

    def test_get_provider_with_kwargs(self):
        from claudetube.providers.registry import get_provider

        provider = get_provider("assemblyai", api_key="test-key")
        assert isinstance(provider, AssemblyAIProvider)
        assert provider._api_key == "test-key"

    def test_package_level_get_provider(self):
        from claudetube.providers import get_provider

        provider = get_provider("assemblyai")
        assert isinstance(provider, AssemblyAIProvider)
