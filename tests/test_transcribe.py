"""Tests for operations/transcribe.py module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from claudetube.exceptions import TranscriptionError
from claudetube.operations.transcribe import (
    TranscribeOperation,
    transcribe_audio,
    transcribe_video,
)
from claudetube.providers.types import TranscriptionResult, TranscriptionSegment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(provider: str = "test-provider") -> TranscriptionResult:
    """Create a minimal TranscriptionResult for tests."""
    return TranscriptionResult(
        text="Hello world",
        segments=[TranscriptionSegment(start=0.0, end=1.5, text="Hello world")],
        language="en",
        duration=1.5,
        provider=provider,
    )


def _make_transcriber(result: TranscriptionResult | None = None) -> AsyncMock:
    """Create a mock Transcriber with an async transcribe method."""
    mock = AsyncMock()
    mock.transcribe.return_value = result or _make_result()
    return mock


# ---------------------------------------------------------------------------
# transcribe_audio (sync WhisperTool wrapper)
# ---------------------------------------------------------------------------


class TestTranscribeAudio:
    """Tests for the transcribe_audio() sync wrapper."""

    @patch("claudetube.operations.transcribe.WhisperTool")
    def test_returns_whisper_result(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        mock_tool.transcribe.return_value = {"srt": "1\n...", "txt": "Hello"}

        result = transcribe_audio(tmp_path / "audio.mp3", model_size="tiny")

        mock_cls.assert_called_once_with(model_size="tiny")
        mock_tool.transcribe.assert_called_once_with(tmp_path / "audio.mp3")
        assert result == {"srt": "1\n...", "txt": "Hello"}

    @patch("claudetube.operations.transcribe.WhisperTool")
    def test_passes_model_size(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        mock_tool.transcribe.return_value = {}

        transcribe_audio(tmp_path / "audio.mp3", model_size="large")

        mock_cls.assert_called_once_with(model_size="large")

    @patch("claudetube.operations.transcribe.WhisperTool")
    def test_default_model_size(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        mock_tool.transcribe.return_value = {}

        transcribe_audio(tmp_path / "audio.mp3")

        mock_cls.assert_called_once_with(model_size="tiny")

    @patch("claudetube.operations.transcribe.WhisperTool")
    def test_propagates_transcription_error(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        mock_tool.transcribe.side_effect = TranscriptionError("Whisper failed")

        with pytest.raises(TranscriptionError, match="Whisper failed"):
            transcribe_audio(tmp_path / "audio.mp3")


# ---------------------------------------------------------------------------
# TranscribeOperation
# ---------------------------------------------------------------------------


class TestTranscribeOperation:
    """Tests for TranscribeOperation.execute()."""

    @pytest.mark.asyncio
    async def test_execute_saves_srt_and_txt(self, tmp_path):
        result = _make_result(provider="whisper-local")
        transcriber = _make_transcriber(result)
        (tmp_path / "vid123").mkdir()

        op = TranscribeOperation(transcriber)
        out = await op.execute("vid123", tmp_path / "audio.mp3", cache_dir=tmp_path)

        srt_path = tmp_path / "vid123" / "audio.srt"
        txt_path = tmp_path / "vid123" / "audio.txt"

        assert out["success"] is True
        assert out["video_id"] == "vid123"
        assert out["source"] == "whisper-local"
        assert out["segments"] == 1
        assert out["duration"] == 1.5
        # Files should be written
        assert srt_path.exists()
        assert txt_path.exists()
        assert "Hello world" in txt_path.read_text()

    @pytest.mark.asyncio
    async def test_execute_updates_video_state(self, tmp_path):
        """When state.json exists, execute() should mark transcript_complete."""
        from claudetube.cache.manager import CacheManager
        from claudetube.models.state import VideoState

        cache = CacheManager(tmp_path)
        state = VideoState(
            video_id="vid123",
            url="https://example.com/v",
            title="Test",
            transcript_complete=False,
        )
        cache.save_state("vid123", state)

        transcriber = _make_transcriber()
        op = TranscribeOperation(transcriber)
        await op.execute("vid123", tmp_path / "audio.mp3", cache_dir=tmp_path)

        updated = cache.get_state("vid123")
        assert updated.transcript_complete is True
        assert updated.transcript_source == "test-provider"

    @pytest.mark.asyncio
    async def test_execute_passes_language(self, tmp_path):
        transcriber = _make_transcriber()
        (tmp_path / "vid123").mkdir()
        op = TranscribeOperation(transcriber)

        await op.execute(
            "vid123", tmp_path / "audio.mp3", language="es", cache_dir=tmp_path
        )

        transcriber.transcribe.assert_called_once_with(
            tmp_path / "audio.mp3", language="es"
        )

    @pytest.mark.asyncio
    async def test_execute_no_state_still_succeeds(self, tmp_path):
        """When no state.json exists, execute() should still succeed."""
        transcriber = _make_transcriber()
        (tmp_path / "vid123").mkdir()
        op = TranscribeOperation(transcriber)

        out = await op.execute("vid123", tmp_path / "audio.mp3", cache_dir=tmp_path)

        assert out["success"] is True

    @pytest.mark.asyncio
    async def test_execute_propagates_transcription_error(self, tmp_path):
        transcriber = AsyncMock()
        transcriber.transcribe.side_effect = TranscriptionError("Provider failed")

        op = TranscribeOperation(transcriber)
        with pytest.raises(TranscriptionError, match="Provider failed"):
            await op.execute("vid123", tmp_path / "audio.mp3", cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# transcribe_video (high-level wrapper)
# ---------------------------------------------------------------------------


class TestTranscribeVideo:
    """Tests for the transcribe_video() async function."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_immediately(self, tmp_path):
        """When SRT and TXT exist and force=False, returns cached result."""
        video_dir = tmp_path / "abc123"
        video_dir.mkdir(parents=True)
        (video_dir / "audio.srt").write_text("1\n00:00:00,000 --> 00:00:01,500\nHi\n")
        (video_dir / "audio.txt").write_text("Hi")

        result = await transcribe_video(
            "abc123", output_base=tmp_path, force=False
        )

        assert result["success"] is True
        assert result["source"] == "cached"
        assert result["message"] == "Returned cached transcript."

    @pytest.mark.asyncio
    async def test_force_bypasses_cache(self, tmp_path):
        """When force=True, should transcribe even if cached files exist."""
        video_dir = tmp_path / "abc123"
        video_dir.mkdir(parents=True)
        (video_dir / "audio.srt").write_text("old srt")
        (video_dir / "audio.txt").write_text("old txt")
        (video_dir / "audio.mp3").write_bytes(b"fake audio")

        transcriber = _make_transcriber(_make_result(provider="whisper-local"))

        result = await transcribe_video(
            "abc123",
            output_base=tmp_path,
            force=True,
            transcriber=transcriber,
        )

        assert result["success"] is True
        assert result["source"] == "whisper-local"
        transcriber.transcribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_audio_no_url_returns_error(self, tmp_path):
        """When no audio file and no URL available, returns failure."""
        video_dir = tmp_path / "abc123"
        video_dir.mkdir(parents=True)

        result = await transcribe_video("abc123", output_base=tmp_path)

        assert result["success"] is False
        assert "no URL available" in result["message"].lower() or "No audio file" in result["message"]

    @pytest.mark.asyncio
    async def test_no_audio_downloads_from_state_url(self, tmp_path):
        """When audio missing but state has URL, downloads audio then transcribes."""
        from claudetube.cache.manager import CacheManager
        from claudetube.models.state import VideoState

        cache = CacheManager(tmp_path)
        state = VideoState(
            video_id="abc123",
            url="https://example.com/video",
            title="Test Video",
        )
        cache.save_state("abc123", state)

        transcriber = _make_transcriber(_make_result(provider="whisper-local"))

        with patch("claudetube.operations.transcribe.download_audio") as mock_dl:
            # Simulate download_audio creating the audio file
            audio_path = tmp_path / "abc123" / "audio.mp3"

            def fake_download(url, path):
                path.write_bytes(b"fake audio")
                return path

            mock_dl.side_effect = fake_download

            result = await transcribe_video(
                "abc123",
                output_base=tmp_path,
                transcriber=transcriber,
            )

        mock_dl.assert_called_once_with("https://example.com/video", audio_path)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_no_audio_downloads_from_url_input(self, tmp_path):
        """When audio missing and input is a URL, uses that URL for download."""
        video_dir = tmp_path / "dQw4w9WgXcQ"
        video_dir.mkdir(parents=True)

        transcriber = _make_transcriber(_make_result(provider="whisper-local"))

        with patch("claudetube.operations.transcribe.download_audio") as mock_dl:
            def fake_download(url, path):
                path.write_bytes(b"fake audio")
                return path

            mock_dl.side_effect = fake_download

            result = await transcribe_video(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                output_base=tmp_path,
                transcriber=transcriber,
            )

        assert result["success"] is True
        assert mock_dl.called

    @pytest.mark.asyncio
    async def test_audio_download_failure_returns_error(self, tmp_path):
        """When audio download fails, returns failure dict."""
        from claudetube.cache.manager import CacheManager
        from claudetube.models.state import VideoState

        cache = CacheManager(tmp_path)
        state = VideoState(
            video_id="abc123",
            url="https://example.com/video",
            title="Test Video",
        )
        cache.save_state("abc123", state)

        with patch("claudetube.operations.transcribe.download_audio") as mock_dl:
            mock_dl.side_effect = RuntimeError("Network error")

            result = await transcribe_video("abc123", output_base=tmp_path)

        assert result["success"] is False
        assert "Audio download failed" in result["message"]

    @pytest.mark.asyncio
    async def test_transcription_error_returns_failure(self, tmp_path):
        """When transcription raises TranscriptionError, returns failure dict."""
        video_dir = tmp_path / "abc123"
        video_dir.mkdir(parents=True)
        (video_dir / "audio.mp3").write_bytes(b"fake audio")

        transcriber = AsyncMock()
        transcriber.transcribe.side_effect = TranscriptionError("Whisper crashed")

        result = await transcribe_video(
            "abc123",
            output_base=tmp_path,
            transcriber=transcriber,
        )

        assert result["success"] is False
        assert "Whisper crashed" in result["message"]

    @pytest.mark.asyncio
    async def test_file_not_found_error_returns_failure(self, tmp_path):
        """When transcription raises FileNotFoundError, returns failure dict."""
        video_dir = tmp_path / "abc123"
        video_dir.mkdir(parents=True)
        (video_dir / "audio.mp3").write_bytes(b"fake audio")

        transcriber = AsyncMock()
        transcriber.transcribe.side_effect = FileNotFoundError("audio.mp3 not found")

        result = await transcribe_video(
            "abc123",
            output_base=tmp_path,
            transcriber=transcriber,
        )

        assert result["success"] is False
        assert "audio.mp3 not found" in result["message"]

    @pytest.mark.asyncio
    async def test_default_provider_used_when_none(self, tmp_path):
        """When no transcriber provided, should create whisper-local provider."""
        video_dir = tmp_path / "abc123"
        video_dir.mkdir(parents=True)
        (video_dir / "audio.mp3").write_bytes(b"fake audio")

        mock_provider = _make_transcriber(_make_result(provider="whisper-local"))

        with patch("claudetube.providers.get_provider") as mock_get:
            mock_get.return_value = mock_provider

            result = await transcribe_video(
                "abc123",
                whisper_model="small",
                output_base=tmp_path,
            )

        mock_get.assert_called_once_with("whisper-local", model_size="small")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_successful_transcription_returns_full_result(self, tmp_path):
        """Verify all expected keys in a successful transcription result."""
        video_dir = tmp_path / "abc123"
        video_dir.mkdir(parents=True)
        (video_dir / "audio.mp3").write_bytes(b"fake audio")

        transcriber = _make_transcriber(
            _make_result(provider="whisper-local")
        )

        result = await transcribe_video(
            "abc123",
            output_base=tmp_path,
            transcriber=transcriber,
        )

        assert result["success"] is True
        assert result["video_id"] == "abc123"
        assert result["transcript_srt"] is not None
        assert result["transcript_txt"] is not None
        assert result["source"] == "whisper-local"
        assert result["segments"] == 1
        assert result["duration"] == 1.5

    @pytest.mark.asyncio
    async def test_cache_hit_only_when_both_files_exist(self, tmp_path):
        """If only SRT exists (no TXT), should NOT return cached."""
        video_dir = tmp_path / "abc123"
        video_dir.mkdir(parents=True)
        (video_dir / "audio.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")
        (video_dir / "audio.mp3").write_bytes(b"fake audio")
        # Note: audio.txt does NOT exist

        transcriber = _make_transcriber(_make_result(provider="whisper-local"))

        result = await transcribe_video(
            "abc123",
            output_base=tmp_path,
            transcriber=transcriber,
        )

        # Should NOT return cached since txt is missing
        assert result["source"] != "cached"
        transcriber.transcribe.assert_called_once()
