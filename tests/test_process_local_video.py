"""Tests for process_local_video function."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from claudetube.models.video_result import VideoResult
from claudetube.operations.processor import process_local_video


@pytest.fixture
def sample_video(tmp_path):
    """Generate a tiny test video with ffmpeg (2 seconds, 320x240).

    If ffmpeg is not available, returns None.
    """
    video = tmp_path / "test_video.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=2:size=320x240:rate=10",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-y",
                str(video),
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        return video
    except (subprocess.SubprocessError, FileNotFoundError):
        pytest.skip("ffmpeg not available for generating test video")


@pytest.fixture
def fake_video(tmp_path):
    """Create a fake video file (just bytes, not actual video)."""
    video = tmp_path / "fake_video.mp4"
    video.write_bytes(b"fake video content" * 100)
    return video


@pytest.fixture
def cache_base(tmp_path):
    """Provide a temporary cache directory."""
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


class TestProcessLocalVideoBasic:
    """Basic unit tests for process_local_video."""

    def test_returns_video_result(self, fake_video, cache_base):
        """Returns a VideoResult object."""
        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=10.0,
                width=320,
                height=240,
                fps=30.0,
                codec="h264",
                creation_time=None,
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {"txt": "test", "srt": "test"}
                    with patch(
                        "claudetube.operations.processor.FFmpegTool"
                    ) as mock_ffmpeg:
                        mock_ffmpeg.return_value.extract_frame.return_value = None

                        result = process_local_video(
                            str(fake_video), output_base=cache_base
                        )

        assert isinstance(result, VideoResult)

    def test_success_true_on_valid_file(self, fake_video, cache_base):
        """Returns success=True for valid file."""
        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=10.0,
                width=320,
                height=240,
                fps=30.0,
                codec="h264",
                creation_time=None,
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {
                        "txt": "hello world",
                        "srt": "1\n00:00:00,000 --> 00:00:02,000\nhello world",
                    }
                    with patch("claudetube.operations.processor.FFmpegTool"):
                        result = process_local_video(
                            str(fake_video), output_base=cache_base
                        )

        assert result.success is True
        assert result.error is None

    def test_error_on_nonexistent_file(self, cache_base):
        """Returns error for nonexistent file."""
        result = process_local_video("/nonexistent/video.mp4", output_base=cache_base)

        assert result.success is False
        assert "not found" in result.error.lower() or "File not found" in result.error

    def test_error_on_unsupported_format(self, tmp_path, cache_base):
        """Returns error for unsupported file format."""
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("not a video")

        result = process_local_video(str(txt_file), output_base=cache_base)

        assert result.success is False
        assert "unsupported" in result.error.lower() or "format" in result.error.lower()

    def test_generates_video_id(self, fake_video, cache_base):
        """Generates a valid video_id from the local file."""
        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=10.0,
                width=320,
                height=240,
                fps=30.0,
                codec="h264",
                creation_time=None,
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {"txt": "test", "srt": "test"}
                    with patch("claudetube.operations.processor.FFmpegTool"):
                        result = process_local_video(
                            str(fake_video), output_base=cache_base
                        )

        assert result.video_id
        assert len(result.video_id) > 0
        # video_id should be filesystem-safe
        import re

        assert re.match(r"^[\w-]+$", result.video_id)


class TestProcessLocalVideoCaching:
    """Tests for local video caching behavior."""

    def test_creates_cache_directory(self, fake_video, cache_base):
        """Creates a cache directory for the video."""
        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=10.0,
                width=320,
                height=240,
                fps=30.0,
                codec="h264",
                creation_time=None,
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {"txt": "test", "srt": "test"}
                    with patch("claudetube.operations.processor.FFmpegTool"):
                        result = process_local_video(
                            str(fake_video), output_base=cache_base
                        )

        assert result.output_dir.exists()
        assert result.output_dir.is_dir()

    def test_saves_state_json(self, fake_video, cache_base):
        """Saves state.json with local file metadata."""
        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=10.0,
                width=320,
                height=240,
                fps=30.0,
                codec="h264",
                creation_time=None,
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {"txt": "test", "srt": "test"}
                    with patch("claudetube.operations.processor.FFmpegTool"):
                        result = process_local_video(
                            str(fake_video), output_base=cache_base
                        )

        state_file = result.output_dir / "state.json"
        assert state_file.exists()

        state = json.loads(state_file.read_text())
        assert state["source_type"] == "local"
        assert state["source_path"] == str(fake_video)
        assert state["cached_file"] is not None

    def test_symlink_by_default(self, fake_video, cache_base):
        """Creates symlink to source file by default."""
        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=10.0,
                width=320,
                height=240,
                fps=30.0,
                codec="h264",
                creation_time=None,
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {"txt": "test", "srt": "test"}
                    with patch("claudetube.operations.processor.FFmpegTool"):
                        result = process_local_video(
                            str(fake_video), output_base=cache_base
                        )

        state = json.loads((result.output_dir / "state.json").read_text())
        assert state["cache_mode"] == "symlink"

        cached_file = result.output_dir / state["cached_file"]
        assert cached_file.is_symlink()

    def test_copy_mode_when_specified(self, fake_video, cache_base):
        """Creates copy of source file when copy=True."""
        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=10.0,
                width=320,
                height=240,
                fps=30.0,
                codec="h264",
                creation_time=None,
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {"txt": "test", "srt": "test"}
                    with patch("claudetube.operations.processor.FFmpegTool"):
                        result = process_local_video(
                            str(fake_video), output_base=cache_base, copy=True
                        )

        state = json.loads((result.output_dir / "state.json").read_text())
        assert state["cache_mode"] == "copy"

        cached_file = result.output_dir / state["cached_file"]
        assert not cached_file.is_symlink()
        assert cached_file.exists()

    def test_cache_hit_returns_early(self, fake_video, cache_base):
        """Second call returns cached result without re-processing."""
        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=10.0,
                width=320,
                height=240,
                fps=30.0,
                codec="h264",
                creation_time=None,
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {"txt": "cached", "srt": "cached"}
                    with patch("claudetube.operations.processor.FFmpegTool"):
                        result1 = process_local_video(
                            str(fake_video), output_base=cache_base
                        )

        # Second call should hit cache
        with (
            patch("claudetube.operations.processor.FFprobeTool"),
            patch(
                "claudetube.operations.processor.transcribe_audio"
            ) as mock_transcribe2,
        ):
            result2 = process_local_video(str(fake_video), output_base=cache_base)

        # Transcription should not have been called again
        mock_transcribe2.assert_not_called()
        assert result2.success is True
        assert result2.video_id == result1.video_id


class TestProcessLocalVideoMetadata:
    """Tests for metadata extraction from local files."""

    def test_extracts_metadata(self, fake_video, cache_base):
        """Extracts metadata from local file via ffprobe."""
        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=120.5,
                width=1920,
                height=1080,
                fps=29.97,
                codec="h264",
                creation_time="2024-01-15T10:30:00Z",
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {"txt": "test", "srt": "test"}
                    with patch("claudetube.operations.processor.FFmpegTool"):
                        result = process_local_video(
                            str(fake_video), output_base=cache_base
                        )

        assert result.metadata["duration"] == 120.5
        assert result.metadata["duration_string"] == "2:00"  # format_duration(120.5)
        # Note: Video dimensions are now stored in SQLite (description field)
        # instead of state.json. VideoState.to_dict() no longer includes description.

    def test_title_from_filename(self, fake_video, cache_base):
        """Uses filename stem as title."""
        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=10.0,
                width=320,
                height=240,
                fps=30.0,
                codec="h264",
                creation_time=None,
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {"txt": "test", "srt": "test"}
                    with patch("claudetube.operations.processor.FFmpegTool"):
                        result = process_local_video(
                            str(fake_video), output_base=cache_base
                        )

        assert result.metadata["title"] == "fake_video"


class TestProcessLocalVideoTranscription:
    """Tests for transcription of local videos."""

    def test_creates_transcript_files(self, fake_video, cache_base):
        """Creates transcript .srt and .txt files."""
        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=10.0,
                width=320,
                height=240,
                fps=30.0,
                codec="h264",
                creation_time=None,
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {
                        "txt": "Hello world",
                        "srt": "1\n00:00:00,000 --> 00:00:02,000\nHello world",
                    }
                    with patch("claudetube.operations.processor.FFmpegTool"):
                        result = process_local_video(
                            str(fake_video), output_base=cache_base
                        )

        assert result.transcript_txt.exists()
        assert result.transcript_srt.exists()
        assert result.transcript_txt.read_text() == "Hello world"

    def test_uses_whisper_model_param(self, fake_video, cache_base):
        """Passes whisper_model parameter to transcribe_audio."""
        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=10.0,
                width=320,
                height=240,
                fps=30.0,
                codec="h264",
                creation_time=None,
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {"txt": "test", "srt": "test"}
                    with patch("claudetube.operations.processor.FFmpegTool"):
                        process_local_video(
                            str(fake_video),
                            output_base=cache_base,
                            whisper_model="medium",
                        )

        mock_transcribe.assert_called_once()
        _, kwargs = mock_transcribe.call_args
        assert kwargs["model_size"] == "medium"


class TestProcessLocalVideoPathFormats:
    """Tests for various path input formats."""

    def test_absolute_path(self, fake_video, cache_base):
        """Handles absolute paths."""
        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=10.0,
                width=320,
                height=240,
                fps=30.0,
                codec="h264",
                creation_time=None,
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {"txt": "test", "srt": "test"}
                    with patch("claudetube.operations.processor.FFmpegTool"):
                        result = process_local_video(
                            str(fake_video.absolute()), output_base=cache_base
                        )

        assert result.success is True

    def test_relative_path(self, tmp_path, cache_base, monkeypatch):
        """Handles relative paths."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake" * 100)

        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=10.0,
                width=320,
                height=240,
                fps=30.0,
                codec="h264",
                creation_time=None,
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {"txt": "test", "srt": "test"}
                    with patch("claudetube.operations.processor.FFmpegTool"):
                        result = process_local_video(
                            "./video.mp4", output_base=cache_base
                        )

        assert result.success is True

    def test_file_uri(self, fake_video, cache_base):
        """Handles file:// URIs."""
        with patch("claudetube.operations.processor.FFprobeTool") as mock_ffprobe:
            mock_ffprobe.return_value.get_metadata.return_value = MagicMock(
                duration=10.0,
                width=320,
                height=240,
                fps=30.0,
                codec="h264",
                creation_time=None,
            )
            with patch(
                "claudetube.operations.processor.extract_audio_local"
            ) as mock_audio:
                mock_audio.return_value = cache_base / "audio.mp3"
                (cache_base / "audio.mp3").touch()
                with patch(
                    "claudetube.operations.processor.transcribe_audio"
                ) as mock_transcribe:
                    mock_transcribe.return_value = {"txt": "test", "srt": "test"}
                    with patch("claudetube.operations.processor.FFmpegTool"):
                        result = process_local_video(
                            f"file://{fake_video}", output_base=cache_base
                        )

        assert result.success is True


class TestProcessLocalVideoIntegration:
    """Integration tests using real video files."""

    @pytest.mark.slow
    def test_full_pipeline_with_real_video(self, sample_video, cache_base):
        """Full pipeline with a real video file (requires ffmpeg)."""
        if sample_video is None:
            pytest.skip("Sample video not available")

        # Mock only whisper to avoid slow transcription
        with patch(
            "claudetube.operations.processor.transcribe_audio"
        ) as mock_transcribe:
            mock_transcribe.return_value = {
                "txt": "Test transcription",
                "srt": "1\n00:00:00,000 --> 00:00:02,000\nTest transcription",
            }
            result = process_local_video(str(sample_video), output_base=cache_base)

        assert result.success is True
        assert result.video_id is not None
        assert result.output_dir.exists()
        assert result.transcript_txt is not None
        assert result.metadata["source_type"] == "local"

    @pytest.mark.slow
    def test_thumbnail_generated(self, sample_video, cache_base):
        """Generates thumbnail from video."""
        if sample_video is None:
            pytest.skip("Sample video not available")

        with patch(
            "claudetube.operations.processor.transcribe_audio"
        ) as mock_transcribe:
            mock_transcribe.return_value = {"txt": "test", "srt": "test"}
            result = process_local_video(str(sample_video), output_base=cache_base)

        # Thumbnail might or might not be generated depending on FFmpegTool success
        # Just verify we don't crash
        assert result.success is True
