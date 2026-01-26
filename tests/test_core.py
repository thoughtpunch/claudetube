"""Tests for claudetube."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from claudetube.fast import (
    extract_video_id,
    process_video,
    get_frames_at,
    VideoResult,
    _format_srt_time,
    _get_metadata,
    _find_tool,
)


class TestExtractVideoId:
    """Tests for video ID extraction."""

    # Positive cases - should extract valid IDs
    def test_standard_url(self):
        assert extract_video_id("https://youtube.com/watch?v=dYP2V_nK8o0") == "dYP2V_nK8o0"

    def test_standard_url_www(self):
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert extract_video_id("https://youtu.be/dYP2V_nK8o0") == "dYP2V_nK8o0"

    def test_embed_url(self):
        assert extract_video_id("https://youtube.com/embed/dYP2V_nK8o0") == "dYP2V_nK8o0"

    def test_just_id(self):
        assert extract_video_id("dYP2V_nK8o0") == "dYP2V_nK8o0"

    def test_with_extra_params(self):
        assert extract_video_id("https://youtube.com/watch?v=dYP2V_nK8o0&t=120") == "dYP2V_nK8o0"

    def test_with_playlist(self):
        assert extract_video_id("https://youtube.com/watch?v=dYP2V_nK8o0&list=PLtest") == "dYP2V_nK8o0"

    def test_mobile_url(self):
        assert extract_video_id("https://m.youtube.com/watch?v=dYP2V_nK8o0") == "dYP2V_nK8o0"

    def test_id_with_underscore(self):
        assert extract_video_id("abc_def_1234") == "abc_def_1234"

    def test_id_with_hyphen(self):
        assert extract_video_id("abc-def-1234") == "abc-def-1234"

    # Edge cases - function sanitizes invalid URLs instead of raising
    def test_invalid_url_gets_sanitized(self):
        # The function returns a sanitized version for invalid URLs
        result = extract_video_id("https://example.com/video")
        assert "/" not in result
        assert ":" not in result


class TestFormatSrtTime:
    """Tests for SRT timestamp formatting."""

    def test_zero(self):
        assert _format_srt_time(0) == "00:00:00,000"

    def test_seconds_only(self):
        assert _format_srt_time(45) == "00:00:45,000"

    def test_seconds_with_millis(self):
        assert _format_srt_time(45.5) == "00:00:45,500"

    def test_minutes_and_seconds(self):
        assert _format_srt_time(90.123) == "00:01:30,123"

    def test_hours_minutes_seconds(self):
        assert _format_srt_time(3661.5) == "01:01:01,500"

    def test_precise_milliseconds(self):
        assert _format_srt_time(1.234) == "00:00:01,234"

    def test_large_hours(self):
        assert _format_srt_time(7261) == "02:01:01,000"  # 2 hours, 1 min, 1 sec


class TestVideoResult:
    """Tests for VideoResult dataclass."""

    def test_create_success_result(self, tmp_path):
        result = VideoResult(
            success=True,
            video_id="test123",
            output_dir=tmp_path,
            transcript_srt=tmp_path / "audio.srt",
            transcript_txt=tmp_path / "audio.txt",
            metadata={"title": "Test Video"},
        )
        assert result.success is True
        assert result.video_id == "test123"
        assert result.error is None

    def test_create_error_result(self, tmp_path):
        result = VideoResult(
            success=False,
            video_id="test123",
            output_dir=tmp_path,
            error="Download failed",
        )
        assert result.success is False
        assert result.error == "Download failed"

    def test_default_fields(self, tmp_path):
        result = VideoResult(
            success=True,
            video_id="test123",
            output_dir=tmp_path,
        )
        assert result.frames == []
        assert result.metadata == {}
        assert result.transcript_srt is None


class TestFindTool:
    """Tests for tool finding."""

    @patch("shutil.which")
    def test_finds_system_tool(self, mock_which):
        mock_which.return_value = "/usr/bin/ffmpeg"
        result = _find_tool("ffmpeg")
        assert result == "/usr/bin/ffmpeg"

    @patch("shutil.which")
    def test_returns_name_when_not_found(self, mock_which):
        mock_which.return_value = None
        result = _find_tool("nonexistent")
        assert result == "nonexistent"


class TestProcessVideo:
    """Tests for video processing (mocked)."""

    def test_cache_hit_returns_immediately(self, tmp_path):
        """When transcript is already complete, should return cached result."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()

        # Create cached state
        state = {
            "video_id": "test12345678",
            "title": "Cached Video",
            "transcript_complete": True,
        }
        (video_dir / "state.json").write_text(json.dumps(state))
        (video_dir / "audio.srt").write_text("1\n00:00:00,000 --> 00:00:05,000\nTest\n")
        (video_dir / "audio.txt").write_text("Test")

        result = process_video(
            "test12345678",
            output_base=tmp_path
        )

        assert result.success is True
        assert result.video_id == "test12345678"
        assert result.metadata["title"] == "Cached Video"

    @patch("claudetube.fast._get_metadata")
    def test_returns_error_when_metadata_fails(self, mock_meta, tmp_path):
        mock_meta.return_value = {}

        result = process_video(
            "https://youtube.com/watch?v=test12345678",
            output_base=tmp_path
        )

        assert result.success is False
        assert "metadata" in result.error.lower()


class TestGetFramesAt:
    """Tests for frame extraction drill-in feature."""

    def test_returns_empty_when_no_video(self, tmp_path):
        """When video doesn't exist and can't be re-downloaded, returns empty."""
        video_dir = tmp_path / "nonexistent11"
        video_dir.mkdir()
        # No state.json means no URL to re-download from

        frames = get_frames_at(
            "nonexistent11",
            start_time=0,
            output_base=tmp_path
        )

        assert frames == []

    @patch("subprocess.run")
    def test_extracts_frames_at_intervals(self, mock_run, tmp_path):
        """Should extract frames at the specified interval."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        (video_dir / "video.mp4").write_bytes(b"fake video")

        # Mock ffmpeg success
        mock_run.return_value = MagicMock(returncode=0)

        # Create expected output files
        drill_dir = video_dir / "drill"
        drill_dir.mkdir()

        # Simulate ffmpeg creating files
        def create_frame(*args, **kwargs):
            cmd = args[0]
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = create_frame

        frames = get_frames_at(
            "test12345678",
            start_time=0,
            duration=3,
            interval=1,
            output_base=tmp_path
        )

        # Should have called ffmpeg 3 times (0s, 1s, 2s)
        assert mock_run.call_count == 3


class TestGetMetadata:
    """Tests for metadata fetching."""

    @patch("subprocess.run")
    def test_returns_dict_on_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"title": "Test", "duration": 120}'
        )

        result = _get_metadata("https://youtube.com/watch?v=test")

        assert result["title"] == "Test"
        assert result["duration"] == 120

    @patch("subprocess.run")
    def test_returns_empty_dict_on_failure(self, mock_run):
        mock_run.side_effect = Exception("Network error")

        result = _get_metadata("https://youtube.com/watch?v=test")

        assert result == {}

    @patch("subprocess.run")
    def test_returns_empty_dict_on_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)

        result = _get_metadata("https://youtube.com/watch?v=test")

        assert result == {}


class TestIntegration:
    """Integration tests (require network, skip in CI)."""

    @pytest.mark.skip(reason="Requires network and yt-dlp/ffmpeg")
    def test_full_pipeline(self, tmp_path):
        """Test the full download -> transcribe pipeline."""
        result = process_video(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            output_base=tmp_path,
            whisper_model="tiny",
        )

        assert result.success is True
        assert result.video_id == "dQw4w9WgXcQ"
        assert result.transcript_srt.exists()
        assert result.transcript_txt.exists()
        assert (result.output_dir / "state.json").exists()
