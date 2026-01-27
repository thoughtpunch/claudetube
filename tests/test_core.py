"""Tests for claudetube."""

import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claudetube.core import (
    QUALITY_LADDER,
    QUALITY_TIERS,
    VideoResult,
    _fetch_subtitles,
    _find_tool,
    _format_srt_time,
    _get_metadata,
    _log,
    extract_playlist_id,
    extract_url_context,
    extract_video_id,
    get_frames_at,
    get_hq_frames_at,
    next_quality,
    process_video,
)


class TestLogDoesNotWriteStdout:
    """Ensure _log() never writes to stdout (MCP stdio safety)."""

    def test_log_does_not_write_to_stdout(self):
        """_log() must not produce any stdout output."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            _log("test message")
            _log("test with time", start_time=0.0)
        finally:
            sys.stdout = old_stdout
        assert captured.getvalue() == "", "_log() wrote to stdout"


class TestExtractVideoId:
    """Tests for video ID extraction."""

    # Positive cases - should extract valid IDs
    def test_standard_url(self):
        assert (
            extract_video_id("https://youtube.com/watch?v=dYP2V_nK8o0") == "dYP2V_nK8o0"
        )

    def test_standard_url_www(self):
        assert (
            extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            == "dQw4w9WgXcQ"
        )

    def test_short_url(self):
        assert extract_video_id("https://youtu.be/dYP2V_nK8o0") == "dYP2V_nK8o0"

    def test_embed_url(self):
        assert (
            extract_video_id("https://youtube.com/embed/dYP2V_nK8o0") == "dYP2V_nK8o0"
        )

    def test_just_id(self):
        assert extract_video_id("dYP2V_nK8o0") == "dYP2V_nK8o0"

    def test_with_extra_params(self):
        assert (
            extract_video_id("https://youtube.com/watch?v=dYP2V_nK8o0&t=120")
            == "dYP2V_nK8o0"
        )

    def test_with_playlist(self):
        assert (
            extract_video_id("https://youtube.com/watch?v=dYP2V_nK8o0&list=PLtest")
            == "dYP2V_nK8o0"
        )

    def test_mobile_url(self):
        assert (
            extract_video_id("https://m.youtube.com/watch?v=dYP2V_nK8o0")
            == "dYP2V_nK8o0"
        )

    def test_id_with_underscore(self):
        assert extract_video_id("abc_def_1234") == "abc_def_1234"

    def test_id_with_hyphen(self):
        assert extract_video_id("abc-def-1234") == "abc-def-1234"

    # Negative cases
    def test_invalid_url_gets_sanitized(self):
        result = extract_video_id("https://example.com/video")
        assert "/" not in result
        assert ":" not in result

    def test_empty_string_returns_sanitized(self):
        result = extract_video_id("")
        assert isinstance(result, str)

    def test_too_short_id_not_matched_as_bare_id(self):
        # Bare IDs must be exactly 11 chars
        result = extract_video_id("abc")
        assert result == "abc"  # falls through to sanitize, but "abc" has no / or :

    def test_too_long_bare_id_not_matched(self):
        result = extract_video_id("a" * 12)
        assert result == "a" * 12  # sanitized but no slashes/colons to strip

    def test_url_with_no_video_id(self):
        result = extract_video_id("https://youtube.com/channel/UCxyz")
        assert isinstance(result, str)
        assert len(result) <= 20


class TestExtractPlaylistId:
    """Tests for playlist ID extraction."""

    # Positive cases
    def test_extracts_from_list_param(self):
        url = "https://youtube.com/watch?v=abc12345678&list=PLtest123"
        assert extract_playlist_id(url) == "PLtest123"

    def test_extracts_from_list_as_first_param(self):
        url = "https://youtube.com/watch?list=PLxyz&v=abc12345678"
        assert extract_playlist_id(url) == "PLxyz"

    def test_extracts_long_playlist_id(self):
        url = "https://youtube.com/watch?v=x&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        assert extract_playlist_id(url) == "PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"

    # Negative cases
    def test_returns_none_without_list(self):
        url = "https://youtube.com/watch?v=abc12345678"
        assert extract_playlist_id(url) is None

    def test_returns_none_for_empty_string(self):
        assert extract_playlist_id("") is None

    def test_returns_none_for_bare_id(self):
        assert extract_playlist_id("dYP2V_nK8o0") is None


class TestExtractUrlContext:
    """Tests for URL context extraction."""

    # Positive cases
    def test_extracts_full_context(self):
        url = "https://youtube.com/watch?v=dYP2V_nK8o0&list=PLtest"
        ctx = extract_url_context(url)
        assert ctx["video_id"] == "dYP2V_nK8o0"
        assert ctx["playlist_id"] == "PLtest"
        assert ctx["original_url"] == url
        assert "list=" not in ctx["clean_url"]

    def test_no_playlist(self):
        url = "https://youtube.com/watch?v=dYP2V_nK8o0"
        ctx = extract_url_context(url)
        assert ctx["video_id"] == "dYP2V_nK8o0"
        assert ctx["playlist_id"] is None
        assert ctx["clean_url"] == url

    # Negative cases
    def test_invalid_url_still_returns_dict(self):
        ctx = extract_url_context("not-a-url")
        assert "video_id" in ctx
        assert "playlist_id" in ctx
        assert ctx["playlist_id"] is None


class TestFormatSrtTime:
    """Tests for SRT timestamp formatting."""

    # Positive cases
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
        assert _format_srt_time(7261) == "02:01:01,000"

    # Negative / edge cases
    def test_fractional_seconds_near_boundary(self):
        result = _format_srt_time(59.999)
        assert result == "00:00:59,999"

    def test_exactly_one_hour(self):
        assert _format_srt_time(3600) == "01:00:00,000"

    def test_small_float(self):
        result = _format_srt_time(0.001)
        assert result == "00:00:00,001"


class TestVideoResult:
    """Tests for VideoResult dataclass."""

    # Positive cases
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
        assert result.transcript_srt == tmp_path / "audio.srt"
        assert result.transcript_txt == tmp_path / "audio.txt"
        assert result.metadata["title"] == "Test Video"

    def test_create_error_result(self, tmp_path):
        result = VideoResult(
            success=False,
            video_id="test123",
            output_dir=tmp_path,
            error="Download failed",
        )
        assert result.success is False
        assert result.error == "Download failed"
        assert result.transcript_srt is None
        assert result.transcript_txt is None

    def test_default_fields(self, tmp_path):
        result = VideoResult(
            success=True,
            video_id="test123",
            output_dir=tmp_path,
        )
        assert result.frames == []
        assert result.metadata == {}
        assert result.transcript_srt is None
        assert result.transcript_txt is None
        assert result.error is None

    # Negative cases
    def test_error_result_has_no_transcript(self, tmp_path):
        result = VideoResult(
            success=False,
            video_id="fail",
            output_dir=tmp_path,
            error="Network error",
        )
        assert result.success is False
        assert result.frames == []
        assert result.metadata == {}

    def test_frames_list_is_independent(self, tmp_path):
        """Default frames list should not be shared between instances."""
        r1 = VideoResult(success=True, video_id="a", output_dir=tmp_path)
        r2 = VideoResult(success=True, video_id="b", output_dir=tmp_path)
        r1.frames.append(Path("/fake"))
        assert r2.frames == []


class TestFindTool:
    """Tests for tool finding."""

    # Positive cases
    @patch("shutil.which")
    def test_finds_system_tool(self, mock_which):
        mock_which.return_value = "/usr/bin/ffmpeg"
        result = _find_tool("ffmpeg")
        assert result == "/usr/bin/ffmpeg"

    # Negative cases
    @patch("shutil.which")
    def test_returns_name_when_not_found(self, mock_which):
        mock_which.return_value = None
        result = _find_tool("nonexistent")
        assert result == "nonexistent"

    @patch("shutil.which")
    def test_returns_string_type(self, mock_which):
        mock_which.return_value = None
        result = _find_tool("anything")
        assert isinstance(result, str)


class TestProcessVideo:
    """Tests for video processing (mocked)."""

    # Positive cases
    def test_cache_hit_returns_immediately(self, tmp_path):
        """When transcript is already complete, should return cached result."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()

        state = {
            "video_id": "test12345678",
            "title": "Cached Video",
            "transcript_complete": True,
        }
        (video_dir / "state.json").write_text(json.dumps(state))
        (video_dir / "audio.srt").write_text("1\n00:00:00,000 --> 00:00:05,000\nTest\n")
        (video_dir / "audio.txt").write_text("Test")

        result = process_video("test12345678", output_base=tmp_path)

        assert result.success is True
        assert result.video_id == "test12345678"
        assert result.metadata["title"] == "Cached Video"
        assert result.transcript_srt is not None
        assert result.transcript_txt is not None

    def test_cache_hit_returns_srt_and_txt_paths(self, tmp_path):
        """Cached result should include paths to transcript files."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()

        state = {"transcript_complete": True}
        (video_dir / "state.json").write_text(json.dumps(state))
        (video_dir / "audio.srt").write_text("subtitle content")
        (video_dir / "audio.txt").write_text("text content")

        result = process_video("test12345678", output_base=tmp_path)

        assert result.transcript_srt == video_dir / "audio.srt"
        assert result.transcript_txt == video_dir / "audio.txt"

    def test_cache_hit_missing_files_returns_none(self, tmp_path):
        """Cache hit but missing files should return None for paths."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()

        state = {"transcript_complete": True}
        (video_dir / "state.json").write_text(json.dumps(state))
        # No audio.srt or audio.txt files

        result = process_video("test12345678", output_base=tmp_path)

        assert result.success is True
        assert result.transcript_srt is None
        assert result.transcript_txt is None

    # Negative cases
    @patch("claudetube.core._get_metadata")
    def test_returns_error_when_metadata_fails(self, mock_meta, tmp_path):
        mock_meta.return_value = {}

        result = process_video(
            "https://youtube.com/watch?v=test12345678", output_base=tmp_path
        )

        assert result.success is False
        assert "metadata" in result.error.lower()

    @patch("claudetube.core._get_metadata")
    def test_error_result_includes_video_id(self, mock_meta, tmp_path):
        """Even on error, result should have video_id set."""
        mock_meta.return_value = {}

        result = process_video(
            "https://youtube.com/watch?v=test1234567", output_base=tmp_path
        )

        assert result.video_id == "test1234567"
        assert result.output_dir == tmp_path / "test1234567"

    def test_incomplete_cache_not_treated_as_hit(self, tmp_path):
        """State with transcript_complete=False should not be a cache hit."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()

        state = {"transcript_complete": False}
        (video_dir / "state.json").write_text(json.dumps(state))

        with patch("claudetube.core._get_metadata") as mock_meta:
            mock_meta.return_value = {}
            result = process_video("test12345678", output_base=tmp_path)

        # Should attempt processing (and fail at metadata)
        assert result.success is False


class TestGetFramesAt:
    """Tests for frame extraction drill-in feature."""

    # Positive cases
    @patch("subprocess.run")
    def test_extracts_frames_at_intervals(self, mock_run, tmp_path):
        """Should extract frames at the specified interval."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        # Segment file matches: segment_lowest_{start}_{start+duration}.mp4
        (video_dir / "segment_lowest_0_3.mp4").write_bytes(b"fake video")

        def create_frame(*args, **kwargs):
            cmd = args[0]
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = create_frame

        frames = get_frames_at(
            "test12345678", start_time=0, duration=3, interval=1, output_base=tmp_path
        )

        assert mock_run.call_count == 3
        assert len(frames) == 3
        for f in frames:
            assert f.exists()
            assert f.suffix == ".jpg"

    @patch("subprocess.run")
    def test_frames_have_timestamp_names(self, mock_run, tmp_path):
        """Frame filenames should contain timestamp info."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        (video_dir / "segment_lowest_90_92.mp4").write_bytes(b"fake video")

        def create_frame(*args, **kwargs):
            cmd = args[0]
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = create_frame

        frames = get_frames_at(
            "test12345678", start_time=90, duration=2, interval=1, output_base=tmp_path
        )

        assert len(frames) == 2
        assert "drill_01-30" in frames[0].name
        assert "drill_01-31" in frames[1].name

    # Negative cases
    def test_returns_empty_when_no_video(self, tmp_path):
        """When video doesn't exist and can't be re-downloaded, returns empty."""
        video_dir = tmp_path / "nonexistent11"
        video_dir.mkdir()

        frames = get_frames_at("nonexistent11", start_time=0, output_base=tmp_path)

        assert frames == []
        assert isinstance(frames, list)

    @patch("subprocess.run")
    def test_skips_failed_frames(self, mock_run, tmp_path):
        """Frames that fail to extract should be skipped."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        (video_dir / "segment_lowest_0_3.mp4").write_bytes(b"fake video")

        mock_run.return_value = MagicMock(returncode=1)  # ffmpeg fails

        frames = get_frames_at(
            "test12345678", start_time=0, duration=3, interval=1, output_base=tmp_path
        )

        assert frames == []

    def test_returns_empty_for_nonexistent_id(self, tmp_path):
        """Non-existent video ID with no cache returns empty list."""
        frames = get_frames_at("XXXXXXXXXXX", start_time=0, output_base=tmp_path)
        assert frames == []


class TestGetHqFramesAt:
    """Tests for high-quality frame extraction."""

    # Positive cases
    @patch("subprocess.run")
    def test_extracts_hq_frames(self, mock_run, tmp_path):
        """Should extract HQ frames when state.json and video exist."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()

        state = {"url": "https://youtube.com/watch?v=test12345678"}
        (video_dir / "state.json").write_text(json.dumps(state))
        # Segment: segment_hq_{start}_{start+duration}.mp4
        (video_dir / "segment_hq_0_2.mp4").write_bytes(b"fake hq video")

        def create_frame(*args, **kwargs):
            cmd = args[0]
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake hq frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = create_frame

        frames = get_hq_frames_at(
            "test12345678",
            start_time=0,
            duration=2,
            interval=1,
            output_base=tmp_path,
        )

        assert len(frames) == 2
        for f in frames:
            assert f.exists()
            assert "hq_" in f.name

    # Negative cases
    def test_returns_empty_without_state(self, tmp_path):
        """No state.json means we can't get the URL to download."""
        frames = get_hq_frames_at("nonexistent11", start_time=0, output_base=tmp_path)
        assert frames == []

    def test_returns_empty_without_url_in_state(self, tmp_path):
        """State.json without url field should return empty."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()

        state = {"title": "No URL here"}
        (video_dir / "state.json").write_text(json.dumps(state))

        frames = get_hq_frames_at("test12345678", start_time=0, output_base=tmp_path)
        assert frames == []

    @patch("subprocess.run")
    def test_returns_empty_when_download_fails(self, mock_run, tmp_path):
        """If HQ video download fails, should return empty."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()

        state = {"url": "https://youtube.com/watch?v=test12345678"}
        (video_dir / "state.json").write_text(json.dumps(state))
        # No video_hq.mp4 - download will be attempted

        mock_run.return_value = MagicMock(returncode=1, stderr="download error")

        frames = get_hq_frames_at("test12345678", start_time=0, output_base=tmp_path)
        assert frames == []


class TestGetMetadata:
    """Tests for metadata fetching."""

    # Positive cases
    @patch("subprocess.run")
    def test_returns_dict_on_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"title": "Test", "duration": 120}'
        )

        result = _get_metadata("https://youtube.com/watch?v=test")

        assert result["title"] == "Test"
        assert result["duration"] == 120

    @patch("subprocess.run")
    def test_returns_all_fields(self, mock_run):
        meta = {"title": "T", "duration": 60, "uploader": "U", "tags": ["a"]}
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(meta))

        result = _get_metadata("https://youtube.com/watch?v=test")

        assert result["uploader"] == "U"
        assert result["tags"] == ["a"]

    # Negative cases
    @patch("subprocess.run")
    def test_returns_empty_dict_on_failure(self, mock_run):
        mock_run.side_effect = Exception("Network error")

        result = _get_metadata("https://youtube.com/watch?v=test")

        assert result == {}
        assert isinstance(result, dict)

    @patch("subprocess.run")
    def test_returns_empty_dict_on_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)

        result = _get_metadata("https://youtube.com/watch?v=test")

        assert result == {}

    @patch("subprocess.run")
    def test_returns_empty_dict_on_bad_json(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="not json")

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


class TestQualityTiers:
    """Tests for quality tier system."""

    def test_all_tiers_have_required_keys(self):
        required = {"sort", "width", "jpeg_q", "concurrent_fragments"}
        for name, tier in QUALITY_TIERS.items():
            assert required.issubset(tier.keys()), f"Tier '{name}' missing keys"

    def test_no_tier_has_format_key(self):
        """Tiers use -S sorting, not -f format selectors."""
        for name, tier in QUALITY_TIERS.items():
            assert "format" not in tier, f"Tier '{name}' should not have 'format' key"

    def test_ladder_order_matches_ascending_widths(self):
        widths = [QUALITY_TIERS[q]["width"] for q in QUALITY_LADDER]
        assert widths == sorted(widths), "QUALITY_LADDER widths should be ascending"

    def test_lowest_uses_ascending_sort(self):
        tier = QUALITY_TIERS["lowest"]
        assert tier["sort"].startswith("+"), "lowest should sort ascending (smallest)"
        assert tier["width"] == 480
        assert tier["jpeg_q"] == 5

    def test_highest_uses_res_cap(self):
        tier = QUALITY_TIERS["highest"]
        assert "res:1080" in tier["sort"]
        assert tier["width"] == 1280
        assert tier["jpeg_q"] == 2

    def test_all_tiers_have_sort_string(self):
        for name, tier in QUALITY_TIERS.items():
            assert isinstance(tier["sort"], str), f"Tier '{name}' sort must be string"
            assert len(tier["sort"]) > 0

    def test_all_ladder_entries_exist_in_tiers(self):
        for name in QUALITY_LADDER:
            assert name in QUALITY_TIERS

    def test_next_quality_returns_correct_next(self):
        assert next_quality("lowest") == "low"
        assert next_quality("low") == "medium"
        assert next_quality("medium") == "high"
        assert next_quality("high") == "highest"

    def test_next_quality_returns_none_at_highest(self):
        assert next_quality("highest") is None

    def test_next_quality_raises_on_invalid(self):
        with pytest.raises(ValueError):
            next_quality("ultra")

    def test_ladder_has_five_entries(self):
        assert len(QUALITY_LADDER) == 5


class TestGetFramesAtQuality:
    """Tests for get_frames_at with quality parameter."""

    @patch("subprocess.run")
    def test_default_quality_uses_lowest(self, mock_run, tmp_path):
        """Default quality should use lowest tier settings."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        (video_dir / "segment_lowest_0_1.mp4").write_bytes(b"fake video")

        def create_frame(*args, **kwargs):
            cmd = args[0]
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = create_frame

        frames = get_frames_at(
            "test12345678", start_time=0, duration=1, interval=1, output_base=tmp_path
        )

        assert len(frames) == 1
        assert "drill_lowest" in str(frames[0].parent)

    @patch("subprocess.run")
    def test_medium_quality_uses_sort_flag(self, mock_run, tmp_path):
        """Medium quality should use -S res:480 sort selector."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        state = {"url": "https://youtube.com/watch?v=test12345678"}
        (video_dir / "state.json").write_text(json.dumps(state))

        captured_cmds = []

        def handle_run(*args, **kwargs):
            cmd = args[0]
            captured_cmds.append(cmd)
            if "yt-dlp" in str(cmd[0]):
                # yt-dlp download call
                for i, arg in enumerate(cmd):
                    if arg == "-o" and i + 1 < len(cmd):
                        Path(cmd[i + 1]).write_bytes(b"fake video")
                        break
                return MagicMock(returncode=0)
            else:
                # ffmpeg frame extraction
                output_path = Path(cmd[-1])
                output_path.write_bytes(b"fake frame")
                return MagicMock(returncode=0)

        mock_run.side_effect = handle_run

        frames = get_frames_at(
            "test12345678",
            start_time=0,
            duration=1,
            interval=1,
            output_base=tmp_path,
            quality="medium",
        )

        assert len(frames) == 1
        assert "drill_medium" in str(frames[0].parent)
        # Verify -S flag used instead of -f
        ytdlp_cmd = captured_cmds[0]
        assert "-S" in ytdlp_cmd
        s_idx = ytdlp_cmd.index("-S")
        assert ytdlp_cmd[s_idx + 1] == "res:480"
        assert "-f" not in ytdlp_cmd

    @patch("subprocess.run")
    def test_different_tiers_produce_isolated_dirs(self, mock_run, tmp_path):
        """Each quality tier should use its own subdirectory."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()

        def create_frame(*args, **kwargs):
            cmd = args[0]
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = create_frame

        for q in ["lowest", "low"]:
            (video_dir / f"segment_{q}_0_1.mp4").write_bytes(b"fake video")
            get_frames_at(
                "test12345678",
                start_time=0,
                duration=1,
                interval=1,
                output_base=tmp_path,
                quality=q,
            )

        assert (video_dir / "drill_lowest").exists()
        assert (video_dir / "drill_low").exists()

    @patch("subprocess.run")
    def test_explicit_width_overrides_tier(self, mock_run, tmp_path):
        """Explicit width parameter should override tier default."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        (video_dir / "segment_lowest_0_1.mp4").write_bytes(b"fake video")

        captured_cmds = []

        def create_frame(*args, **kwargs):
            cmd = args[0]
            captured_cmds.append(cmd)
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = create_frame

        get_frames_at(
            "test12345678",
            start_time=0,
            duration=1,
            interval=1,
            output_base=tmp_path,
            width=320,
            quality="lowest",
        )

        # Check ffmpeg was called with width=320
        ffmpeg_cmd = captured_cmds[0]
        vf_idx = ffmpeg_cmd.index("-vf")
        assert "scale=320:-1" in ffmpeg_cmd[vf_idx + 1]

    @patch("subprocess.run")
    def test_segment_always_cleaned_up(self, mock_run, tmp_path):
        """Segment files should always be cleaned up after extraction."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        video_path = video_dir / "segment_high_0_1.mp4"
        video_path.write_bytes(b"fake video")

        def create_frame(*args, **kwargs):
            cmd = args[0]
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = create_frame

        get_frames_at(
            "test12345678",
            start_time=0,
            duration=1,
            interval=1,
            output_base=tmp_path,
            quality="high",
        )

        assert not video_path.exists(), "segment should be cleaned up"

    @patch("subprocess.run")
    def test_lowest_segment_cleaned_up(self, mock_run, tmp_path):
        """lowest tier segment should be cleaned up after extraction."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        video_path = video_dir / "segment_lowest_0_1.mp4"
        video_path.write_bytes(b"fake video")

        def create_frame(*args, **kwargs):
            cmd = args[0]
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = create_frame

        get_frames_at(
            "test12345678",
            start_time=0,
            duration=1,
            interval=1,
            output_base=tmp_path,
            quality="lowest",
        )

        assert not video_path.exists(), "segment should be cleaned up"

    @patch("subprocess.run")
    def test_quality_tracked_in_state_json(self, mock_run, tmp_path):
        """Quality extraction should be tracked in state.json."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        (video_dir / "segment_lowest_60_61.mp4").write_bytes(b"fake video")
        state = {"url": "https://youtube.com/watch?v=test12345678"}
        (video_dir / "state.json").write_text(json.dumps(state))

        def create_frame(*args, **kwargs):
            cmd = args[0]
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = create_frame

        get_frames_at(
            "test12345678",
            start_time=60,
            duration=1,
            interval=1,
            output_base=tmp_path,
            quality="lowest",
        )

        state = json.loads((video_dir / "state.json").read_text())
        assert "quality_extractions" in state
        assert "lowest" in state["quality_extractions"]
        assert state["quality_extractions"]["lowest"]["start_time"] == 60

    def test_invalid_quality_raises_error(self, tmp_path):
        """Invalid quality value should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid quality"):
            get_frames_at(
                "test12345678",
                start_time=0,
                output_base=tmp_path,
                quality="ultra_hd",
            )

    @patch("subprocess.run")
    def test_redownloads_segment_for_new_quality(self, mock_run, tmp_path):
        """Should download segment when it doesn't exist for a new quality tier."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        state = {"url": "https://youtube.com/watch?v=test12345678"}
        (video_dir / "state.json").write_text(json.dumps(state))

        call_count = [0]

        def handle_run(*args, **kwargs):
            cmd = args[0]
            call_count[0] += 1
            if call_count[0] == 1:
                # yt-dlp download — create segment file
                video_path = video_dir / "segment_medium_0_1.mp4"
                video_path.write_bytes(b"fake video")
                return MagicMock(returncode=0)
            else:
                # ffmpeg
                output_path = Path(cmd[-1])
                output_path.write_bytes(b"fake frame")
                return MagicMock(returncode=0)

        mock_run.side_effect = handle_run

        frames = get_frames_at(
            "test12345678",
            start_time=0,
            duration=1,
            interval=1,
            output_base=tmp_path,
            quality="medium",
        )

        assert len(frames) == 1
        assert call_count[0] >= 2  # at least one yt-dlp + one ffmpeg call


class TestFetchSubtitles:
    """Tests for subtitle fetching."""

    @patch("subprocess.run")
    def test_returns_srt_and_txt_when_subs_found(self, mock_run, tmp_path):
        """Should return parsed SRT and plain text when subtitles exist."""
        srt_content = (
            "1\n00:00:00,000 --> 00:00:05,000\nHello world\n\n"
            "2\n00:00:05,000 --> 00:00:10,000\nSecond line\n"
        )

        def write_sub(*args, **kwargs):
            (tmp_path / "test123.en.srt").write_text(srt_content)
            return MagicMock(returncode=0)

        mock_run.side_effect = write_sub

        result = _fetch_subtitles("https://youtube.com/watch?v=test123", tmp_path, 0)

        assert result is not None
        assert "Hello world" in result["txt"]
        assert "Second line" in result["txt"]
        assert result["source"] == "uploaded"

    @patch("subprocess.run")
    def test_returns_auto_generated_source(self, mock_run, tmp_path):
        """Auto-generated subs should be tagged as such."""
        srt_content = "1\n00:00:00,000 --> 00:00:05,000\nAuto text\n"

        def write_sub(*args, **kwargs):
            (tmp_path / "test123.auto.en.srt").write_text(srt_content)
            return MagicMock(returncode=0)

        mock_run.side_effect = write_sub

        result = _fetch_subtitles("https://youtube.com/watch?v=test123", tmp_path, 0)

        assert result is not None
        assert result["source"] == "auto-generated"

    @patch("subprocess.run")
    def test_returns_none_when_no_subs(self, mock_run, tmp_path):
        """Should return None when no subtitle files are produced."""
        mock_run.return_value = MagicMock(returncode=0)

        result = _fetch_subtitles("https://youtube.com/watch?v=test123", tmp_path, 0)

        assert result is None

    @patch("subprocess.run")
    def test_strips_html_tags(self, mock_run, tmp_path):
        """HTML tags in auto-subs should be stripped."""
        srt_content = (
            "1\n00:00:00,000 --> 00:00:05,000\n"
            "<font color='#ffffff'>Hello</font> <b>world</b>\n"
        )

        def write_sub(*args, **kwargs):
            (tmp_path / "test123.auto.en.srt").write_text(srt_content)
            return MagicMock(returncode=0)

        mock_run.side_effect = write_sub

        result = _fetch_subtitles("https://youtube.com/watch?v=test123", tmp_path, 0)

        assert "<font" not in result["txt"]
        assert "<b>" not in result["txt"]
        assert "Hello" in result["txt"]

    @patch("subprocess.run")
    def test_returns_none_on_empty_subs(self, mock_run, tmp_path):
        """Empty subtitle file should return None."""
        srt_content = "1\n00:00:00,000 --> 00:00:05,000\n\n"

        def write_sub(*args, **kwargs):
            (tmp_path / "test123.en.srt").write_text(srt_content)
            return MagicMock(returncode=0)

        mock_run.side_effect = write_sub

        result = _fetch_subtitles("https://youtube.com/watch?v=test123", tmp_path, 0)

        assert result is None

    @patch("subprocess.run")
    def test_handles_timeout(self, mock_run, tmp_path):
        """Timeout during subtitle fetch should return None."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)

        result = _fetch_subtitles("https://youtube.com/watch?v=test123", tmp_path, 0)

        assert result is None

    @patch("subprocess.run")
    def test_cleans_up_sub_files(self, mock_run, tmp_path):
        """Downloaded subtitle files should be cleaned up."""
        srt_content = "1\n00:00:00,000 --> 00:00:05,000\nHello\n"

        def write_sub(*args, **kwargs):
            (tmp_path / "test123.en.srt").write_text(srt_content)
            return MagicMock(returncode=0)

        mock_run.side_effect = write_sub

        _fetch_subtitles("https://youtube.com/watch?v=test123", tmp_path, 0)

        # The intermediate sub file should be cleaned up
        assert not (tmp_path / "test123.en.srt").exists()


class TestSubtitleFirstPipeline:
    """Tests for subtitle-first process_video pipeline."""

    @patch("claudetube.core._fetch_subtitles")
    @patch("claudetube.core._get_metadata")
    def test_uses_subs_when_available(self, mock_meta, mock_subs, tmp_path):
        """process_video should use subtitles and skip whisper when subs found."""
        mock_meta.return_value = {
            "title": "Test",
            "duration": 120,
            "duration_string": "2:00",
        }
        mock_subs.return_value = {
            "srt": "1\n00:00:00,000 --> 00:00:05,000\nHello\n",
            "txt": "Hello",
            "source": "uploaded",
        }

        result = process_video("test12345678", output_base=tmp_path)

        assert result.success is True
        assert result.transcript_srt is not None
        assert result.transcript_srt.exists()
        assert result.transcript_txt.exists()
        assert result.transcript_txt.read_text() == "Hello"
        state = json.loads((result.output_dir / "state.json").read_text())
        assert state["transcript_source"] == "uploaded"
        assert state["transcript_complete"] is True

    @patch("claudetube.core._transcribe_faster_whisper")
    @patch("claudetube.core._fetch_subtitles")
    @patch("claudetube.core._get_metadata")
    @patch("subprocess.run")
    def test_falls_back_to_whisper_when_no_subs(
        self, mock_run, mock_meta, mock_subs, mock_whisper, tmp_path
    ):
        """Should fall back to whisper when no subtitles available."""
        mock_meta.return_value = {
            "title": "Test",
            "duration": 120,
            "duration_string": "2:00",
        }
        mock_subs.return_value = None
        mock_whisper.return_value = {
            "srt": "1\n00:00:00,000 --> 00:00:05,000\nWhispered\n",
            "txt": "Whispered",
        }

        # Mock yt-dlp audio download to create audio.mp3
        def handle_run(cmd, *args, **kwargs):
            if isinstance(cmd, list) and "yt-dlp" in cmd[0]:
                # Find -o argument and create the file
                for i, arg in enumerate(cmd):
                    if arg == "-o" and i + 1 < len(cmd):
                        Path(cmd[i + 1]).write_bytes(b"fake audio")
                        break
            return MagicMock(returncode=0)

        mock_run.side_effect = handle_run

        result = process_video("test12345678", output_base=tmp_path)

        assert result.success is True
        mock_subs.assert_called_once()
        mock_whisper.assert_called_once()
        state = json.loads((result.output_dir / "state.json").read_text())
        assert state["transcript_source"] == "whisper"

    @patch("claudetube.core._fetch_subtitles")
    @patch("claudetube.core._get_metadata")
    def test_subtitle_source_tracked_in_state(self, mock_meta, mock_subs, tmp_path):
        """state.json should track whether transcript came from subs or whisper."""
        mock_meta.return_value = {
            "title": "Test",
            "duration": 60,
            "duration_string": "1:00",
        }
        mock_subs.return_value = {
            "srt": "1\n00:00:00,000 --> 00:00:05,000\nAuto\n",
            "txt": "Auto",
            "source": "auto-generated",
        }

        result = process_video("test12345678", output_base=tmp_path)

        state = json.loads((result.output_dir / "state.json").read_text())
        assert state["transcript_source"] == "auto-generated"

    @patch("claudetube.core._fetch_subtitles")
    @patch("claudetube.core._get_metadata")
    def test_no_video_downloaded_when_subs_found(self, mock_meta, mock_subs, tmp_path):
        """When subs are found, no video or audio file should be downloaded."""
        mock_meta.return_value = {
            "title": "Test",
            "duration": 60,
            "duration_string": "1:00",
        }
        mock_subs.return_value = {
            "srt": "1\n00:00:00,000 --> 00:00:05,000\nSub\n",
            "txt": "Sub",
            "source": "uploaded",
        }

        result = process_video("test12345678", output_base=tmp_path)

        assert not (result.output_dir / "video.mp4").exists()
        assert not (result.output_dir / "audio.mp3").exists()


class TestDownloadSections:
    """Tests for --download-sections partial video download."""

    @patch("subprocess.run")
    def test_download_sections_flag_in_command(self, mock_run, tmp_path):
        """yt-dlp command should include --download-sections with time range."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        state = {"url": "https://youtube.com/watch?v=test12345678"}
        (video_dir / "state.json").write_text(json.dumps(state))

        captured_cmds = []

        def handle_run(*args, **kwargs):
            cmd = args[0]
            captured_cmds.append(cmd)
            # Create segment file if yt-dlp call
            if "yt-dlp" in str(cmd[0]):
                for i, arg in enumerate(cmd):
                    if arg == "-o" and i + 1 < len(cmd):
                        Path(cmd[i + 1]).write_bytes(b"fake segment")
                        break
                return MagicMock(returncode=0)
            # Create frame if ffmpeg call
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = handle_run

        get_frames_at(
            "test12345678",
            start_time=60,
            duration=5,
            interval=5,
            output_base=tmp_path,
            quality="lowest",
        )

        # First call should be yt-dlp with --download-sections
        ytdlp_cmd = captured_cmds[0]
        assert "--download-sections" in ytdlp_cmd
        # Section should be ~58-67 (60-2 to 60+5+2)
        sections_idx = ytdlp_cmd.index("--download-sections")
        section_arg = ytdlp_cmd[sections_idx + 1]
        assert section_arg.startswith("*")
        assert "--force-keyframes-at-cuts" in ytdlp_cmd

    @patch("subprocess.run")
    def test_buffer_does_not_go_negative(self, mock_run, tmp_path):
        """Buffer padding should clamp to 0 for early timestamps."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        state = {"url": "https://youtube.com/watch?v=test12345678"}
        (video_dir / "state.json").write_text(json.dumps(state))

        captured_cmds = []

        def handle_run(*args, **kwargs):
            cmd = args[0]
            captured_cmds.append(cmd)
            if "yt-dlp" in str(cmd[0]):
                for i, arg in enumerate(cmd):
                    if arg == "-o" and i + 1 < len(cmd):
                        Path(cmd[i + 1]).write_bytes(b"fake segment")
                        break
                return MagicMock(returncode=0)
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = handle_run

        get_frames_at(
            "test12345678",
            start_time=1,
            duration=1,
            interval=1,
            output_base=tmp_path,
        )

        ytdlp_cmd = captured_cmds[0]
        sections_idx = ytdlp_cmd.index("--download-sections")
        section_arg = ytdlp_cmd[sections_idx + 1]
        # start=1, buffer=2 -> max(0, 1-2) = 0
        assert section_arg.startswith("*0")

    @patch("subprocess.run")
    def test_segment_file_naming(self, mock_run, tmp_path):
        """Segment files should include quality, start, and end in name."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        state = {"url": "https://youtube.com/watch?v=test12345678"}
        (video_dir / "state.json").write_text(json.dumps(state))

        captured_cmds = []

        def handle_run(*args, **kwargs):
            cmd = args[0]
            captured_cmds.append(cmd)
            if "yt-dlp" in str(cmd[0]):
                for i, arg in enumerate(cmd):
                    if arg == "-o" and i + 1 < len(cmd):
                        Path(cmd[i + 1]).write_bytes(b"fake segment")
                        break
                return MagicMock(returncode=0)
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = handle_run

        get_frames_at(
            "test12345678",
            start_time=30,
            duration=10,
            interval=10,
            output_base=tmp_path,
            quality="medium",
        )

        # Check segment file name in yt-dlp -o argument
        ytdlp_cmd = captured_cmds[0]
        o_idx = ytdlp_cmd.index("-o")
        output_name = Path(ytdlp_cmd[o_idx + 1]).name
        assert "segment_medium_30_40" in output_name

    @patch("subprocess.run")
    def test_segment_cleaned_up_after_extraction(self, mock_run, tmp_path):
        """Segment files should always be deleted after frame extraction."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        state = {"url": "https://youtube.com/watch?v=test12345678"}
        (video_dir / "state.json").write_text(json.dumps(state))

        def handle_run(*args, **kwargs):
            cmd = args[0]
            if "yt-dlp" in str(cmd[0]):
                for i, arg in enumerate(cmd):
                    if arg == "-o" and i + 1 < len(cmd):
                        Path(cmd[i + 1]).write_bytes(b"fake segment")
                        break
                return MagicMock(returncode=0)
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = handle_run

        get_frames_at(
            "test12345678",
            start_time=0,
            duration=1,
            interval=1,
            output_base=tmp_path,
            quality="highest",
        )

        # No segment files should remain
        segments = list(video_dir.glob("segment_*.mp4"))
        assert len(segments) == 0

    @patch("subprocess.run")
    def test_hq_uses_download_sections(self, mock_run, tmp_path):
        """get_hq_frames_at should also use --download-sections."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        state = {"url": "https://youtube.com/watch?v=test12345678"}
        (video_dir / "state.json").write_text(json.dumps(state))

        captured_cmds = []

        def handle_run(*args, **kwargs):
            cmd = args[0]
            captured_cmds.append(cmd)
            if "yt-dlp" in str(cmd[0]):
                for i, arg in enumerate(cmd):
                    if arg == "-o" and i + 1 < len(cmd):
                        Path(cmd[i + 1]).write_bytes(b"fake segment")
                        break
                return MagicMock(returncode=0)
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = handle_run

        get_hq_frames_at(
            "test12345678",
            start_time=120,
            duration=5,
            interval=5,
            output_base=tmp_path,
        )

        ytdlp_cmd = captured_cmds[0]
        assert "--download-sections" in ytdlp_cmd
        assert "--force-keyframes-at-cuts" in ytdlp_cmd


class TestThumbnailDownload:
    """Tests for automatic thumbnail download."""

    @patch("claudetube.core._fetch_subtitles")
    @patch("claudetube.core._get_metadata")
    @patch("subprocess.run")
    def test_thumbnail_downloaded_during_process(
        self, mock_run, mock_meta, mock_subs, tmp_path
    ):
        """process_video should download thumbnail during processing."""
        mock_meta.return_value = {
            "title": "Test",
            "duration": 60,
            "duration_string": "1:00",
        }
        mock_subs.return_value = {
            "srt": "1\n00:00:00,000 --> 00:00:05,000\nHello\n",
            "txt": "Hello",
            "source": "uploaded",
        }

        def handle_run(cmd, *args, **kwargs):
            if isinstance(cmd, list):
                if "--write-thumbnail" in cmd:
                    for i, arg in enumerate(cmd):
                        if arg == "-o" and i + 1 < len(cmd):
                            thumb_path = Path(cmd[i + 1]).with_suffix(".jpg")
                            thumb_path.parent.mkdir(parents=True, exist_ok=True)
                            thumb_path.write_bytes(b"fake thumbnail")
                            break
                    return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        mock_run.side_effect = handle_run

        result = process_video("test12345678", output_base=tmp_path)

        assert result.thumbnail is not None
        assert result.thumbnail.exists()
        assert result.thumbnail.name == "thumbnail.jpg"

    @patch("claudetube.core._fetch_subtitles")
    @patch("claudetube.core._get_metadata")
    @patch("subprocess.run")
    def test_thumbnail_failure_non_fatal(
        self, mock_run, mock_meta, mock_subs, tmp_path
    ):
        """Thumbnail download failure should not break the pipeline."""
        mock_meta.return_value = {
            "title": "Test",
            "duration": 60,
            "duration_string": "1:00",
        }
        mock_subs.return_value = {
            "srt": "1\n00:00:00,000 --> 00:00:05,000\nHello\n",
            "txt": "Hello",
            "source": "uploaded",
        }

        def handle_run(cmd, *args, **kwargs):
            if isinstance(cmd, list) and "--write-thumbnail" in cmd:
                raise subprocess.TimeoutExpired("cmd", 15)
            return MagicMock(returncode=0)

        mock_run.side_effect = handle_run

        result = process_video("test12345678", output_base=tmp_path)

        assert result.success is True
        assert result.thumbnail is None

    def test_cache_hit_includes_thumbnail(self, tmp_path):
        """Cache hit should return thumbnail path if it exists."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        (video_dir / "thumbnail.jpg").write_bytes(b"fake thumb")
        state = {"transcript_complete": True}
        (video_dir / "state.json").write_text(json.dumps(state))
        (video_dir / "audio.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
        (video_dir / "audio.txt").write_text("hi")

        result = process_video("test12345678", output_base=tmp_path)

        assert result.thumbnail is not None
        assert result.thumbnail.name == "thumbnail.jpg"

    def test_cache_hit_no_thumbnail(self, tmp_path):
        """Cache hit without thumbnail should return None."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        state = {"transcript_complete": True}
        (video_dir / "state.json").write_text(json.dumps(state))

        result = process_video("test12345678", output_base=tmp_path)

        assert result.thumbnail is None

    @patch("claudetube.core._fetch_subtitles")
    @patch("claudetube.core._get_metadata")
    @patch("subprocess.run")
    def test_thumbnail_tracked_in_state(self, mock_run, mock_meta, mock_subs, tmp_path):
        """state.json should track thumbnail existence."""
        mock_meta.return_value = {
            "title": "Test",
            "duration": 60,
            "duration_string": "1:00",
        }
        mock_subs.return_value = {
            "srt": "1\n00:00:00,000 --> 00:00:05,000\nHello\n",
            "txt": "Hello",
            "source": "uploaded",
        }

        def handle_run(cmd, *args, **kwargs):
            if isinstance(cmd, list) and "--write-thumbnail" in cmd:
                for i, arg in enumerate(cmd):
                    if arg == "-o" and i + 1 < len(cmd):
                        Path(cmd[i + 1]).with_suffix(".jpg").write_bytes(b"thumb")
                        break
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        mock_run.side_effect = handle_run

        result = process_video("test12345678", output_base=tmp_path)

        state = json.loads((result.output_dir / "state.json").read_text())
        assert state.get("has_thumbnail") is True


class TestConcurrentFragments:
    """Tests for -N concurrent fragment downloads."""

    def test_lowest_uses_one_fragment(self):
        assert QUALITY_TIERS["lowest"]["concurrent_fragments"] == 1

    def test_low_uses_two_fragments(self):
        assert QUALITY_TIERS["low"]["concurrent_fragments"] == 2

    def test_medium_uses_four_fragments(self):
        assert QUALITY_TIERS["medium"]["concurrent_fragments"] == 4

    def test_highest_uses_four_fragments(self):
        assert QUALITY_TIERS["highest"]["concurrent_fragments"] == 4

    @patch("subprocess.run")
    def test_n_flag_in_ytdlp_command(self, mock_run, tmp_path):
        """yt-dlp command should include -N with tier's fragment count."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        state = {"url": "https://youtube.com/watch?v=test12345678"}
        (video_dir / "state.json").write_text(json.dumps(state))

        captured_cmds = []

        def handle_run(*args, **kwargs):
            cmd = args[0]
            captured_cmds.append(cmd)
            if "yt-dlp" in str(cmd[0]):
                for i, arg in enumerate(cmd):
                    if arg == "-o" and i + 1 < len(cmd):
                        Path(cmd[i + 1]).write_bytes(b"fake video")
                        break
                return MagicMock(returncode=0)
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = handle_run

        get_frames_at(
            "test12345678",
            start_time=0,
            duration=1,
            interval=1,
            output_base=tmp_path,
            quality="high",
        )

        ytdlp_cmd = captured_cmds[0]
        assert "-N" in ytdlp_cmd
        n_idx = ytdlp_cmd.index("-N")
        assert ytdlp_cmd[n_idx + 1] == "4"

    @patch("subprocess.run")
    def test_lowest_n_flag_is_one(self, mock_run, tmp_path):
        """Lowest quality should use -N 1."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        state = {"url": "https://youtube.com/watch?v=test12345678"}
        (video_dir / "state.json").write_text(json.dumps(state))

        captured_cmds = []

        def handle_run(*args, **kwargs):
            cmd = args[0]
            captured_cmds.append(cmd)
            if "yt-dlp" in str(cmd[0]):
                for i, arg in enumerate(cmd):
                    if arg == "-o" and i + 1 < len(cmd):
                        Path(cmd[i + 1]).write_bytes(b"fake video")
                        break
                return MagicMock(returncode=0)
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = handle_run

        get_frames_at(
            "test12345678",
            start_time=0,
            duration=1,
            interval=1,
            output_base=tmp_path,
            quality="lowest",
        )

        ytdlp_cmd = captured_cmds[0]
        assert "-N" in ytdlp_cmd
        n_idx = ytdlp_cmd.index("-N")
        assert ytdlp_cmd[n_idx + 1] == "1"

    @patch("subprocess.run")
    def test_hq_uses_n_flag(self, mock_run, tmp_path):
        """get_hq_frames_at should use -N 4."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        state = {"url": "https://youtube.com/watch?v=test12345678"}
        (video_dir / "state.json").write_text(json.dumps(state))

        captured_cmds = []

        def handle_run(*args, **kwargs):
            cmd = args[0]
            captured_cmds.append(cmd)
            if "yt-dlp" in str(cmd[0]):
                for i, arg in enumerate(cmd):
                    if arg == "-o" and i + 1 < len(cmd):
                        Path(cmd[i + 1]).write_bytes(b"fake segment")
                        break
                return MagicMock(returncode=0)
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = handle_run

        get_hq_frames_at(
            "test12345678",
            start_time=0,
            duration=1,
            interval=1,
            output_base=tmp_path,
        )

        ytdlp_cmd = captured_cmds[0]
        assert "-N" in ytdlp_cmd
        n_idx = ytdlp_cmd.index("-N")
        assert ytdlp_cmd[n_idx + 1] == "4"
