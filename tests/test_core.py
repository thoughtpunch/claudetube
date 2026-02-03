"""Tests for claudetube."""

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Public API imports
from claudetube import (
    QUALITY_LADDER,
    QUALITY_TIERS,
    VideoResult,
    extract_playlist_id,
    extract_url_context,
    extract_video_id,
    get_frames_at,
    get_hq_frames_at,
    next_quality,
    process_video,
)

# Internal imports for testing internals
from claudetube.utils.formatting import format_srt_time
from claudetube.utils.logging import log_timed
from claudetube.utils.system import find_tool


class TestLogDoesNotWriteStdout:
    """Ensure log_timed() never writes to stdout (MCP stdio safety)."""

    def test_log_does_not_write_to_stdout(self):
        """log_timed() must not produce any stdout output."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            log_timed("test message")
            log_timed("test with time", start_time=0.0)
        finally:
            sys.stdout = old_stdout
        assert captured.getvalue() == "", "log_timed() wrote to stdout"


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
        assert format_srt_time(0) == "00:00:00,000"

    def test_seconds_only(self):
        assert format_srt_time(45) == "00:00:45,000"

    def test_seconds_with_millis(self):
        assert format_srt_time(45.5) == "00:00:45,500"

    def test_minutes_and_seconds(self):
        assert format_srt_time(90.123) == "00:01:30,123"

    def test_hours_minutes_seconds(self):
        assert format_srt_time(3661.5) == "01:01:01,500"

    def test_precise_milliseconds(self):
        assert format_srt_time(1.234) == "00:00:01,234"

    def test_large_hours(self):
        assert format_srt_time(7261) == "02:01:01,000"

    # Negative / edge cases
    def test_fractional_seconds_near_boundary(self):
        result = format_srt_time(59.999)
        assert result == "00:00:59,999"

    def test_exactly_one_hour(self):
        assert format_srt_time(3600) == "01:00:00,000"

    def test_small_float(self):
        result = format_srt_time(0.001)
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
        result = find_tool("ffmpeg")
        assert result == "/usr/bin/ffmpeg"

    # Negative cases
    @patch("shutil.which")
    def test_returns_name_when_not_found(self, mock_which):
        mock_which.return_value = None
        result = find_tool("nonexistent")
        assert result == "nonexistent"

    @patch("shutil.which")
    def test_returns_string_type(self, mock_which):
        mock_which.return_value = None
        result = find_tool("anything")
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
    @patch("claudetube.operations.processor.fetch_metadata")
    def test_returns_error_when_metadata_fails(self, mock_meta, tmp_path):
        from claudetube.exceptions import MetadataError

        mock_meta.side_effect = MetadataError("Failed to fetch")

        result = process_video(
            "https://youtube.com/watch?v=test12345678", output_base=tmp_path
        )

        assert result.success is False
        assert "fetch" in result.error.lower() or "failed" in result.error.lower()

    @patch("claudetube.operations.processor.fetch_metadata")
    def test_error_result_includes_video_id(self, mock_meta, tmp_path):
        """Even on error, result should have video_id set."""
        from claudetube.exceptions import MetadataError

        mock_meta.side_effect = MetadataError("Failed")

        result = process_video(
            "https://youtube.com/watch?v=test1234567", output_base=tmp_path
        )

        assert result.video_id == "test1234567"
        # With hierarchical paths, the output_dir is youtube/no_channel/no_playlist/video_id
        assert (
            result.output_dir
            == tmp_path / "youtube" / "no_channel" / "no_playlist" / "test1234567"
        )

    def test_incomplete_cache_not_treated_as_hit(self, tmp_path):
        """State with transcript_complete=False should not be a cache hit."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()

        state = {"transcript_complete": False}
        (video_dir / "state.json").write_text(json.dumps(state))

        with patch("claudetube.operations.processor.fetch_metadata") as mock_meta:
            from claudetube.exceptions import MetadataError

            mock_meta.side_effect = MetadataError("Failed")
            result = process_video("test12345678", output_base=tmp_path)

        # Should attempt processing (and fail at metadata)
        assert result.success is False


class TestGetFramesAt:
    """Tests for frame extraction drill-in feature."""

    # Positive cases
    @patch("claudetube.tools.ffmpeg.subprocess.run")
    def test_extracts_frames_at_intervals(self, mock_run, tmp_path):
        """Should extract frames at the specified interval."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        # Segment file matches: segment_lowest_{start}_{start+duration}.mp4
        (video_dir / "segment_lowest_0_3.mp4").write_bytes(b"fake video")

        def create_frame(*args, **kwargs):
            cmd = args[0]
            output_path = Path(cmd[-1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
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

    @patch("claudetube.tools.ffmpeg.subprocess.run")
    def test_frames_have_timestamp_names(self, mock_run, tmp_path):
        """Frame filenames should contain timestamp info."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        (video_dir / "segment_lowest_90_92.mp4").write_bytes(b"fake video")

        def create_frame(*args, **kwargs):
            cmd = args[0]
            output_path = Path(cmd[-1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
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

    @patch("claudetube.tools.ffmpeg.subprocess.run")
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
    @patch("claudetube.tools.ffmpeg.subprocess.run")
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
            output_path.parent.mkdir(parents=True, exist_ok=True)
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

    def test_next_quality_returns_none_on_invalid(self):
        # Now returns None instead of raising
        assert next_quality("ultra") is None

    def test_ladder_has_five_entries(self):
        assert len(QUALITY_LADDER) == 5


class TestGetFramesAtQuality:
    """Tests for get_frames_at with quality parameter."""

    @patch("claudetube.tools.ffmpeg.subprocess.run")
    def test_default_quality_uses_lowest(self, mock_run, tmp_path):
        """Default quality should use lowest tier settings."""
        video_dir = tmp_path / "test12345678"
        video_dir.mkdir()
        (video_dir / "segment_lowest_0_1.mp4").write_bytes(b"fake video")

        def create_frame(*args, **kwargs):
            cmd = args[0]
            output_path = Path(cmd[-1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fake frame")
            return MagicMock(returncode=0)

        mock_run.side_effect = create_frame

        frames = get_frames_at(
            "test12345678", start_time=0, duration=1, interval=1, output_base=tmp_path
        )

        assert len(frames) == 1
        assert "drill_lowest" in str(frames[0].parent)

    def test_invalid_quality_raises_error(self, tmp_path):
        """Invalid quality value should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid quality"):
            get_frames_at(
                "test12345678",
                start_time=0,
                output_base=tmp_path,
                quality="ultra_hd",
            )


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
