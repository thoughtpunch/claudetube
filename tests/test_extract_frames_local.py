"""Tests for local file frame extraction."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claudetube.cache.manager import CacheManager
from claudetube.models.state import VideoState
from claudetube.operations.extract_frames import (
    QUALITY_WIDTHS,
    extract_frames_local,
    extract_hq_frames_local,
)


class TestQualityWidths:
    """Tests for quality width mapping."""

    def test_all_quality_tiers_defined(self):
        """All quality tiers have width mappings."""
        expected_tiers = ["lowest", "low", "medium", "high", "highest"]
        for tier in expected_tiers:
            assert tier in QUALITY_WIDTHS

    def test_widths_increase_with_quality(self):
        """Higher quality = higher width."""
        assert QUALITY_WIDTHS["lowest"] < QUALITY_WIDTHS["low"]
        assert QUALITY_WIDTHS["low"] < QUALITY_WIDTHS["medium"]
        assert QUALITY_WIDTHS["medium"] < QUALITY_WIDTHS["high"]
        assert QUALITY_WIDTHS["high"] <= QUALITY_WIDTHS["highest"]

    def test_specific_widths(self):
        """Width values match spec."""
        assert QUALITY_WIDTHS["lowest"] == 480
        assert QUALITY_WIDTHS["low"] == 640
        assert QUALITY_WIDTHS["medium"] == 854
        assert QUALITY_WIDTHS["high"] == 1280
        assert QUALITY_WIDTHS["highest"] == 1920


class TestExtractFramesLocal:
    """Tests for extract_frames_local function."""

    def test_raises_on_invalid_quality(self, tmp_path):
        """Invalid quality tier raises ValueError."""
        with pytest.raises(ValueError, match="Invalid quality 'ultra'"):
            extract_frames_local(
                video_id="test",
                start_time=0,
                quality="ultra",
                output_base=tmp_path,
            )

    def test_raises_on_missing_video(self, tmp_path):
        """Missing video raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Video not cached"):
            extract_frames_local(
                video_id="nonexistent",
                start_time=0,
                output_base=tmp_path,
            )

    def test_raises_on_url_video(self, tmp_path):
        """URL-type video raises ValueError."""
        cache = CacheManager(tmp_path)
        state = VideoState(
            video_id="url_video",
            source_type="url",
            url="https://youtube.com/watch?v=abc123",
        )
        cache.save_state("url_video", state)

        with pytest.raises(ValueError, match="not a local file"):
            extract_frames_local(
                video_id="url_video",
                start_time=0,
                output_base=tmp_path,
            )

    def test_raises_on_missing_cached_file(self, tmp_path):
        """Missing cached_file in state raises FileNotFoundError."""
        cache = CacheManager(tmp_path)
        state = VideoState(
            video_id="local_video",
            source_type="local",
            source_path="/path/to/video.mp4",
            # No cached_file set
        )
        cache.save_state("local_video", state)

        with pytest.raises(FileNotFoundError, match="No cached source file"):
            extract_frames_local(
                video_id="local_video",
                start_time=0,
                output_base=tmp_path,
            )

    def test_raises_on_broken_symlink(self, tmp_path):
        """Broken symlink raises FileNotFoundError."""
        # Create source and cache it
        source = tmp_path / "videos" / "test.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"fake video content")

        cache = CacheManager(tmp_path / "cache")
        dest, mode = cache.cache_local_file("local_video", source)

        state = VideoState(
            video_id="local_video",
            source_type="local",
            source_path=str(source),
            cache_mode=mode,
            cached_file="source.mp4",
        )
        cache.save_state("local_video", state)

        # Delete source to break symlink
        source.unlink()

        with pytest.raises(FileNotFoundError, match="Source file unavailable"):
            extract_frames_local(
                video_id="local_video",
                start_time=0,
                output_base=tmp_path / "cache",
            )

    @patch("claudetube.operations.extract_frames.FFmpegTool")
    def test_extracts_frames_with_correct_params(self, mock_ffmpeg_class, tmp_path):
        """Calls FFmpegTool with correct parameters."""
        # Create source and cache it
        source = tmp_path / "videos" / "test.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"fake video content")

        cache = CacheManager(tmp_path / "cache")
        dest, mode = cache.cache_local_file("local_video", source)

        state = VideoState(
            video_id="local_video",
            source_type="local",
            source_path=str(source),
            cache_mode=mode,
            cached_file="source.mp4",
        )
        cache.save_state("local_video", state)

        # Mock FFmpegTool
        mock_ffmpeg = MagicMock()
        mock_ffmpeg.extract_frames_range.return_value = [
            tmp_path / "frame_00-00.jpg",
            tmp_path / "frame_00-01.jpg",
        ]
        mock_ffmpeg_class.return_value = mock_ffmpeg

        frames = extract_frames_local(
            video_id="local_video",
            start_time=10.0,
            duration=5.0,
            interval=1.0,
            quality="medium",
            output_base=tmp_path / "cache",
        )

        # Verify call
        mock_ffmpeg.extract_frames_range.assert_called_once()
        call_kwargs = mock_ffmpeg.extract_frames_range.call_args.kwargs

        assert call_kwargs["video_path"] == dest
        assert call_kwargs["start_time"] == 10.0
        assert call_kwargs["duration"] == 5.0
        assert call_kwargs["interval"] == 1.0
        assert call_kwargs["width"] == 854  # medium quality
        assert call_kwargs["seek_offset"] == 0.0  # No offset for local
        assert call_kwargs["prefix"] == "frame"

    @patch("claudetube.operations.extract_frames.FFmpegTool")
    def test_creates_drill_directory(self, mock_ffmpeg_class, tmp_path):
        """Creates drill_QUALITY directory."""
        source = tmp_path / "videos" / "test.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"fake video content")

        cache = CacheManager(tmp_path / "cache")
        dest, mode = cache.cache_local_file("local_video", source)

        state = VideoState(
            video_id="local_video",
            source_type="local",
            source_path=str(source),
            cache_mode=mode,
            cached_file="source.mp4",
        )
        cache.save_state("local_video", state)

        mock_ffmpeg = MagicMock()
        mock_ffmpeg.extract_frames_range.return_value = []
        mock_ffmpeg_class.return_value = mock_ffmpeg

        extract_frames_local(
            video_id="local_video",
            start_time=0,
            quality="high",
            output_base=tmp_path / "cache",
        )

        # Check directory was created
        drill_dir = tmp_path / "cache" / "local_video" / "drill_high"
        assert drill_dir.exists()

    @patch("claudetube.operations.extract_frames.FFmpegTool")
    def test_tracks_extraction_in_state(self, mock_ffmpeg_class, tmp_path):
        """Updates state.json with extraction info."""
        source = tmp_path / "videos" / "test.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"fake video content")

        cache = CacheManager(tmp_path / "cache")
        dest, mode = cache.cache_local_file("local_video", source)

        state = VideoState(
            video_id="local_video",
            source_type="local",
            source_path=str(source),
            cache_mode=mode,
            cached_file="source.mp4",
        )
        cache.save_state("local_video", state)

        mock_ffmpeg = MagicMock()
        mock_ffmpeg.extract_frames_range.return_value = [
            tmp_path / "frame_00-00.jpg",
            tmp_path / "frame_00-01.jpg",
            tmp_path / "frame_00-02.jpg",
        ]
        mock_ffmpeg_class.return_value = mock_ffmpeg

        extract_frames_local(
            video_id="local_video",
            start_time=30.0,
            duration=3.0,
            quality="low",
            output_base=tmp_path / "cache",
        )

        # Check state was updated
        state_file = tmp_path / "cache" / "local_video" / "state.json"
        state_data = json.loads(state_file.read_text())

        assert "quality_extractions" in state_data
        assert "low" in state_data["quality_extractions"]
        extraction = state_data["quality_extractions"]["low"]
        assert extraction["start_time"] == 30.0
        assert extraction["duration"] == 3.0
        assert extraction["frames"] == 3
        assert extraction["width"] == 640
        assert extraction["local"] is True


class TestExtractHqFramesLocal:
    """Tests for extract_hq_frames_local function."""

    def test_raises_on_missing_video(self, tmp_path):
        """Missing video raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Video not cached"):
            extract_hq_frames_local(
                video_id="nonexistent",
                start_time=0,
                output_base=tmp_path,
            )

    def test_raises_on_url_video(self, tmp_path):
        """URL-type video raises ValueError."""
        cache = CacheManager(tmp_path)
        state = VideoState(
            video_id="url_video",
            source_type="url",
            url="https://youtube.com/watch?v=abc123",
        )
        cache.save_state("url_video", state)

        with pytest.raises(ValueError, match="not a local file"):
            extract_hq_frames_local(
                video_id="url_video",
                start_time=0,
                output_base=tmp_path,
            )

    @patch("claudetube.operations.extract_frames.FFmpegTool")
    def test_extracts_hq_frames_with_correct_params(self, mock_ffmpeg_class, tmp_path):
        """Calls FFmpegTool with HQ parameters."""
        source = tmp_path / "videos" / "test.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"fake video content")

        cache = CacheManager(tmp_path / "cache")
        dest, mode = cache.cache_local_file("local_video", source)

        state = VideoState(
            video_id="local_video",
            source_type="local",
            source_path=str(source),
            cache_mode=mode,
            cached_file="source.mp4",
        )
        cache.save_state("local_video", state)

        mock_ffmpeg = MagicMock()
        mock_ffmpeg.extract_frames_range.return_value = []
        mock_ffmpeg_class.return_value = mock_ffmpeg

        extract_hq_frames_local(
            video_id="local_video",
            start_time=20.0,
            duration=10.0,
            interval=2.0,
            width=1920,
            output_base=tmp_path / "cache",
        )

        call_kwargs = mock_ffmpeg.extract_frames_range.call_args.kwargs

        assert call_kwargs["video_path"] == dest
        assert call_kwargs["start_time"] == 20.0
        assert call_kwargs["duration"] == 10.0
        assert call_kwargs["interval"] == 2.0
        assert call_kwargs["width"] == 1920
        assert call_kwargs["jpeg_quality"] == 2  # HQ setting
        assert call_kwargs["seek_offset"] == 0.0
        assert call_kwargs["prefix"] == "hq"

    @patch("claudetube.operations.extract_frames.FFmpegTool")
    def test_creates_hq_directory(self, mock_ffmpeg_class, tmp_path):
        """Creates hq/ directory."""
        source = tmp_path / "videos" / "test.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"fake video content")

        cache = CacheManager(tmp_path / "cache")
        dest, mode = cache.cache_local_file("local_video", source)

        state = VideoState(
            video_id="local_video",
            source_type="local",
            source_path=str(source),
            cache_mode=mode,
            cached_file="source.mp4",
        )
        cache.save_state("local_video", state)

        mock_ffmpeg = MagicMock()
        mock_ffmpeg.extract_frames_range.return_value = []
        mock_ffmpeg_class.return_value = mock_ffmpeg

        extract_hq_frames_local(
            video_id="local_video",
            start_time=0,
            output_base=tmp_path / "cache",
        )

        hq_dir = tmp_path / "cache" / "local_video" / "hq"
        assert hq_dir.exists()

    @patch("claudetube.operations.extract_frames.FFmpegTool")
    def test_default_width_is_1280(self, mock_ffmpeg_class, tmp_path):
        """Default HQ width is 1280."""
        source = tmp_path / "videos" / "test.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"fake video content")

        cache = CacheManager(tmp_path / "cache")
        dest, mode = cache.cache_local_file("local_video", source)

        state = VideoState(
            video_id="local_video",
            source_type="local",
            source_path=str(source),
            cache_mode=mode,
            cached_file="source.mp4",
        )
        cache.save_state("local_video", state)

        mock_ffmpeg = MagicMock()
        mock_ffmpeg.extract_frames_range.return_value = []
        mock_ffmpeg_class.return_value = mock_ffmpeg

        extract_hq_frames_local(
            video_id="local_video",
            start_time=0,
            output_base=tmp_path / "cache",
        )

        call_kwargs = mock_ffmpeg.extract_frames_range.call_args.kwargs
        assert call_kwargs["width"] == 1280
