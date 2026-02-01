"""Tests for pause-based boundary detection."""

import tempfile
from pathlib import Path

import pytest

from claudetube.analysis.pause import (
    BASE_CONFIDENCE,
    CONFIDENCE_PER_SECOND,
    MAX_CONFIDENCE,
    MIN_PAUSE_SECONDS,
    detect_pause_boundaries,
    parse_srt_file,
    parse_srt_timestamp,
)


class TestParseSrtTimestamp:
    """Tests for SRT timestamp parsing."""

    def test_basic_timestamp(self):
        """Parse a basic timestamp."""
        assert parse_srt_timestamp("00:00:01,000") == 1.0

    def test_with_hours(self):
        """Parse timestamp with hours."""
        assert parse_srt_timestamp("01:30:45,500") == 5445.5

    def test_milliseconds(self):
        """Parse timestamp with milliseconds."""
        assert parse_srt_timestamp("00:00:00,123") == 0.123

    def test_zero_timestamp(self):
        """Parse zero timestamp."""
        assert parse_srt_timestamp("00:00:00,000") == 0.0

    def test_max_values(self):
        """Parse max reasonable values."""
        result = parse_srt_timestamp("23:59:59,999")
        assert result == pytest.approx(86399.999)

    def test_invalid_format_raises(self):
        """Invalid format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid SRT timestamp"):
            parse_srt_timestamp("invalid")

    def test_incomplete_timestamp_raises(self):
        """Incomplete timestamp should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid SRT timestamp"):
            parse_srt_timestamp("00:00:01")


class TestParseSrtFile:
    """Tests for SRT file parsing."""

    def test_basic_srt(self):
        """Parse a basic SRT file."""
        srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello world

2
00:00:05,000 --> 00:00:08,000
Second subtitle
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()

            segments = parse_srt_file(f.name)

            assert len(segments) == 2
            assert segments[0]["start"] == 1.0
            assert segments[0]["end"] == 4.0
            assert segments[0]["text"] == "Hello world"
            assert segments[1]["start"] == 5.0
            assert segments[1]["end"] == 8.0

            Path(f.name).unlink()

    def test_multiline_text(self):
        """Parse SRT with multiline subtitle text."""
        srt_content = """1
00:00:01,000 --> 00:00:04,000
Line one
Line two
Line three
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()

            segments = parse_srt_file(f.name)

            assert len(segments) == 1
            assert segments[0]["text"] == "Line one Line two Line three"

            Path(f.name).unlink()

    def test_nonexistent_file(self):
        """Nonexistent file should return empty list."""
        segments = parse_srt_file("/nonexistent/path.srt")
        assert segments == []

    def test_empty_file(self):
        """Empty file should return empty list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write("")
            f.flush()

            segments = parse_srt_file(f.name)
            assert segments == []

            Path(f.name).unlink()

    def test_path_object(self):
        """Should accept Path objects."""
        srt_content = """1
00:00:01,000 --> 00:00:02,000
Test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()

            segments = parse_srt_file(Path(f.name))
            assert len(segments) == 1

            Path(f.name).unlink()


class TestDetectPauseBoundaries:
    """Tests for pause boundary detection."""

    def test_no_pause(self):
        """Consecutive segments should not create boundaries."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 5.0, "end": 10.0, "text": "Second"},
        ]
        boundaries = detect_pause_boundaries(segments=segments)
        assert len(boundaries) == 0

    def test_small_gap(self):
        """Gaps under 2 seconds should not create boundaries."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 6.5, "end": 10.0, "text": "Second"},  # 1.5s gap
        ]
        boundaries = detect_pause_boundaries(segments=segments)
        assert len(boundaries) == 0

    def test_exactly_2_seconds(self):
        """Exactly 2 second gap should not create boundary (must be >2)."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 7.0, "end": 10.0, "text": "Second"},  # exactly 2s gap
        ]
        boundaries = detect_pause_boundaries(segments=segments)
        assert len(boundaries) == 0

    def test_just_over_2_seconds(self):
        """Gap just over 2 seconds should create boundary."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 7.1, "end": 10.0, "text": "Second"},  # 2.1s gap
        ]
        boundaries = detect_pause_boundaries(segments=segments)
        assert len(boundaries) == 1
        assert boundaries[0].timestamp == 7.1
        assert boundaries[0].type == "pause"

    def test_confidence_base(self):
        """Short pause should have base confidence."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 7.5, "end": 10.0, "text": "Second"},  # 2.5s gap
        ]
        boundaries = detect_pause_boundaries(segments=segments)
        # 0.5 + 2.5 * 0.03 = 0.575, rounds to 0.57 (banker's rounding)
        assert boundaries[0].confidence == pytest.approx(0.57, rel=0.01)

    def test_confidence_scales_with_duration(self):
        """Longer pauses should have higher confidence."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 10.0, "end": 15.0, "text": "Second"},  # 5s gap
        ]
        boundaries = detect_pause_boundaries(segments=segments)
        # 0.5 + 5 * 0.03 = 0.65
        assert boundaries[0].confidence == pytest.approx(0.65, rel=0.01)

    def test_confidence_max(self):
        """Very long pauses should cap at max confidence."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 25.0, "end": 30.0, "text": "Second"},  # 20s gap
        ]
        boundaries = detect_pause_boundaries(segments=segments)
        # 0.5 + 20 * 0.03 = 1.1, but capped at 0.8
        assert boundaries[0].confidence == MAX_CONFIDENCE

    def test_multiple_pauses(self):
        """Multiple pauses should create multiple boundaries."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 10.0, "end": 15.0, "text": "Second"},  # 5s gap
            {"start": 20.0, "end": 25.0, "text": "Third"},  # 5s gap
        ]
        boundaries = detect_pause_boundaries(segments=segments)
        assert len(boundaries) == 2
        assert boundaries[0].timestamp == 10.0
        assert boundaries[1].timestamp == 20.0

    def test_trigger_text_format(self):
        """Trigger text should show gap duration."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 8.5, "end": 10.0, "text": "Second"},  # 3.5s gap
        ]
        boundaries = detect_pause_boundaries(segments=segments)
        assert boundaries[0].trigger_text == "3.5s pause"

    def test_boundary_type(self):
        """Boundary type should be 'pause'."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 10.0, "end": 15.0, "text": "Second"},
        ]
        boundaries = detect_pause_boundaries(segments=segments)
        assert boundaries[0].type == "pause"

    def test_empty_segments(self):
        """Empty segments should return empty list."""
        boundaries = detect_pause_boundaries(segments=[])
        assert boundaries == []

    def test_single_segment(self):
        """Single segment should return empty list."""
        segments = [{"start": 0.0, "end": 5.0, "text": "Only one"}]
        boundaries = detect_pause_boundaries(segments=segments)
        assert boundaries == []

    def test_missing_end_time(self):
        """Segments missing 'end' should be skipped."""
        segments = [
            {"start": 0.0, "text": "First"},  # missing end
            {"start": 10.0, "end": 15.0, "text": "Second"},
        ]
        boundaries = detect_pause_boundaries(segments=segments)
        assert len(boundaries) == 0

    def test_missing_start_time(self):
        """Segments missing 'start' should be skipped."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"end": 15.0, "text": "Second"},  # missing start
        ]
        boundaries = detect_pause_boundaries(segments=segments)
        assert len(boundaries) == 0

    def test_no_args_returns_empty(self):
        """Calling with no args should return empty list."""
        boundaries = detect_pause_boundaries()
        assert boundaries == []


class TestDetectPauseBoundariesFromFile:
    """Tests for pause detection from SRT file."""

    def test_from_srt_file(self):
        """Detect pauses from SRT file."""
        srt_content = """1
00:00:01,000 --> 00:00:05,000
First segment

2
00:00:10,000 --> 00:00:15,000
Second segment after 5s pause
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()

            boundaries = detect_pause_boundaries(srt_path=f.name)

            assert len(boundaries) == 1
            assert boundaries[0].timestamp == 10.0
            assert boundaries[0].type == "pause"
            assert "5.0s" in boundaries[0].trigger_text

            Path(f.name).unlink()

    def test_segments_override_file(self):
        """Segments parameter should override srt_path."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 10.0, "end": 15.0, "text": "Second"},
        ]
        # Pass both, segments should win
        boundaries = detect_pause_boundaries(
            srt_path="/nonexistent/file.srt",
            segments=segments,
        )
        assert len(boundaries) == 1


class TestConstants:
    """Tests for module constants."""

    def test_min_pause_is_2_seconds(self):
        """Minimum pause should be 2 seconds."""
        assert MIN_PAUSE_SECONDS == 2.0

    def test_base_confidence(self):
        """Base confidence should be 0.5."""
        assert BASE_CONFIDENCE == 0.5

    def test_confidence_per_second(self):
        """Confidence per second should be 0.03."""
        assert CONFIDENCE_PER_SECOND == 0.03

    def test_max_confidence(self):
        """Max confidence should be 0.8."""
        assert MAX_CONFIDENCE == 0.8


class TestBoundaryCompatibility:
    """Tests ensuring Boundary compatibility with linguistic module."""

    def test_same_boundary_type(self):
        """Pause detection should use same Boundary class as linguistic."""
        from claudetube.analysis.linguistic import Boundary as LinguisticBoundary
        from claudetube.analysis.pause import detect_pause_boundaries

        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 10.0, "end": 15.0, "text": "Second"},
        ]
        boundaries = detect_pause_boundaries(segments=segments)

        # Should be the same type
        assert isinstance(boundaries[0], LinguisticBoundary)

    def test_boundary_fields_match(self):
        """Boundary fields should match expected structure."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 10.0, "end": 15.0, "text": "Second"},
        ]
        boundaries = detect_pause_boundaries(segments=segments)

        b = boundaries[0]
        assert hasattr(b, "timestamp")
        assert hasattr(b, "type")
        assert hasattr(b, "trigger_text")
        assert hasattr(b, "confidence")
