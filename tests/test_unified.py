"""Tests for unified cheap boundary detection."""

import tempfile
from pathlib import Path

from claudetube.analysis.linguistic import Boundary
from claudetube.analysis.unified import (
    CONFIDENCE_BOOST,
    MAX_CONFIDENCE,
    MERGE_THRESHOLD_SECONDS,
    _chapter_to_boundary,
    _merge_group,
    detect_boundaries_cheap,
    merge_nearby_boundaries,
)


class TestConstants:
    """Tests for module constants."""

    def test_merge_threshold(self):
        """Merge threshold should be 5 seconds."""
        assert MERGE_THRESHOLD_SECONDS == 5.0

    def test_confidence_boost(self):
        """Confidence boost should be 0.1."""
        assert CONFIDENCE_BOOST == 0.1

    def test_max_confidence(self):
        """Max confidence should be 0.95."""
        assert MAX_CONFIDENCE == 0.95


class TestChapterToBoundary:
    """Tests for chapter to boundary conversion."""

    def test_basic_conversion(self):
        """Convert a basic chapter to boundary."""
        from claudetube.models import Chapter

        ch = Chapter(
            title="Introduction",
            start=0.0,
            source="youtube_chapters",
            confidence=0.95,
        )
        b = _chapter_to_boundary(ch)

        assert b.timestamp == 0.0
        assert b.type == "chapter"
        assert "Introduction" in b.trigger_text
        assert b.confidence == 0.95

    def test_trigger_text_includes_source(self):
        """Trigger text should include source."""
        from claudetube.models import Chapter

        ch = Chapter(
            title="Setup",
            start=60.0,
            source="description_parsed",
            confidence=0.9,
        )
        b = _chapter_to_boundary(ch)

        assert "description_parsed" in b.trigger_text
        assert "Setup" in b.trigger_text

    def test_long_title_truncated(self):
        """Long titles should be truncated to 50 chars."""
        from claudetube.models import Chapter

        ch = Chapter(
            title="A" * 100,
            start=0.0,
            source="youtube_chapters",
            confidence=0.95,
        )
        b = _chapter_to_boundary(ch)

        assert len(b.trigger_text) == 50


class TestMergeGroup:
    """Tests for _merge_group helper."""

    def test_single_boundary(self):
        """Single boundary should be returned unchanged."""
        b = Boundary(10.0, "chapter", "Intro", 0.95)
        merged = _merge_group([b])

        assert merged == b

    def test_two_boundaries(self):
        """Two boundaries should merge with boosted confidence."""
        b1 = Boundary(10.0, "chapter", "Intro", 0.95)
        b2 = Boundary(12.0, "linguistic_cue", "now let's", 0.7)
        merged = _merge_group([b1, b2])

        assert merged.timestamp == 10.0  # Uses highest confidence timestamp
        assert "chapter" in merged.type
        assert "linguistic_cue" in merged.type
        assert merged.confidence == 0.95  # 0.95 + 0.1 = 1.05, capped to 0.95
        assert merged.trigger_text == "Intro"  # From highest confidence

    def test_three_boundaries(self):
        """Three boundaries should merge with double boost."""
        b1 = Boundary(10.0, "pause", "3.0s pause", 0.59)
        b2 = Boundary(11.0, "linguistic_cue", "now let's", 0.7)
        b3 = Boundary(12.0, "vocabulary_shift", "vocab shift", 0.6)
        merged = _merge_group([b1, b2, b3])

        # Highest confidence is linguistic_cue at 0.7
        assert merged.timestamp == 11.0
        # 0.7 + 2*0.1 = 0.9
        assert merged.confidence == 0.9
        assert "linguistic_cue" in merged.type

    def test_confidence_capped(self):
        """Confidence should be capped at MAX_CONFIDENCE."""
        b1 = Boundary(10.0, "chapter", "Intro", 0.95)
        b2 = Boundary(11.0, "linguistic_cue", "now let's", 0.7)
        b3 = Boundary(12.0, "pause", "pause", 0.6)
        b4 = Boundary(13.0, "vocabulary_shift", "vocab", 0.6)
        merged = _merge_group([b1, b2, b3, b4])

        # 0.95 + 3*0.1 = 1.25, capped to 0.95
        assert merged.confidence == MAX_CONFIDENCE

    def test_types_combined(self):
        """Types should be combined with + separator."""
        b1 = Boundary(10.0, "chapter", "ch", 0.95)
        b2 = Boundary(11.0, "pause", "pause", 0.6)
        merged = _merge_group([b1, b2])

        assert merged.type == "chapter+pause"

    def test_duplicate_types_deduplicated(self):
        """Duplicate types should be deduplicated."""
        b1 = Boundary(10.0, "pause", "pause1", 0.6)
        b2 = Boundary(11.0, "pause", "pause2", 0.7)
        merged = _merge_group([b1, b2])

        # Should only have one "pause"
        assert merged.type == "pause"


class TestMergeNearbyBoundaries:
    """Tests for merge_nearby_boundaries function."""

    def test_empty_list(self):
        """Empty list should return empty list."""
        assert merge_nearby_boundaries([]) == []

    def test_single_boundary(self):
        """Single boundary should be returned unchanged."""
        b = Boundary(10.0, "chapter", "Intro", 0.95)
        merged = merge_nearby_boundaries([b])

        assert len(merged) == 1
        assert merged[0] == b

    def test_no_merge_needed(self):
        """Boundaries far apart should not merge."""
        b1 = Boundary(0.0, "chapter", "Intro", 0.95)
        b2 = Boundary(60.0, "chapter", "Main", 0.95)
        b3 = Boundary(120.0, "chapter", "End", 0.95)
        merged = merge_nearby_boundaries([b1, b2, b3])

        assert len(merged) == 3

    def test_merge_two_nearby(self):
        """Two nearby boundaries should merge."""
        b1 = Boundary(10.0, "chapter", "Intro", 0.95)
        b2 = Boundary(12.0, "linguistic_cue", "now let's", 0.7)
        merged = merge_nearby_boundaries([b1, b2])

        assert len(merged) == 1
        assert merged[0].confidence == 0.95  # 0.95 + 0.1 = 1.05, capped at 0.95

    def test_merge_respects_threshold(self):
        """Boundaries exactly at threshold should not merge."""
        b1 = Boundary(10.0, "chapter", "Intro", 0.95)
        b2 = Boundary(15.0, "linguistic_cue", "now let's", 0.7)  # Exactly 5s apart
        merged = merge_nearby_boundaries([b1, b2], threshold=5.0)

        # 15.0 - 10.0 = 5.0, which is NOT < 5.0, so should NOT merge
        assert len(merged) == 2

    def test_merge_just_under_threshold(self):
        """Boundaries just under threshold should merge."""
        b1 = Boundary(10.0, "chapter", "Intro", 0.95)
        b2 = Boundary(14.9, "linguistic_cue", "now let's", 0.7)  # 4.9s apart
        merged = merge_nearby_boundaries([b1, b2], threshold=5.0)

        assert len(merged) == 1

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        b1 = Boundary(10.0, "chapter", "Intro", 0.95)
        b2 = Boundary(12.0, "linguistic_cue", "now let's", 0.7)

        # With 1s threshold, should not merge
        merged_1s = merge_nearby_boundaries([b1, b2], threshold=1.0)
        assert len(merged_1s) == 2

        # With 10s threshold, should merge
        merged_10s = merge_nearby_boundaries([b1, b2], threshold=10.0)
        assert len(merged_10s) == 1

    def test_sorted_output(self):
        """Output should be sorted by timestamp."""
        b1 = Boundary(100.0, "chapter", "End", 0.95)
        b2 = Boundary(0.0, "chapter", "Intro", 0.95)
        b3 = Boundary(50.0, "chapter", "Middle", 0.95)
        merged = merge_nearby_boundaries([b1, b2, b3])

        assert merged[0].timestamp == 0.0
        assert merged[1].timestamp == 50.0
        assert merged[2].timestamp == 100.0

    def test_chain_merge(self):
        """Boundaries that chain together should form groups."""
        # Three boundaries, each 3s apart
        b1 = Boundary(10.0, "a", "a", 0.5)
        b2 = Boundary(13.0, "b", "b", 0.6)
        b3 = Boundary(16.0, "c", "c", 0.7)
        merged = merge_nearby_boundaries([b1, b2, b3], threshold=5.0)

        # 10 -> 13 (3s apart, merge)
        # 13 -> 16 (3s apart, merge with group)
        # All three should merge into one
        assert len(merged) == 1
        assert merged[0].type == "a+b+c"


class TestDetectBoundariesCheap:
    """Tests for detect_boundaries_cheap main function."""

    def test_empty_inputs(self):
        """Empty inputs should return empty list."""
        boundaries = detect_boundaries_cheap()
        assert boundaries == []

    def test_video_info_only(self):
        """Should extract chapters from video_info."""
        video_info = {
            "chapters": [
                {"title": "Intro", "start_time": 0.0, "end_time": 60.0},
                {"title": "Main", "start_time": 60.0, "end_time": 120.0},
            ]
        }
        boundaries = detect_boundaries_cheap(video_info=video_info)

        assert len(boundaries) == 2
        assert boundaries[0].type == "chapter"
        assert boundaries[0].timestamp == 0.0
        assert boundaries[1].timestamp == 60.0

    def test_transcript_segments_only(self):
        """Should detect linguistic cues from transcript."""
        segments = [
            {"start": 0.0, "text": "Welcome to the tutorial"},
            {"start": 60.0, "text": "Now let's talk about setup"},
            {"start": 120.0, "text": "Finally, let's wrap up"},
        ]
        boundaries = detect_boundaries_cheap(transcript_segments=segments)

        # Should detect "Now let's" and "Finally"
        assert len(boundaries) >= 2

    def test_srt_path_only(self):
        """Should detect pauses from SRT file."""
        srt_content = """1
00:00:01,000 --> 00:00:05,000
First segment

2
00:00:10,000 --> 00:00:15,000
Second segment after pause
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()

            boundaries = detect_boundaries_cheap(srt_path=f.name)

            # Should detect 5s pause
            assert len(boundaries) == 1
            assert boundaries[0].type == "pause"
            assert boundaries[0].timestamp == 10.0

            Path(f.name).unlink()

    def test_combined_inputs(self):
        """Should combine multiple detection methods."""
        video_info = {
            "chapters": [
                {"title": "Intro", "start_time": 0.0, "end_time": 60.0},
            ]
        }
        segments = [
            {"start": 0.0, "text": "Welcome to the tutorial"},
            {"start": 60.0, "text": "Now let's talk about setup"},
        ]
        boundaries = detect_boundaries_cheap(
            video_info=video_info,
            transcript_segments=segments,
        )

        # Should have chapter at 0.0 and linguistic cue at 60.0
        assert len(boundaries) >= 2

    def test_nearby_boundaries_merged(self):
        """Nearby boundaries from different sources should merge."""
        video_info = {
            "chapters": [
                {"title": "Setup", "start_time": 60.0, "end_time": 120.0},
            ]
        }
        segments = [
            {"start": 0.0, "text": "Welcome"},
            {"start": 62.0, "text": "Now let's talk about setup"},  # 2s after chapter
        ]
        boundaries = detect_boundaries_cheap(
            video_info=video_info,
            transcript_segments=segments,
        )

        # 60.0 chapter and 62.0 linguistic cue should merge
        merged_at_60 = [b for b in boundaries if 58 < b.timestamp < 65]
        assert len(merged_at_60) == 1
        assert "chapter" in merged_at_60[0].type

    def test_custom_merge_threshold(self):
        """Custom merge threshold should be respected."""
        video_info = {
            "chapters": [
                {"title": "Ch1", "start_time": 0.0, "end_time": 10.0},
                {"title": "Ch2", "start_time": 3.0, "end_time": 20.0},  # 3s apart
            ]
        }

        # With 5s threshold, should merge
        merged_5s = detect_boundaries_cheap(video_info=video_info, merge_threshold=5.0)
        assert len(merged_5s) == 1

        # With 1s threshold, should not merge
        merged_1s = detect_boundaries_cheap(video_info=video_info, merge_threshold=1.0)
        assert len(merged_1s) == 2

    def test_vocabulary_shifts_included(self):
        """Should include vocabulary shifts in detection."""
        segments = [
            {
                "start": 0.0,
                "text": "Python programming language functions classes modules",
            },
            {
                "start": 35.0,
                "text": "Cooking recipes kitchen ingredients chef food dishes",
            },
        ]
        boundaries = detect_boundaries_cheap(transcript_segments=segments)

        # Should detect vocabulary shift around 35.0
        vocab_boundaries = [b for b in boundaries if "vocabulary_shift" in b.type]
        assert len(vocab_boundaries) >= 1

    def test_output_sorted_by_timestamp(self):
        """Output should be sorted by timestamp."""
        video_info = {
            "chapters": [
                {"title": "End", "start_time": 120.0, "end_time": 180.0},
                {"title": "Intro", "start_time": 0.0, "end_time": 60.0},
                {"title": "Middle", "start_time": 60.0, "end_time": 120.0},
            ]
        }
        boundaries = detect_boundaries_cheap(video_info=video_info)

        timestamps = [b.timestamp for b in boundaries]
        assert timestamps == sorted(timestamps)

    def test_path_object_accepted(self):
        """Should accept Path objects for srt_path."""
        srt_content = """1
00:00:01,000 --> 00:00:05,000
First segment

2
00:00:10,000 --> 00:00:15,000
Second segment
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()

            boundaries = detect_boundaries_cheap(srt_path=Path(f.name))
            assert len(boundaries) >= 1

            Path(f.name).unlink()


class TestBoundaryCompatibility:
    """Tests ensuring Boundary compatibility."""

    def test_returns_boundary_type(self):
        """Should return Boundary namedtuples."""
        video_info = {
            "chapters": [
                {"title": "Intro", "start_time": 0.0, "end_time": 60.0},
            ]
        }
        boundaries = detect_boundaries_cheap(video_info=video_info)

        assert all(isinstance(b, Boundary) for b in boundaries)

    def test_boundary_fields(self):
        """Boundary should have expected fields."""
        video_info = {
            "chapters": [
                {"title": "Intro", "start_time": 0.0, "end_time": 60.0},
            ]
        }
        boundaries = detect_boundaries_cheap(video_info=video_info)

        b = boundaries[0]
        assert hasattr(b, "timestamp")
        assert hasattr(b, "type")
        assert hasattr(b, "trigger_text")
        assert hasattr(b, "confidence")


class TestModuleExport:
    """Tests for module exports."""

    def test_import_from_analysis(self):
        """Should be importable from analysis package."""
        from claudetube.analysis import (
            detect_boundaries_cheap,
            merge_nearby_boundaries,
        )

        assert callable(detect_boundaries_cheap)
        assert callable(merge_nearby_boundaries)
