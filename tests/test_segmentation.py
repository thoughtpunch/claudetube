"""Tests for smart video segmentation."""

import json

from claudetube.analysis.linguistic import Boundary
from claudetube.cache.scenes import load_scenes_data
from claudetube.operations.segmentation import (
    boundaries_to_segments,
    segment_video_smart,
)


class TestBoundariesToSegments:
    """Tests for boundaries_to_segments function."""

    def test_no_boundaries_returns_single_segment(self):
        """No boundaries means one segment spanning the whole video."""
        segments = boundaries_to_segments([], 300.0)
        assert len(segments) == 1
        assert segments[0].scene_id == 0
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 300.0
        assert segments[0].title is None

    def test_single_boundary_creates_two_segments(self):
        """One boundary at 60s creates segments 0-60 and 60-end."""
        boundaries = [Boundary(60.0, "chapter", "[native] Intro", 0.95)]
        segments = boundaries_to_segments(boundaries, 180.0)

        assert len(segments) == 2
        # First segment: 0-60
        assert segments[0].scene_id == 0
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 60.0
        # Second segment: 60-180
        assert segments[1].scene_id == 1
        assert segments[1].start_time == 60.0
        assert segments[1].end_time == 180.0
        assert segments[1].title == "Intro"  # Extracted from chapter

    def test_multiple_boundaries_create_multiple_segments(self):
        """Multiple boundaries create proper segments."""
        boundaries = [
            Boundary(60.0, "chapter", "[native] Setup", 0.95),
            Boundary(120.0, "chapter", "[native] Main Content", 0.95),
            Boundary(240.0, "chapter", "[native] Conclusion", 0.95),
        ]
        segments = boundaries_to_segments(boundaries, 300.0)

        assert len(segments) == 4
        assert (segments[0].start_time, segments[0].end_time) == (0.0, 60.0)
        assert (segments[1].start_time, segments[1].end_time) == (60.0, 120.0)
        assert (segments[2].start_time, segments[2].end_time) == (120.0, 240.0)
        assert (segments[3].start_time, segments[3].end_time) == (240.0, 300.0)

    def test_boundaries_sorted_by_timestamp(self):
        """Boundaries should be sorted regardless of input order."""
        boundaries = [
            Boundary(120.0, "chapter", "[native] Middle", 0.95),
            Boundary(60.0, "chapter", "[native] Start", 0.95),
            Boundary(180.0, "chapter", "[native] End", 0.95),
        ]
        segments = boundaries_to_segments(boundaries, 240.0)

        # Verify correct ordering
        assert segments[1].title == "Start"  # First boundary
        assert segments[2].title == "Middle"  # Second boundary
        assert segments[3].title == "End"  # Third boundary

    def test_boundary_at_zero_skipped_for_first_segment(self):
        """Boundary very close to start doesn't create an empty first segment."""
        boundaries = [
            Boundary(0.2, "chapter", "[native] Very Start", 0.95),
            Boundary(60.0, "chapter", "[native] Next", 0.95),
        ]
        segments = boundaries_to_segments(boundaries, 120.0)

        # Should NOT have an empty 0-0.2 segment
        assert segments[0].start_time == 0.2
        assert segments[0].title == "Very Start"

    def test_non_chapter_boundary_no_title(self):
        """Non-chapter boundaries don't extract titles."""
        boundaries = [
            Boundary(60.0, "linguistic_cue", "now let's talk about", 0.7),
            Boundary(120.0, "pause", "3.5s pause", 0.65),
        ]
        segments = boundaries_to_segments(boundaries, 180.0)

        # Non-chapters have no title
        assert segments[1].title is None
        assert segments[2].title is None

    def test_chapter_title_extraction(self):
        """Chapter trigger_text format is correctly parsed."""
        boundaries = [
            Boundary(60.0, "chapter", "[native] Introduction to Python", 0.95),
            Boundary(120.0, "chapter", "[description] Advanced Topics", 0.9),
        ]
        segments = boundaries_to_segments(boundaries, 180.0)

        assert segments[1].title == "Introduction to Python"
        assert segments[2].title == "Advanced Topics"

    def test_chapter_title_without_bracket_format(self):
        """Handle chapter trigger_text without bracket format."""
        boundaries = [Boundary(60.0, "chapter", "Simple Title", 0.95)]
        segments = boundaries_to_segments(boundaries, 120.0)

        assert segments[1].title == "Simple Title"

    def test_combined_boundary_type(self):
        """Combined boundary types (chapter+linguistic_cue) extract title."""
        boundaries = [
            Boundary(60.0, "chapter+linguistic_cue", "[native] Main Topic", 0.95)
        ]
        segments = boundaries_to_segments(boundaries, 120.0)

        assert segments[1].title == "Main Topic"


class TestSegmentVideoSmart:
    """Tests for segment_video_smart function."""

    def test_returns_cached_data(self, tmp_path):
        """If scenes.json exists, return cached data."""
        cache_dir = tmp_path / "test_video"
        cache_dir.mkdir()
        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir()

        # Create cached scenes.json
        cached_data = {
            "video_id": "test_video",
            "method": "transcript",
            "scenes": [
                {"scene_id": 0, "start_time": 0, "end_time": 60},
                {"scene_id": 1, "start_time": 60, "end_time": 120},
            ],
        }
        (scenes_dir / "scenes.json").write_text(json.dumps(cached_data))

        result = segment_video_smart(
            video_id="test_video",
            video_path=None,
            transcript_segments=None,
            video_info=None,
            cache_dir=cache_dir,
        )

        assert result.video_id == "test_video"
        assert result.method == "transcript"
        assert len(result.scenes) == 2

    def test_force_ignores_cache(self, tmp_path):
        """With force=True, re-run segmentation even if cached."""
        cache_dir = tmp_path / "test_video"
        cache_dir.mkdir()
        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir()

        # Create cached scenes.json
        cached_data = {
            "video_id": "test_video",
            "method": "visual",  # Old cached method
            "scenes": [{"scene_id": 0, "start_time": 0, "end_time": 100}],
        }
        (scenes_dir / "scenes.json").write_text(json.dumps(cached_data))

        # Provide video_info with chapters
        video_info = {
            "duration": 300,
            "chapters": [
                {"title": "Intro", "start_time": 0, "end_time": 60},
                {"title": "Main", "start_time": 60, "end_time": 200},
                {"title": "Outro", "start_time": 200, "end_time": 300},
            ],
        }

        result = segment_video_smart(
            video_id="test_video",
            video_path=None,
            transcript_segments=None,
            video_info=video_info,
            cache_dir=cache_dir,
            force=True,
        )

        # Should have re-run with new data
        assert result.method == "transcript"  # Not "visual" from cache
        assert len(result.scenes) >= 3  # At least the 3 chapters

    def test_cheap_detection_with_chapters(self, tmp_path):
        """With good YouTube chapters, use cheap detection only."""
        cache_dir = tmp_path / "test_video"
        cache_dir.mkdir()

        # 6 chapters - should skip visual detection
        video_info = {
            "duration": 600,
            "chapters": [
                {
                    "title": f"Chapter {i}",
                    "start_time": i * 100,
                    "end_time": (i + 1) * 100,
                }
                for i in range(6)
            ],
        }

        result = segment_video_smart(
            video_id="test_video",
            video_path=None,
            transcript_segments=None,
            video_info=video_info,
            cache_dir=cache_dir,
        )

        assert result.method == "transcript"
        assert len(result.scenes) >= 6

    def test_saves_to_cache(self, tmp_path):
        """Results are saved to scenes/scenes.json."""
        cache_dir = tmp_path / "test_video"
        cache_dir.mkdir()

        video_info = {
            "duration": 120,
            "chapters": [
                {"title": "Intro", "start_time": 0, "end_time": 60},
                {"title": "Main", "start_time": 60, "end_time": 120},
            ],
        }

        segment_video_smart(
            video_id="test_video",
            video_path=None,
            transcript_segments=None,
            video_info=video_info,
            cache_dir=cache_dir,
        )

        # Verify saved
        scenes_json = cache_dir / "scenes" / "scenes.json"
        assert scenes_json.exists()

        # Verify can load
        loaded = load_scenes_data(cache_dir)
        assert loaded is not None
        assert loaded.video_id == "test_video"

    def test_no_metadata_uses_default_duration(self, tmp_path):
        """Without duration info, uses default 3600s."""
        cache_dir = tmp_path / "test_video"
        cache_dir.mkdir()

        result = segment_video_smart(
            video_id="test_video",
            video_path=None,
            transcript_segments=None,
            video_info=None,
            cache_dir=cache_dir,
        )

        # Should create single segment with default duration
        assert len(result.scenes) == 1
        assert result.scenes[0].end_time == 3600.0

    def test_transcript_segments_detected(self, tmp_path):
        """Transcript segments trigger linguistic/pause/vocabulary detection."""
        cache_dir = tmp_path / "test_video"
        cache_dir.mkdir()

        transcript_segments = [
            {"start": 0.0, "text": "Welcome to this tutorial"},
            {"start": 30.0, "text": "Now let's talk about the setup"},
            {"start": 60.0, "text": "Step 1 is to install dependencies"},
            {"start": 90.0, "text": "Moving on to step 2"},
            {"start": 120.0, "text": "Finally let's wrap up"},
        ]

        result = segment_video_smart(
            video_id="test_video",
            video_path=None,
            transcript_segments=transcript_segments,
            video_info={"duration": 150},
            cache_dir=cache_dir,
        )

        # Should detect linguistic boundaries from transcript
        assert result.method == "transcript"
        # Should have found at least some boundaries from "Now let's", "Step 1", "Moving on", "Finally"
        assert len(result.scenes) >= 2


class TestVisualFallbackDecision:
    """Tests for visual detection fallback logic."""

    def test_skips_visual_with_many_chapters(self, tmp_path):
        """5+ chapters skip visual detection entirely."""
        cache_dir = tmp_path / "test_video"
        cache_dir.mkdir()

        # 5 chapters - threshold to skip visual
        video_info = {
            "duration": 1800,  # 30 minutes
            "chapters": [
                {
                    "title": f"Chapter {i}",
                    "start_time": i * 360,
                    "end_time": (i + 1) * 360,
                }
                for i in range(5)
            ],
        }

        result = segment_video_smart(
            video_id="test_video",
            video_path=None,  # No video path provided
            transcript_segments=None,
            video_info=video_info,
            cache_dir=cache_dir,
        )

        # Should NOT attempt visual (no video_path wouldn't matter if visual was needed)
        assert result.method == "transcript"

    def test_short_video_no_visual(self, tmp_path):
        """Videos < 5 min don't need visual fallback."""
        cache_dir = tmp_path / "test_video"
        cache_dir.mkdir()

        # 4 minute video with no boundaries
        video_info = {"duration": 240}

        result = segment_video_smart(
            video_id="test_video",
            video_path=None,
            transcript_segments=None,
            video_info=video_info,
            cache_dir=cache_dir,
        )

        # Should stay with transcript method despite no boundaries
        assert result.method == "transcript"
