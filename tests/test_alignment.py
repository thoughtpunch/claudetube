"""Tests for transcript-to-scene alignment."""

from claudetube.analysis.alignment import (
    align_transcript_to_scenes,
    align_transcript_to_scenes_simple,
)
from claudetube.cache.scenes import SceneBoundary


class TestAlignTranscriptToScenes:
    """Tests for align_transcript_to_scenes function."""

    def test_empty_scenes_returns_empty(self):
        """Empty scenes list returns empty."""
        segments = [{"start": 0, "end": 10, "text": "Hello"}]
        result = align_transcript_to_scenes(segments, [])
        assert result == []

    def test_empty_transcript_returns_scenes_unchanged(self):
        """Empty transcript keeps scenes with empty transcript fields."""
        scenes = [
            SceneBoundary(scene_id=0, start_time=0, end_time=30),
            SceneBoundary(scene_id=1, start_time=30, end_time=60),
        ]
        result = align_transcript_to_scenes([], scenes)
        assert len(result) == 2
        assert result[0].transcript == []
        assert result[0].transcript_text == ""
        assert result[1].transcript == []
        assert result[1].transcript_text == ""

    def test_single_segment_single_scene(self):
        """Single segment goes to single scene."""
        segments = [{"start": 5, "end": 10, "text": "Hello world"}]
        scenes = [SceneBoundary(scene_id=0, start_time=0, end_time=30)]

        result = align_transcript_to_scenes(segments, scenes)

        assert len(result[0].transcript) == 1
        assert result[0].transcript[0]["text"] == "Hello world"
        assert result[0].transcript_text == "Hello world"

    def test_multiple_segments_single_scene(self):
        """Multiple segments in one scene get joined."""
        segments = [
            {"start": 5, "end": 10, "text": "Hello"},
            {"start": 15, "end": 20, "text": "World"},
        ]
        scenes = [SceneBoundary(scene_id=0, start_time=0, end_time=30)]

        result = align_transcript_to_scenes(segments, scenes)

        assert len(result[0].transcript) == 2
        assert result[0].transcript_text == "Hello World"

    def test_segments_distributed_across_scenes(self):
        """Segments are distributed to correct scenes by midpoint."""
        segments = [
            {"start": 5, "end": 10, "text": "First scene"},
            {"start": 35, "end": 40, "text": "Second scene"},
            {"start": 65, "end": 70, "text": "Third scene"},
        ]
        scenes = [
            SceneBoundary(scene_id=0, start_time=0, end_time=30),
            SceneBoundary(scene_id=1, start_time=30, end_time=60),
            SceneBoundary(scene_id=2, start_time=60, end_time=90),
        ]

        result = align_transcript_to_scenes(segments, scenes)

        assert result[0].transcript_text == "First scene"
        assert result[1].transcript_text == "Second scene"
        assert result[2].transcript_text == "Third scene"

    def test_midpoint_assignment_when_spanning_boundary(self):
        """Segment spanning boundary assigned by midpoint."""
        # Segment from 25-35, midpoint is 30 - should go to scene 1 (30-60)
        segments = [{"start": 25, "end": 35, "text": "Spans boundary"}]
        scenes = [
            SceneBoundary(scene_id=0, start_time=0, end_time=30),
            SceneBoundary(scene_id=1, start_time=30, end_time=60),
        ]

        result = align_transcript_to_scenes(segments, scenes)

        # Midpoint 30.0 is >= scene 1's start (30) and < end (60)
        assert result[0].transcript == []
        assert result[1].transcript_text == "Spans boundary"

    def test_midpoint_just_before_boundary(self):
        """Segment with midpoint just before boundary stays in earlier scene."""
        # Segment from 24-34, midpoint is 29 - should go to scene 0 (0-30)
        segments = [{"start": 24, "end": 34, "text": "Before boundary"}]
        scenes = [
            SceneBoundary(scene_id=0, start_time=0, end_time=30),
            SceneBoundary(scene_id=1, start_time=30, end_time=60),
        ]

        result = align_transcript_to_scenes(segments, scenes)

        assert result[0].transcript_text == "Before boundary"
        assert result[1].transcript == []

    def test_segment_without_end_uses_start_as_midpoint(self):
        """Segment without 'end' key uses 'start' as midpoint."""
        segments = [{"start": 15, "text": "No end timestamp"}]
        scenes = [
            SceneBoundary(scene_id=0, start_time=0, end_time=30),
            SceneBoundary(scene_id=1, start_time=30, end_time=60),
        ]

        result = align_transcript_to_scenes(segments, scenes)

        assert result[0].transcript_text == "No end timestamp"

    def test_empty_text_segments_skipped(self):
        """Segments with empty text are skipped."""
        segments = [
            {"start": 5, "end": 10, "text": "Valid"},
            {"start": 15, "end": 20, "text": ""},
            {"start": 25, "end": 28, "text": "   "},
        ]
        scenes = [SceneBoundary(scene_id=0, start_time=0, end_time=30)]

        result = align_transcript_to_scenes(segments, scenes)

        assert len(result[0].transcript) == 1
        assert result[0].transcript_text == "Valid"

    def test_segment_outside_all_scenes_ignored(self):
        """Segment outside all scene bounds is ignored."""
        segments = [
            {"start": 100, "end": 110, "text": "Beyond scenes"},
        ]
        scenes = [
            SceneBoundary(scene_id=0, start_time=0, end_time=30),
            SceneBoundary(scene_id=1, start_time=30, end_time=60),
        ]

        result = align_transcript_to_scenes(segments, scenes)

        assert result[0].transcript == []
        assert result[1].transcript == []

    def test_segment_before_first_scene_ignored(self):
        """Segment before first scene start is ignored."""
        segments = [{"start": -10, "end": -5, "text": "Before video"}]
        scenes = [SceneBoundary(scene_id=0, start_time=0, end_time=30)]

        result = align_transcript_to_scenes(segments, scenes)

        assert result[0].transcript == []

    def test_preserves_original_timestamps(self):
        """Original segment timestamps are preserved in transcript list."""
        segments = [
            {"start": 5.5, "end": 10.2, "text": "Hello", "extra_field": "preserved"},
        ]
        scenes = [SceneBoundary(scene_id=0, start_time=0, end_time=30)]

        result = align_transcript_to_scenes(segments, scenes)

        assert result[0].transcript[0]["start"] == 5.5
        assert result[0].transcript[0]["end"] == 10.2
        assert result[0].transcript[0]["extra_field"] == "preserved"

    def test_binary_search_performance_with_many_scenes(self):
        """Binary search works efficiently with many scenes."""
        # Create 100 scenes
        scenes = [
            SceneBoundary(scene_id=i, start_time=i * 10, end_time=(i + 1) * 10)
            for i in range(100)
        ]
        # Create 500 transcript segments
        segments = [
            {"start": i * 2, "end": i * 2 + 1, "text": f"Segment {i}"}
            for i in range(500)
        ]

        result = align_transcript_to_scenes(segments, scenes)

        # Verify a sample - segment at time 50-51 should be in scene 5 (50-60)
        assert "Segment 25" in result[5].transcript_text

    def test_transcript_text_joins_with_space(self):
        """Multiple segments joined with single space."""
        segments = [
            {"start": 5, "end": 8, "text": "First part."},
            {"start": 10, "end": 15, "text": "Second part."},
        ]
        scenes = [SceneBoundary(scene_id=0, start_time=0, end_time=30)]

        result = align_transcript_to_scenes(segments, scenes)

        assert result[0].transcript_text == "First part. Second part."


class TestAlignTranscriptToScenesSimple:
    """Tests for align_transcript_to_scenes_simple (dict-based version)."""

    def test_basic_alignment_with_dicts(self):
        """Basic alignment works with plain dicts."""
        segments = [
            {"start": 5, "end": 10, "text": "First"},
            {"start": 35, "end": 40, "text": "Second"},
        ]
        scenes = [
            {"start_time": 0, "end_time": 30, "title": "Scene 1"},
            {"start_time": 30, "end_time": 60, "title": "Scene 2"},
        ]

        result = align_transcript_to_scenes_simple(segments, scenes)

        assert result[0]["transcript_text"] == "First"
        assert result[1]["transcript_text"] == "Second"
        assert len(result[0]["transcript"]) == 1
        assert len(result[1]["transcript"]) == 1

    def test_empty_inputs(self):
        """Handles empty inputs."""
        assert align_transcript_to_scenes_simple([], []) == []

        scenes = [{"start_time": 0, "end_time": 30}]
        result = align_transcript_to_scenes_simple([], scenes)
        assert result[0]["transcript"] == []
        assert result[0]["transcript_text"] == ""


class TestSceneBoundaryTranscriptFields:
    """Tests for SceneBoundary transcript field serialization."""

    def test_to_dict_includes_transcript(self):
        """to_dict includes transcript and transcript_text when set."""
        scene = SceneBoundary(scene_id=0, start_time=0, end_time=30)
        scene.transcript = [{"start": 5, "end": 10, "text": "Hello"}]
        scene.transcript_text = "Hello"

        d = scene.to_dict()

        assert d["transcript"] == [{"start": 5, "end": 10, "text": "Hello"}]
        assert d["transcript_text"] == "Hello"

    def test_to_dict_omits_empty_transcript(self):
        """to_dict omits transcript fields when empty."""
        scene = SceneBoundary(scene_id=0, start_time=0, end_time=30)

        d = scene.to_dict()

        assert "transcript" not in d
        assert "transcript_text" not in d

    def test_from_dict_loads_transcript(self):
        """from_dict loads transcript and transcript_text."""
        d = {
            "scene_id": 0,
            "start_time": 0,
            "end_time": 30,
            "transcript": [{"start": 5, "end": 10, "text": "Hello"}],
            "transcript_text": "Hello",
        }

        scene = SceneBoundary.from_dict(d)

        assert scene.transcript == [{"start": 5, "end": 10, "text": "Hello"}]
        assert scene.transcript_text == "Hello"

    def test_from_dict_backward_compat_no_transcript(self):
        """from_dict handles missing transcript fields (backward compat)."""
        d = {
            "scene_id": 0,
            "start_time": 0,
            "end_time": 30,
        }

        scene = SceneBoundary.from_dict(d)

        assert scene.transcript == []
        assert scene.transcript_text == ""
