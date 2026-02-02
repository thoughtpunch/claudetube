"""
Tests for scene change detection.
"""

import json

import numpy as np

from claudetube.operations.change_detection import (
    ChangesData,
    SceneChange,
    VisualChanges,
    _cosine_similarity,
    _extract_objects,
    _get_content_type,
    detect_scene_changes,
    get_changes_json_path,
    get_major_transitions,
    get_scene_changes,
)


class TestVisualChanges:
    """Tests for VisualChanges dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        changes = VisualChanges(
            added=["code_editor", "terminal"],
            removed=["title_slide"],
            persistent=["presenter"],
        )
        data = changes.to_dict()
        assert data["added"] == ["code_editor", "terminal"]
        assert data["removed"] == ["title_slide"]
        assert data["persistent"] == ["presenter"]

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "added": ["diagram"],
            "removed": ["code"],
            "persistent": ["person"],
        }
        changes = VisualChanges.from_dict(data)
        assert changes.added == ["diagram"]
        assert changes.removed == ["code"]
        assert changes.persistent == ["person"]

    def test_from_dict_defaults(self):
        """Should handle missing keys with defaults."""
        changes = VisualChanges.from_dict({})
        assert changes.added == []
        assert changes.removed == []
        assert changes.persistent == []


class TestSceneChange:
    """Tests for SceneChange dataclass."""

    def test_is_major_transition_high_topic_shift(self):
        """Should identify major transition from high topic shift."""
        change = SceneChange(
            scene_a_id=0,
            scene_b_id=1,
            visual_changes=VisualChanges(),
            topic_shift_score=0.7,
            content_type_change=False,
        )
        assert change.is_major_transition is True

    def test_is_major_transition_content_type_change(self):
        """Should identify major transition from content type change."""
        change = SceneChange(
            scene_a_id=0,
            scene_b_id=1,
            visual_changes=VisualChanges(),
            topic_shift_score=0.1,
            content_type_change=True,
        )
        assert change.is_major_transition is True

    def test_is_not_major_transition(self):
        """Should not identify minor changes as major transition."""
        change = SceneChange(
            scene_a_id=0,
            scene_b_id=1,
            visual_changes=VisualChanges(added=["small_change"]),
            topic_shift_score=0.3,
            content_type_change=False,
        )
        assert change.is_major_transition is False

    def test_to_dict(self):
        """Should convert to dictionary."""
        change = SceneChange(
            scene_a_id=0,
            scene_b_id=1,
            visual_changes=VisualChanges(added=["code"]),
            topic_shift_score=0.35,
            content_type_change=True,
            content_type_from="slides",
            content_type_to="code",
        )
        data = change.to_dict()
        assert data["scene_a_id"] == 0
        assert data["scene_b_id"] == 1
        assert data["topic_shift_score"] == 0.35
        assert data["content_type_change"] is True
        assert data["content_type_from"] == "slides"
        assert data["content_type_to"] == "code"
        assert data["is_major_transition"] is True

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "scene_a_id": 5,
            "scene_b_id": 6,
            "visual_changes": {"added": ["x"], "removed": ["y"], "persistent": []},
            "topic_shift_score": 0.2,
            "content_type_change": False,
            "content_type_from": "code",
            "content_type_to": "code",
        }
        change = SceneChange.from_dict(data)
        assert change.scene_a_id == 5
        assert change.scene_b_id == 6
        assert change.visual_changes.added == ["x"]
        assert change.visual_changes.removed == ["y"]


class TestChangesData:
    """Tests for ChangesData container."""

    def test_to_dict_generates_summary(self):
        """Should generate accurate summary."""
        data = ChangesData(
            video_id="test123",
            changes=[
                SceneChange(0, 1, VisualChanges(), 0.6, True, "slides", "code"),
                SceneChange(1, 2, VisualChanges(), 0.2, False),
                SceneChange(2, 3, VisualChanges(), 0.8, False),
            ],
        )
        result = data.to_dict()

        assert result["video_id"] == "test123"
        assert len(result["changes"]) == 3
        assert result["summary"]["total_changes"] == 3
        assert result["summary"]["major_transition_count"] == 2  # scene 1 and 3
        assert result["summary"]["content_type_changes"] == 1
        # avg = (0.6 + 0.2 + 0.8) / 3 = 0.533
        assert 0.53 <= result["summary"]["avg_topic_shift"] <= 0.54

    def test_roundtrip(self):
        """Should serialize and deserialize correctly."""
        original = ChangesData(
            video_id="vid123",
            changes=[
                SceneChange(0, 1, VisualChanges(added=["a"]), 0.5, True),
            ],
        )
        data = original.to_dict()
        restored = ChangesData.from_dict(data)

        assert restored.video_id == original.video_id
        assert len(restored.changes) == 1
        assert restored.changes[0].visual_changes.added == ["a"]


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self):
        """Should return 1 for identical vectors."""
        a = np.array([1.0, 2.0, 3.0])
        similarity = _cosine_similarity(a, a)
        assert abs(similarity - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        """Should return 0 for orthogonal vectors."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        similarity = _cosine_similarity(a, b)
        assert abs(similarity) < 0.001

    def test_opposite_vectors(self):
        """Should return -1 for opposite vectors."""
        a = np.array([1.0, 2.0])
        b = np.array([-1.0, -2.0])
        similarity = _cosine_similarity(a, b)
        assert abs(similarity - (-1.0)) < 0.001

    def test_zero_vector(self):
        """Should return 0 for zero vector."""
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        similarity = _cosine_similarity(a, b)
        assert similarity == 0.0


class TestExtractObjects:
    """Tests for object extraction from scene data."""

    def test_extracts_objects_from_visual_data(self):
        """Should extract objects from visual.json."""
        visual_data = {
            "objects": ["laptop", "whiteboard", "marker"],
            "people": ["presenter in blue shirt"],
            "description": "A presenter showing code in an editor",
        }
        objects = _extract_objects(visual_data, None)

        assert "laptop" in objects
        assert "whiteboard" in objects
        assert "person:presenter in blue shirt" in objects
        assert "content:code" in objects
        assert "content:editor" in objects

    def test_extracts_content_type_from_technical(self):
        """Should extract content type from technical.json."""
        technical_data = {
            "content_type": "code",
            "frames": [
                {
                    "content_type": "code",
                    "code_blocks": [{"content": "def foo(): pass"}],
                },
            ],
        }
        objects = _extract_objects(None, technical_data)

        assert "content_type:code" in objects
        assert "content:code" in objects

    def test_handles_empty_data(self):
        """Should handle empty or missing data."""
        objects = _extract_objects(None, None)
        assert len(objects) == 0

    def test_handles_objects_as_dicts(self):
        """Should handle objects in dict format."""
        visual_data = {
            "objects": [
                {"name": "laptop", "confidence": 0.95},
                {"label": "keyboard"},
            ],
        }
        objects = _extract_objects(visual_data, None)
        assert "laptop" in objects
        assert "keyboard" in objects


class TestGetContentType:
    """Tests for content type extraction."""

    def test_prefers_technical_data(self):
        """Should prefer content_type from technical.json."""
        technical_data = {"content_type": "code"}
        visual_data = {"description": "A presenter talking"}
        content_type = _get_content_type(visual_data, technical_data)
        assert content_type == "code"

    def test_uses_frame_types(self):
        """Should use frame content types when main type missing."""
        technical_data = {
            "frames": [
                {"content_type": "slides"},
                {"content_type": "slides"},
                {"content_type": "code"},
            ],
        }
        content_type = _get_content_type(None, technical_data)
        assert content_type == "slides"  # Most common

    def test_falls_back_to_visual_description(self):
        """Should detect content type from visual description."""
        visual_data = {
            "description": "The presenter shows a diagram explaining the architecture"
        }
        content_type = _get_content_type(visual_data, None)
        assert content_type == "diagram"

    def test_returns_unknown_for_empty_data(self):
        """Should return 'unknown' when no data available."""
        content_type = _get_content_type(None, None)
        assert content_type == "unknown"


class TestGetChangesJsonPath:
    """Tests for cache path generation."""

    def test_creates_structure_dir(self, tmp_path):
        """Should create structure directory if needed."""
        path = get_changes_json_path(tmp_path)
        assert path.parent.name == "structure"
        assert path.parent.exists()
        assert path.name == "changes.json"


class TestDetectSceneChanges:
    """Integration tests for detect_scene_changes."""

    def test_returns_error_for_uncached_video(self, tmp_path):
        """Should return error if video not cached."""
        result = detect_scene_changes("nonexistent", output_base=tmp_path)
        assert "error" in result
        assert "not cached" in result["error"]

    def test_handles_single_scene_video(self, tmp_path):
        """Should handle videos with only one scene."""
        video_id = "single_scene"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir(parents=True)

        # Create minimal structure
        (cache_dir / "state.json").write_text(json.dumps({"video_id": video_id}))

        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir()
        (scenes_dir / "scenes.json").write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "method": "transcript",
                    "scenes": [
                        {
                            "scene_id": 0,
                            "start_time": 0,
                            "end_time": 30,
                            "title": "Only scene",
                        },
                    ],
                }
            )
        )

        result = detect_scene_changes(video_id, output_base=tmp_path)
        assert "error" not in result
        assert result["summary"]["total_changes"] == 0
        assert "fewer than 2 scenes" in result.get("message", "")

    def test_detects_changes_between_scenes(self, tmp_path):
        """Should detect changes between consecutive scenes."""
        video_id = "test_video"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir(parents=True)

        # Create state.json
        (cache_dir / "state.json").write_text(json.dumps({"video_id": video_id}))

        # Create scenes data
        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir()
        (scenes_dir / "scenes.json").write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "method": "transcript",
                    "scenes": [
                        {
                            "scene_id": 0,
                            "start_time": 0,
                            "end_time": 30,
                            "title": "Intro",
                        },
                        {
                            "scene_id": 1,
                            "start_time": 30,
                            "end_time": 60,
                            "title": "Code demo",
                        },
                        {
                            "scene_id": 2,
                            "start_time": 60,
                            "end_time": 90,
                            "title": "Summary",
                        },
                    ],
                }
            )
        )

        # Create scene directories with visual/technical data
        for i, (ctype, objs) in enumerate(
            [
                ("slides", ["title_slide", "presenter"]),
                ("code", ["code_editor", "presenter"]),
                ("slides", ["summary_slide"]),
            ]
        ):
            scene_dir = scenes_dir / f"scene_{i:03d}"
            scene_dir.mkdir()
            (scene_dir / "visual.json").write_text(
                json.dumps(
                    {
                        "description": f"Scene {i}",
                        "objects": objs,
                        "people": ["presenter"] if "presenter" in objs else [],
                    }
                )
            )
            (scene_dir / "technical.json").write_text(
                json.dumps(
                    {
                        "content_type": ctype,
                        "frames": [],
                    }
                )
            )

        result = detect_scene_changes(video_id, output_base=tmp_path)

        assert "error" not in result
        assert result["summary"]["total_changes"] == 2

        # Scene 0->1: slides to code (content type change)
        change_0_1 = result["changes"][0]
        assert change_0_1["scene_a_id"] == 0
        assert change_0_1["scene_b_id"] == 1
        assert change_0_1["content_type_change"] is True
        assert "code_editor" in change_0_1["visual_changes"]["added"]
        assert "title_slide" in change_0_1["visual_changes"]["removed"]

        # Scene 1->2: code to slides (content type change)
        change_1_2 = result["changes"][1]
        assert change_1_2["content_type_change"] is True

    def test_caches_results(self, tmp_path):
        """Should cache results and return from cache on second call."""
        video_id = "cache_test"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir(parents=True)

        (cache_dir / "state.json").write_text(json.dumps({"video_id": video_id}))

        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir()
        (scenes_dir / "scenes.json").write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "method": "transcript",
                    "scenes": [
                        {"scene_id": 0, "start_time": 0, "end_time": 30},
                        {"scene_id": 1, "start_time": 30, "end_time": 60},
                    ],
                }
            )
        )

        # First call - generates data
        result1 = detect_scene_changes(video_id, output_base=tmp_path)
        assert "error" not in result1

        # Verify cached
        changes_path = get_changes_json_path(cache_dir)
        assert changes_path.exists()

        # Second call - from cache
        result2 = detect_scene_changes(video_id, output_base=tmp_path)
        assert result2["video_id"] == result1["video_id"]
        assert (
            result2["summary"]["total_changes"] == result1["summary"]["total_changes"]
        )

    def test_uses_embeddings_for_topic_shift(self, tmp_path):
        """Should use cached embeddings for topic shift detection."""
        video_id = "embed_test"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir(parents=True)

        (cache_dir / "state.json").write_text(json.dumps({"video_id": video_id}))

        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir()
        (scenes_dir / "scenes.json").write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "method": "transcript",
                    "scenes": [
                        {"scene_id": 0, "start_time": 0, "end_time": 30},
                        {"scene_id": 1, "start_time": 30, "end_time": 60},
                    ],
                }
            )
        )

        # Create embeddings (very different vectors)
        emb_dir = cache_dir / "embeddings"
        emb_dir.mkdir()
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Scene 0
                [0.0, 1.0, 0.0],  # Scene 1 - orthogonal (different topic)
            ],
            dtype=np.float32,
        )
        np.save(emb_dir / "scene_embeddings.npy", embeddings)
        (emb_dir / "scene_ids.json").write_text(json.dumps([0, 1]))

        result = detect_scene_changes(video_id, output_base=tmp_path)

        assert "error" not in result
        # Orthogonal vectors should have similarity ~0, so shift ~1
        topic_shift = result["changes"][0]["topic_shift_score"]
        assert topic_shift > 0.9


class TestGetSceneChanges:
    """Tests for retrieving cached scene changes."""

    def test_returns_cached_data(self, tmp_path):
        """Should return cached changes data."""
        video_id = "cached_video"
        cache_dir = tmp_path / video_id
        structure_dir = cache_dir / "structure"
        structure_dir.mkdir(parents=True)

        changes_data = {
            "video_id": video_id,
            "changes": [
                {
                    "scene_a_id": 0,
                    "scene_b_id": 1,
                    "visual_changes": {"added": [], "removed": [], "persistent": []},
                    "topic_shift_score": 0.3,
                    "content_type_change": False,
                },
            ],
            "summary": {"total_changes": 1},
        }
        (structure_dir / "changes.json").write_text(json.dumps(changes_data))

        result = get_scene_changes(video_id, output_base=tmp_path)
        assert "error" not in result
        assert result["video_id"] == video_id
        assert result["summary"]["total_changes"] == 1

    def test_returns_error_for_missing_data(self, tmp_path):
        """Should return error if no changes data."""
        result = get_scene_changes("nonexistent", output_base=tmp_path)
        assert "error" in result


class TestGetMajorTransitions:
    """Tests for getting major transition scene IDs."""

    def test_returns_major_transitions(self, tmp_path):
        """Should return scene IDs of major transitions."""
        video_id = "transition_test"
        cache_dir = tmp_path / video_id
        structure_dir = cache_dir / "structure"
        structure_dir.mkdir(parents=True)

        changes_data = {
            "video_id": video_id,
            "changes": [],
            "summary": {
                "major_transitions": [1, 5, 12],
            },
        }
        (structure_dir / "changes.json").write_text(json.dumps(changes_data))

        transitions = get_major_transitions(video_id, output_base=tmp_path)
        assert transitions == [1, 5, 12]

    def test_returns_empty_list_for_missing_data(self, tmp_path):
        """Should return empty list if no data."""
        transitions = get_major_transitions("nonexistent", output_base=tmp_path)
        assert transitions == []
