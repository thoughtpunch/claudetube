"""Tests for entity tracking (objects and concepts across scenes)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# Check if sklearn is available for concept tests
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from claudetube.cache.entities import (
    ConceptMention,
    ObjectAppearance,
    TrackedConcept,
    TrackedObject,
    get_concepts_json_path,
    get_entities_dir,
    get_objects_json_path,
    load_concepts,
    load_objects,
    save_concepts,
    save_objects,
    track_concepts_from_scenes,
    track_entities,
    track_objects_from_scenes,
)


class TestObjectAppearance:
    """Tests for ObjectAppearance dataclass."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        appearance = ObjectAppearance(scene_id=1, timestamp=30.0)
        d = appearance.to_dict()
        assert d["scene_id"] == 1
        assert d["timestamp"] == 30.0

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {"scene_id": 2, "timestamp": 45.5}
        appearance = ObjectAppearance.from_dict(d)
        assert appearance.scene_id == 2
        assert appearance.timestamp == 45.5


class TestTrackedObject:
    """Tests for TrackedObject dataclass."""

    def test_empty_appearances(self):
        """Object with no appearances should have safe defaults."""
        obj = TrackedObject(name="laptop")
        assert obj.first_seen == 0.0
        assert obj.last_seen == 0.0
        assert obj.frequency == 0

    def test_single_appearance(self):
        """Object with single appearance."""
        obj = TrackedObject(
            name="laptop", appearances=[ObjectAppearance(scene_id=0, timestamp=10.0)]
        )
        assert obj.first_seen == 10.0
        assert obj.last_seen == 10.0
        assert obj.frequency == 1

    def test_multiple_appearances(self):
        """Object appearing in multiple scenes."""
        obj = TrackedObject(
            name="whiteboard",
            appearances=[
                ObjectAppearance(scene_id=0, timestamp=0.0),
                ObjectAppearance(scene_id=2, timestamp=60.0),
                ObjectAppearance(scene_id=5, timestamp=150.0),
            ],
        )
        assert obj.first_seen == 0.0
        assert obj.last_seen == 150.0
        assert obj.frequency == 3

    def test_to_dict(self):
        """Should serialize to dictionary."""
        obj = TrackedObject(
            name="laptop",
            appearances=[ObjectAppearance(scene_id=0, timestamp=0.0)],
        )
        d = obj.to_dict()
        assert d["name"] == "laptop"
        assert d["first_seen"] == 0.0
        assert d["last_seen"] == 0.0
        assert d["frequency"] == 1
        assert len(d["appearances"]) == 1

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            "name": "monitor",
            "appearances": [{"scene_id": 1, "timestamp": 30.0}],
        }
        obj = TrackedObject.from_dict(d)
        assert obj.name == "monitor"
        assert len(obj.appearances) == 1
        assert obj.appearances[0].timestamp == 30.0


class TestConceptMention:
    """Tests for ConceptMention dataclass."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        mention = ConceptMention(scene_id=1, timestamp=30.0, score=0.75)
        d = mention.to_dict()
        assert d["scene_id"] == 1
        assert d["timestamp"] == 30.0
        assert d["score"] == 0.75

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {"scene_id": 2, "timestamp": 45.5, "score": 0.5}
        mention = ConceptMention.from_dict(d)
        assert mention.scene_id == 2
        assert mention.timestamp == 45.5
        assert mention.score == 0.5


class TestTrackedConcept:
    """Tests for TrackedConcept dataclass."""

    def test_empty_mentions(self):
        """Concept with no mentions should have safe defaults."""
        concept = TrackedConcept(term="python")
        assert concept.first_mention == 0.0
        assert concept.frequency == 0
        assert concept.avg_score == 0.0

    def test_single_mention(self):
        """Concept with single mention."""
        concept = TrackedConcept(
            term="python",
            mentions=[ConceptMention(scene_id=0, timestamp=10.0, score=0.8)],
        )
        assert concept.first_mention == 10.0
        assert concept.frequency == 1
        assert concept.avg_score == 0.8

    def test_multiple_mentions(self):
        """Concept mentioned in multiple scenes."""
        concept = TrackedConcept(
            term="machine learning",
            mentions=[
                ConceptMention(scene_id=0, timestamp=0.0, score=0.8),
                ConceptMention(scene_id=2, timestamp=60.0, score=0.6),
                ConceptMention(scene_id=5, timestamp=150.0, score=0.7),
            ],
        )
        assert concept.first_mention == 0.0
        assert concept.frequency == 3
        assert concept.avg_score == pytest.approx(0.7, rel=0.01)

    def test_to_dict(self):
        """Should serialize to dictionary."""
        concept = TrackedConcept(
            term="python",
            mentions=[ConceptMention(scene_id=0, timestamp=0.0, score=0.5)],
        )
        d = concept.to_dict()
        assert d["term"] == "python"
        assert d["first_mention"] == 0.0
        assert d["frequency"] == 1
        assert "avg_score" in d
        assert len(d["mentions"]) == 1

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            "term": "neural network",
            "mentions": [{"scene_id": 1, "timestamp": 30.0, "score": 0.65}],
        }
        concept = TrackedConcept.from_dict(d)
        assert concept.term == "neural network"
        assert len(concept.mentions) == 1
        assert concept.mentions[0].score == 0.65


class TestPathHelpers:
    """Tests for path helper functions."""

    def test_get_entities_dir(self, tmp_path):
        """Should create and return entities directory."""
        entities_dir = get_entities_dir(tmp_path)
        assert entities_dir.exists()
        assert entities_dir == tmp_path / "entities"

    def test_get_objects_json_path(self, tmp_path):
        """Should return path to objects.json."""
        path = get_objects_json_path(tmp_path)
        assert path == tmp_path / "entities" / "objects.json"

    def test_get_concepts_json_path(self, tmp_path):
        """Should return path to concepts.json."""
        path = get_concepts_json_path(tmp_path)
        assert path == tmp_path / "entities" / "concepts.json"


class TestSaveLoad:
    """Tests for save/load functions."""

    def test_save_and_load_objects(self, tmp_path):
        """Should save and load objects correctly."""
        objects = {
            "laptop": TrackedObject(
                name="laptop",
                appearances=[ObjectAppearance(scene_id=0, timestamp=0.0)],
            ),
        }
        save_objects(tmp_path, objects, "test-video")
        loaded = load_objects(tmp_path)

        assert loaded is not None
        assert "laptop" in loaded
        assert loaded["laptop"].frequency == 1

    def test_save_and_load_concepts(self, tmp_path):
        """Should save and load concepts correctly."""
        concepts = {
            "python": TrackedConcept(
                term="python",
                mentions=[ConceptMention(scene_id=0, timestamp=0.0, score=0.8)],
            ),
        }
        save_concepts(tmp_path, concepts, "test-video")
        loaded = load_concepts(tmp_path)

        assert loaded is not None
        assert "python" in loaded
        assert loaded["python"].frequency == 1

    def test_load_missing_objects(self, tmp_path):
        """Should return None for missing objects file."""
        assert load_objects(tmp_path) is None

    def test_load_missing_concepts(self, tmp_path):
        """Should return None for missing concepts file."""
        assert load_concepts(tmp_path) is None

    def test_load_invalid_json_objects(self, tmp_path):
        """Should return None for invalid objects.json."""
        path = get_objects_json_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json")
        assert load_objects(tmp_path) is None

    def test_load_invalid_json_concepts(self, tmp_path):
        """Should return None for invalid concepts.json."""
        path = get_concepts_json_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json")
        assert load_concepts(tmp_path) is None


class TestTrackObjectsFromScenes:
    """Tests for object tracking from scenes."""

    def test_empty_scenes(self):
        """Empty scenes should return empty objects."""
        objects = track_objects_from_scenes([])
        assert objects == {}

    def test_no_visual_data(self):
        """Scenes without visual data should return empty objects."""
        scenes = [{"scene_id": 0, "start_time": 0.0}]
        objects = track_objects_from_scenes(scenes)
        assert objects == {}

    def test_single_object_single_scene(self):
        """Single object in single scene."""
        scenes = [
            {
                "scene_id": 0,
                "start_time": 0.0,
                "visual": {"objects": ["laptop"]},
            }
        ]
        objects = track_objects_from_scenes(scenes)
        assert "laptop" in objects
        assert objects["laptop"].frequency == 1
        assert objects["laptop"].first_seen == 0.0

    def test_object_across_multiple_scenes(self):
        """Object appearing in multiple scenes."""
        scenes = [
            {"scene_id": 0, "start_time": 0.0, "visual": {"objects": ["laptop"]}},
            {"scene_id": 1, "start_time": 30.0, "visual": {"objects": ["laptop", "monitor"]}},
            {"scene_id": 2, "start_time": 60.0, "visual": {"objects": ["laptop"]}},
        ]
        objects = track_objects_from_scenes(scenes)
        assert objects["laptop"].frequency == 3
        assert objects["laptop"].first_seen == 0.0
        assert objects["laptop"].last_seen == 60.0
        assert objects["monitor"].frequency == 1

    def test_object_name_normalization(self):
        """Object names should be normalized (lowercase, stripped)."""
        scenes = [
            {"scene_id": 0, "start_time": 0.0, "visual": {"objects": ["Laptop"]}},
            {"scene_id": 1, "start_time": 30.0, "visual": {"objects": [" laptop "]}},
            {"scene_id": 2, "start_time": 60.0, "visual": {"objects": ["LAPTOP"]}},
        ]
        objects = track_objects_from_scenes(scenes)
        assert len(objects) == 1
        assert "laptop" in objects
        assert objects["laptop"].frequency == 3

    def test_empty_object_name_skipped(self):
        """Empty object names should be skipped."""
        scenes = [
            {"scene_id": 0, "start_time": 0.0, "visual": {"objects": ["", "laptop", "  "]}},
        ]
        objects = track_objects_from_scenes(scenes)
        assert len(objects) == 1
        assert "laptop" in objects


class TestTrackConceptsFromScenes:
    """Tests for concept tracking from scenes."""

    def test_empty_scenes(self):
        """Empty scenes should return empty concepts."""
        concepts = track_concepts_from_scenes([])
        assert concepts == {}

    def test_single_scene_insufficient(self):
        """Single scene with text is insufficient for TF-IDF."""
        scenes = [{"scene_id": 0, "start_time": 0.0, "transcript_text": "Python programming"}]
        concepts = track_concepts_from_scenes(scenes)
        assert concepts == {}

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_basic_concept_extraction(self):
        """Should extract concepts from multiple scenes."""
        scenes = [
            {"scene_id": 0, "start_time": 0.0, "transcript_text": "Python programming language"},
            {"scene_id": 1, "start_time": 30.0, "transcript_text": "Python functions and classes"},
            {"scene_id": 2, "start_time": 60.0, "transcript_text": "Machine learning with Python"},
        ]
        concepts = track_concepts_from_scenes(scenes, top_n=10)
        # "python" should be a top concept
        assert len(concepts) > 0

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_top_n_limit(self):
        """Should respect top_n limit."""
        scenes = [
            {"scene_id": 0, "start_time": 0.0, "transcript_text": "Python Java Ruby Go Rust"},
            {"scene_id": 1, "start_time": 30.0, "transcript_text": "Python Java Ruby frameworks"},
            {"scene_id": 2, "start_time": 60.0, "transcript_text": "Python Java libraries tools"},
        ]
        concepts = track_concepts_from_scenes(scenes, top_n=3)
        assert len(concepts) <= 3

    def test_fallback_to_transcript_segment(self):
        """Should fallback to transcript_segment if transcript_text missing."""
        scenes = [
            {"scene_id": 0, "start_time": 0.0, "transcript_segment": "Python programming"},
            {"scene_id": 1, "start_time": 30.0, "transcript_segment": "Python development"},
        ]
        concepts = track_concepts_from_scenes(scenes)
        # Should not crash with fallback
        assert isinstance(concepts, dict)

    def test_empty_transcript_handled(self):
        """Should handle empty transcript text gracefully."""
        scenes = [
            {"scene_id": 0, "start_time": 0.0, "transcript_text": ""},
            {"scene_id": 1, "start_time": 30.0, "transcript_text": ""},
        ]
        concepts = track_concepts_from_scenes(scenes)
        assert concepts == {}


class TestTrackEntities:
    """Tests for main track_entities function."""

    def test_no_scenes_data(self, tmp_path):
        """Should return error if no scenes data."""
        result = track_entities("test-video", tmp_path)
        assert "error" in result
        assert "scenes" in result["error"].lower()

    def test_cached_entities_returned(self, tmp_path):
        """Should return cached entities if available."""
        # Setup cached objects and concepts
        objects = {
            "laptop": TrackedObject(
                name="laptop", appearances=[ObjectAppearance(scene_id=0, timestamp=0.0)]
            )
        }
        concepts = {
            "python": TrackedConcept(
                term="python", mentions=[ConceptMention(scene_id=0, timestamp=0.0, score=0.8)]
            )
        }
        save_objects(tmp_path, objects, "test-video")
        save_concepts(tmp_path, concepts, "test-video")

        result = track_entities("test-video", tmp_path)
        assert result.get("from_cache") is True
        assert "laptop" in result["objects"]
        assert "python" in result["concepts"]

    def test_force_reprocess(self, tmp_path):
        """Force should reprocess even if cached."""
        # Setup cached entities
        objects = {"old": TrackedObject(name="old")}
        save_objects(tmp_path, objects, "test-video")

        # Setup minimal scenes data
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        scenes_json = scenes_dir / "scenes.json"
        scenes_json.write_text(
            json.dumps(
                {
                    "video_id": "test-video",
                    "method": "transcript",
                    "scenes": [
                        {
                            "scene_id": 0,
                            "start_time": 0.0,
                            "end_time": 30.0,
                            "transcript_text": "Python programming",
                        },
                        {
                            "scene_id": 1,
                            "start_time": 30.0,
                            "end_time": 60.0,
                            "transcript_text": "Python development",
                        },
                    ],
                }
            )
        )

        result = track_entities("test-video", tmp_path, force=True)
        assert result.get("from_cache") is not True


class TestModuleExport:
    """Tests for module exports."""

    def test_import_from_cache(self):
        """Should be importable from cache package."""
        from claudetube.cache import (
            TrackedConcept,
            TrackedObject,
            track_entities,
        )

        assert TrackedObject is not None
        assert TrackedConcept is not None
        assert callable(track_entities)
