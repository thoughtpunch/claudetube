"""Tests for narrative structure detection.

Verifies:
1. Embedding-based scene clustering
2. Transcript-based fallback clustering
3. Section labeling (intro/main/conclusion)
4. Video type classification
5. Cache read/write
6. Edge cases (single scene, no scenes, no embeddings)
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from claudetube.cache.scenes import SceneBoundary, ScenesData
from claudetube.operations.narrative_structure import (
    NarrativeStructure,
    Section,
    _build_sections,
    _cluster_scenes_by_transcript,
    _cluster_scenes_with_embeddings,
    _cosine_similarity,
    _label_sections,
    classify_video_type,
    detect_narrative_structure,
    get_narrative_json_path,
    get_narrative_structure,
)


def _make_scene(
    scene_id: int,
    start: float,
    end: float,
    transcript: str = "",
    title: str | None = None,
) -> SceneBoundary:
    """Helper to create a SceneBoundary."""
    return SceneBoundary(
        scene_id=scene_id,
        start_time=start,
        end_time=end,
        title=title,
        transcript_text=transcript,
    )


def _make_scenes_data(video_id: str, scenes: list[SceneBoundary]) -> ScenesData:
    """Helper to create ScenesData."""
    return ScenesData(video_id=video_id, method="transcript", scenes=scenes)


class TestCosineSimlarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(a, b) == 0.0


class TestClusterScenesByTranscript:
    def test_similar_scenes_grouped(self):
        scenes = [
            _make_scene(0, 0, 10, "python functions classes methods"),
            _make_scene(1, 10, 20, "python functions variables types"),
            _make_scene(2, 20, 30, "cooking recipes ingredients mixing"),
            _make_scene(3, 30, 40, "cooking baking oven temperature"),
        ]
        labels = _cluster_scenes_by_transcript(scenes)
        assert len(labels) == 4
        # First two should share a label, last two should share a different one
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]

    def test_single_scene(self):
        scenes = [_make_scene(0, 0, 10, "hello world")]
        labels = _cluster_scenes_by_transcript(scenes)
        assert labels == [0]

    def test_two_scenes(self):
        scenes = [
            _make_scene(0, 0, 10, "hello"),
            _make_scene(1, 10, 20, "world"),
        ]
        labels = _cluster_scenes_by_transcript(scenes)
        assert len(labels) == 2

    def test_empty_transcript(self):
        scenes = [
            _make_scene(0, 0, 10, ""),
            _make_scene(1, 10, 20, ""),
            _make_scene(2, 20, 30, ""),
        ]
        labels = _cluster_scenes_by_transcript(scenes)
        assert len(labels) == 3


class TestClusterScenesWithEmbeddings:
    def test_clusters_similar_embeddings(self):
        scenes = [
            _make_scene(0, 0, 10),
            _make_scene(1, 10, 20),
            _make_scene(2, 20, 30),
            _make_scene(3, 30, 40),
        ]
        # Two clusters: scenes 0,1 similar and scenes 2,3 similar
        embeddings = {
            0: np.array([1.0, 0.0, 0.0]),
            1: np.array([0.9, 0.1, 0.0]),
            2: np.array([0.0, 0.0, 1.0]),
            3: np.array([0.0, 0.1, 0.9]),
        }
        labels = _cluster_scenes_with_embeddings(scenes, embeddings)
        assert len(labels) == 4
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_insufficient_embeddings(self):
        scenes = [
            _make_scene(0, 0, 10),
            _make_scene(1, 10, 20),
        ]
        embeddings = {
            0: np.array([1.0, 0.0]),
            1: np.array([0.0, 1.0]),
        }
        # Not enough for clustering (< 3)
        labels = _cluster_scenes_with_embeddings(scenes, embeddings)
        assert labels == [0, 0]

    def test_missing_embeddings_filled(self):
        scenes = [
            _make_scene(0, 0, 10),
            _make_scene(1, 10, 20),
            _make_scene(2, 20, 30),
            _make_scene(3, 30, 40),
        ]
        # Scene 1 missing - should inherit from neighbor
        embeddings = {
            0: np.array([1.0, 0.0, 0.0]),
            2: np.array([0.0, 0.0, 1.0]),
            3: np.array([0.0, 0.1, 0.9]),
        }
        labels = _cluster_scenes_with_embeddings(scenes, embeddings)
        assert len(labels) == 4
        assert labels[1] != -1  # Should have been filled


class TestBuildSections:
    def test_groups_consecutive_labels(self, tmp_path):
        scenes = [
            _make_scene(0, 0, 10, "intro welcome"),
            _make_scene(1, 10, 30, "main content here"),
            _make_scene(2, 30, 50, "more main content"),
            _make_scene(3, 50, 60, "conclusion goodbye"),
        ]
        labels = [0, 1, 1, 2]
        sections = _build_sections(scenes, labels, tmp_path)

        assert len(sections) == 3
        assert sections[0].scene_ids == [0]
        assert sections[1].scene_ids == [1, 2]
        assert sections[2].scene_ids == [3]

    def test_preserves_temporal_order(self, tmp_path):
        scenes = [
            _make_scene(0, 0, 10, "topic A"),
            _make_scene(1, 10, 20, "topic B"),
            _make_scene(2, 20, 30, "topic A again"),
        ]
        # Same label for 0 and 2 but separated = different sections
        labels = [0, 1, 0]
        sections = _build_sections(scenes, labels, tmp_path)

        assert len(sections) == 3
        assert sections[0].scene_ids == [0]
        assert sections[1].scene_ids == [1]
        assert sections[2].scene_ids == [2]

    def test_empty_scenes(self, tmp_path):
        sections = _build_sections([], [], tmp_path)
        assert sections == []


class TestLabelSections:
    def test_intro_detection(self):
        scenes = [_make_scene(i, i * 10, (i + 1) * 10) for i in range(5)]
        sections = [
            Section(0, "main_content", 0, 10, [0], "welcome to this video today we"),
            Section(1, "main_content", 10, 40, [1, 2, 3], "deep technical content"),
            Section(2, "main_content", 40, 50, [4], "thanks for watching subscribe"),
        ]
        _label_sections(sections, scenes)

        assert sections[0].label == "introduction"
        assert sections[1].label == "main_content"
        assert sections[2].label == "conclusion"

    def test_short_first_section_is_intro(self):
        scenes = [_make_scene(i, i * 10, (i + 1) * 10) for i in range(10)]
        sections = [
            Section(0, "main_content", 0, 5, [0], "some text"),
            Section(1, "main_content", 5, 90, [1, 2, 3, 4, 5, 6, 7, 8], "long content"),
            Section(2, "main_content", 90, 100, [9], "ending"),
        ]
        _label_sections(sections, scenes)
        assert sections[0].label == "introduction"

    def test_transition_detection(self):
        scenes = [_make_scene(i, i * 30, (i + 1) * 30) for i in range(5)]
        sections = [
            Section(0, "main_content", 0, 30, [0], "part one content"),
            Section(1, "main_content", 30, 40, [1], "brief transition"),  # 10s, 1 scene
            Section(2, "main_content", 40, 120, [2, 3, 4], "part two content"),
        ]
        # Override scene 1 to be short for transition detection
        sections[1] = Section(1, "main_content", 30, 40, [1], "brief")
        _label_sections(sections, scenes)
        assert sections[1].label == "transition"


class TestClassifyVideoType:
    def test_coding_tutorial(self, tmp_path):
        scenes = [_make_scene(i, i * 10, (i + 1) * 10) for i in range(10)]
        sections = []

        # Create technical.json with code content type for 4/10 scenes
        for i in range(10):
            scene_dir = tmp_path / "scenes" / f"scene_{i:03d}"
            scene_dir.mkdir(parents=True)
            content_type = "code" if i < 4 else "talking_head"
            (scene_dir / "technical.json").write_text(
                json.dumps({"content_type": content_type})
            )

        result = classify_video_type(scenes, sections, tmp_path)
        assert result == "coding_tutorial"

    def test_lecture(self, tmp_path):
        scenes = [_make_scene(i, i * 10, (i + 1) * 10) for i in range(10)]
        sections = []

        for i in range(10):
            scene_dir = tmp_path / "scenes" / f"scene_{i:03d}"
            scene_dir.mkdir(parents=True)
            content_type = "slides" if i < 6 else "talking_head"
            (scene_dir / "technical.json").write_text(
                json.dumps({"content_type": content_type})
            )

        result = classify_video_type(scenes, sections, tmp_path)
        assert result == "lecture"

    def test_interview(self, tmp_path):
        scenes = [_make_scene(i, i * 10, (i + 1) * 10) for i in range(10)]
        sections = []

        for i in range(10):
            scene_dir = tmp_path / "scenes" / f"scene_{i:03d}"
            scene_dir.mkdir(parents=True)
            (scene_dir / "technical.json").write_text(
                json.dumps({"content_type": "talking_head"})
            )

        result = classify_video_type(scenes, sections, tmp_path)
        assert result == "interview"

    def test_unknown_with_no_scenes(self, tmp_path):
        result = classify_video_type([], [], tmp_path)
        assert result == "unknown"


class TestSectionDataclass:
    def test_to_dict_round_trip(self):
        section = Section(
            section_id=0,
            label="introduction",
            start_time=0.0,
            end_time=30.5,
            scene_ids=[0, 1, 2],
            summary="Welcome to the video",
        )
        data = section.to_dict()
        restored = Section.from_dict(data)

        assert restored.section_id == section.section_id
        assert restored.label == section.label
        assert restored.start_time == section.start_time
        assert restored.end_time == section.end_time
        assert restored.scene_ids == section.scene_ids
        assert restored.summary == section.summary

    def test_duration(self):
        section = Section(0, "main", 10.0, 45.5, [])
        assert section.duration() == pytest.approx(35.5)


class TestNarrativeStructureDataclass:
    def test_to_dict_round_trip(self):
        structure = NarrativeStructure(
            video_id="test123",
            video_type="tutorial",
            sections=[
                Section(0, "introduction", 0, 10, [0], "intro"),
                Section(1, "main_content", 10, 50, [1, 2, 3], "main"),
                Section(2, "conclusion", 50, 60, [4], "outro"),
            ],
            cluster_count=3,
        )
        data = structure.to_dict()
        restored = NarrativeStructure.from_dict(data)

        assert restored.video_id == "test123"
        assert restored.video_type == "tutorial"
        assert len(restored.sections) == 3
        assert restored.cluster_count == 3

    def test_summary_in_dict(self):
        structure = NarrativeStructure(
            video_id="test",
            video_type="demo",
            sections=[Section(0, "main_content", 0, 100, [0], "content")],
            cluster_count=1,
        )
        data = structure.to_dict()
        assert data["summary"]["section_count"] == 1
        assert data["summary"]["video_type"] == "demo"
        assert data["summary"]["section_labels"] == ["main_content"]


class TestDetectNarrativeStructure:
    def test_returns_error_for_uncached_video(self, tmp_path):
        result = detect_narrative_structure("nonexistent", output_base=tmp_path)
        assert "error" in result

    def test_returns_error_for_no_scenes(self, tmp_path):
        video_dir = tmp_path / "test_video"
        video_dir.mkdir()
        result = detect_narrative_structure("test_video", output_base=tmp_path)
        assert "error" in result

    def test_single_scene_video(self, tmp_path):
        video_dir = tmp_path / "test_video"
        video_dir.mkdir()
        scenes_dir = video_dir / "scenes"
        scenes_dir.mkdir()

        scenes_data = _make_scenes_data(
            "test_video",
            [_make_scene(0, 0, 60, "the full video content")],
        )
        (scenes_dir / "scenes.json").write_text(json.dumps(scenes_data.to_dict()))

        result = detect_narrative_structure("test_video", output_base=tmp_path)
        assert "error" not in result
        assert len(result["sections"]) == 1
        assert result["sections"][0]["label"] == "main_content"

    def test_caches_result(self, tmp_path):
        video_dir = tmp_path / "test_video"
        video_dir.mkdir()
        scenes_dir = video_dir / "scenes"
        scenes_dir.mkdir()

        scenes_data = _make_scenes_data(
            "test_video",
            [
                _make_scene(0, 0, 10, "intro welcome"),
                _make_scene(1, 10, 50, "main content long"),
                _make_scene(2, 50, 60, "conclusion thanks"),
            ],
        )
        (scenes_dir / "scenes.json").write_text(json.dumps(scenes_data.to_dict()))

        # First call generates
        result1 = detect_narrative_structure("test_video", output_base=tmp_path)
        assert "error" not in result1

        # Second call returns from cache
        result2 = detect_narrative_structure("test_video", output_base=tmp_path)
        assert result2 == result1

    def test_force_regenerates(self, tmp_path):
        video_dir = tmp_path / "test_video"
        video_dir.mkdir()
        scenes_dir = video_dir / "scenes"
        scenes_dir.mkdir()

        scenes_data = _make_scenes_data(
            "test_video",
            [
                _make_scene(0, 0, 10, "hello"),
                _make_scene(1, 10, 20, "world"),
                _make_scene(2, 20, 30, "end"),
            ],
        )
        (scenes_dir / "scenes.json").write_text(json.dumps(scenes_data.to_dict()))

        detect_narrative_structure("test_video", output_base=tmp_path)
        # Force re-generation
        result = detect_narrative_structure(
            "test_video", force=True, output_base=tmp_path
        )
        assert "error" not in result

    def test_uses_embeddings_when_available(self, tmp_path):
        video_dir = tmp_path / "test_video"
        video_dir.mkdir()
        scenes_dir = video_dir / "scenes"
        scenes_dir.mkdir()
        emb_dir = video_dir / "embeddings"
        emb_dir.mkdir()

        scenes = [
            _make_scene(0, 0, 10, "topic A content"),
            _make_scene(1, 10, 20, "topic A more"),
            _make_scene(2, 20, 30, "topic B different"),
            _make_scene(3, 30, 40, "topic B more different"),
        ]
        scenes_data = _make_scenes_data("test_video", scenes)
        (scenes_dir / "scenes.json").write_text(json.dumps(scenes_data.to_dict()))

        # Save embeddings: first two similar, last two similar
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.1, 0.9],
            ]
        )
        np.save(emb_dir / "scene_embeddings.npy", embeddings)
        (emb_dir / "scene_ids.json").write_text(json.dumps([0, 1, 2, 3]))
        (emb_dir / "metadata.json").write_text(
            json.dumps({"model": "test", "embedding_dim": 3, "num_scenes": 4})
        )

        result = detect_narrative_structure("test_video", output_base=tmp_path)
        assert "error" not in result
        assert result["cluster_count"] >= 2


class TestGetNarrativeStructure:
    def test_returns_cached_data(self, tmp_path):
        video_dir = tmp_path / "test_video"
        video_dir.mkdir()
        structure_dir = video_dir / "structure"
        structure_dir.mkdir()

        data = {"video_id": "test_video", "video_type": "tutorial", "sections": []}
        (structure_dir / "narrative.json").write_text(json.dumps(data))

        result = get_narrative_structure("test_video", output_base=tmp_path)
        assert result == data

    def test_returns_error_when_not_cached(self, tmp_path):
        video_dir = tmp_path / "test_video"
        video_dir.mkdir()

        result = get_narrative_structure("test_video", output_base=tmp_path)
        assert "error" in result

    def test_returns_error_for_uncached_video(self, tmp_path):
        result = get_narrative_structure("nonexistent", output_base=tmp_path)
        assert "error" in result


class TestGetNarrativeJsonPath:
    def test_creates_structure_dir(self, tmp_path):
        path = get_narrative_json_path(tmp_path)
        assert path.name == "narrative.json"
        assert path.parent.name == "structure"
        assert path.parent.exists()
