"""Tests for cross-video knowledge graph (cache/knowledge_graph.py)."""

import json
from pathlib import Path

from claudetube.cache.knowledge_graph import (
    ConceptNode,
    EntityNode,
    RelatedVideoMatch,
    VideoKnowledgeGraph,
    VideoNode,
    get_knowledge_graph,
    index_video_to_graph,
)


class TestVideoNode:
    """Test VideoNode dataclass."""

    def test_to_dict(self):
        node = VideoNode(
            video_id="abc123",
            title="Test Video",
            channel="TestChannel",
            indexed_at="2024-01-01T12:00:00",
        )
        d = node.to_dict()
        assert d["video_id"] == "abc123"
        assert d["title"] == "Test Video"
        assert d["channel"] == "TestChannel"
        assert d["indexed_at"] == "2024-01-01T12:00:00"

    def test_from_dict(self):
        d = {
            "video_id": "xyz789",
            "title": "Another Video",
            "channel": "AnotherChannel",
            "indexed_at": "2024-02-01T10:00:00",
        }
        node = VideoNode.from_dict(d)
        assert node.video_id == "xyz789"
        assert node.title == "Another Video"
        assert node.channel == "AnotherChannel"


class TestEntityNode:
    """Test EntityNode dataclass."""

    def test_to_dict(self):
        node = EntityNode(
            name="python",
            entity_type="technology",
            video_ids=["vid1", "vid2"],
        )
        d = node.to_dict()
        assert d["name"] == "python"
        assert d["type"] == "technology"
        assert d["videos"] == ["vid1", "vid2"]

    def test_from_dict(self):
        d = {
            "name": "docker",
            "type": "technology",
            "videos": ["a", "b", "c"],
        }
        node = EntityNode.from_dict(d)
        assert node.name == "docker"
        assert node.entity_type == "technology"
        assert len(node.video_ids) == 3


class TestConceptNode:
    """Test ConceptNode dataclass."""

    def test_to_dict(self):
        node = ConceptNode(name="machine learning", video_ids=["v1", "v2"])
        d = node.to_dict()
        assert d["name"] == "machine learning"
        assert d["videos"] == ["v1", "v2"]

    def test_from_dict(self):
        d = {"name": "web development", "videos": ["x", "y"]}
        node = ConceptNode.from_dict(d)
        assert node.name == "web development"
        assert len(node.video_ids) == 2


class TestRelatedVideoMatch:
    """Test RelatedVideoMatch dataclass."""

    def test_to_dict(self):
        match = RelatedVideoMatch(
            video_id="vid123",
            video_title="Python Tutorial",
            match_type="concept",
            matched_term="python",
        )
        d = match.to_dict()
        assert d["video_id"] == "vid123"
        assert d["video_title"] == "Python Tutorial"
        assert d["match_type"] == "concept"
        assert d["matched"] == "python"


class TestVideoKnowledgeGraph:
    """Test VideoKnowledgeGraph class."""

    def test_init_creates_directory(self, tmp_path: Path):
        graph_dir = tmp_path / "video_knowledge"
        graph = VideoKnowledgeGraph(graph_dir)
        assert graph.graph_dir.exists()
        assert graph.video_count == 0

    def test_add_video(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        graph.add_video(
            video_id="vid1",
            metadata={"title": "Python Basics", "channel": "CodeChannel"},
            entities={"technology": ["Python", "Django"], "person": ["Guido"]},
            concepts=["web development", "programming"],
        )
        assert graph.video_count == 1
        assert graph.entity_count == 3  # python, django, guido
        assert graph.concept_count == 2

        video = graph.get_video("vid1")
        assert video is not None
        assert video.title == "Python Basics"

    def test_add_video_updates_existing(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        graph.add_video(
            video_id="vid1",
            metadata={"title": "Old Title"},
            entities={},
            concepts=[],
        )
        graph.add_video(
            video_id="vid1",
            metadata={"title": "New Title"},
            entities={"technology": ["React"]},
            concepts=["frontend"],
        )
        assert graph.video_count == 1
        assert graph.get_video("vid1").title == "New Title"

    def test_remove_video(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        graph.add_video(
            video_id="vid1",
            metadata={"title": "Test"},
            entities={"tech": ["python"]},
            concepts=["coding"],
        )
        assert graph.video_count == 1

        removed = graph.remove_video("vid1")
        assert removed is True
        assert graph.video_count == 0
        # Entities/concepts should also be removed if no other videos use them
        assert graph.entity_count == 0
        assert graph.concept_count == 0

    def test_remove_video_not_found(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        removed = graph.remove_video("nonexistent")
        assert removed is False

    def test_find_related_videos_by_entity(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        graph.add_video(
            video_id="vid1",
            metadata={"title": "Python Intro"},
            entities={"technology": ["Python"]},
            concepts=[],
        )
        graph.add_video(
            video_id="vid2",
            metadata={"title": "Python Advanced"},
            entities={"technology": ["Python", "Django"]},
            concepts=[],
        )
        graph.add_video(
            video_id="vid3",
            metadata={"title": "JavaScript Basics"},
            entities={"technology": ["JavaScript"]},
            concepts=[],
        )

        matches = graph.find_related_videos("python")
        assert len(matches) == 2
        video_ids = [m.video_id for m in matches]
        assert "vid1" in video_ids
        assert "vid2" in video_ids
        assert "vid3" not in video_ids

    def test_find_related_videos_by_concept(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        graph.add_video(
            video_id="vid1",
            metadata={"title": "ML Course"},
            entities={},
            concepts=["machine learning", "deep learning"],
        )
        graph.add_video(
            video_id="vid2",
            metadata={"title": "AI Tutorial"},
            entities={},
            concepts=["machine learning", "neural networks"],
        )

        matches = graph.find_related_videos("learning")
        assert len(matches) == 2

    def test_find_related_videos_deduplicates(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        # Video has both entity and concept matching "python"
        graph.add_video(
            video_id="vid1",
            metadata={"title": "Python Tutorial"},
            entities={"technology": ["Python"]},
            concepts=["python basics"],
        )

        matches = graph.find_related_videos("python")
        # Should only return vid1 once
        assert len(matches) == 1
        assert matches[0].video_id == "vid1"

    def test_find_related_videos_case_insensitive(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        graph.add_video(
            video_id="vid1",
            metadata={"title": "React App"},
            entities={"technology": ["REACT"]},
            concepts=["FRONTEND"],
        )

        # Lowercase query should match uppercase entries
        matches = graph.find_related_videos("react")
        assert len(matches) == 1

    def test_get_video_connections(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        graph.add_video(
            video_id="vid1",
            metadata={"title": "Python Basics"},
            entities={"technology": ["Python"]},
            concepts=["programming"],
        )
        graph.add_video(
            video_id="vid2",
            metadata={"title": "Python Web"},
            entities={"technology": ["Python", "Flask"]},
            concepts=["web development"],
        )
        graph.add_video(
            video_id="vid3",
            metadata={"title": "Unrelated"},
            entities={"technology": ["Rust"]},
            concepts=["systems programming"],
        )

        connections = graph.get_video_connections("vid1")
        # vid1 shares "python" with vid2, but not with vid3
        assert "vid2" in connections
        assert "vid3" not in connections
        assert "vid1" not in connections  # Should not include self

    def test_get_video_connections_not_found(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        connections = graph.get_video_connections("nonexistent")
        assert connections == []

    def test_persistence_save_and_load(self, tmp_path: Path):
        graph1 = VideoKnowledgeGraph(tmp_path)
        graph1.add_video(
            video_id="vid1",
            metadata={"title": "Persistent Video", "channel": "TestChan"},
            entities={"tech": ["docker"]},
            concepts=["containers"],
        )

        # Create new instance - should load from disk
        graph2 = VideoKnowledgeGraph(tmp_path)
        assert graph2.video_count == 1
        assert graph2.entity_count == 1
        assert graph2.concept_count == 1

        video = graph2.get_video("vid1")
        assert video is not None
        assert video.title == "Persistent Video"

    def test_get_stats(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        graph.add_video(
            video_id="v1",
            metadata={"title": "T1"},
            entities={"t": ["a", "b"]},
            concepts=["c1", "c2", "c3"],
        )
        stats = graph.get_stats()
        assert stats["video_count"] == 1
        assert stats["entity_count"] == 2
        assert stats["concept_count"] == 3
        assert "graph_path" in stats

    def test_clear(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        graph.add_video(
            video_id="v1",
            metadata={"title": "T1"},
            entities={"t": ["a"]},
            concepts=["c1"],
        )
        assert graph.video_count == 1

        graph.clear()
        assert graph.video_count == 0
        assert graph.entity_count == 0
        assert graph.concept_count == 0
        assert not graph.graph_path.exists()

    def test_get_all_videos(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        graph.add_video("v1", {"title": "V1"}, {}, [])
        graph.add_video("v2", {"title": "V2"}, {}, [])

        all_videos = graph.get_all_videos()
        assert len(all_videos) == 2
        titles = [v.title for v in all_videos]
        assert "V1" in titles
        assert "V2" in titles

    def test_get_entity(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        graph.add_video("v1", {}, {"tech": ["Python"]}, [])

        entity = graph.get_entity("python")
        assert entity is not None
        assert entity.entity_type == "tech"

        assert graph.get_entity("nonexistent") is None

    def test_get_concept(self, tmp_path: Path):
        graph = VideoKnowledgeGraph(tmp_path)
        graph.add_video("v1", {}, {}, ["machine learning"])

        concept = graph.get_concept("machine learning")
        assert concept is not None
        assert "v1" in concept.video_ids

        assert graph.get_concept("nonexistent") is None


class TestGetKnowledgeGraph:
    """Test get_knowledge_graph factory function."""

    def test_returns_graph_instance(self, tmp_path: Path):
        graph = get_knowledge_graph(tmp_path)
        assert isinstance(graph, VideoKnowledgeGraph)


class TestIndexVideoToGraph:
    """Test index_video_to_graph function."""

    def test_returns_error_for_uncached_video(self, tmp_path: Path):
        result = index_video_to_graph("nonexistent", tmp_path / "uncached", tmp_path)
        assert "error" in result

    def test_indexes_video_with_entities(self, tmp_path: Path):
        # Create mock video cache
        video_dir = tmp_path / "vid123"
        video_dir.mkdir()

        # Create state.json (must have video_id)
        state = {"video_id": "vid123", "title": "Test Video", "channel": "TestChannel"}
        (video_dir / "state.json").write_text(json.dumps(state))

        # Create entities
        entities_dir = video_dir / "entities"
        entities_dir.mkdir()

        objects = {
            "video_id": "vid123",
            "object_count": 2,
            "objects": {
                "laptop": {"name": "laptop", "appearances": [], "frequency": 3},
                "whiteboard": {"name": "whiteboard", "appearances": [], "frequency": 1},
            },
        }
        (entities_dir / "objects.json").write_text(json.dumps(objects))

        concepts = {
            "video_id": "vid123",
            "concept_count": 2,
            "concepts": {
                "programming": {"term": "programming", "mentions": [], "frequency": 5},
                "tutorial": {"term": "tutorial", "mentions": [], "frequency": 2},
            },
        }
        (entities_dir / "concepts.json").write_text(json.dumps(concepts))

        # Index the video
        result = index_video_to_graph("vid123", video_dir, tmp_path)

        assert result["status"] == "indexed"
        assert result["video_id"] == "vid123"
        assert result["entities_count"] == 2  # laptop, whiteboard
        assert result["concepts_count"] == 2  # programming, tutorial

        # Verify graph has the video
        graph = get_knowledge_graph(tmp_path)
        assert graph.video_count == 1

    def test_skips_already_indexed(self, tmp_path: Path):
        # Create mock video cache with entities
        video_dir = tmp_path / "vid456"
        video_dir.mkdir()
        (video_dir / "state.json").write_text(
            json.dumps({"video_id": "vid456", "title": "Test"})
        )

        # Add entity data to avoid no_data status
        entities_dir = video_dir / "entities"
        entities_dir.mkdir()
        (entities_dir / "concepts.json").write_text(
            json.dumps({"concepts": {"python": {"term": "python", "mentions": []}}})
        )

        # Index once
        result1 = index_video_to_graph("vid456", video_dir, tmp_path)
        assert result1["status"] == "indexed"

        # Try to index again
        result2 = index_video_to_graph("vid456", video_dir, tmp_path)
        assert result2["status"] == "already_indexed"
        assert result2["from_cache"] is True

    def test_force_reindexes(self, tmp_path: Path):
        # Create mock video cache with entities
        video_dir = tmp_path / "vid789"
        video_dir.mkdir()
        (video_dir / "state.json").write_text(
            json.dumps({"video_id": "vid789", "title": "Original"})
        )

        # Add entity data to avoid no_data status
        entities_dir = video_dir / "entities"
        entities_dir.mkdir()
        (entities_dir / "concepts.json").write_text(
            json.dumps({"concepts": {"coding": {"term": "coding", "mentions": []}}})
        )

        # Index once
        index_video_to_graph("vid789", video_dir, tmp_path)

        # Update state
        (video_dir / "state.json").write_text(
            json.dumps({"video_id": "vid789", "title": "Updated"})
        )

        # Force re-index
        result = index_video_to_graph("vid789", video_dir, tmp_path, force=True)
        assert result["status"] == "indexed"

        # Verify updated
        graph = get_knowledge_graph(tmp_path)
        video = graph.get_video("vid789")
        assert video.title == "Updated"

    def test_warns_when_no_entities(self, tmp_path: Path):
        # Create mock video cache WITHOUT entities
        video_dir = tmp_path / "vid_empty"
        video_dir.mkdir()
        (video_dir / "state.json").write_text(
            json.dumps({"video_id": "vid_empty", "title": "No Entities"})
        )

        # Index should return no_data status
        result = index_video_to_graph("vid_empty", video_dir, tmp_path)
        assert result["status"] == "no_data"
        assert "warning" in result
        assert "extract_entities_tool" in result["warning"]
        assert result["entities_count"] == 0
        assert result["concepts_count"] == 0

        # Verify video was NOT added to graph
        graph = get_knowledge_graph(tmp_path)
        assert graph.get_video("vid_empty") is None
