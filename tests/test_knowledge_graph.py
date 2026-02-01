"""Tests for cross-video knowledge graph."""

from pathlib import Path


class TestExtractTopicKeywords:
    """Test TF-IDF keyword extraction."""

    def test_extracts_keywords_from_texts(self):
        from claudetube.operations.knowledge_graph import extract_topic_keywords

        texts = [
            "Introduction to Python programming",
            "Python functions and methods",
            "Advanced Python classes and objects",
        ]
        keywords = extract_topic_keywords(texts, top_n=5)

        assert len(keywords) > 0
        assert all("keyword" in k and "score" in k for k in keywords)
        # "python" should be highly ranked as it appears in all texts
        keyword_texts = [k["keyword"] for k in keywords]
        assert "python" in keyword_texts

    def test_handles_empty_texts(self):
        from claudetube.operations.knowledge_graph import extract_topic_keywords

        assert extract_topic_keywords([]) == []
        assert extract_topic_keywords(["", "  "]) == []

    def test_handles_single_text(self):
        from claudetube.operations.knowledge_graph import extract_topic_keywords

        keywords = extract_topic_keywords(["React hooks and state management"])
        assert len(keywords) > 0

    def test_includes_bigrams(self):
        from claudetube.operations.knowledge_graph import extract_topic_keywords

        texts = [
            "Machine learning basics",
            "Deep learning fundamentals",
            "Machine learning advanced",
        ]
        keywords = extract_topic_keywords(texts, top_n=10)
        keyword_texts = [k["keyword"] for k in keywords]
        # Should include bigrams like "machine learning"
        assert any(" " in kw for kw in keyword_texts) or "learning" in keyword_texts


class TestExtractSharedEntities:
    """Test shared entity extraction."""

    def test_finds_shared_technology_terms(self):
        from claudetube.operations.knowledge_graph import extract_shared_entities

        videos = [
            {"title": "Python basics", "description": "Learn Python programming"},
            {"title": "Python functions", "description": "Functions in Python"},
            {"title": "Python classes", "description": "OOP with Python"},
        ]
        entities = extract_shared_entities(videos)

        # Python appears in all videos
        python_entity = next((e for e in entities if e["text"] == "python"), None)
        assert python_entity is not None
        assert python_entity["video_count"] == 3
        assert python_entity["type"] == "technology"

    def test_filters_single_occurrence(self):
        from claudetube.operations.knowledge_graph import extract_shared_entities

        videos = [
            {"title": "Python basics"},
            {"title": "JavaScript intro"},
            {"title": "Rust fundamentals"},
        ]
        entities = extract_shared_entities(videos)

        # Each tech only appears once, so none should be shared
        assert len(entities) == 0

    def test_handles_empty_videos(self):
        from claudetube.operations.knowledge_graph import extract_shared_entities

        assert extract_shared_entities([]) == []
        assert extract_shared_entities([{"title": ""}]) == []


class TestBuildPrerequisiteChain:
    """Test prerequisite chain building."""

    def test_course_has_prerequisites(self):
        from claudetube.operations.knowledge_graph import build_prerequisite_chain

        videos = [
            {"video_id": "vid1", "position": 0, "title": "Intro"},
            {"video_id": "vid2", "position": 1, "title": "Basics"},
            {"video_id": "vid3", "position": 2, "title": "Advanced"},
        ]
        enriched = build_prerequisite_chain(videos, "course")

        # First video has no prerequisites
        assert enriched[0]["prerequisites"] == []
        assert enriched[0]["previous"] is None
        assert enriched[0]["next"] == "vid2"

        # Second video requires first
        assert enriched[1]["prerequisites"] == ["vid1"]
        assert enriched[1]["previous"] == "vid1"
        assert enriched[1]["next"] == "vid3"

        # Third video requires first two
        assert enriched[2]["prerequisites"] == ["vid1", "vid2"]
        assert enriched[2]["previous"] == "vid2"
        assert enriched[2]["next"] is None

    def test_series_has_prerequisites(self):
        from claudetube.operations.knowledge_graph import build_prerequisite_chain

        videos = [
            {"video_id": "ep1", "position": 0},
            {"video_id": "ep2", "position": 1},
        ]
        enriched = build_prerequisite_chain(videos, "series")

        assert enriched[0]["prerequisites"] == []
        assert enriched[1]["prerequisites"] == ["ep1"]

    def test_collection_has_no_prerequisites(self):
        from claudetube.operations.knowledge_graph import build_prerequisite_chain

        videos = [
            {"video_id": "vid1", "position": 0},
            {"video_id": "vid2", "position": 1},
        ]
        enriched = build_prerequisite_chain(videos, "collection")

        assert enriched[0]["prerequisites"] == []
        assert enriched[1]["prerequisites"] == []
        assert enriched[0]["next"] is None

    def test_conference_has_no_prerequisites(self):
        from claudetube.operations.knowledge_graph import build_prerequisite_chain

        videos = [
            {"video_id": "talk1", "position": 0},
            {"video_id": "talk2", "position": 1},
        ]
        enriched = build_prerequisite_chain(videos, "conference")

        assert enriched[0]["prerequisites"] == []
        assert enriched[1]["prerequisites"] == []

    def test_handles_unsorted_positions(self):
        from claudetube.operations.knowledge_graph import build_prerequisite_chain

        # Videos out of order by position
        videos = [
            {"video_id": "vid3", "position": 2},
            {"video_id": "vid1", "position": 0},
            {"video_id": "vid2", "position": 1},
        ]
        enriched = build_prerequisite_chain(videos, "course")

        # Should be sorted by position
        assert enriched[0]["video_id"] == "vid1"
        assert enriched[1]["video_id"] == "vid2"
        assert enriched[2]["video_id"] == "vid3"


class TestCreateVideoSymlinks:
    """Test video symlink creation."""

    def test_creates_symlinks_for_cached_videos(self, tmp_path: Path):
        from claudetube.operations.knowledge_graph import create_video_symlinks

        # Create mock video cache
        video_cache = tmp_path / "vid123"
        video_cache.mkdir()
        (video_cache / "state.json").write_text("{}")

        videos = [{"video_id": "vid123"}]
        symlinks = create_video_symlinks(videos, "playlist1", tmp_path)

        assert symlinks["vid123"] is not None
        assert symlinks["vid123"].is_symlink()
        assert symlinks["vid123"].resolve() == video_cache

    def test_returns_none_for_uncached_videos(self, tmp_path: Path):
        from claudetube.operations.knowledge_graph import create_video_symlinks

        videos = [{"video_id": "not_cached"}]
        symlinks = create_video_symlinks(videos, "playlist1", tmp_path)

        assert symlinks["not_cached"] is None

    def test_updates_existing_symlinks(self, tmp_path: Path):
        from claudetube.operations.knowledge_graph import create_video_symlinks

        # Create video cache
        video_cache = tmp_path / "vid123"
        video_cache.mkdir()

        # Create old symlink
        playlist_dir = tmp_path / "playlists" / "playlist1" / "videos"
        playlist_dir.mkdir(parents=True)
        old_target = tmp_path / "old_target"
        old_target.mkdir()
        (playlist_dir / "vid123").symlink_to(old_target)

        videos = [{"video_id": "vid123"}]
        symlinks = create_video_symlinks(videos, "playlist1", tmp_path)

        # Symlink should now point to correct location
        assert symlinks["vid123"].resolve() == video_cache


class TestBuildKnowledgeGraph:
    """Test full knowledge graph building."""

    def test_builds_complete_graph(self, tmp_path: Path):
        from claudetube.operations.knowledge_graph import build_knowledge_graph

        playlist_data = {
            "playlist_id": "PL123",
            "title": "Python Course",
            "description": "Learn Python programming",
            "inferred_type": "course",
            "videos": [
                {"video_id": "vid1", "position": 0, "title": "Python Introduction"},
                {"video_id": "vid2", "position": 1, "title": "Python Variables"},
                {"video_id": "vid3", "position": 2, "title": "Python Functions"},
            ],
        }

        graph = build_knowledge_graph(playlist_data, tmp_path)

        assert graph["playlist"] == playlist_data
        assert "common_topics" in graph
        assert "shared_entities" in graph
        assert len(graph["videos"]) == 3

        # Videos should have prerequisites
        assert graph["videos"][0]["prerequisites"] == []
        assert graph["videos"][1]["prerequisites"] == ["vid1"]
        assert graph["videos"][2]["prerequisites"] == ["vid1", "vid2"]

    def test_includes_cached_video_ids(self, tmp_path: Path):
        from claudetube.operations.knowledge_graph import build_knowledge_graph

        # Create one cached video
        (tmp_path / "vid1").mkdir()

        playlist_data = {
            "playlist_id": "PL456",
            "title": "Mixed Playlist",
            "inferred_type": "collection",
            "videos": [
                {"video_id": "vid1", "position": 0, "title": "Cached Video"},
                {"video_id": "vid2", "position": 1, "title": "Not Cached"},
            ],
        }

        graph = build_knowledge_graph(playlist_data, tmp_path)

        assert "vid1" in graph["cached_videos"]
        assert "vid2" not in graph["cached_videos"]


class TestSaveLoadKnowledgeGraph:
    """Test knowledge graph persistence."""

    def test_save_and_load(self, tmp_path: Path):
        from claudetube.operations.knowledge_graph import (
            load_knowledge_graph,
            save_knowledge_graph,
        )

        graph = {
            "playlist": {"playlist_id": "PL789", "title": "Test Playlist"},
            "common_topics": [{"keyword": "test", "score": 1.0}],
            "shared_entities": [],
            "videos": [{"video_id": "v1", "prerequisites": []}],
            "cached_videos": [],
        }

        save_path = save_knowledge_graph(graph, tmp_path)
        assert save_path.exists()
        assert save_path.name == "knowledge_graph.json"

        loaded = load_knowledge_graph("PL789", tmp_path)
        assert loaded is not None
        assert loaded["playlist"]["title"] == "Test Playlist"

    def test_load_missing_returns_none(self, tmp_path: Path):
        from claudetube.operations.knowledge_graph import load_knowledge_graph

        result = load_knowledge_graph("nonexistent", tmp_path)
        assert result is None


class TestGetVideoContext:
    """Test video context retrieval."""

    def test_returns_context_for_video(self, tmp_path: Path):
        from claudetube.operations.knowledge_graph import (
            get_video_context,
            save_knowledge_graph,
        )

        graph = {
            "playlist": {
                "playlist_id": "PL_CTX",
                "title": "Context Test",
                "inferred_type": "course",
            },
            "common_topics": [{"keyword": "python", "score": 2.5}],
            "shared_entities": [{"text": "function", "type": "concept", "video_count": 3}],
            "videos": [
                {"video_id": "v1", "position": 0, "title": "First", "prerequisites": [], "next": "v2", "previous": None},
                {"video_id": "v2", "position": 1, "title": "Second", "prerequisites": ["v1"], "next": "v3", "previous": "v1"},
                {"video_id": "v3", "position": 2, "title": "Third", "prerequisites": ["v1", "v2"], "next": None, "previous": "v2"},
            ],
            "cached_videos": ["v1"],
        }
        save_knowledge_graph(graph, tmp_path)

        context = get_video_context("v2", "PL_CTX", tmp_path)

        assert context is not None
        assert context["video"]["video_id"] == "v2"
        assert context["playlist_title"] == "Context Test"
        assert context["playlist_type"] == "course"
        assert context["position"] == 1
        assert context["total_videos"] == 3
        assert context["prerequisites"] == ["v1"]
        assert context["next"] == "v3"
        assert context["previous"] == "v1"
        assert len(context["prerequisite_titles"]) == 1
        assert context["prerequisite_titles"][0]["title"] == "First"

    def test_returns_none_for_missing_video(self, tmp_path: Path):
        from claudetube.operations.knowledge_graph import (
            get_video_context,
            save_knowledge_graph,
        )

        graph = {
            "playlist": {"playlist_id": "PL_MISS", "title": "Test"},
            "common_topics": [],
            "shared_entities": [],
            "videos": [{"video_id": "v1", "position": 0}],
            "cached_videos": [],
        }
        save_knowledge_graph(graph, tmp_path)

        context = get_video_context("v999", "PL_MISS", tmp_path)
        assert context is None

    def test_returns_none_for_missing_graph(self, tmp_path: Path):
        from claudetube.operations.knowledge_graph import get_video_context

        context = get_video_context("v1", "nonexistent", tmp_path)
        assert context is None
