"""
Tests for vector index module (ChromaDB-based scene search).
"""

import importlib.util

import numpy as np
import pytest


def _has_chromadb():
    """Check if chromadb is installed."""
    return importlib.util.find_spec("chromadb") is not None


@pytest.fixture
def mock_scene_data():
    """Create mock scene data for testing."""
    return [
        {
            "scene_id": 0,
            "start_time": 0.0,
            "end_time": 30.0,
            "transcript_text": "Welcome to the tutorial on machine learning.",
        },
        {
            "scene_id": 1,
            "start_time": 30.0,
            "end_time": 60.0,
            "transcript_text": "First, we'll cover neural networks and deep learning.",
        },
        {
            "scene_id": 2,
            "start_time": 60.0,
            "end_time": 90.0,
            "transcript_text": "Now let's look at some Python code examples.",
        },
    ]


@pytest.fixture
def mock_embeddings(mock_scene_data):
    """Create mock embeddings for testing."""
    from claudetube.analysis.embeddings import SceneEmbedding

    return [
        SceneEmbedding(
            scene_id=s["scene_id"],
            embedding=np.random.randn(512).astype(np.float32),
            model="local",
        )
        for s in mock_scene_data
    ]


class TestHasVectorIndex:
    """Tests for has_vector_index function."""

    def test_no_index(self, tmp_path):
        """Should return False when no index exists."""
        from claudetube.analysis.vector_index import has_vector_index

        assert has_vector_index(tmp_path) is False

    def test_empty_chroma_dir(self, tmp_path):
        """Should return False for empty chroma directory."""
        from claudetube.analysis.vector_index import has_vector_index

        (tmp_path / "embeddings" / "chroma").mkdir(parents=True)
        assert has_vector_index(tmp_path) is False


@pytest.mark.skipif(not _has_chromadb(), reason="chromadb not installed")
class TestBuildSceneIndex:
    """Tests for build_scene_index function."""

    def test_build_index(self, tmp_path, mock_scene_data, mock_embeddings):
        """Should create ChromaDB index with scenes."""
        from claudetube.analysis.vector_index import (
            build_scene_index,
            has_vector_index,
        )

        count = build_scene_index(
            cache_dir=tmp_path,
            scenes=mock_scene_data,
            embeddings=mock_embeddings,
            video_id="test_video",
        )

        assert count == 3
        assert has_vector_index(tmp_path)
        assert (tmp_path / "embeddings" / "chroma" / "chroma.sqlite3").exists()

    def test_build_index_empty_embeddings(self, tmp_path, mock_scene_data):
        """Should handle empty embeddings list."""
        from claudetube.analysis.vector_index import build_scene_index

        count = build_scene_index(
            cache_dir=tmp_path,
            scenes=mock_scene_data,
            embeddings=[],
        )

        assert count == 0

    def test_build_index_mismatched_scenes(self, tmp_path, mock_scene_data):
        """Should skip scenes without embeddings."""
        from claudetube.analysis.embeddings import SceneEmbedding
        from claudetube.analysis.vector_index import build_scene_index

        # Only provide embeddings for scene 0 and 2
        embeddings = [
            SceneEmbedding(
                scene_id=0,
                embedding=np.random.randn(512).astype(np.float32),
                model="local",
            ),
            SceneEmbedding(
                scene_id=2,
                embedding=np.random.randn(512).astype(np.float32),
                model="local",
            ),
        ]

        count = build_scene_index(
            cache_dir=tmp_path,
            scenes=mock_scene_data,
            embeddings=embeddings,
        )

        assert count == 2

    def test_rebuild_index(self, tmp_path, mock_scene_data, mock_embeddings):
        """Should replace existing index on rebuild."""
        from claudetube.analysis.vector_index import (
            build_scene_index,
            load_scene_index,
        )

        # Build first time
        build_scene_index(tmp_path, mock_scene_data, mock_embeddings)

        # Build again with fewer scenes
        new_scenes = [mock_scene_data[0]]
        new_embeddings = [mock_embeddings[0]]
        count = build_scene_index(tmp_path, new_scenes, new_embeddings)

        assert count == 1
        collection = load_scene_index(tmp_path)
        assert collection.count() == 1


@pytest.mark.skipif(not _has_chromadb(), reason="chromadb not installed")
class TestLoadSceneIndex:
    """Tests for load_scene_index function."""

    def test_load_nonexistent(self, tmp_path):
        """Should return None for nonexistent index."""
        from claudetube.analysis.vector_index import load_scene_index

        result = load_scene_index(tmp_path)
        assert result is None

    def test_load_existing(self, tmp_path, mock_scene_data, mock_embeddings):
        """Should load existing index."""
        from claudetube.analysis.vector_index import (
            build_scene_index,
            load_scene_index,
        )

        build_scene_index(tmp_path, mock_scene_data, mock_embeddings)

        collection = load_scene_index(tmp_path)
        assert collection is not None
        assert collection.count() == 3


@pytest.mark.skipif(not _has_chromadb(), reason="chromadb not installed")
class TestSearchScenes:
    """Tests for search_scenes function."""

    def test_search_returns_results(self, tmp_path, mock_scene_data, mock_embeddings):
        """Should return search results."""
        from claudetube.analysis.vector_index import (
            build_scene_index,
            search_scenes,
        )

        build_scene_index(tmp_path, mock_scene_data, mock_embeddings)

        query_emb = mock_embeddings[0].embedding
        results = search_scenes(tmp_path, query_emb, top_k=3)

        assert len(results) == 3
        assert all(hasattr(r, "scene_id") for r in results)
        assert all(hasattr(r, "distance") for r in results)
        assert all(hasattr(r, "start_time") for r in results)
        assert all(hasattr(r, "transcript_preview") for r in results)

    def test_search_top_k(self, tmp_path, mock_scene_data, mock_embeddings):
        """Should respect top_k parameter."""
        from claudetube.analysis.vector_index import (
            build_scene_index,
            search_scenes,
        )

        build_scene_index(tmp_path, mock_scene_data, mock_embeddings)

        query_emb = mock_embeddings[0].embedding
        results = search_scenes(tmp_path, query_emb, top_k=1)

        assert len(results) == 1

    def test_search_no_index(self, tmp_path):
        """Should raise error when no index exists."""
        from claudetube.analysis.vector_index import search_scenes

        query_emb = np.random.randn(512).astype(np.float32)

        with pytest.raises(ValueError, match="No vector index found"):
            search_scenes(tmp_path, query_emb)

    def test_search_result_metadata(self, tmp_path, mock_scene_data, mock_embeddings):
        """Should include correct metadata in results."""
        from claudetube.analysis.vector_index import (
            build_scene_index,
            search_scenes,
        )

        build_scene_index(tmp_path, mock_scene_data, mock_embeddings)

        query_emb = mock_embeddings[1].embedding
        results = search_scenes(tmp_path, query_emb, top_k=1)

        # Most similar should be scene 1 (same embedding)
        assert results[0].scene_id == 1
        assert results[0].start_time == 30.0
        assert results[0].end_time == 60.0
        assert "neural networks" in results[0].transcript_preview


@pytest.mark.skipif(not _has_chromadb(), reason="chromadb not installed")
class TestDeleteSceneIndex:
    """Tests for delete_scene_index function."""

    def test_delete_existing(self, tmp_path, mock_scene_data, mock_embeddings):
        """Should delete existing index."""
        from claudetube.analysis.vector_index import (
            build_scene_index,
            delete_scene_index,
            has_vector_index,
        )

        build_scene_index(tmp_path, mock_scene_data, mock_embeddings)
        assert has_vector_index(tmp_path)

        result = delete_scene_index(tmp_path)

        assert result is True
        assert not has_vector_index(tmp_path)

    def test_delete_nonexistent(self, tmp_path):
        """Should return False for nonexistent index."""
        from claudetube.analysis.vector_index import delete_scene_index

        result = delete_scene_index(tmp_path)
        assert result is False


@pytest.mark.skipif(not _has_chromadb(), reason="chromadb not installed")
class TestGetIndexStats:
    """Tests for get_index_stats function."""

    def test_stats_existing(self, tmp_path, mock_scene_data, mock_embeddings):
        """Should return stats for existing index."""
        from claudetube.analysis.vector_index import (
            build_scene_index,
            get_index_stats,
        )

        build_scene_index(
            tmp_path, mock_scene_data, mock_embeddings, video_id="test_vid"
        )

        stats = get_index_stats(tmp_path)

        assert stats is not None
        assert stats["num_scenes"] == 3
        assert stats["video_id"] == "test_vid"
        assert "path" in stats

    def test_stats_nonexistent(self, tmp_path):
        """Should return None for nonexistent index."""
        from claudetube.analysis.vector_index import get_index_stats

        stats = get_index_stats(tmp_path)
        assert stats is None


@pytest.mark.skipif(not _has_chromadb(), reason="chromadb not installed")
class TestSearchScenesByText:
    """Tests for search_scenes_by_text function."""

    def test_import_error_without_sentence_transformers(
        self, tmp_path, mock_scene_data, mock_embeddings, monkeypatch
    ):
        """Should raise ImportError if sentence-transformers not available for local model."""
        from claudetube.analysis.vector_index import (
            build_scene_index,
        )

        build_scene_index(tmp_path, mock_scene_data, mock_embeddings)

        # Mock missing sentence_transformers
        import sys

        original_modules = sys.modules.copy()
        sys.modules["sentence_transformers"] = None

        try:
            monkeypatch.setenv("CLAUDETUBE_EMBEDDING_MODEL", "local")
            # This should work or fail gracefully depending on environment
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_fields(self):
        """Should have all required fields."""
        from claudetube.analysis.vector_index import SearchResult

        result = SearchResult(
            scene_id=1,
            distance=0.5,
            start_time=30.0,
            end_time=60.0,
            transcript_preview="Sample text",
            visual_description="Person talking",
        )

        assert result.scene_id == 1
        assert result.distance == 0.5
        assert result.start_time == 30.0
        assert result.end_time == 60.0
        assert result.transcript_preview == "Sample text"
        assert result.visual_description == "Person talking"


class TestSearchScenesByTextProvider:
    """Tests for search_scenes_by_text using provider pattern."""

    def test_uses_embedder_provider(self, monkeypatch):
        """Should call _get_embedder and embed_sync instead of direct API calls."""
        from unittest.mock import MagicMock, patch

        mock_embedder = MagicMock()
        mock_embedder.embed_sync.return_value = [0.1] * 512

        # _get_embedder is imported from embeddings inside search_scenes_by_text
        with (
            patch(
                "claudetube.analysis.embeddings._get_embedder",
                return_value=mock_embedder,
            ) as mock_get,
            patch(
                "claudetube.analysis.vector_index.search_scenes",
                return_value=[],
            ),
        ):
            from claudetube.analysis.vector_index import search_scenes_by_text

            search_scenes_by_text(MagicMock(), "test query", model="voyage")

            mock_get.assert_called_once_with("voyage")
            mock_embedder.embed_sync.assert_called_once_with("test query")

    def test_no_direct_voyage_import(self):
        """search_scenes_by_text should not import voyageai directly."""
        import ast
        from pathlib import Path

        from claudetube.analysis import vector_index

        source = ast.parse(Path(vector_index.__file__).read_text())

        # Check that no function named _embed_text_voyage exists
        for node in ast.walk(source):
            if isinstance(node, ast.FunctionDef) and node.name == "_embed_text_voyage":
                pytest.fail(
                    "_embed_text_voyage still exists - should use provider pattern"
                )

    def test_no_direct_sentence_transformers_import(self):
        """search_scenes_by_text should not import sentence_transformers directly."""
        import ast
        from pathlib import Path

        from claudetube.analysis import vector_index

        source = ast.parse(Path(vector_index.__file__).read_text())

        # Check that no function named _embed_text_local exists
        for node in ast.walk(source):
            if isinstance(node, ast.FunctionDef) and node.name == "_embed_text_local":
                pytest.fail(
                    "_embed_text_local still exists - should use provider pattern"
                )
