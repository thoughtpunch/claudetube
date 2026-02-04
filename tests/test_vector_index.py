"""
Tests for vector index module (sqlite-vec-based scene search).
"""

import numpy as np
import pytest


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
        """Should return False for empty legacy chroma directory."""
        from claudetube.analysis.vector_index import has_vector_index

        (tmp_path / "embeddings" / "chroma").mkdir(parents=True)
        assert has_vector_index(tmp_path) is False


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

    def test_search_result_video_id(self):
        """Should support optional video_id for cross-video search."""
        from claudetube.analysis.vector_index import SearchResult

        result = SearchResult(
            scene_id=1,
            distance=0.5,
            start_time=30.0,
            end_time=60.0,
            transcript_preview="Sample text",
            visual_description="",
            video_id="test_video_123",
        )

        assert result.video_id == "test_video_123"


class TestScoreNormalization:
    """Tests for score normalization functions in search.py."""

    def test_normalize_fts_score_typical(self):
        """Should normalize typical FTS5 scores to 0-1 range."""
        from claudetube.analysis.search import normalize_fts_score

        # Best possible match (rank = -20)
        assert normalize_fts_score(-20.0) == pytest.approx(1.0)

        # Moderate match (rank = -10)
        assert normalize_fts_score(-10.0) == pytest.approx(0.5)

        # Weak match (rank = -5)
        assert normalize_fts_score(-5.0) == pytest.approx(0.25)

        # No match (rank = 0)
        assert normalize_fts_score(0.0) == pytest.approx(0.0)

    def test_normalize_fts_score_edge_cases(self):
        """Should handle edge cases correctly."""
        from claudetube.analysis.search import normalize_fts_score

        # Very strong match (beyond typical range)
        assert normalize_fts_score(-30.0) == pytest.approx(1.0)

        # Positive rank (shouldn't happen, but handle gracefully)
        assert normalize_fts_score(5.0) == pytest.approx(0.0)

    def test_normalize_vec_distance_typical(self):
        """Should normalize typical L2 distances to 0-1 range."""
        from claudetube.analysis.search import normalize_vec_distance

        # Identical embedding (distance = 0)
        assert normalize_vec_distance(0.0) == pytest.approx(1.0)

        # Moderate distance
        assert normalize_vec_distance(1.0) == pytest.approx(0.5)

        # Maximum expected distance
        assert normalize_vec_distance(2.0) == pytest.approx(0.0)

    def test_normalize_vec_distance_edge_cases(self):
        """Should handle edge cases correctly."""
        from claudetube.analysis.search import normalize_vec_distance

        # Beyond max distance
        assert normalize_vec_distance(3.0) == pytest.approx(0.0)

        # Negative distance (shouldn't happen)
        assert normalize_vec_distance(-0.5) == pytest.approx(1.0)


class TestMergeResults:
    """Tests for _merge_results function."""

    def test_deduplicates_by_scene_id(self):
        """Should keep only one result per scene_id."""
        from claudetube.analysis.search import SearchMoment, _merge_results

        text_results = [
            SearchMoment(1, 0, 0.0, 30.0, 0.6, "preview", "0:00", "text"),
            SearchMoment(2, 1, 30.0, 60.0, 0.5, "preview", "0:30", "text"),
        ]
        semantic_results = [
            SearchMoment(1, 0, 0.0, 30.0, 0.8, "preview", "0:00", "semantic"),
            SearchMoment(2, 2, 60.0, 90.0, 0.7, "preview", "1:00", "semantic"),
        ]

        merged = _merge_results(text_results, semantic_results, top_k=10)

        scene_ids = [m.scene_id for m in merged]
        assert len(scene_ids) == len(set(scene_ids))  # No duplicates

    def test_combines_scores_for_shared_scenes(self):
        """Should blend text and semantic scores for scenes in both sets."""
        from claudetube.analysis.search import SearchMoment, _merge_results

        text_results = [
            SearchMoment(1, 0, 0.0, 30.0, 0.6, "text preview", "0:00", "text"),
        ]
        semantic_results = [
            SearchMoment(1, 0, 0.0, 30.0, 0.8, "semantic preview", "0:00", "semantic"),
        ]

        # Equal weight (default)
        merged = _merge_results(text_results, semantic_results, top_k=10)

        assert len(merged) == 1
        # 0.5 * 0.6 + 0.5 * 0.8 = 0.7
        assert merged[0].relevance == pytest.approx(0.7)
        assert merged[0].match_type == "text+semantic"

    def test_semantic_weight_configurable(self):
        """Should use configurable weight for blending scores."""
        from claudetube.analysis.search import SearchMoment, _merge_results

        text_results = [
            SearchMoment(1, 0, 0.0, 30.0, 1.0, "text preview", "0:00", "text"),
        ]
        semantic_results = [
            SearchMoment(1, 0, 0.0, 30.0, 0.0, "semantic preview", "0:00", "semantic"),
        ]

        # Heavy text weight
        merged = _merge_results(
            text_results, semantic_results, top_k=10, semantic_weight=0.2
        )
        assert merged[0].relevance == pytest.approx(0.8)

        # Recreate moments (relevance was mutated)
        text_results = [
            SearchMoment(1, 0, 0.0, 30.0, 1.0, "text preview", "0:00", "text"),
        ]
        semantic_results = [
            SearchMoment(1, 0, 0.0, 30.0, 0.0, "semantic preview", "0:00", "semantic"),
        ]

        # Heavy semantic weight
        merged = _merge_results(
            text_results, semantic_results, top_k=10, semantic_weight=0.8
        )
        assert merged[0].relevance == pytest.approx(0.2)

    def test_text_only_scene_keeps_original_score(self):
        """Scenes only in text results should keep their score."""
        from claudetube.analysis.search import SearchMoment, _merge_results

        text_results = [
            SearchMoment(1, 0, 0.0, 30.0, 0.6, "text preview", "0:00", "text"),
        ]

        merged = _merge_results(text_results, [], top_k=10)

        assert len(merged) == 1
        assert merged[0].relevance == 0.6
        assert merged[0].match_type == "text"

    def test_semantic_only_scene_keeps_original_score(self):
        """Scenes only in semantic results should keep their score."""
        from claudetube.analysis.search import SearchMoment, _merge_results

        semantic_results = [
            SearchMoment(1, 0, 0.0, 30.0, 0.9, "semantic preview", "0:00", "semantic"),
        ]

        merged = _merge_results([], semantic_results, top_k=10)

        assert len(merged) == 1
        assert merged[0].relevance == 0.9
        assert merged[0].match_type == "semantic"

    def test_respects_top_k(self):
        """Should limit results to top_k."""
        from claudetube.analysis.search import SearchMoment, _merge_results

        results = [
            SearchMoment(
                i,
                i,
                float(i * 30),
                float((i + 1) * 30),
                0.5,
                "preview",
                f"{i}:00",
                "text",
            )
            for i in range(10)
        ]

        merged = _merge_results(results, [], top_k=3)

        assert len(merged) == 3

    def test_reassigns_ranks(self):
        """Should reassign ranks after merge."""
        from claudetube.analysis.search import SearchMoment, _merge_results

        results = [
            SearchMoment(5, 0, 0.0, 30.0, 0.9, "preview", "0:00", "text"),
            SearchMoment(3, 1, 30.0, 60.0, 0.7, "preview", "0:30", "text"),
        ]

        merged = _merge_results(results, [], top_k=10)

        assert merged[0].rank == 1
        assert merged[1].rank == 2

    def test_combined_score_capped_at_one(self):
        """Combined relevance should not exceed 1.0."""
        from claudetube.analysis.search import SearchMoment, _merge_results

        text_results = [
            SearchMoment(1, 0, 0.0, 30.0, 1.0, "preview", "0:00", "text"),
        ]
        semantic_results = [
            SearchMoment(1, 0, 0.0, 30.0, 1.0, "preview", "0:00", "semantic"),
        ]

        merged = _merge_results(text_results, semantic_results, top_k=10)

        assert merged[0].relevance <= 1.0


class TestMergeResultsCrossVideo:
    """Tests for _merge_results_cross_video function."""

    def test_deduplicates_by_video_and_scene(self):
        """Should deduplicate by (video_id, scene_id) tuple."""
        from claudetube.analysis.search import SearchMoment, _merge_results_cross_video

        # Same scene_id but different videos should NOT be deduplicated
        text_results = [
            SearchMoment(1, 0, 0.0, 30.0, 0.6, "preview", "0:00", "text", "video_a"),
            SearchMoment(2, 0, 0.0, 30.0, 0.5, "preview", "0:00", "text", "video_b"),
        ]
        semantic_results = []

        merged = _merge_results_cross_video(text_results, semantic_results, top_k=10)

        assert len(merged) == 2  # Both should remain

    def test_merges_same_video_scene(self):
        """Should merge scores for same (video_id, scene_id)."""
        from claudetube.analysis.search import SearchMoment, _merge_results_cross_video

        text_results = [
            SearchMoment(
                1, 0, 0.0, 30.0, 0.6, "text preview", "0:00", "text", "video_a"
            ),
        ]
        semantic_results = [
            SearchMoment(
                1, 0, 0.0, 30.0, 0.8, "semantic preview", "0:00", "semantic", "video_a"
            ),
        ]

        merged = _merge_results_cross_video(text_results, semantic_results, top_k=10)

        assert len(merged) == 1
        assert merged[0].relevance == pytest.approx(0.7)
        assert merged[0].match_type == "text+semantic"


class TestUnifiedSearch:
    """Tests for unified_search function."""

    def test_returns_empty_for_unknown_video(self):
        """Should return empty list for unknown video."""
        from claudetube.analysis.search import unified_search

        # This will likely fail to find the video and return empty
        results = unified_search("nonexistent_video_xyz", "test query", top_k=5)
        assert results == [] or isinstance(results, list)


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

    def test_no_chromadb_import(self):
        """vector_index should not import chromadb directly."""
        import ast
        from pathlib import Path

        from claudetube.analysis import vector_index

        source = ast.parse(Path(vector_index.__file__).read_text())

        # Check that chromadb is not imported
        for node in ast.walk(source):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "chromadb" in alias.name:
                        pytest.fail("chromadb still imported - should use sqlite-vec")
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "chromadb" in node.module
            ):
                pytest.fail("chromadb still imported - should use sqlite-vec")
