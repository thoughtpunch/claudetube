"""
Tests for temporal grounding search module.
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
        {
            "scene_id": 3,
            "start_time": 90.0,
            "end_time": 120.0,
            "transcript_text": "We fix the bug by changing the learning rate parameter.",
        },
    ]


@pytest.fixture
def video_cache_dir(tmp_path, mock_scene_data):
    """Create a mock video cache with scenes data."""
    from claudetube.cache.scenes import SceneBoundary, ScenesData, save_scenes_data

    video_dir = tmp_path / "test_video_123"
    video_dir.mkdir()

    # Create scenes data
    # Note: transcript list must be non-empty for transcript_text to be saved
    scenes = [
        SceneBoundary(
            scene_id=s["scene_id"],
            start_time=s["start_time"],
            end_time=s["end_time"],
            transcript=[{"start": s["start_time"], "text": s["transcript_text"]}],
            transcript_text=s["transcript_text"],
        )
        for s in mock_scene_data
    ]

    scenes_data = ScenesData(
        video_id="test_video_123",
        method="transcript",
        scenes=scenes,
    )

    save_scenes_data(video_dir, scenes_data)

    return video_dir


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_seconds_only(self):
        """Should format seconds under a minute."""
        from claudetube.analysis.search import format_timestamp

        assert format_timestamp(45) == "0:45"
        assert format_timestamp(0) == "0:00"

    def test_minutes_and_seconds(self):
        """Should format minutes and seconds."""
        from claudetube.analysis.search import format_timestamp

        assert format_timestamp(90) == "1:30"
        assert format_timestamp(125) == "2:05"

    def test_hours(self):
        """Should include hours when needed."""
        from claudetube.analysis.search import format_timestamp

        assert format_timestamp(3600) == "1:00:00"
        assert format_timestamp(3661) == "1:01:01"
        assert format_timestamp(7325) == "2:02:05"

    def test_decimal_seconds(self):
        """Should truncate decimal seconds."""
        from claudetube.analysis.search import format_timestamp

        assert format_timestamp(90.5) == "1:30"
        assert format_timestamp(90.9) == "1:30"


class TestSearchMoment:
    """Tests for SearchMoment dataclass."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        from claudetube.analysis.search import SearchMoment

        moment = SearchMoment(
            rank=1,
            scene_id=2,
            start_time=60.0,
            end_time=90.0,
            relevance=0.85,
            preview="Sample text...",
            timestamp_str="1:00",
            match_type="text",
        )

        result = moment.to_dict()

        assert result["rank"] == 1
        assert result["scene_id"] == 2
        assert result["start_time"] == 60.0
        assert result["end_time"] == 90.0
        assert result["relevance"] == 0.85
        assert result["preview"] == "Sample text..."
        assert result["timestamp_str"] == "1:00"
        assert result["match_type"] == "text"


class TestSearchTranscriptText:
    """Tests for _search_transcript_text function."""

    def test_exact_phrase_match(self, video_cache_dir):
        """Should find exact phrase matches with high relevance."""
        from claudetube.analysis.search import _search_transcript_text

        results = _search_transcript_text(video_cache_dir, "fix the bug")

        assert len(results) >= 1
        assert results[0].scene_id == 3
        assert results[0].relevance > 0.4
        assert results[0].match_type == "text"

    def test_word_match(self, video_cache_dir):
        """Should find word matches."""
        from claudetube.analysis.search import _search_transcript_text

        results = _search_transcript_text(video_cache_dir, "machine learning")

        assert len(results) >= 1
        # Scene 0 has "machine learning"
        scene_ids = [r.scene_id for r in results]
        assert 0 in scene_ids

    def test_partial_word_match(self, video_cache_dir):
        """Should find partial word matches."""
        from claudetube.analysis.search import _search_transcript_text

        results = _search_transcript_text(video_cache_dir, "neural")

        assert len(results) >= 1
        scene_ids = [r.scene_id for r in results]
        assert 1 in scene_ids

    def test_case_insensitive(self, video_cache_dir):
        """Should match case-insensitively."""
        from claudetube.analysis.search import _search_transcript_text

        results_lower = _search_transcript_text(video_cache_dir, "python")
        results_upper = _search_transcript_text(video_cache_dir, "PYTHON")

        assert len(results_lower) == len(results_upper)
        assert results_lower[0].scene_id == results_upper[0].scene_id

    def test_no_match(self, video_cache_dir):
        """Should return empty list for no matches."""
        from claudetube.analysis.search import _search_transcript_text

        results = _search_transcript_text(video_cache_dir, "xyzzy nonexistent")

        assert len(results) == 0

    def test_top_k_limit(self, video_cache_dir):
        """Should respect top_k parameter."""
        from claudetube.analysis.search import _search_transcript_text

        results = _search_transcript_text(video_cache_dir, "the", top_k=2)

        assert len(results) <= 2

    def test_ranks_assigned(self, video_cache_dir):
        """Should assign consecutive ranks starting from 1."""
        from claudetube.analysis.search import _search_transcript_text

        results = _search_transcript_text(video_cache_dir, "learning", top_k=5)

        for i, result in enumerate(results):
            assert result.rank == i + 1


class TestCreatePreview:
    """Tests for _create_preview function."""

    def test_short_transcript(self):
        """Should return full text if under max_len."""
        from claudetube.analysis.search import _create_preview

        text = "Short transcript text"
        preview = _create_preview(text, "short", max_len=100)

        assert preview == text

    def test_centered_on_match(self):
        """Should center preview on query match."""
        from claudetube.analysis.search import _create_preview

        text = "A" * 100 + " MATCH " + "B" * 100
        preview = _create_preview(text, "MATCH", max_len=50)

        assert "MATCH" in preview
        assert preview.startswith("...")
        assert preview.endswith("...")

    def test_no_match_returns_start(self):
        """Should return start of text if no match found."""
        from claudetube.analysis.search import _create_preview

        text = "A" * 200
        preview = _create_preview(text, "nonexistent", max_len=50)

        assert preview.startswith("A")
        assert preview.endswith("...")


class TestFindMoments:
    """Tests for find_moments function."""

    def test_video_not_found(self, tmp_path):
        """Should raise FileNotFoundError for missing video."""
        from claudetube.analysis.search import find_moments

        with pytest.raises(FileNotFoundError, match="not found in cache"):
            find_moments("nonexistent_video", "query", cache_dir=tmp_path)

    def test_no_scenes(self, tmp_path):
        """Should raise ValueError if no scene data."""
        from claudetube.analysis.search import find_moments

        video_dir = tmp_path / "video_no_scenes"
        video_dir.mkdir()

        with pytest.raises(ValueError, match="no scene data"):
            find_moments("video_no_scenes", "query", cache_dir=tmp_path)

    def test_text_search_strategy(self, tmp_path, video_cache_dir):
        """Should use text search when strategy='text'."""
        from claudetube.analysis.search import find_moments

        results = find_moments(
            "test_video_123",
            "fix the bug",
            cache_dir=tmp_path,
            strategy="text",
        )

        assert len(results) >= 1
        assert results[0].match_type == "text"

    def test_auto_strategy_text_first(self, tmp_path, video_cache_dir):
        """Should try text search first in auto mode."""
        from claudetube.analysis.search import find_moments

        results = find_moments(
            "test_video_123",
            "Python code",
            cache_dir=tmp_path,
            strategy="auto",
        )

        assert len(results) >= 1

    def test_returns_search_moment_objects(self, tmp_path, video_cache_dir):
        """Should return SearchMoment objects with correct fields."""
        from claudetube.analysis.search import SearchMoment, find_moments

        results = find_moments(
            "test_video_123",
            "neural networks",
            cache_dir=tmp_path,
        )

        assert len(results) >= 1
        assert isinstance(results[0], SearchMoment)
        assert results[0].rank == 1
        assert results[0].scene_id is not None
        assert results[0].start_time >= 0
        assert results[0].end_time > results[0].start_time
        assert 0 <= results[0].relevance <= 1
        assert results[0].preview
        assert results[0].timestamp_str
        assert results[0].match_type in ("text", "semantic")

    def test_top_k_parameter(self, tmp_path, video_cache_dir):
        """Should respect top_k parameter."""
        from claudetube.analysis.search import find_moments

        results = find_moments(
            "test_video_123",
            "the",  # Common word
            top_k=2,
            cache_dir=tmp_path,
        )

        assert len(results) <= 2

    def test_semantic_strategy_no_index(self, tmp_path, video_cache_dir):
        """Should raise ValueError for semantic search without index."""
        from claudetube.analysis.search import find_moments

        with pytest.raises(ValueError, match="no vector index"):
            find_moments(
                "test_video_123",
                "query",
                cache_dir=tmp_path,
                strategy="semantic",
            )


@pytest.mark.skipif(not _has_chromadb(), reason="chromadb not installed")
class TestFindMomentsWithEmbeddings:
    """Tests for find_moments with vector index."""

    @pytest.fixture
    def indexed_video_cache(self, video_cache_dir, mock_scene_data):
        """Create a video cache with vector index."""
        from claudetube.analysis.embeddings import SceneEmbedding
        from claudetube.analysis.vector_index import build_scene_index

        # Create mock embeddings
        embeddings = [
            SceneEmbedding(
                scene_id=s["scene_id"],
                embedding=np.random.randn(512).astype(np.float32),
                model="local",
            )
            for s in mock_scene_data
        ]

        # Build vector index
        build_scene_index(
            cache_dir=video_cache_dir,
            scenes=mock_scene_data,
            embeddings=embeddings,
            video_id="test_video_123",
        )

        return video_cache_dir

    def test_semantic_search_with_index(self, tmp_path, indexed_video_cache):
        """Should use semantic search when index available."""
        from claudetube.analysis.search import find_moments

        results = find_moments(
            "test_video_123",
            "debugging code issues",
            cache_dir=tmp_path,
            strategy="semantic",
        )

        # Should return results (exact content depends on embeddings)
        assert isinstance(results, list)

    def test_auto_merges_text_and_semantic(self, tmp_path, indexed_video_cache):
        """Should merge text and semantic results in auto mode."""
        from claudetube.analysis.search import find_moments

        # Use a query that won't have strong text match
        results = find_moments(
            "test_video_123",
            "programming concepts",
            cache_dir=tmp_path,
            strategy="auto",
        )

        # Should return results from either method
        assert isinstance(results, list)


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

    def test_prefers_higher_relevance(self):
        """Should keep result with higher relevance."""
        from claudetube.analysis.search import SearchMoment, _merge_results

        text_results = [
            SearchMoment(1, 0, 0.0, 30.0, 0.3, "text preview", "0:00", "text"),
        ]
        semantic_results = [
            SearchMoment(1, 0, 0.0, 30.0, 0.9, "semantic preview", "0:00", "semantic"),
        ]

        merged = _merge_results(text_results, semantic_results, top_k=10)

        assert len(merged) == 1
        assert merged[0].relevance == 0.9
        assert merged[0].match_type == "semantic"

    def test_respects_top_k(self):
        """Should limit results to top_k."""
        from claudetube.analysis.search import SearchMoment, _merge_results

        results = [
            SearchMoment(i, i, float(i * 30), float((i + 1) * 30), 0.5, "preview", f"{i}:00", "text")
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


class TestExpandQuery:
    """Tests for expand_query function."""

    @pytest.mark.asyncio
    async def test_returns_expanded_terms(self):
        """Should return list of expanded query strings."""
        from unittest.mock import AsyncMock

        from claudetube.analysis.search import expand_query

        mock_reasoner = AsyncMock()
        mock_reasoner.reason.return_value = (
            "debugging code\n"
            "fixing errors\n"
            "resolving bugs\n"
            "troubleshooting issues\n"
            "patching defects"
        )

        result = await expand_query("fix the bug", mock_reasoner)

        assert len(result) == 5
        assert "debugging code" in result
        assert "fixing errors" in result

    @pytest.mark.asyncio
    async def test_excludes_original_query(self):
        """Should not include the original query in expanded terms."""
        from unittest.mock import AsyncMock

        from claudetube.analysis.search import expand_query

        mock_reasoner = AsyncMock()
        mock_reasoner.reason.return_value = (
            "fix the bug\n"
            "debugging code\n"
            "resolving errors"
        )

        result = await expand_query("fix the bug", mock_reasoner)

        assert "fix the bug" not in result

    @pytest.mark.asyncio
    async def test_limits_to_five_terms(self):
        """Should return at most 5 terms."""
        from unittest.mock import AsyncMock

        from claudetube.analysis.search import expand_query

        mock_reasoner = AsyncMock()
        mock_reasoner.reason.return_value = "\n".join(
            [f"term {i}" for i in range(10)]
        )

        result = await expand_query("query", mock_reasoner)

        assert len(result) <= 5

    @pytest.mark.asyncio
    async def test_handles_reasoner_failure(self):
        """Should return empty list on reasoner failure."""
        from unittest.mock import AsyncMock

        from claudetube.analysis.search import expand_query

        mock_reasoner = AsyncMock()
        mock_reasoner.reason.side_effect = Exception("API error")

        result = await expand_query("query", mock_reasoner)

        assert result == []

    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        """Should handle empty reasoner response."""
        from unittest.mock import AsyncMock

        from claudetube.analysis.search import expand_query

        mock_reasoner = AsyncMock()
        mock_reasoner.reason.return_value = ""

        result = await expand_query("query", mock_reasoner)

        assert result == []

    @pytest.mark.asyncio
    async def test_strips_whitespace(self):
        """Should strip whitespace from expanded terms."""
        from unittest.mock import AsyncMock

        from claudetube.analysis.search import expand_query

        mock_reasoner = AsyncMock()
        mock_reasoner.reason.return_value = (
            "  term one  \n"
            "  term two  \n"
            "  \n"
            "term three"
        )

        result = await expand_query("query", mock_reasoner)

        assert "term one" in result
        assert "term two" in result
        assert "term three" in result
        assert "" not in result


class TestSearchWithExpandedQueries:
    """Tests for _search_with_expanded_queries function."""

    def test_includes_original_results(self, video_cache_dir):
        """Should include results from original query."""
        from claudetube.analysis.search import _search_with_expanded_queries

        results = _search_with_expanded_queries(
            video_cache_dir, "fix the bug", [], top_k=5
        )

        assert len(results) >= 1
        assert results[0].scene_id == 3

    def test_expanded_queries_add_results(self, video_cache_dir):
        """Should find additional results from expanded queries."""
        from claudetube.analysis.search import _search_with_expanded_queries

        # Original query won't match scene 1, but expanded query will
        results = _search_with_expanded_queries(
            video_cache_dir,
            "fix the bug",
            ["neural networks deep learning"],
            top_k=10,
        )

        scene_ids = [r.scene_id for r in results]
        assert 3 in scene_ids  # Original match
        assert 1 in scene_ids  # Expanded match

    def test_expanded_results_discounted(self, video_cache_dir):
        """Should discount expanded query results by 0.8."""
        from claudetube.analysis.search import _search_with_expanded_queries

        # "machine learning" matches scene 0 directly
        results_original = _search_with_expanded_queries(
            video_cache_dir, "machine learning", [], top_k=5
        )
        # Same query as expanded should be discounted
        results_expanded = _search_with_expanded_queries(
            video_cache_dir, "nonexistent query xyz", ["machine learning"], top_k=5
        )

        # Find scene 0 in both
        orig_scene0 = next((r for r in results_original if r.scene_id == 0), None)
        expanded_scene0 = next((r for r in results_expanded if r.scene_id == 0), None)

        assert orig_scene0 is not None
        assert expanded_scene0 is not None
        assert expanded_scene0.relevance < orig_scene0.relevance

    def test_deduplicates_across_queries(self, video_cache_dir):
        """Should deduplicate results across original and expanded queries."""
        from claudetube.analysis.search import _search_with_expanded_queries

        # Both queries match the same scene
        results = _search_with_expanded_queries(
            video_cache_dir,
            "machine learning",
            ["machine learning tutorial"],
            top_k=10,
        )

        scene_ids = [r.scene_id for r in results]
        assert len(scene_ids) == len(set(scene_ids))  # No duplicates

    def test_expanded_match_type(self, video_cache_dir):
        """Expanded results should have text+expanded match type."""
        from claudetube.analysis.search import _search_with_expanded_queries

        results = _search_with_expanded_queries(
            video_cache_dir,
            "nonexistent xyz",
            ["neural networks"],
            top_k=5,
        )

        # Results from expanded query only
        expanded = [r for r in results if r.match_type == "text+expanded"]
        assert len(expanded) >= 1


class TestFindMomentsWithReasoner:
    """Tests for find_moments with reasoner parameter."""

    def test_without_reasoner_works(self, tmp_path, video_cache_dir):
        """Should work without reasoner (backward compatible)."""
        from claudetube.analysis.search import find_moments

        results = find_moments(
            "test_video_123",
            "fix the bug",
            cache_dir=tmp_path,
        )

        assert len(results) >= 1
        assert results[0].scene_id == 3

    def test_with_reasoner_uses_expansion(self, tmp_path, video_cache_dir):
        """Should use query expansion when reasoner is provided."""
        from unittest.mock import AsyncMock

        from claudetube.analysis.search import find_moments

        mock_reasoner = AsyncMock()
        mock_reasoner.reason.return_value = (
            "neural networks\n"
            "deep learning models\n"
            "AI training"
        )

        results = find_moments(
            "test_video_123",
            "fix the bug",
            cache_dir=tmp_path,
            strategy="text",
            reasoner=mock_reasoner,
        )

        # Should still find the bug fix scene
        scene_ids = [r.scene_id for r in results]
        assert 3 in scene_ids
        # May also find expanded results
        mock_reasoner.reason.assert_called_once()

    def test_reasoner_failure_falls_back(self, tmp_path, video_cache_dir):
        """Should fall back to regular search if reasoner fails."""
        from unittest.mock import AsyncMock

        from claudetube.analysis.search import find_moments

        mock_reasoner = AsyncMock()
        mock_reasoner.reason.side_effect = Exception("LLM unavailable")

        results = find_moments(
            "test_video_123",
            "fix the bug",
            cache_dir=tmp_path,
            reasoner=mock_reasoner,
        )

        # Should still return results from text search
        assert len(results) >= 1
        assert results[0].scene_id == 3
