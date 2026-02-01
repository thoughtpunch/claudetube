"""Tests for cache enrichment functionality."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from claudetube.cache.enrichment import (
    boost_scene_relevance,
    find_relevant_scenes,
    find_scene_at_timestamp,
    get_boosted_relevance,
    get_enrichment_stats,
    get_relevance_boosts,
    get_scene_context,
    record_frame_examination,
    record_qa_interaction,
    save_relevance_boosts,
    search_cached_qa,
)
from claudetube.cache.scenes import SceneBoundary, ScenesData, save_scenes_data


def _make_scene(
    scene_id: int,
    start_time: float,
    end_time: float,
    title: str,
    transcript_text: str,
) -> SceneBoundary:
    """Create a SceneBoundary with transcript list populated for proper serialization."""
    # The transcript list must be non-empty for transcript_text to be serialized
    return SceneBoundary(
        scene_id=scene_id,
        start_time=start_time,
        end_time=end_time,
        title=title,
        transcript=[{"text": transcript_text, "start": start_time, "end": end_time}],
        transcript_text=transcript_text,
    )


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Create a cache directory with scene data."""
    cache = tmp_path / "test_video"
    cache.mkdir()

    # Create state.json
    state = {"video_id": "test_video", "title": "Test Video"}
    (cache / "state.json").write_text(json.dumps(state))

    # Create scenes data with transcript
    scenes = ScenesData(
        video_id="test_video",
        method="transcript",
        scenes=[
            _make_scene(
                scene_id=0,
                start_time=0.0,
                end_time=30.0,
                title="Introduction",
                transcript_text="Welcome to the tutorial. We'll cover authentication and security basics.",
            ),
            _make_scene(
                scene_id=1,
                start_time=30.0,
                end_time=60.0,
                title="Authentication Flow",
                transcript_text="The authentication flow starts with the login form. Users enter credentials.",
            ),
            _make_scene(
                scene_id=2,
                start_time=60.0,
                end_time=90.0,
                title="Security Best Practices",
                transcript_text="Always hash passwords. Never store them in plain text.",
            ),
            _make_scene(
                scene_id=3,
                start_time=90.0,
                end_time=120.0,
                title="Conclusion",
                transcript_text="That concludes our tutorial on authentication and security.",
            ),
        ],
    )
    save_scenes_data(cache, scenes)

    return cache


class TestFindSceneAtTimestamp:
    """Tests for find_scene_at_timestamp."""

    def test_finds_scene_at_start(self, cache_dir: Path) -> None:
        """Should find scene when timestamp is at scene start."""
        result = find_scene_at_timestamp(cache_dir, 0.0)
        assert result == 0

    def test_finds_scene_in_middle(self, cache_dir: Path) -> None:
        """Should find scene when timestamp is in the middle."""
        result = find_scene_at_timestamp(cache_dir, 45.0)
        assert result == 1

    def test_finds_scene_at_boundary(self, cache_dir: Path) -> None:
        """Should find correct scene at boundary."""
        # 30.0 is the start of scene 1, not the end of scene 0
        result = find_scene_at_timestamp(cache_dir, 30.0)
        assert result == 1

    def test_finds_last_scene_at_end(self, cache_dir: Path) -> None:
        """Should find last scene for timestamps at or past the end."""
        result = find_scene_at_timestamp(cache_dir, 119.0)
        assert result == 3

        # Even past the end
        result = find_scene_at_timestamp(cache_dir, 200.0)
        assert result == 3

    def test_returns_none_for_no_scenes(self, tmp_path: Path) -> None:
        """Should return None when no scenes exist."""
        result = find_scene_at_timestamp(tmp_path, 10.0)
        assert result is None


class TestFindRelevantScenes:
    """Tests for find_relevant_scenes."""

    def test_finds_scenes_by_question(self, cache_dir: Path) -> None:
        """Should find scenes relevant to a question."""
        result = find_relevant_scenes(cache_dir, "How does authentication work?")
        assert 1 in result  # "Authentication Flow" scene
        assert 0 in result  # "Introduction" mentions authentication

    def test_finds_scenes_by_answer(self, cache_dir: Path) -> None:
        """Should use answer text to improve relevance."""
        result = find_relevant_scenes(
            cache_dir,
            "How should I store passwords?",
            "Always hash passwords and never store them in plain text.",
        )
        assert 2 in result  # "Security Best Practices" scene

    def test_returns_top_5(self, cache_dir: Path) -> None:
        """Should return at most 5 relevant scenes."""
        result = find_relevant_scenes(cache_dir, "tutorial authentication security")
        assert len(result) <= 5

    def test_handles_no_matches(self, cache_dir: Path) -> None:
        """Should return empty list for no matches."""
        result = find_relevant_scenes(cache_dir, "quantum computing blockchain")
        assert result == []

    def test_filters_stop_words(self, cache_dir: Path) -> None:
        """Should filter out common stop words."""
        # "what" and "this" are stop words
        result = find_relevant_scenes(cache_dir, "what is this about")
        # Should still find something based on "about"
        assert isinstance(result, list)


class TestRelevanceBoosts:
    """Tests for relevance boost functions."""

    def test_get_empty_boosts(self, cache_dir: Path) -> None:
        """Should return empty dict when no boosts exist."""
        result = get_relevance_boosts(cache_dir)
        assert result == {}

    def test_save_and_load_boosts(self, cache_dir: Path) -> None:
        """Should save and load boosts correctly."""
        boosts = {"0": 1.1, "1": 1.2, "2": 1.0}
        save_relevance_boosts(cache_dir, boosts)

        result = get_relevance_boosts(cache_dir)
        assert result == boosts

    def test_boost_scene_relevance(self, cache_dir: Path) -> None:
        """Should increment scene relevance boost."""
        new_boost = boost_scene_relevance(cache_dir, 0, boost=0.1)
        assert new_boost == 1.1

        # Boost again
        new_boost = boost_scene_relevance(cache_dir, 0, boost=0.1)
        assert new_boost == pytest.approx(1.2)

    def test_get_boosted_relevance(self, cache_dir: Path) -> None:
        """Should apply boost to relevance score."""
        # Set up a boost
        boost_scene_relevance(cache_dir, 0, boost=0.5)  # Now 1.5

        # Check boosted relevance
        result = get_boosted_relevance(cache_dir, 0, base_relevance=0.8)
        assert result == pytest.approx(1.2)  # 0.8 * 1.5

    def test_unboosted_scene_returns_base(self, cache_dir: Path) -> None:
        """Should return base relevance for unboosted scene."""
        result = get_boosted_relevance(cache_dir, 99, base_relevance=0.5)
        assert result == 0.5


class TestRecordFrameExamination:
    """Tests for record_frame_examination."""

    def test_records_examination(self, cache_dir: Path) -> None:
        """Should record frame examination with scene and boost."""
        result = record_frame_examination(
            "test_video", cache_dir, start_time=45.0, duration=5.0, quality="standard"
        )

        assert result is not None
        assert result["scene_id"] == 1
        assert result["new_boost"] == pytest.approx(1.1)

    def test_records_observation_in_memory(self, cache_dir: Path) -> None:
        """Should record observation in VideoMemory."""
        record_frame_examination(
            "test_video", cache_dir, start_time=45.0, duration=5.0, quality="hq"
        )

        # Check memory file
        memory_file = cache_dir / "memory" / "observations.json"
        assert memory_file.exists()

        observations = json.loads(memory_file.read_text())
        assert "1" in observations
        assert observations["1"][0]["type"] == "frames_examined"
        assert "hq" in observations["1"][0]["content"]

    def test_returns_none_for_no_scene(self, tmp_path: Path) -> None:
        """Should return None when no scene found."""
        result = record_frame_examination(
            "test", tmp_path, start_time=100.0, duration=5.0
        )
        assert result is None


class TestRecordQAInteraction:
    """Tests for record_qa_interaction."""

    def test_records_qa(self, cache_dir: Path) -> None:
        """Should record Q&A and return result."""
        result = record_qa_interaction(
            "test_video",
            cache_dir,
            question="How does authentication work?",
            answer="The authentication flow starts with a login form.",
        )

        assert result["cached"] is True
        assert result["qa_count"] == 1
        assert len(result["scenes"]) > 0

    def test_auto_finds_relevant_scenes(self, cache_dir: Path) -> None:
        """Should auto-find relevant scenes if not provided."""
        result = record_qa_interaction(
            "test_video",
            cache_dir,
            question="What about password hashing?",
            answer="Always hash passwords before storing.",
        )

        assert 2 in result["scenes"]  # Security scene mentions hashing

    def test_uses_provided_scenes(self, cache_dir: Path) -> None:
        """Should use provided scene IDs."""
        result = record_qa_interaction(
            "test_video",
            cache_dir,
            question="Random question",
            answer="Random answer",
            relevant_scene_ids=[0, 3],
        )

        assert result["scenes"] == [0, 3]

    def test_boosts_relevant_scenes(self, cache_dir: Path) -> None:
        """Should boost relevance for relevant scenes."""
        record_qa_interaction(
            "test_video",
            cache_dir,
            question="Test",
            answer="Test",
            relevant_scene_ids=[0, 1],
        )

        boosts = get_relevance_boosts(cache_dir)
        assert boosts.get("0", 1.0) > 1.0
        assert boosts.get("1", 1.0) > 1.0


class TestSearchCachedQA:
    """Tests for search_cached_qa."""

    def test_searches_qa_history(self, cache_dir: Path) -> None:
        """Should find matching Q&A from history."""
        # First record some Q&A
        record_qa_interaction(
            "test_video",
            cache_dir,
            question="How do I authenticate users?",
            answer="Use the login form with credentials.",
            relevant_scene_ids=[1],
        )
        record_qa_interaction(
            "test_video",
            cache_dir,
            question="How do I hash passwords?",
            answer="Use bcrypt or argon2.",
            relevant_scene_ids=[2],
        )

        # Search for related question
        results = search_cached_qa("test_video", cache_dir, "authenticate")
        assert len(results) == 1
        assert "authenticate" in results[0]["question"].lower()

    def test_returns_empty_for_no_match(self, cache_dir: Path) -> None:
        """Should return empty list for no matches."""
        results = search_cached_qa("test_video", cache_dir, "blockchain")
        assert results == []


class TestGetSceneContext:
    """Tests for get_scene_context."""

    def test_gets_empty_context(self, cache_dir: Path) -> None:
        """Should return empty context for new scene."""
        context = get_scene_context("test_video", cache_dir, 0)
        assert context["observations"] == []
        assert context["related_qa"] == []
        assert context["boost"] == 1.0

    def test_gets_full_context(self, cache_dir: Path) -> None:
        """Should return all context after interactions."""
        # Record frame examination
        record_frame_examination(
            "test_video", cache_dir, start_time=10.0, duration=5.0
        )

        # Record Q&A
        record_qa_interaction(
            "test_video",
            cache_dir,
            question="What is in the intro?",
            answer="Tutorial overview.",
            relevant_scene_ids=[0],
        )

        context = get_scene_context("test_video", cache_dir, 0)
        assert len(context["observations"]) == 1
        assert len(context["related_qa"]) == 1
        assert context["boost"] > 1.0


class TestGetEnrichmentStats:
    """Tests for get_enrichment_stats."""

    def test_empty_stats(self, cache_dir: Path) -> None:
        """Should return zero stats for new video."""
        stats = get_enrichment_stats(cache_dir)
        assert stats["observation_count"] == 0
        assert stats["qa_count"] == 0
        assert stats["boosted_scenes"] == 0
        assert stats["has_enrichment"] is False

    def test_stats_after_enrichment(self, cache_dir: Path) -> None:
        """Should track enrichment statistics."""
        # Add some interactions
        record_frame_examination("test_video", cache_dir, 10.0, 5.0)
        record_frame_examination("test_video", cache_dir, 45.0, 5.0)
        record_qa_interaction(
            "test_video",
            cache_dir,
            question="Q1",
            answer="A1",
            relevant_scene_ids=[0],
        )
        record_qa_interaction(
            "test_video",
            cache_dir,
            question="Q2",
            answer="A2",
            relevant_scene_ids=[1, 2],
        )

        stats = get_enrichment_stats(cache_dir)
        assert stats["observation_count"] == 2
        assert stats["qa_count"] == 2
        assert stats["boosted_scenes"] > 0
        assert stats["has_enrichment"] is True
