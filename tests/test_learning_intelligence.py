"""Tests for learning intelligence functionality."""

from __future__ import annotations

import json

import pytest

from claudetube.navigation.learning import (
    PrerequisiteWarning,
    Recommendation,
    TopicCoverage,
    VideoContext,
    analyze_topic_coverage,
    check_prerequisites,
    get_learning_recommendations,
    get_video_context,
)


class TestDataclasses:
    """Tests for dataclass serialization."""

    def test_prerequisite_warning_to_dict(self):
        """Test PrerequisiteWarning serialization."""
        warning = PrerequisiteWarning(
            video_id="vid3",
            video_title="Video 3",
            missing_prerequisites=[
                {"video_id": "vid1", "title": "Video 1", "position": 1},
                {"video_id": "vid2", "title": "Video 2", "position": 2},
            ],
            total_prerequisites=2,
            watched_prerequisites=0,
        )

        d = warning.to_dict()
        assert d["video_id"] == "vid3"
        assert d["missing_count"] == 2
        assert len(d["missing_prerequisites"]) == 2
        assert d["total_prerequisites"] == 2

    def test_recommendation_to_dict(self):
        """Test Recommendation serialization."""
        rec = Recommendation(
            video_id="vid2",
            video_title="Functions",
            video_position=2,
            reason_type="sequential",
            reason="Next video in sequence",
            priority=1,
        )

        d = rec.to_dict()
        assert d["video_id"] == "vid2"
        assert d["reason_type"] == "sequential"
        assert d["priority"] == 1

    def test_topic_coverage_to_dict(self):
        """Test TopicCoverage serialization."""
        coverage = TopicCoverage(
            topic="functions",
            videos_covering=[
                {"video_id": "v1", "title": "Intro", "position": 1, "mentions": 5},
                {"video_id": "v2", "title": "Advanced", "position": 2, "mentions": 10},
            ],
            total_mentions=15,
            chapters_matching=[
                {
                    "video_id": "v1",
                    "chapter_title": "Basic Functions",
                    "start_time": 120.0,
                }
            ],
        )

        d = coverage.to_dict()
        assert d["topic"] == "functions"
        assert d["video_count"] == 2
        assert d["total_mentions"] == 15
        assert d["chapter_count"] == 1

    def test_video_context_to_dict(self):
        """Test VideoContext serialization."""
        ctx = VideoContext(
            current_video_id="vid3",
            previous_videos_summary=[
                {
                    "video_id": "vid1",
                    "title": "Intro",
                    "position": 1,
                    "topics": ["python"],
                },
            ],
            relevant_prior_content=[
                {"video_id": "vid1", "shared_topics": ["functions"]},
            ],
            prerequisites_covered=["vid1", "vid2"],
        )

        d = ctx.to_dict()
        assert d["current_video_id"] == "vid3"
        assert len(d["previous_videos_summary"]) == 1
        assert len(d["prerequisites_covered"]) == 2


class TestPrerequisiteChecking:
    """Tests for prerequisite checking functionality."""

    @pytest.fixture
    def course_playlist(self, tmp_path):
        """Create a course playlist with prerequisites."""
        # Create playlist directory
        playlist_dir = tmp_path / "playlists" / "course123"
        playlist_dir.mkdir(parents=True)

        # Create playlist metadata
        (playlist_dir / "playlist.json").write_text(
            json.dumps(
                {
                    "playlist_id": "course123",
                    "title": "Python Course",
                    "inferred_type": "course",
                    "videos": [
                        {"video_id": "v1", "title": "Introduction", "position": 0},
                        {"video_id": "v2", "title": "Variables", "position": 1},
                        {"video_id": "v3", "title": "Functions", "position": 2},
                    ],
                }
            )
        )

        # Create knowledge graph with prerequisites
        (playlist_dir / "knowledge_graph.json").write_text(
            json.dumps(
                {
                    "playlist": {"playlist_id": "course123", "inferred_type": "course"},
                    "common_topics": [],
                    "shared_entities": [],
                    "videos": [
                        {
                            "video_id": "v1",
                            "title": "Introduction",
                            "prerequisites": [],
                        },
                        {
                            "video_id": "v2",
                            "title": "Variables",
                            "prerequisites": ["v1"],
                        },
                        {
                            "video_id": "v3",
                            "title": "Functions",
                            "prerequisites": ["v1", "v2"],
                        },
                    ],
                }
            )
        )

        # Create progress (no videos watched yet)
        (playlist_dir / "progress.json").write_text(
            json.dumps(
                {
                    "playlist_id": "course123",
                    "watched_videos": [],
                    "watch_times": {},
                    "current_video": None,
                }
            )
        )

        return tmp_path

    def test_check_prerequisites_none_watched(self, course_playlist):
        """Test prerequisite check when no videos are watched."""
        warning = check_prerequisites("v3", "course123", cache_base=course_playlist)

        assert warning is not None
        assert warning.video_id == "v3"
        assert len(warning.missing_prerequisites) == 2
        assert warning.watched_prerequisites == 0

    def test_check_prerequisites_some_watched(self, course_playlist):
        """Test prerequisite check when some prerequisites are watched."""
        # Update progress to mark v1 as watched
        progress_file = course_playlist / "playlists" / "course123" / "progress.json"
        progress_file.write_text(
            json.dumps(
                {
                    "playlist_id": "course123",
                    "watched_videos": ["v1"],
                    "watch_times": {"v1": 1234567890},
                    "current_video": "v1",
                }
            )
        )

        warning = check_prerequisites("v3", "course123", cache_base=course_playlist)

        assert warning is not None
        assert len(warning.missing_prerequisites) == 1
        assert warning.missing_prerequisites[0]["video_id"] == "v2"
        assert warning.watched_prerequisites == 1

    def test_check_prerequisites_all_watched(self, course_playlist):
        """Test prerequisite check when all prerequisites are watched."""
        # Update progress to mark all as watched
        progress_file = course_playlist / "playlists" / "course123" / "progress.json"
        progress_file.write_text(
            json.dumps(
                {
                    "playlist_id": "course123",
                    "watched_videos": ["v1", "v2"],
                    "watch_times": {"v1": 1234567890, "v2": 1234567891},
                    "current_video": "v2",
                }
            )
        )

        warning = check_prerequisites("v3", "course123", cache_base=course_playlist)

        assert warning is None

    def test_check_prerequisites_first_video(self, course_playlist):
        """Test that first video has no prerequisites."""
        warning = check_prerequisites("v1", "course123", cache_base=course_playlist)
        assert warning is None


class TestLearningRecommendations:
    """Tests for learning recommendations functionality."""

    @pytest.fixture
    def playlist_with_progress(self, tmp_path):
        """Create a playlist with some progress."""
        playlist_dir = tmp_path / "playlists" / "series123"
        playlist_dir.mkdir(parents=True)

        # Playlist metadata
        (playlist_dir / "playlist.json").write_text(
            json.dumps(
                {
                    "playlist_id": "series123",
                    "title": "Tutorial Series",
                    "inferred_type": "series",
                    "videos": [
                        {"video_id": "v1", "title": "Part 1: Basics", "position": 0},
                        {
                            "video_id": "v2",
                            "title": "Part 2: Intermediate",
                            "position": 1,
                        },
                        {"video_id": "v3", "title": "Part 3: Advanced", "position": 2},
                    ],
                }
            )
        )

        # Knowledge graph
        (playlist_dir / "knowledge_graph.json").write_text(
            json.dumps(
                {
                    "playlist": {"playlist_id": "series123", "inferred_type": "series"},
                    "common_topics": [{"keyword": "python"}],
                    "shared_entities": [{"text": "function", "type": "concept"}],
                    "videos": [
                        {"video_id": "v1", "title": "Part 1", "prerequisites": []},
                        {"video_id": "v2", "title": "Part 2", "prerequisites": ["v1"]},
                        {
                            "video_id": "v3",
                            "title": "Part 3",
                            "prerequisites": ["v1", "v2"],
                        },
                    ],
                }
            )
        )

        # Progress - watched first video
        (playlist_dir / "progress.json").write_text(
            json.dumps(
                {
                    "playlist_id": "series123",
                    "watched_videos": ["v1"],
                    "watch_times": {"v1": 1234567890},
                    "current_video": "v1",
                }
            )
        )

        return tmp_path

    def test_get_recommendations_sequential(self, playlist_with_progress):
        """Test that sequential recommendation is provided."""
        recs = get_learning_recommendations(
            "series123", cache_base=playlist_with_progress
        )

        assert len(recs) > 0
        # Should have a sequential recommendation for v2
        seq_recs = [r for r in recs if r.reason_type == "sequential"]
        assert len(seq_recs) > 0
        assert seq_recs[0].video_id == "v2"

    def test_get_recommendations_sorted_by_priority(self, playlist_with_progress):
        """Test that recommendations are sorted by priority."""
        recs = get_learning_recommendations(
            "series123", cache_base=playlist_with_progress
        )

        for i in range(len(recs) - 1):
            assert recs[i].priority <= recs[i + 1].priority

    def test_get_recommendations_no_duplicates(self, playlist_with_progress):
        """Test that recommendations don't have duplicates."""
        recs = get_learning_recommendations(
            "series123", cache_base=playlist_with_progress
        )

        video_ids = [r.video_id for r in recs]
        assert len(video_ids) == len(set(video_ids))


class TestTopicCoverage:
    """Tests for topic coverage analysis."""

    @pytest.fixture
    def playlist_with_content(self, tmp_path):
        """Create a playlist with searchable content."""
        playlist_dir = tmp_path / "playlists" / "course"
        playlist_dir.mkdir(parents=True)

        # Playlist
        (playlist_dir / "playlist.json").write_text(
            json.dumps(
                {
                    "playlist_id": "course",
                    "title": "Python Course",
                    "videos": [
                        {
                            "video_id": "v1",
                            "title": "Intro to Functions",
                            "position": 0,
                        },
                        {
                            "video_id": "v2",
                            "title": "Advanced Functions",
                            "position": 1,
                        },
                    ],
                }
            )
        )

        # Chapter index
        (playlist_dir / "chapter_index.json").write_text(
            json.dumps(
                {
                    "playlist_id": "course",
                    "chapters": [
                        {
                            "video_id": "v1",
                            "video_title": "Intro to Functions",
                            "video_position": 0,
                            "chapter_title": "What are Functions",
                            "chapter_index": 0,
                            "start_time": 0.0,
                            "end_time": 60.0,
                            "timestamp_str": "0:00",
                        },
                        {
                            "video_id": "v2",
                            "video_title": "Advanced Functions",
                            "video_position": 1,
                            "chapter_title": "Lambda Functions",
                            "chapter_index": 0,
                            "start_time": 0.0,
                            "end_time": 90.0,
                            "timestamp_str": "0:00",
                        },
                    ],
                }
            )
        )

        # Create video scene data
        v1_dir = tmp_path / "v1" / "scenes"
        v1_dir.mkdir(parents=True)
        (v1_dir / "scenes.json").write_text(
            json.dumps(
                {
                    "video_id": "v1",
                    "method": "transcript",
                    "scenes": [
                        {
                            "scene_id": 0,
                            "start_time": 0.0,
                            "end_time": 60.0,
                            "title": None,
                            "transcript_text": "Functions are reusable blocks of code.",
                        }
                    ],
                }
            )
        )

        return tmp_path

    def test_analyze_topic_coverage(self, playlist_with_content):
        """Test topic coverage analysis."""
        coverage = analyze_topic_coverage(
            "course", "functions", cache_base=playlist_with_content
        )

        assert coverage.topic == "functions"
        # Should find chapters matching "functions"
        assert len(coverage.chapters_matching) > 0


class TestVideoContext:
    """Tests for video context bridging."""

    @pytest.fixture
    def playlist_with_context(self, tmp_path):
        """Create a playlist with watched videos for context."""
        playlist_dir = tmp_path / "playlists" / "course"
        playlist_dir.mkdir(parents=True)

        # Playlist
        (playlist_dir / "playlist.json").write_text(
            json.dumps(
                {
                    "playlist_id": "course",
                    "title": "Course",
                    "inferred_type": "course",
                    "videos": [
                        {"video_id": "v1", "title": "Intro", "position": 0},
                        {"video_id": "v2", "title": "Basics", "position": 1},
                        {"video_id": "v3", "title": "Advanced", "position": 2},
                    ],
                }
            )
        )

        # Progress - watched first two
        (playlist_dir / "progress.json").write_text(
            json.dumps(
                {
                    "playlist_id": "course",
                    "watched_videos": ["v1", "v2"],
                    "watch_times": {"v1": 1234567890, "v2": 1234567891},
                    "current_video": "v2",
                }
            )
        )

        # Knowledge graph
        (playlist_dir / "knowledge_graph.json").write_text(
            json.dumps(
                {
                    "playlist": {"playlist_id": "course"},
                    "common_topics": [],
                    "shared_entities": [{"text": "python"}],
                    "videos": [
                        {"video_id": "v1", "prerequisites": []},
                        {"video_id": "v2", "prerequisites": ["v1"]},
                        {"video_id": "v3", "prerequisites": ["v1", "v2"]},
                    ],
                }
            )
        )

        return tmp_path

    def test_get_video_context_previous_summaries(self, playlist_with_context):
        """Test that previous video summaries are included."""
        ctx = get_video_context("v3", "course", cache_base=playlist_with_context)

        assert ctx.current_video_id == "v3"
        assert len(ctx.previous_videos_summary) == 2
        video_ids = [s["video_id"] for s in ctx.previous_videos_summary]
        assert "v1" in video_ids
        assert "v2" in video_ids

    def test_get_video_context_prerequisites_covered(self, playlist_with_context):
        """Test that covered prerequisites are listed."""
        ctx = get_video_context("v3", "course", cache_base=playlist_with_context)

        # v3 requires v1 and v2, both are watched
        assert len(ctx.prerequisites_covered) == 2
        assert "v1" in ctx.prerequisites_covered
        assert "v2" in ctx.prerequisites_covered
