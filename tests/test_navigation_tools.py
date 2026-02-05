"""Tests for Phase 4 navigation MCP tools."""

from __future__ import annotations

import json

import pytest


class TestPlaylistProgressData:
    """Tests for progress data management."""

    @pytest.fixture
    def playlist_with_progress(self, tmp_path):
        """Create a playlist with some progress."""
        from claudetube.navigation.progress import PlaylistProgress

        playlist_dir = tmp_path / "playlists" / "test123"
        playlist_dir.mkdir(parents=True)

        # Playlist metadata
        (playlist_dir / "playlist.json").write_text(
            json.dumps(
                {
                    "playlist_id": "test123",
                    "title": "Test Course",
                    "inferred_type": "course",
                    "videos": [
                        {"video_id": "v1", "title": "Introduction", "position": 0},
                        {"video_id": "v2", "title": "Basics", "position": 1},
                        {"video_id": "v3", "title": "Advanced", "position": 2},
                        {"video_id": "v4", "title": "Authentication", "position": 3},
                        {"video_id": "v5", "title": "Conclusion", "position": 4},
                    ],
                }
            )
        )

        # Progress - watched first two videos
        progress = PlaylistProgress(
            playlist_id="test123",
            watched_videos=["v1", "v2"],
            watch_times={"v1": 1000, "v2": 2000},
            current_video="v2",
            current_timestamp=120.5,
        )
        progress.save(tmp_path)

        return tmp_path


class TestPlaylistContext:
    """Tests for PlaylistContext navigation helpers."""

    @pytest.fixture
    def context_fixture(self, tmp_path):
        """Create a playlist context for testing."""
        playlist_dir = tmp_path / "playlists" / "nav123"
        playlist_dir.mkdir(parents=True)

        (playlist_dir / "playlist.json").write_text(
            json.dumps(
                {
                    "playlist_id": "nav123",
                    "title": "Navigation Test",
                    "inferred_type": "series",
                    "videos": [
                        {"video_id": "v1", "title": "Part 1", "position": 0},
                        {"video_id": "v2", "title": "Part 2", "position": 1},
                        {"video_id": "v3", "title": "Part 3", "position": 2},
                    ],
                }
            )
        )

        (playlist_dir / "progress.json").write_text(
            json.dumps(
                {
                    "playlist_id": "nav123",
                    "watched_videos": ["v1"],
                    "watch_times": {"v1": 1000},
                    "current_video": "v1",
                    "current_timestamp": None,
                    "bookmarks": [],
                    "last_accessed": 1000,
                }
            )
        )

        return tmp_path

    def test_load_context(self, context_fixture):
        """Test loading playlist context."""
        from claudetube.navigation.context import PlaylistContext

        context = PlaylistContext.load("nav123", context_fixture)

        assert context is not None
        assert context.playlist_id == "nav123"
        assert context.total_videos == 3
        assert context.current_position == 0  # v1 is at position 0

    def test_next_video(self, context_fixture):
        """Test getting next video."""
        from claudetube.navigation.context import PlaylistContext

        context = PlaylistContext.load("nav123", context_fixture)

        next_video = context.next_video
        assert next_video is not None
        assert next_video["video_id"] == "v2"
        assert next_video["title"] == "Part 2"

    def test_previous_video(self, context_fixture):
        """Test getting previous video."""
        from claudetube.navigation.context import PlaylistContext

        context = PlaylistContext.load("nav123", context_fixture)

        # At v1, no previous video
        prev_video = context.previous_video
        assert prev_video is None

        # Update to v2
        context.progress.current_video = "v2"
        context.current_position = 1

        prev_video = context.previous_video
        assert prev_video is not None
        assert prev_video["video_id"] == "v1"

    def test_get_video_at_position(self, context_fixture):
        """Test getting video by position."""
        from claudetube.navigation.context import PlaylistContext

        context = PlaylistContext.load("nav123", context_fixture)

        video = context.get_video_at_position(1)
        assert video is not None
        assert video["video_id"] == "v2"

        # Out of range
        video = context.get_video_at_position(10)
        assert video is None

    def test_get_video_by_id(self, context_fixture):
        """Test getting video by ID."""
        from claudetube.navigation.context import PlaylistContext

        context = PlaylistContext.load("nav123", context_fixture)

        video = context.get_video_by_id("v3")
        assert video is not None
        assert video["title"] == "Part 3"

        video = context.get_video_by_id("nonexistent")
        assert video is None


class TestProgressSummary:
    """Tests for progress summary generation."""

    @pytest.fixture
    def progress_context(self, tmp_path):
        """Create context with various progress states."""
        playlist_dir = tmp_path / "playlists" / "prog123"
        playlist_dir.mkdir(parents=True)

        (playlist_dir / "playlist.json").write_text(
            json.dumps(
                {
                    "playlist_id": "prog123",
                    "title": "Progress Test",
                    "inferred_type": "course",
                    "videos": [
                        {"video_id": f"v{i}", "title": f"Video {i}", "position": i - 1}
                        for i in range(1, 11)  # 10 videos
                    ],
                }
            )
        )

        # 4 of 10 watched
        (playlist_dir / "progress.json").write_text(
            json.dumps(
                {
                    "playlist_id": "prog123",
                    "watched_videos": ["v1", "v2", "v3", "v4"],
                    "watch_times": {f"v{i}": i * 1000 for i in range(1, 5)},
                    "current_video": "v4",
                    "current_timestamp": 300.0,
                    "bookmarks": [],
                    "last_accessed": 5000,
                }
            )
        )

        return tmp_path

    def test_progress_summary(self, progress_context):
        """Test progress summary calculation."""
        from claudetube.navigation.context import PlaylistContext

        context = PlaylistContext.load("prog123", progress_context)
        summary = context.get_progress_summary()

        assert summary["completed"] == 4
        assert summary["total"] == 10
        assert summary["percentage"] == 40.0
        assert summary["current_position"] == 3  # 0-indexed position of v4

    def test_completion_stats(self, progress_context):
        """Test completion statistics."""
        from claudetube.navigation.context import PlaylistContext

        context = PlaylistContext.load("prog123", progress_context)
        stats = context.progress.get_completion_stats(10)

        assert stats["completed"] == 4
        assert stats["total"] == 10
        assert stats["percentage"] == 40.0
        assert stats["remaining"] == 6


class TestBookmarks:
    """Tests for bookmark functionality."""

    @pytest.fixture
    def bookmark_context(self, tmp_path):
        """Create context for bookmark testing."""
        playlist_dir = tmp_path / "playlists" / "bm123"
        playlist_dir.mkdir(parents=True)

        (playlist_dir / "playlist.json").write_text(
            json.dumps(
                {
                    "playlist_id": "bm123",
                    "title": "Bookmark Test",
                    "videos": [
                        {"video_id": "v1", "title": "Video 1", "position": 0},
                        {"video_id": "v2", "title": "Video 2", "position": 1},
                    ],
                }
            )
        )

        (playlist_dir / "progress.json").write_text(
            json.dumps(
                {
                    "playlist_id": "bm123",
                    "watched_videos": ["v1"],
                    "watch_times": {"v1": 1000},
                    "current_video": "v1",
                    "current_timestamp": 60.0,
                    "bookmarks": [
                        {
                            "video_id": "v1",
                            "timestamp": 30.0,
                            "note": "Important concept",
                            "created_at": 1000,
                        }
                    ],
                    "last_accessed": 1000,
                }
            )
        )

        return tmp_path

    def test_add_bookmark(self, bookmark_context):
        """Test adding a bookmark."""
        from claudetube.navigation.context import PlaylistContext

        context = PlaylistContext.load("bm123", bookmark_context)

        context.progress.add_bookmark("v1", 120.0, "Another bookmark")
        context.progress.save(bookmark_context)

        # Reload and verify
        context = PlaylistContext.load("bm123", bookmark_context)
        assert len(context.progress.bookmarks) == 2

    def test_get_bookmarks_for_video(self, bookmark_context):
        """Test filtering bookmarks by video."""
        from claudetube.navigation.context import PlaylistContext

        context = PlaylistContext.load("bm123", bookmark_context)

        # Add bookmark to different video
        context.progress.add_bookmark("v2", 45.0, "V2 bookmark")

        v1_bookmarks = context.progress.get_bookmarks_for_video("v1")
        assert len(v1_bookmarks) == 1
        assert v1_bookmarks[0]["note"] == "Important concept"

        v2_bookmarks = context.progress.get_bookmarks_for_video("v2")
        assert len(v2_bookmarks) == 1
        assert v2_bookmarks[0]["note"] == "V2 bookmark"


class TestMarkWatched:
    """Tests for marking videos as watched."""

    @pytest.fixture
    def watch_context(self, tmp_path):
        """Create context for watch testing."""
        playlist_dir = tmp_path / "playlists" / "watch123"
        playlist_dir.mkdir(parents=True)

        (playlist_dir / "playlist.json").write_text(
            json.dumps(
                {
                    "playlist_id": "watch123",
                    "title": "Watch Test",
                    "videos": [
                        {"video_id": "v1", "title": "Video 1", "position": 0},
                        {"video_id": "v2", "title": "Video 2", "position": 1},
                        {"video_id": "v3", "title": "Video 3", "position": 2},
                    ],
                }
            )
        )

        (playlist_dir / "progress.json").write_text(
            json.dumps(
                {
                    "playlist_id": "watch123",
                    "watched_videos": [],
                    "watch_times": {},
                    "current_video": None,
                    "current_timestamp": None,
                    "bookmarks": [],
                    "last_accessed": 1000,
                }
            )
        )

        return tmp_path

    def test_mark_watched_updates_progress(self, watch_context):
        """Test that marking watched updates progress correctly."""
        from claudetube.navigation.context import PlaylistContext

        context = PlaylistContext.load("watch123", watch_context)

        assert len(context.progress.watched_videos) == 0
        assert context.progress.current_video is None

        context.mark_watched("v1", save=True, cache_base=watch_context)

        assert "v1" in context.progress.watched_videos
        assert context.progress.current_video == "v1"
        assert context.current_position == 0

    def test_mark_watched_idempotent(self, watch_context):
        """Test that marking same video twice doesn't duplicate."""
        from claudetube.navigation.context import PlaylistContext

        context = PlaylistContext.load("watch123", watch_context)

        context.mark_watched("v1", save=False)
        context.mark_watched("v1", save=False)

        assert context.progress.watched_videos.count("v1") == 1

    def test_get_unwatched_videos(self, watch_context):
        """Test getting unwatched videos."""
        from claudetube.navigation.context import PlaylistContext

        context = PlaylistContext.load("watch123", watch_context)

        unwatched = context.get_unwatched_videos()
        assert len(unwatched) == 3

        context.mark_watched("v1", save=False)
        context.mark_watched("v2", save=False)

        unwatched = context.get_unwatched_videos()
        assert len(unwatched) == 1
        assert unwatched[0]["video_id"] == "v3"
