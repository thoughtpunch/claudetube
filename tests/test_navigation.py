"""Tests for playlist navigation and progress tracking."""

import json
import time

import pytest


class TestPlaylistProgress:
    """Test PlaylistProgress persistence and operations."""

    def test_load_nonexistent_returns_empty(self, tmp_path):
        """Loading progress for nonexistent playlist returns empty progress."""
        from claudetube.navigation.progress import PlaylistProgress

        progress = PlaylistProgress.load("nonexistent-playlist", tmp_path)

        assert progress.playlist_id == "nonexistent-playlist"
        assert progress.watched_videos == []
        assert progress.current_video is None
        assert progress.bookmarks == []

    def test_save_and_load_roundtrip(self, tmp_path):
        """Progress can be saved and loaded correctly."""
        from claudetube.navigation.progress import PlaylistProgress

        # Create and populate progress
        progress = PlaylistProgress(
            playlist_id="test-playlist",
            watched_videos=["video1", "video2"],
            watch_times={"video1": 1000.0, "video2": 2000.0},
            current_video="video2",
            current_timestamp=125.5,
        )

        # Save to temp directory
        path = progress.save(tmp_path)
        assert path.exists()

        # Load and verify
        loaded = PlaylistProgress.load("test-playlist", tmp_path)
        assert loaded.playlist_id == "test-playlist"
        assert loaded.watched_videos == ["video1", "video2"]
        assert loaded.current_video == "video2"
        assert loaded.current_timestamp == 125.5
        assert loaded.watch_times["video1"] == 1000.0

    def test_mark_watched(self, tmp_path):
        """mark_watched adds video to watched list and sets current."""
        from claudetube.navigation.progress import PlaylistProgress

        progress = PlaylistProgress(playlist_id="test")

        progress.mark_watched("video1")
        assert "video1" in progress.watched_videos
        assert progress.current_video == "video1"
        assert "video1" in progress.watch_times

        # Mark same video again - should not duplicate
        progress.mark_watched("video1")
        assert progress.watched_videos.count("video1") == 1

    def test_mark_watched_without_setting_current(self, tmp_path):
        """mark_watched can skip setting current video."""
        from claudetube.navigation.progress import PlaylistProgress

        progress = PlaylistProgress(playlist_id="test", current_video="existing")

        progress.mark_watched("video1", set_current=False)
        assert "video1" in progress.watched_videos
        assert progress.current_video == "existing"

    def test_is_watched(self, tmp_path):
        """is_watched returns correct boolean."""
        from claudetube.navigation.progress import PlaylistProgress

        progress = PlaylistProgress(
            playlist_id="test", watched_videos=["video1", "video2"]
        )

        assert progress.is_watched("video1") is True
        assert progress.is_watched("video2") is True
        assert progress.is_watched("video3") is False

    def test_add_bookmark(self, tmp_path):
        """Bookmarks can be added and retrieved."""
        from claudetube.navigation.progress import PlaylistProgress

        progress = PlaylistProgress(playlist_id="test")

        progress.add_bookmark("video1", 125.5, "Important concept")
        progress.add_bookmark("video1", 200.0, "Another note")
        progress.add_bookmark("video2", 50.0)

        assert len(progress.bookmarks) == 3
        video1_bookmarks = progress.get_bookmarks_for_video("video1")
        assert len(video1_bookmarks) == 2
        assert video1_bookmarks[0]["note"] == "Important concept"

    def test_completion_stats(self, tmp_path):
        """Completion stats are calculated correctly."""
        from claudetube.navigation.progress import PlaylistProgress

        progress = PlaylistProgress(
            playlist_id="test", watched_videos=["v1", "v2", "v3"]
        )

        stats = progress.get_completion_stats(total_videos=10)
        assert stats["completed"] == 3
        assert stats["total"] == 10
        assert stats["percentage"] == 30.0
        assert stats["remaining"] == 7

    def test_completion_stats_empty_playlist(self, tmp_path):
        """Completion stats handle empty playlist."""
        from claudetube.navigation.progress import PlaylistProgress

        progress = PlaylistProgress(playlist_id="test")
        stats = progress.get_completion_stats(total_videos=0)

        assert stats["percentage"] == 0

    def test_update_position(self, tmp_path):
        """Position can be updated."""
        from claudetube.navigation.progress import PlaylistProgress

        progress = PlaylistProgress(playlist_id="test")

        progress.update_position("video1", 125.5)
        assert progress.current_video == "video1"
        assert progress.current_timestamp == 125.5

    def test_atomic_save(self, tmp_path):
        """Save uses atomic write (temp file + rename)."""
        from claudetube.navigation.progress import PlaylistProgress

        # Create playlist directory
        playlist_dir = tmp_path / "playlists" / "test"
        playlist_dir.mkdir(parents=True)

        progress = PlaylistProgress(playlist_id="test")
        progress.save(tmp_path)

        # Verify only progress.json exists (no temp files left)
        files = list(playlist_dir.iterdir())
        assert len(files) == 1
        assert files[0].name == "progress.json"


class TestPlaylistContext:
    """Test PlaylistContext navigation operations."""

    @pytest.fixture
    def mock_playlist(self, tmp_path):
        """Create a mock playlist in cache."""
        playlist_id = "test-playlist"
        playlist_dir = tmp_path / "playlists" / playlist_id
        playlist_dir.mkdir(parents=True)

        # Create playlist metadata
        playlist_data = {
            "playlist_id": playlist_id,
            "title": "Test Course",
            "inferred_type": "course",
            "video_count": 5,
            "videos": [
                {
                    "video_id": "v1",
                    "title": "Lesson 1",
                    "duration": 600,
                    "position": 0,
                    "url": "https://youtube.com/watch?v=v1",
                },
                {
                    "video_id": "v2",
                    "title": "Lesson 2",
                    "duration": 720,
                    "position": 1,
                    "url": "https://youtube.com/watch?v=v2",
                },
                {
                    "video_id": "v3",
                    "title": "Lesson 3",
                    "duration": 540,
                    "position": 2,
                    "url": "https://youtube.com/watch?v=v3",
                },
                {
                    "video_id": "v4",
                    "title": "Lesson 4",
                    "duration": 900,
                    "position": 3,
                    "url": "https://youtube.com/watch?v=v4",
                },
                {
                    "video_id": "v5",
                    "title": "Lesson 5",
                    "duration": 480,
                    "position": 4,
                    "url": "https://youtube.com/watch?v=v5",
                },
            ],
        }
        (playlist_dir / "playlist.json").write_text(json.dumps(playlist_data))

        return playlist_id, tmp_path

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading nonexistent playlist returns None."""
        from claudetube.navigation.context import PlaylistContext

        context = PlaylistContext.load("nonexistent", tmp_path)
        assert context is None

    def test_load_existing_playlist(self, mock_playlist):
        """Loading existing playlist returns context with metadata."""
        from claudetube.navigation.context import PlaylistContext

        playlist_id, cache_base = mock_playlist
        context = PlaylistContext.load(playlist_id, cache_base)

        assert context is not None
        assert context.playlist_id == playlist_id
        assert context.title == "Test Course"
        assert context.playlist_type == "course"
        assert context.total_videos == 5
        assert len(context.videos) == 5

    def test_get_video_at_position(self, mock_playlist):
        """Can get video by position."""
        from claudetube.navigation.context import PlaylistContext

        playlist_id, cache_base = mock_playlist
        context = PlaylistContext.load(playlist_id, cache_base)

        video = context.get_video_at_position(0)
        assert video["video_id"] == "v1"

        video = context.get_video_at_position(4)
        assert video["video_id"] == "v5"

        video = context.get_video_at_position(10)
        assert video is None

    def test_get_video_by_id(self, mock_playlist):
        """Can get video by ID."""
        from claudetube.navigation.context import PlaylistContext

        playlist_id, cache_base = mock_playlist
        context = PlaylistContext.load(playlist_id, cache_base)

        video = context.get_video_by_id("v3")
        assert video["title"] == "Lesson 3"

        video = context.get_video_by_id("nonexistent")
        assert video is None

    def test_next_video_from_start(self, mock_playlist):
        """Next video from start returns first video."""
        from claudetube.navigation.context import PlaylistContext

        playlist_id, cache_base = mock_playlist
        context = PlaylistContext.load(playlist_id, cache_base)

        # No current video - should return first
        next_video = context.next_video
        assert next_video["video_id"] == "v1"

    def test_next_video_from_middle(self, mock_playlist):
        """Next video from middle position works correctly."""
        from claudetube.navigation.context import PlaylistContext

        playlist_id, cache_base = mock_playlist
        context = PlaylistContext.load(playlist_id, cache_base)

        # Set current to v2 (position 1)
        context.mark_watched("v2", save=False)
        assert context.current_position == 1

        next_video = context.next_video
        assert next_video["video_id"] == "v3"

    def test_next_video_at_end(self, mock_playlist):
        """Next video at end returns None."""
        from claudetube.navigation.context import PlaylistContext

        playlist_id, cache_base = mock_playlist
        context = PlaylistContext.load(playlist_id, cache_base)

        # Set current to last video
        context.mark_watched("v5", save=False)

        next_video = context.next_video
        assert next_video is None

    def test_previous_video(self, mock_playlist):
        """Previous video navigation works correctly."""
        from claudetube.navigation.context import PlaylistContext

        playlist_id, cache_base = mock_playlist
        context = PlaylistContext.load(playlist_id, cache_base)

        # No current - no previous
        assert context.previous_video is None

        # At first video - no previous
        context.mark_watched("v1", save=False)
        assert context.previous_video is None

        # At middle - has previous
        context.mark_watched("v3", save=False)
        assert context.previous_video["video_id"] == "v2"

    def test_progress_summary(self, mock_playlist):
        """Progress summary includes all relevant info."""
        from claudetube.navigation.context import PlaylistContext

        playlist_id, cache_base = mock_playlist
        context = PlaylistContext.load(playlist_id, cache_base)

        context.mark_watched("v1", save=False)
        context.mark_watched("v2", save=False)

        summary = context.get_progress_summary()
        assert summary["completed"] == 2
        assert summary["total"] == 5
        assert summary["percentage"] == 40.0
        assert summary["current_position"] == 1  # 0-indexed
        assert summary["current_video"] == "v2"
        assert "Video 2 of 5" in summary["display"]

    def test_mark_watched_updates_position(self, mock_playlist):
        """Marking video watched updates current position."""
        from claudetube.navigation.context import PlaylistContext

        playlist_id, cache_base = mock_playlist
        context = PlaylistContext.load(playlist_id, cache_base)

        context.mark_watched("v3", save=False)

        assert context.current_position == 2
        assert context.progress.current_video == "v3"
        assert context.progress.is_watched("v3")

    def test_get_unwatched_videos(self, mock_playlist):
        """Unwatched videos list is correct."""
        from claudetube.navigation.context import PlaylistContext

        playlist_id, cache_base = mock_playlist
        context = PlaylistContext.load(playlist_id, cache_base)

        context.mark_watched("v1", save=False)
        context.mark_watched("v3", save=False)

        unwatched = context.get_unwatched_videos()
        unwatched_ids = [v["video_id"] for v in unwatched]

        assert "v1" not in unwatched_ids
        assert "v2" in unwatched_ids
        assert "v3" not in unwatched_ids
        assert "v4" in unwatched_ids
        assert "v5" in unwatched_ids

    def test_get_watched_videos(self, mock_playlist):
        """Watched videos list returns in watch order."""
        from claudetube.navigation.context import PlaylistContext

        playlist_id, cache_base = mock_playlist
        context = PlaylistContext.load(playlist_id, cache_base)

        # Watch in non-sequential order
        context.mark_watched("v3", save=False)
        time.sleep(0.01)  # Ensure different timestamps
        context.mark_watched("v1", save=False)
        time.sleep(0.01)
        context.mark_watched("v5", save=False)

        watched = context.get_watched_videos()
        watched_ids = [v["video_id"] for v in watched]

        # Should be in watch order: v3, v1, v5
        assert watched_ids == ["v3", "v1", "v5"]


class TestPlaylistContextDetection:
    """Test automatic playlist context detection for videos."""

    def test_playlist_context_structure(self):
        """Verify playlist context has correct structure for navigation."""
        # This tests the data structure that process_video_tool returns
        # when a video is part of a playlist

        # Mock playlist context as returned by _get_playlist_context_for_video
        context = {
            "playlist_id": "test-playlist",
            "playlist_title": "Python Tutorial Series",
            "playlist_type": "series",
            "position": 2,
            "total_videos": 4,
            "display": 'Video 2 of 4 in "Python Tutorial Series"',
            "previous_video": {"video_id": "vid_intro", "title": "Introduction"},
            "next_video": {"video_id": "vid_functions", "title": "Functions"},
            "hint": "Use watch_next or watch_video_in_playlist to continue the series.",
        }

        # Verify structure provides Claude with navigation info
        assert context["position"] == 2
        assert context["total_videos"] == 4
        assert "previous_video" in context
        assert "next_video" in context
        assert "hint" in context
        assert "watch_next" in context["hint"]

    def test_first_video_no_previous(self):
        """First video in playlist has no previous_video."""
        context = {
            "playlist_id": "test-playlist",
            "position": 1,
            "total_videos": 4,
            "next_video": {"video_id": "vid_basics", "title": "Basics"},
        }

        assert "previous_video" not in context
        assert "next_video" in context

    def test_last_video_no_next(self):
        """Last video in playlist has no next_video."""
        context = {
            "playlist_id": "test-playlist",
            "position": 4,
            "total_videos": 4,
            "previous_video": {"video_id": "vid_functions", "title": "Functions"},
        }

        assert "previous_video" in context
        assert "next_video" not in context


class TestNavigationIntegration:
    """Integration tests for navigation workflow."""

    @pytest.fixture
    def setup_playlist_cache(self, tmp_path, monkeypatch):
        """Set up a complete playlist cache environment."""
        # Mock get_cache_dir to use tmp_path
        monkeypatch.setattr(
            "claudetube.navigation.progress.get_cache_dir", lambda: tmp_path
        )
        monkeypatch.setattr(
            "claudetube.navigation.context.get_cache_dir", lambda: tmp_path
        )
        monkeypatch.setattr(
            "claudetube.operations.playlist.get_cache_dir", lambda: tmp_path
        )

        # Create playlist
        playlist_id = "integration-test"
        playlist_dir = tmp_path / "playlists" / playlist_id
        playlist_dir.mkdir(parents=True)

        playlist_data = {
            "playlist_id": playlist_id,
            "title": "Integration Test Course",
            "inferred_type": "series",
            "video_count": 3,
            "videos": [
                {
                    "video_id": "vid1",
                    "title": "Part 1",
                    "duration": 300,
                    "position": 0,
                    "url": "https://example.com/vid1",
                },
                {
                    "video_id": "vid2",
                    "title": "Part 2",
                    "duration": 400,
                    "position": 1,
                    "url": "https://example.com/vid2",
                },
                {
                    "video_id": "vid3",
                    "title": "Part 3",
                    "duration": 500,
                    "position": 2,
                    "url": "https://example.com/vid3",
                },
            ],
        }
        (playlist_dir / "playlist.json").write_text(json.dumps(playlist_data))

        return playlist_id, tmp_path

    def test_full_navigation_workflow(self, setup_playlist_cache):
        """Test complete navigation workflow from start to finish."""
        from claudetube.navigation.context import PlaylistContext

        playlist_id, cache_base = setup_playlist_cache

        # 1. Load context
        context = PlaylistContext.load(playlist_id)
        assert context is not None
        assert context.total_videos == 3

        # 2. Check initial state - next should be first video
        assert context.next_video["video_id"] == "vid1"
        assert context.current_position is None

        # 3. Watch first video
        context.mark_watched("vid1")
        assert context.current_position == 0
        assert context.next_video["video_id"] == "vid2"

        # 4. Watch second video
        context.mark_watched("vid2")
        assert context.current_position == 1
        assert context.next_video["video_id"] == "vid3"

        # 5. Check progress
        summary = context.get_progress_summary()
        assert summary["completed"] == 2
        assert summary["percentage"] == pytest.approx(66.7, 0.1)

        # 6. Add a bookmark
        context.progress.add_bookmark("vid2", 120.0, "Key concept")
        context.progress.save()

        # 7. Reload and verify persistence
        context2 = PlaylistContext.load(playlist_id)
        assert context2.progress.is_watched("vid1")
        assert context2.progress.is_watched("vid2")
        assert not context2.progress.is_watched("vid3")
        assert len(context2.progress.bookmarks) == 1

        # 8. Watch final video
        context2.mark_watched("vid3")
        assert context2.next_video is None  # End of playlist

        # 9. Final progress
        final_summary = context2.get_progress_summary()
        assert final_summary["completed"] == 3
        assert final_summary["percentage"] == 100.0
