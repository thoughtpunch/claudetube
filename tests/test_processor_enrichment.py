"""Tests for progressive enrichment integration in processor.py."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from claudetube.db import get_database, reset_database
from claudetube.db.repos.videos import VideoRepository
from claudetube.models.state import VideoState
from claudetube.models.video_path import VideoPath
from claudetube.operations.processor import _try_progressive_enrichment


@pytest.fixture(autouse=True)
def reset_db():
    """Reset database singleton before each test."""
    reset_database()
    yield
    reset_database()


@pytest.fixture
def in_memory_db():
    """Get an in-memory database for testing."""
    return get_database(":memory:")


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestTryProgressiveEnrichment:
    """Tests for _try_progressive_enrichment() helper."""

    def test_returns_original_dir_when_path_unchanged(
        self, in_memory_db, temp_cache_dir: Path
    ):
        """Returns original cache_dir if path doesn't improve."""
        # Setup: create initial path and directory
        video_path = VideoPath(
            domain="youtube",
            channel="UCexisting",
            playlist="PLexisting",
            video_id="vid123",
        )
        cache_dir = video_path.cache_dir(temp_cache_dir)
        cache_dir.mkdir(parents=True)
        (cache_dir / "state.json").write_text("{}")

        # Create video in database
        repo = VideoRepository(in_memory_db)
        repo.insert(
            video_id="vid123",
            domain="youtube",
            cache_path=str(video_path.relative_path()),
            channel="UCexisting",
            playlist="PLexisting",
        )

        state = VideoState(
            video_id="vid123",
            url="https://youtube.com/watch?v=vid123",
            domain="youtube",
        )

        # Metadata with same channel/playlist (no improvement)
        meta = {
            "channel_id": "UCexisting",
            "playlist_id": "PLexisting",
        }

        result = _try_progressive_enrichment(
            video_id="vid123",
            video_path=video_path,
            meta=meta,
            cache_dir=cache_dir,
            cache_base=temp_cache_dir,
            state=state,
        )

        # Should return original cache_dir
        assert result == cache_dir
        assert cache_dir.exists()

    def test_moves_directory_when_channel_improves(
        self, in_memory_db, temp_cache_dir: Path
    ):
        """Moves directory when metadata provides channel_id we didn't have."""
        # Setup: create initial path with no_channel
        video_path = VideoPath(
            domain="youtube",
            channel=None,  # no channel
            playlist=None,
            video_id="vid456",
        )
        cache_dir = video_path.cache_dir(temp_cache_dir)
        cache_dir.mkdir(parents=True)
        (cache_dir / "state.json").write_text("{}")

        # Create video in database with no_channel path
        repo = VideoRepository(in_memory_db)
        repo.insert(
            video_id="vid456",
            domain="youtube",
            cache_path=str(video_path.relative_path()),
        )

        state = VideoState(
            video_id="vid456",
            url="https://youtube.com/watch?v=vid456",
            domain="youtube",
        )

        # Metadata provides channel info
        meta = {
            "channel_id": "UCnewchannel",
        }

        result = _try_progressive_enrichment(
            video_id="vid456",
            video_path=video_path,
            meta=meta,
            cache_dir=cache_dir,
            cache_base=temp_cache_dir,
            state=state,
        )

        # Should return new cache_dir
        expected_new_dir = (
            temp_cache_dir / "youtube" / "UCnewchannel" / "no_playlist" / "vid456"
        )
        assert result == expected_new_dir
        assert expected_new_dir.exists()
        assert not cache_dir.exists()

        # Database should be updated
        record = repo.get_by_video_id("vid456")
        assert record["cache_path"] == "youtube/UCnewchannel/no_playlist/vid456"
        assert record["channel"] == "UCnewchannel"

    def test_moves_directory_when_playlist_improves(
        self, in_memory_db, temp_cache_dir: Path
    ):
        """Moves directory when metadata provides playlist_id we didn't have."""
        # Setup: path with channel but no playlist
        video_path = VideoPath(
            domain="youtube",
            channel="UCchannel",
            playlist=None,  # no playlist
            video_id="vid789",
        )
        cache_dir = video_path.cache_dir(temp_cache_dir)
        cache_dir.mkdir(parents=True)
        (cache_dir / "state.json").write_text("{}")

        repo = VideoRepository(in_memory_db)
        repo.insert(
            video_id="vid789",
            domain="youtube",
            cache_path=str(video_path.relative_path()),
            channel="UCchannel",
        )

        state = VideoState(
            video_id="vid789",
            url="https://youtube.com/watch?v=vid789",
            domain="youtube",
        )

        # Metadata provides playlist info
        meta = {
            "channel_id": "UCchannel",
            "playlist_id": "PLnewplaylist",
        }

        result = _try_progressive_enrichment(
            video_id="vid789",
            video_path=video_path,
            meta=meta,
            cache_dir=cache_dir,
            cache_base=temp_cache_dir,
            state=state,
        )

        expected_new_dir = (
            temp_cache_dir / "youtube" / "UCchannel" / "PLnewplaylist" / "vid789"
        )
        assert result == expected_new_dir
        assert expected_new_dir.exists()
        assert not cache_dir.exists()

    def test_moves_directory_when_both_improve(
        self, in_memory_db, temp_cache_dir: Path
    ):
        """Moves directory when both channel and playlist improve at once."""
        # Setup: path with neither channel nor playlist
        video_path = VideoPath(
            domain="youtube",
            channel=None,
            playlist=None,
            video_id="vidABC",
        )
        cache_dir = video_path.cache_dir(temp_cache_dir)
        cache_dir.mkdir(parents=True)
        (cache_dir / "state.json").write_text("{}")

        repo = VideoRepository(in_memory_db)
        repo.insert(
            video_id="vidABC",
            domain="youtube",
            cache_path=str(video_path.relative_path()),
        )

        state = VideoState(
            video_id="vidABC",
            url="https://youtube.com/watch?v=vidABC",
            domain="youtube",
        )

        # Metadata provides both
        meta = {
            "channel_id": "UCfullchannel",
            "playlist_id": "PLfullplaylist",
        }

        result = _try_progressive_enrichment(
            video_id="vidABC",
            video_path=video_path,
            meta=meta,
            cache_dir=cache_dir,
            cache_base=temp_cache_dir,
            state=state,
        )

        expected = (
            temp_cache_dir / "youtube" / "UCfullchannel" / "PLfullplaylist" / "vidABC"
        )
        assert result == expected
        assert expected.exists()
        assert not cache_dir.exists()

        record = repo.get_by_video_id("vidABC")
        assert record["channel"] == "UCfullchannel"
        assert record["playlist"] == "PLfullplaylist"

    def test_preserves_explicit_playlist_from_url(
        self, in_memory_db, temp_cache_dir: Path
    ):
        """If video_path has playlist from URL, preserves it over metadata."""
        # Setup: explicit playlist from URL
        video_path = VideoPath(
            domain="youtube",
            channel=None,
            playlist="PLfromurl",  # explicit from URL
            video_id="vidXYZ",
        )
        cache_dir = video_path.cache_dir(temp_cache_dir)
        cache_dir.mkdir(parents=True)
        (cache_dir / "state.json").write_text("{}")

        repo = VideoRepository(in_memory_db)
        repo.insert(
            video_id="vidXYZ",
            domain="youtube",
            cache_path=str(video_path.relative_path()),
            playlist="PLfromurl",
        )

        state = VideoState(
            video_id="vidXYZ",
            url="https://youtube.com/watch?v=vidXYZ&list=PLfromurl",
            domain="youtube",
        )

        # Metadata provides channel, but no playlist
        meta = {
            "channel_id": "UCchannel",
        }

        result = _try_progressive_enrichment(
            video_id="vidXYZ",
            video_path=video_path,
            meta=meta,
            cache_dir=cache_dir,
            cache_base=temp_cache_dir,
            state=state,
        )

        # Should have channel from metadata, playlist from URL
        expected = temp_cache_dir / "youtube" / "UCchannel" / "PLfromurl" / "vidXYZ"
        assert result == expected
        assert expected.exists()

    def test_cleans_up_empty_parent_dirs(self, in_memory_db, temp_cache_dir: Path):
        """Empty no_channel/no_playlist dirs are cleaned up after move."""
        video_path = VideoPath(
            domain="youtube",
            channel=None,
            playlist=None,
            video_id="vidClean",
        )
        cache_dir = video_path.cache_dir(temp_cache_dir)
        cache_dir.mkdir(parents=True)
        (cache_dir / "state.json").write_text("{}")

        repo = VideoRepository(in_memory_db)
        repo.insert(
            video_id="vidClean",
            domain="youtube",
            cache_path=str(video_path.relative_path()),
        )

        state = VideoState(
            video_id="vidClean",
            url="https://youtube.com/watch?v=vidClean",
            domain="youtube",
        )

        meta = {"channel_id": "UCchannel"}

        _try_progressive_enrichment(
            video_id="vidClean",
            video_path=video_path,
            meta=meta,
            cache_dir=cache_dir,
            cache_base=temp_cache_dir,
            state=state,
        )

        # Old empty directories should be gone
        assert not (temp_cache_dir / "youtube" / "no_channel" / "no_playlist").exists()
        assert not (temp_cache_dir / "youtube" / "no_channel").exists()

    def test_returns_original_on_move_failure(self, in_memory_db, temp_cache_dir: Path):
        """Returns original cache_dir if move fails."""
        video_path = VideoPath(
            domain="youtube",
            channel=None,
            playlist=None,
            video_id="vidFail",
        )
        cache_dir = video_path.cache_dir(temp_cache_dir)
        cache_dir.mkdir(parents=True)
        (cache_dir / "state.json").write_text("{}")

        repo = VideoRepository(in_memory_db)
        repo.insert(
            video_id="vidFail",
            domain="youtube",
            cache_path=str(video_path.relative_path()),
        )

        state = VideoState(
            video_id="vidFail",
            url="https://youtube.com/watch?v=vidFail",
            domain="youtube",
        )

        meta = {"channel_id": "UCnew"}

        # Mock shutil.move to fail
        with patch("shutil.move", side_effect=OSError("Permission denied")):
            result = _try_progressive_enrichment(
                video_id="vidFail",
                video_path=video_path,
                meta=meta,
                cache_dir=cache_dir,
                cache_base=temp_cache_dir,
                state=state,
            )

        # Should return original
        assert result == cache_dir
        assert cache_dir.exists()

    def test_fire_and_forget_on_any_error(self, in_memory_db, temp_cache_dir: Path):
        """Returns original cache_dir on any exception (fire-and-forget)."""
        video_path = VideoPath(
            domain="youtube",
            channel=None,
            playlist=None,
            video_id="vidErr",
        )
        cache_dir = video_path.cache_dir(temp_cache_dir)
        cache_dir.mkdir(parents=True)
        (cache_dir / "state.json").write_text("{}")

        state = VideoState(
            video_id="vidErr",
            url="https://youtube.com/watch?v=vidErr",
            domain="youtube",
        )

        meta = {"channel_id": "UCnew"}

        # Mock VideoPath.from_url to raise
        with patch(
            "claudetube.operations.processor.VideoPath.from_url",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = _try_progressive_enrichment(
                video_id="vidErr",
                video_path=video_path,
                meta=meta,
                cache_dir=cache_dir,
                cache_base=temp_cache_dir,
                state=state,
            )

        # Should return original without raising
        assert result == cache_dir

    def test_updates_state_after_successful_move(
        self, in_memory_db, temp_cache_dir: Path
    ):
        """Updates VideoState fields after successful directory move."""
        video_path = VideoPath(
            domain="youtube",
            channel=None,
            playlist=None,
            video_id="vidState",
        )
        cache_dir = video_path.cache_dir(temp_cache_dir)
        cache_dir.mkdir(parents=True)
        (cache_dir / "state.json").write_text("{}")

        repo = VideoRepository(in_memory_db)
        repo.insert(
            video_id="vidState",
            domain="youtube",
            cache_path=str(video_path.relative_path()),
        )

        state = VideoState(
            video_id="vidState",
            url="https://youtube.com/watch?v=vidState",
            domain="youtube",
        )

        meta = {"channel_id": "UCnewchan", "playlist_id": "PLnewlist"}

        result = _try_progressive_enrichment(
            video_id="vidState",
            video_path=video_path,
            meta=meta,
            cache_dir=cache_dir,
            cache_base=temp_cache_dir,
            state=state,
        )

        # State should be updated
        assert state.domain == "youtube"
        assert state.channel_id == "UCnewchan"
        assert state.playlist_id == "PLnewlist"

        # State should be saved to new location
        new_state_path = result / "state.json"
        assert new_state_path.exists()


class TestProcessVideoPlaylistIdFromUrl:
    """Tests for playlist_id extraction from URL in cache hit path."""

    def test_cache_hit_updates_playlist_id_from_url(self, temp_cache_dir: Path):
        """Cache hit should update state.playlist_id if URL has playlist but state doesn't."""
        from claudetube.cache.storage import load_state, save_state
        from claudetube.models.state import VideoState
        from claudetube.operations.processor import process_video
        from unittest.mock import patch, MagicMock

        # Setup: create cached video WITHOUT playlist_id
        video_id = "testvidXYZ"
        cache_dir = temp_cache_dir / "youtube" / "no_channel" / "no_playlist" / video_id
        cache_dir.mkdir(parents=True)

        state = VideoState(
            video_id=video_id,
            url="https://youtube.com/watch?v=testvidXYZ",
            domain="youtube",
            transcript_complete=True,
            playlist_id=None,  # No playlist_id initially
        )
        save_state(state, cache_dir / "state.json")

        # Create transcript files so cache hit returns
        (cache_dir / "audio.srt").write_text("test srt")
        (cache_dir / "audio.txt").write_text("test transcript")

        # Mock CacheManager to return our temp cache dir
        mock_cache = MagicMock()
        mock_cache.get_cache_dir.return_value = cache_dir

        with patch(
            "claudetube.operations.processor.CacheManager", return_value=mock_cache
        ), patch("claudetube.operations.processor.get_cache_dir", return_value=temp_cache_dir):
            # Process video with playlist URL
            result = process_video(
                "https://youtube.com/watch?v=testvidXYZ&list=PLtestplaylist123",
                output_base=temp_cache_dir,
            )

        assert result.success is True
        assert result.video_id == video_id

        # State should now have playlist_id
        updated_state = load_state(cache_dir / "state.json")
        assert updated_state.playlist_id == "PLtestplaylist123"

        # Metadata returned should include playlist_id
        assert result.metadata.get("playlist_id") == "PLtestplaylist123"

    def test_cache_hit_does_not_overwrite_existing_playlist_id(
        self, temp_cache_dir: Path
    ):
        """Cache hit should NOT overwrite existing playlist_id with URL playlist."""
        from claudetube.cache.storage import load_state, save_state
        from claudetube.models.state import VideoState
        from claudetube.operations.processor import process_video
        from unittest.mock import patch, MagicMock

        # Setup: create cached video WITH existing playlist_id
        video_id = "testvidABC"
        cache_dir = (
            temp_cache_dir / "youtube" / "no_channel" / "PLoriginal" / video_id
        )
        cache_dir.mkdir(parents=True)

        state = VideoState(
            video_id=video_id,
            url="https://youtube.com/watch?v=testvidABC",
            domain="youtube",
            transcript_complete=True,
            playlist_id="PLoriginal",  # Already has playlist_id
        )
        save_state(state, cache_dir / "state.json")

        # Create transcript files
        (cache_dir / "audio.srt").write_text("test srt")
        (cache_dir / "audio.txt").write_text("test transcript")

        mock_cache = MagicMock()
        mock_cache.get_cache_dir.return_value = cache_dir

        with patch(
            "claudetube.operations.processor.CacheManager", return_value=mock_cache
        ), patch("claudetube.operations.processor.get_cache_dir", return_value=temp_cache_dir):
            # Process video with DIFFERENT playlist URL
            result = process_video(
                "https://youtube.com/watch?v=testvidABC&list=PLdifferent",
                output_base=temp_cache_dir,
            )

        assert result.success is True

        # State should KEEP original playlist_id
        updated_state = load_state(cache_dir / "state.json")
        assert updated_state.playlist_id == "PLoriginal"

    def test_cache_hit_without_playlist_url_does_not_change_state(
        self, temp_cache_dir: Path
    ):
        """Cache hit without playlist in URL should not change state."""
        from claudetube.cache.storage import load_state, save_state
        from claudetube.models.state import VideoState
        from claudetube.operations.processor import process_video
        from unittest.mock import patch, MagicMock

        video_id = "testvidNoChange"
        cache_dir = (
            temp_cache_dir / "youtube" / "no_channel" / "no_playlist" / video_id
        )
        cache_dir.mkdir(parents=True)

        state = VideoState(
            video_id=video_id,
            url="https://youtube.com/watch?v=testvidNoChange",
            domain="youtube",
            transcript_complete=True,
            playlist_id=None,
        )
        save_state(state, cache_dir / "state.json")

        (cache_dir / "audio.srt").write_text("test srt")
        (cache_dir / "audio.txt").write_text("test transcript")

        mock_cache = MagicMock()
        mock_cache.get_cache_dir.return_value = cache_dir

        with patch(
            "claudetube.operations.processor.CacheManager", return_value=mock_cache
        ), patch("claudetube.operations.processor.get_cache_dir", return_value=temp_cache_dir):
            # Process video WITHOUT playlist in URL
            result = process_video(
                "https://youtube.com/watch?v=testvidNoChange",
                output_base=temp_cache_dir,
            )

        assert result.success is True

        # State should still have no playlist_id
        updated_state = load_state(cache_dir / "state.json")
        assert updated_state.playlist_id is None
