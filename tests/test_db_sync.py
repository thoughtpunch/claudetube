"""Tests for the db/sync.py dual-write and progressive enrichment module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from claudetube.db import get_database, reset_database
from claudetube.db.repos.audio_tracks import AudioTrackRepository
from claudetube.db.repos.transcriptions import TranscriptionRepository
from claudetube.db.repos.videos import VideoRepository
from claudetube.db.sync import (
    _cleanup_empty_parents,
    _embed_async_fire_and_forget,
    _extract_channel_from_metadata,
    _extract_playlist_from_metadata,
    _get_db,
    embed_observation,
    embed_qa,
    embed_scene_transcript,
    embed_technical_content,
    embed_transcription,
    embed_visual_description,
    enrich_video,
    get_video_uuid,
    sync_audio_track,
    sync_transcription,
    sync_video,
)
from claudetube.models.state import VideoState


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


class TestGetDb:
    """Tests for _get_db() helper."""

    def test_get_db_returns_database(self, in_memory_db):
        """_get_db() returns a Database instance when available."""
        db = _get_db()
        assert db is not None

    def test_get_db_returns_none_on_error(self):
        """_get_db() returns None when database unavailable."""
        # Patch the module that get_database is imported from
        with patch("claudetube.db.get_database", side_effect=RuntimeError("DB error")):
            result = _get_db()
            assert result is None


class TestCleanupEmptyParents:
    """Tests for _cleanup_empty_parents() helper."""

    def test_removes_empty_directories(self, temp_cache_dir: Path):
        """Empty parent directories are removed up to cache_base."""
        # Create nested empty directories
        nested = temp_cache_dir / "youtube" / "no_channel" / "no_playlist"
        nested.mkdir(parents=True)

        # Start cleanup from the leaf directory (no_playlist)
        # This simulates what happens after moving a video out
        _cleanup_empty_parents(nested, temp_cache_dir)

        # All empty dirs should be removed
        assert not nested.exists()
        assert not (temp_cache_dir / "youtube" / "no_channel").exists()
        assert not (temp_cache_dir / "youtube").exists()
        # Base should still exist
        assert temp_cache_dir.exists()

    def test_stops_at_non_empty_directory(self, temp_cache_dir: Path):
        """Stops cleaning when a non-empty directory is found."""
        # Create structure: youtube/no_channel/no_playlist/
        nested = temp_cache_dir / "youtube" / "no_channel" / "no_playlist"
        nested.mkdir(parents=True)

        # Put another video in no_channel so it's not empty
        other_video = (
            temp_cache_dir / "youtube" / "no_channel" / "other_playlist" / "vid456"
        )
        other_video.mkdir(parents=True)

        # Clean up starting from no_playlist
        _cleanup_empty_parents(nested, temp_cache_dir)

        # no_playlist should be removed, but no_channel should remain (has other_playlist)
        assert not nested.exists()
        assert (temp_cache_dir / "youtube" / "no_channel").exists()

    def test_does_not_fail_on_nonexistent_dir(self, temp_cache_dir: Path):
        """Does not raise when directory doesn't exist."""
        nonexistent = temp_cache_dir / "does" / "not" / "exist"
        # Should not raise
        _cleanup_empty_parents(nonexistent, temp_cache_dir)


class TestSyncVideo:
    """Tests for sync_video() function."""

    def test_sync_video_creates_record(self, in_memory_db):
        """sync_video() creates a new video record from VideoState."""
        state = VideoState(
            video_id="abc123",
            url="https://youtube.com/watch?v=abc123",
            domain="youtube",
            channel_id="UCtest",
            title="Test Video",
            duration=120.5,
            source_type="url",
        )

        sync_video(state, "youtube/UCtest/no_playlist/abc123")

        # Verify record was created
        repo = VideoRepository(in_memory_db)
        record = repo.get_by_video_id("abc123")
        assert record is not None
        assert record["video_id"] == "abc123"
        assert record["domain"] == "youtube"
        assert record["channel"] == "UCtest"
        assert record["title"] == "Test Video"
        assert record["duration"] == 120.5
        assert record["cache_path"] == "youtube/UCtest/no_playlist/abc123"

    def test_sync_video_upserts_existing(self, in_memory_db):
        """sync_video() updates existing record, filling NULL fields."""
        # First sync with partial data
        state1 = VideoState(
            video_id="abc123",
            domain="youtube",
        )
        sync_video(state1, "youtube/no_channel/no_playlist/abc123")

        # Second sync with more data
        state2 = VideoState(
            video_id="abc123",
            domain="youtube",
            title="New Title",
            duration=60.0,
        )
        sync_video(state2, "youtube/no_channel/no_playlist/abc123")

        # Verify title and duration were added
        repo = VideoRepository(in_memory_db)
        record = repo.get_by_video_id("abc123")
        assert record["title"] == "New Title"
        assert record["duration"] == 60.0

    def test_sync_video_extracts_domain_from_path(self, in_memory_db):
        """sync_video() extracts domain from cache_path when not in state."""
        state = VideoState(video_id="xyz789")
        # domain is None in state

        sync_video(state, "vimeo/channel1/no_playlist/xyz789")

        repo = VideoRepository(in_memory_db)
        record = repo.get_by_video_id("xyz789")
        assert record["domain"] == "vimeo"

    def test_sync_video_fire_and_forget(self, in_memory_db):
        """sync_video() does not raise on database errors."""
        state = VideoState(video_id="test123", domain="youtube")

        # Even with invalid data that might cause issues, should not raise
        with patch.object(
            VideoRepository, "upsert", side_effect=RuntimeError("DB error")
        ):
            # Should not raise
            sync_video(state, "youtube/no_channel/no_playlist/test123")


class TestEnrichVideo:
    """Tests for enrich_video() progressive enrichment."""

    def test_enrich_video_moves_directory_on_channel_update(
        self, in_memory_db, temp_cache_dir: Path
    ):
        """enrich_video() moves directory when channel changes from NULL to real."""
        # Setup: create video with no_channel
        repo = VideoRepository(in_memory_db)
        repo.insert(
            video_id="vid123",
            domain="youtube",
            cache_path="youtube/no_channel/no_playlist/vid123",
        )

        # Create the actual directory
        old_dir = temp_cache_dir / "youtube" / "no_channel" / "no_playlist" / "vid123"
        old_dir.mkdir(parents=True)
        (old_dir / "state.json").write_text("{}")  # Some content

        # Enrich with channel info
        metadata = {"channel_id": "UCnewchannel", "title": "Updated Title"}
        enrich_video("vid123", metadata, temp_cache_dir)

        # Check directory was moved
        new_dir = temp_cache_dir / "youtube" / "UCnewchannel" / "no_playlist" / "vid123"
        assert new_dir.exists()
        assert (new_dir / "state.json").exists()
        assert not old_dir.exists()

        # Check database updated
        record = repo.get_by_video_id("vid123")
        assert record["cache_path"] == "youtube/UCnewchannel/no_playlist/vid123"
        assert record["channel"] == "UCnewchannel"

    def test_enrich_video_moves_directory_on_playlist_update(
        self, in_memory_db, temp_cache_dir: Path
    ):
        """enrich_video() moves directory when playlist changes from NULL to real."""
        repo = VideoRepository(in_memory_db)
        repo.insert(
            video_id="vid456",
            domain="youtube",
            cache_path="youtube/UCchannel/no_playlist/vid456",
            channel="UCchannel",
        )

        # Create directory
        old_dir = temp_cache_dir / "youtube" / "UCchannel" / "no_playlist" / "vid456"
        old_dir.mkdir(parents=True)
        (old_dir / "audio.mp3").write_text("audio")

        # Enrich with playlist info
        metadata = {"playlist_id": "PLmyplaylist"}
        enrich_video("vid456", metadata, temp_cache_dir)

        # Check move
        new_dir = temp_cache_dir / "youtube" / "UCchannel" / "PLmyplaylist" / "vid456"
        assert new_dir.exists()
        assert not old_dir.exists()

        record = repo.get_by_video_id("vid456")
        assert record["cache_path"] == "youtube/UCchannel/PLmyplaylist/vid456"
        assert record["playlist"] == "PLmyplaylist"

    def test_enrich_video_does_not_move_if_path_unchanged(
        self, in_memory_db, temp_cache_dir: Path
    ):
        """enrich_video() does not move if path hasn't improved."""
        repo = VideoRepository(in_memory_db)
        repo.insert(
            video_id="vid789",
            domain="youtube",
            cache_path="youtube/UCexisting/PLexisting/vid789",
            channel="UCexisting",
            playlist="PLexisting",
        )

        # Create directory
        existing_dir = (
            temp_cache_dir / "youtube" / "UCexisting" / "PLexisting" / "vid789"
        )
        existing_dir.mkdir(parents=True)

        # Enrich with same info (no improvement)
        metadata = {
            "channel_id": "UCexisting",
            "playlist_id": "PLexisting",
            "title": "New Title",
        }
        enrich_video("vid789", metadata, temp_cache_dir)

        # Directory should still be in same place
        assert existing_dir.exists()

        # But metadata should be updated
        record = repo.get_by_video_id("vid789")
        assert record["title"] == "New Title"

    def test_enrich_video_cleans_up_empty_parents(
        self, in_memory_db, temp_cache_dir: Path
    ):
        """enrich_video() cleans up empty no_channel/no_playlist dirs after move."""
        repo = VideoRepository(in_memory_db)
        repo.insert(
            video_id="vid111",
            domain="youtube",
            cache_path="youtube/no_channel/no_playlist/vid111",
        )

        # Create structure with some content so the video dir is valid
        old_dir = temp_cache_dir / "youtube" / "no_channel" / "no_playlist" / "vid111"
        old_dir.mkdir(parents=True)
        (old_dir / "state.json").write_text("{}")  # Add file so it's a real video dir

        # Enrich
        enrich_video("vid111", {"channel_id": "UCnew"}, temp_cache_dir)

        # Old empty directories should be cleaned up
        # After move, the old vid111 dir is gone, and its parent no_playlist should be empty
        assert not (temp_cache_dir / "youtube" / "no_channel" / "no_playlist").exists()
        assert not (temp_cache_dir / "youtube" / "no_channel").exists()

    def test_enrich_video_keeps_old_path_on_move_failure(
        self, in_memory_db, temp_cache_dir: Path
    ):
        """If shutil.move() fails, database retains old cache_path."""
        repo = VideoRepository(in_memory_db)
        repo.insert(
            video_id="vid222",
            domain="youtube",
            cache_path="youtube/no_channel/no_playlist/vid222",
        )

        # Create directory but make new location unmovable
        old_dir = temp_cache_dir / "youtube" / "no_channel" / "no_playlist" / "vid222"
        old_dir.mkdir(parents=True)

        with patch("shutil.move", side_effect=OSError("Permission denied")):
            enrich_video("vid222", {"channel_id": "UCnew"}, temp_cache_dir)

        # Database should still have old path
        record = repo.get_by_video_id("vid222")
        assert record["cache_path"] == "youtube/no_channel/no_playlist/vid222"

    def test_enrich_video_nonexistent_video(self, in_memory_db, temp_cache_dir: Path):
        """enrich_video() handles non-existent video gracefully."""
        # Should not raise
        enrich_video("nonexistent", {"channel_id": "UCtest"}, temp_cache_dir)

    def test_enrich_video_fire_and_forget(self, in_memory_db, temp_cache_dir: Path):
        """enrich_video() does not raise on errors."""
        with patch("claudetube.db.sync._get_db", side_effect=RuntimeError("DB error")):
            # Should not raise
            enrich_video("vid333", {"channel_id": "UCtest"}, temp_cache_dir)


class TestExtractHelpers:
    """Tests for metadata extraction helper functions."""

    def test_extract_channel_from_channel_id(self):
        """Extracts channel_id as first priority."""
        meta = {
            "channel_id": "UCtest123",
            "uploader_id": "uploader",
            "channel": "Display Name",
        }
        assert _extract_channel_from_metadata(meta) == "UCtest123"

    def test_extract_channel_from_uploader_id(self):
        """Falls back to uploader_id."""
        meta = {"uploader_id": "uploader456", "channel": "Display Name"}
        assert _extract_channel_from_metadata(meta) == "uploader456"

    def test_extract_channel_from_channel_name(self):
        """Falls back to sanitized channel name."""
        meta = {"channel": "My Channel Name!"}
        result = _extract_channel_from_metadata(meta)
        assert result == "My_Channel_Name_"

    def test_extract_channel_returns_none(self):
        """Returns None when no channel info available."""
        assert _extract_channel_from_metadata({}) is None

    def test_extract_playlist_from_playlist_id(self):
        """Extracts playlist_id as first priority."""
        meta = {"playlist_id": "PLtest123", "playlist_title": "My Playlist"}
        assert _extract_playlist_from_metadata(meta) == "PLtest123"

    def test_extract_playlist_from_title(self):
        """Falls back to sanitized playlist title."""
        meta = {"playlist_title": "My Playlist!"}
        result = _extract_playlist_from_metadata(meta)
        assert result == "My_Playlist_"

    def test_extract_playlist_returns_none(self):
        """Returns None when no playlist info available."""
        assert _extract_playlist_from_metadata({}) is None


class TestSyncAudioTrack:
    """Tests for sync_audio_track() function."""

    def test_sync_audio_track_creates_record(self, in_memory_db):
        """sync_audio_track() creates a new audio track record."""
        # First create a video
        video_repo = VideoRepository(in_memory_db)
        video_uuid = video_repo.insert(
            "vid123", "youtube", "youtube/no_channel/no_playlist/vid123"
        )

        track_id = sync_audio_track(
            video_uuid=video_uuid,
            format_="mp3",
            file_path="audio.mp3",
            sample_rate=44100,
            channels=2,
            bitrate_kbps=128,
            duration=120.5,
        )

        assert track_id is not None

        # Verify record
        audio_repo = AudioTrackRepository(in_memory_db)
        record = audio_repo.get_by_uuid(track_id)
        assert record is not None
        assert record["format"] == "mp3"
        assert record["sample_rate"] == 44100

    def test_sync_audio_track_returns_existing_id(self, in_memory_db):
        """sync_audio_track() returns existing ID if track already exists."""
        video_repo = VideoRepository(in_memory_db)
        video_uuid = video_repo.insert("vid123", "youtube", "cache/vid123")

        # Create first
        track_id1 = sync_audio_track(video_uuid, "mp3", "audio.mp3")

        # Try to create again - should return same ID
        track_id2 = sync_audio_track(video_uuid, "mp3", "audio.mp3")

        assert track_id1 == track_id2

    def test_sync_audio_track_fire_and_forget(self, in_memory_db):
        """sync_audio_track() returns None on errors."""
        # Invalid video_uuid should trigger FK constraint
        result = sync_audio_track(
            video_uuid="nonexistent-uuid-0000-0000-000000000000",
            format_="mp3",
            file_path="audio.mp3",
        )
        # Should return None, not raise
        assert result is None


class TestSyncTranscription:
    """Tests for sync_transcription() function."""

    def test_sync_transcription_creates_record(self, in_memory_db):
        """sync_transcription() creates a new transcription record."""
        video_repo = VideoRepository(in_memory_db)
        video_uuid = video_repo.insert("vid123", "youtube", "cache/vid123")

        transcript_id = sync_transcription(
            video_uuid=video_uuid,
            provider="whisper",
            format_="srt",
            file_path="audio.srt",
            model="small",
            language="en",
            full_text="Hello, this is a test transcript.",
            word_count=6,
            is_primary=True,
        )

        assert transcript_id is not None

        # Verify record
        trans_repo = TranscriptionRepository(in_memory_db)
        record = trans_repo.get_by_uuid(transcript_id)
        assert record is not None
        assert record["provider"] == "whisper"
        assert record["model"] == "small"
        assert record["full_text"] == "Hello, this is a test transcript."
        assert record["is_primary"] == 1

    def test_sync_transcription_sets_primary(self, in_memory_db):
        """sync_transcription() with is_primary demotes existing primary."""
        video_repo = VideoRepository(in_memory_db)
        video_uuid = video_repo.insert("vid123", "youtube", "cache/vid123")

        # Create first primary
        id1 = sync_transcription(
            video_uuid=video_uuid,
            provider="youtube_subtitles",
            format_="srt",
            file_path="yt.srt",
            is_primary=True,
        )

        # Create second primary
        id2 = sync_transcription(
            video_uuid=video_uuid,
            provider="whisper",
            format_="srt",
            file_path="whisper.srt",
            is_primary=True,
        )

        trans_repo = TranscriptionRepository(in_memory_db)

        # First should no longer be primary
        record1 = trans_repo.get_by_uuid(id1)
        assert record1["is_primary"] == 0

        # Second should be primary
        record2 = trans_repo.get_by_uuid(id2)
        assert record2["is_primary"] == 1

    def test_sync_transcription_fire_and_forget(self, in_memory_db):
        """sync_transcription() returns None on errors."""
        result = sync_transcription(
            video_uuid="nonexistent-uuid-0000-0000-000000000000",
            provider="whisper",
            format_="srt",
            file_path="audio.srt",
        )
        assert result is None


class TestGetVideoUuid:
    """Tests for get_video_uuid() helper."""

    def test_get_video_uuid_returns_uuid(self, in_memory_db):
        """get_video_uuid() returns UUID for existing video."""
        video_repo = VideoRepository(in_memory_db)
        expected_uuid = video_repo.insert("vid123", "youtube", "cache/vid123")

        result = get_video_uuid("vid123")
        assert result == expected_uuid

    def test_get_video_uuid_returns_none_for_missing(self, in_memory_db):
        """get_video_uuid() returns None for non-existent video."""
        result = get_video_uuid("nonexistent")
        assert result is None

    def test_get_video_uuid_fire_and_forget(self):
        """get_video_uuid() returns None on errors."""
        with patch("claudetube.db.sync._get_db", return_value=None):
            result = get_video_uuid("vid123")
            assert result is None


# ============================================================
# Auto-Embedding Tests
# ============================================================


class TestAutoEmbedding:
    """Tests for auto-embedding helper functions."""

    def test_embed_transcription_calls_helper(self, in_memory_db):
        """embed_transcription() calls the async helper correctly."""
        from claudetube.db.sync import embed_transcription

        with patch("claudetube.db.sync._embed_async_fire_and_forget") as mock_embed:
            embed_transcription("video-uuid", "Hello world transcript", 120.5)
            mock_embed.assert_called_once_with(
                video_uuid="video-uuid",
                scene_id=None,
                source="transcription",
                text="Hello world transcript",
                start_time=0.0,
                end_time=120.5,
            )

    def test_embed_scene_transcript_calls_helper(self, in_memory_db):
        """embed_scene_transcript() calls the async helper correctly."""
        from claudetube.db.sync import embed_scene_transcript

        with patch("claudetube.db.sync._embed_async_fire_and_forget") as mock_embed:
            embed_scene_transcript(
                "video-uuid", 5, "Scene text", start_time=100.0, end_time=150.0
            )
            mock_embed.assert_called_once_with(
                video_uuid="video-uuid",
                scene_id=5,
                source="scene_transcript",
                text="Scene text",
                start_time=100.0,
                end_time=150.0,
            )

    def test_embed_visual_description_calls_helper(self, in_memory_db):
        """embed_visual_description() calls the async helper correctly."""
        from claudetube.db.sync import embed_visual_description

        with patch("claudetube.db.sync._embed_async_fire_and_forget") as mock_embed:
            embed_visual_description("video-uuid", 3, "Visual description text")
            mock_embed.assert_called_once_with(
                video_uuid="video-uuid",
                scene_id=3,
                source="visual",
                text="Visual description text",
                start_time=None,
                end_time=None,
            )

    def test_embed_technical_content_calls_helper(self, in_memory_db):
        """embed_technical_content() calls the async helper correctly."""
        from claudetube.db.sync import embed_technical_content

        with patch("claudetube.db.sync._embed_async_fire_and_forget") as mock_embed:
            embed_technical_content("video-uuid", 2, "OCR text content")
            mock_embed.assert_called_once_with(
                video_uuid="video-uuid",
                scene_id=2,
                source="technical",
                text="OCR text content",
                start_time=None,
                end_time=None,
            )

    def test_embed_qa_combines_question_answer(self, in_memory_db):
        """embed_qa() combines question and answer before embedding."""
        from claudetube.db.sync import embed_qa

        with patch("claudetube.db.sync._embed_async_fire_and_forget") as mock_embed:
            embed_qa("video-uuid", "What happens?", "Something happens.")
            mock_embed.assert_called_once_with(
                video_uuid="video-uuid",
                scene_id=None,
                source="qa",
                text="Q: What happens?\nA: Something happens.",
            )

    def test_embed_observation_calls_helper(self, in_memory_db):
        """embed_observation() calls the async helper correctly."""
        from claudetube.db.sync import embed_observation

        with patch("claudetube.db.sync._embed_async_fire_and_forget") as mock_embed:
            embed_observation("video-uuid", 1, "Observation content")
            mock_embed.assert_called_once_with(
                video_uuid="video-uuid",
                scene_id=1,
                source="observation",
                text="Observation content",
                start_time=None,
                end_time=None,
            )

    def test_embed_helper_skips_empty_text(self, in_memory_db):
        """_embed_async_fire_and_forget() skips empty text."""
        from claudetube.db.sync import _embed_async_fire_and_forget

        # Should not raise, should skip silently
        _embed_async_fire_and_forget("video-uuid", None, "transcription", "")
        _embed_async_fire_and_forget("video-uuid", None, "transcription", "   ")
        # No assertions needed - just verify no exceptions

    def test_embed_helper_fire_and_forget_on_error(self, in_memory_db):
        """_embed_async_fire_and_forget() does not raise on errors."""
        from claudetube.db.sync import _embed_async_fire_and_forget

        # Patch vec.embed_text to raise an error
        with patch(
            "claudetube.db.vec.embed_text",
            side_effect=RuntimeError("Embedding failed"),
        ):
            # Should not raise
            _embed_async_fire_and_forget(
                "video-uuid", None, "transcription", "Some text"
            )


class TestSyncTranscriptionWithEmbedding:
    """Tests that sync_transcription triggers auto-embedding."""

    def test_sync_transcription_triggers_embed(self, in_memory_db):
        """sync_transcription() triggers embed_transcription after successful sync."""
        video_repo = VideoRepository(in_memory_db)
        video_uuid = video_repo.insert("vid123", "youtube", "cache/vid123")

        with patch("claudetube.db.sync.embed_transcription") as mock_embed:
            transcript_id = sync_transcription(
                video_uuid=video_uuid,
                provider="whisper",
                format_="srt",
                file_path="audio.srt",
                full_text="Hello world transcript",
                duration=120.5,
            )

            assert transcript_id is not None
            mock_embed.assert_called_once_with(
                video_uuid, "Hello world transcript", 120.5
            )

    def test_sync_transcription_does_not_embed_if_no_full_text(self, in_memory_db):
        """sync_transcription() does not trigger embed if full_text is None."""
        video_repo = VideoRepository(in_memory_db)
        video_uuid = video_repo.insert("vid123", "youtube", "cache/vid123")

        with patch("claudetube.db.sync.embed_transcription") as mock_embed:
            sync_transcription(
                video_uuid=video_uuid,
                provider="whisper",
                format_="srt",
                file_path="audio.srt",
                full_text=None,
            )

            mock_embed.assert_not_called()


class TestSyncSceneWithEmbedding:
    """Tests that sync_scene triggers auto-embedding."""

    def test_sync_scene_triggers_embed(self, in_memory_db):
        """sync_scene() triggers embed_scene_transcript after successful sync."""
        from claudetube.db.sync import sync_scene

        video_repo = VideoRepository(in_memory_db)
        video_uuid = video_repo.insert("vid123", "youtube", "cache/vid123")

        with patch("claudetube.db.sync.embed_scene_transcript") as mock_embed:
            scene_uuid = sync_scene(
                video_uuid=video_uuid,
                scene_id=0,
                start_time=0.0,
                end_time=60.0,
                transcript_text="Scene transcript text",
            )

            assert scene_uuid is not None
            mock_embed.assert_called_once_with(
                video_uuid, 0, "Scene transcript text", 0.0, 60.0
            )

    def test_sync_scene_does_not_embed_if_no_transcript(self, in_memory_db):
        """sync_scene() does not trigger embed if transcript_text is None."""
        from claudetube.db.sync import sync_scene

        video_repo = VideoRepository(in_memory_db)
        video_uuid = video_repo.insert("vid123", "youtube", "cache/vid123")

        with patch("claudetube.db.sync.embed_scene_transcript") as mock_embed:
            sync_scene(
                video_uuid=video_uuid,
                scene_id=0,
                start_time=0.0,
                end_time=60.0,
                transcript_text=None,
            )

            mock_embed.assert_not_called()


class TestSyncQAWithEmbedding:
    """Tests that sync_qa triggers auto-embedding."""

    def test_sync_qa_triggers_embed(self, in_memory_db):
        """sync_qa() triggers embed_qa after successful sync."""
        from claudetube.db.sync import sync_qa

        video_repo = VideoRepository(in_memory_db)
        video_uuid = video_repo.insert("vid123", "youtube", "cache/vid123")

        with patch("claudetube.db.sync.embed_qa") as mock_embed:
            qa_uuid = sync_qa(
                video_uuid=video_uuid,
                question="What happens?",
                answer="Something happens.",
            )

            assert qa_uuid is not None
            mock_embed.assert_called_once_with(
                video_uuid, "What happens?", "Something happens."
            )
