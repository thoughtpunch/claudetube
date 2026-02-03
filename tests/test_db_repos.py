"""Tests for claudetube database repository classes."""

import sqlite3
import uuid

import pytest

from claudetube.db.connection import Database
from claudetube.db.migrate import run_migrations
from claudetube.db.repos import (
    AudioTrackRepository,
    TranscriptionRepository,
    VideoRepository,
)


@pytest.fixture
def db():
    """Create an in-memory database with migrations applied."""
    database = Database(":memory:")
    run_migrations(database)
    yield database
    database.close()


@pytest.fixture
def video_repo(db):
    """Create a VideoRepository instance."""
    return VideoRepository(db)


@pytest.fixture
def audio_repo(db):
    """Create an AudioTrackRepository instance."""
    return AudioTrackRepository(db)


@pytest.fixture
def transcription_repo(db):
    """Create a TranscriptionRepository instance."""
    return TranscriptionRepository(db)


# ============================================================
# VideoRepository Tests
# ============================================================


class TestVideoRepositoryInsert:
    """Tests for VideoRepository.insert()."""

    def test_insert_minimal(self, video_repo):
        """Test inserting a video with only required fields."""
        uuid_ = video_repo.insert(
            video_id="dQw4w9WgXcQ",
            domain="youtube",
            cache_path="/cache/dQw4w9WgXcQ",
        )
        assert uuid_ is not None
        assert len(uuid_) == 36  # UUID format

    def test_insert_with_all_fields(self, video_repo):
        """Test inserting a video with all fields."""
        uuid_ = video_repo.insert(
            video_id="abc123",
            domain="youtube",
            cache_path="/cache/abc123",
            channel="somechannel",
            playlist="someplaylist",
            url="https://youtube.com/watch?v=abc123",
            title="Test Video",
            duration=180.5,
            duration_string="3:00",
            uploader="TestUser",
            channel_name="Test Channel",
            upload_date="2024-01-15",
            description="A test video description",
            language="en",
            view_count=1000000,
            like_count=50000,
            source_type="url",
        )

        video = video_repo.get_by_uuid(uuid_)
        assert video["video_id"] == "abc123"
        assert video["title"] == "Test Video"
        assert video["duration"] == 180.5
        assert video["view_count"] == 1000000

    def test_insert_duplicate_video_id_fails(self, video_repo):
        """Test that inserting duplicate video_id raises IntegrityError."""
        video_repo.insert(
            video_id="duplicate123",
            domain="youtube",
            cache_path="/cache/dup1",
        )

        with pytest.raises(sqlite3.IntegrityError):
            video_repo.insert(
                video_id="duplicate123",
                domain="youtube",
                cache_path="/cache/dup2",
            )

    def test_insert_empty_video_id_fails(self, video_repo):
        """Test that empty video_id violates CHECK constraint."""
        with pytest.raises(sqlite3.IntegrityError):
            video_repo.insert(
                video_id="",
                domain="youtube",
                cache_path="/cache/empty",
            )

    def test_insert_local_source_type(self, video_repo):
        """Test inserting a local video."""
        uuid_ = video_repo.insert(
            video_id="local-file-hash",
            domain="local",
            cache_path="/cache/local",
            source_type="local",
        )

        video = video_repo.get_by_uuid(uuid_)
        assert video["source_type"] == "local"


class TestVideoRepositoryUpsert:
    """Tests for VideoRepository.upsert() progressive enrichment."""

    def test_upsert_creates_new_video(self, video_repo):
        """Test upsert creates a new video when none exists."""
        uuid_ = video_repo.upsert(
            video_id="newvideo123",
            domain="youtube",
            cache_path="/cache/new",
            title="New Video",
        )

        video = video_repo.get_by_video_id("newvideo123")
        assert video is not None
        assert video["title"] == "New Video"

    def test_upsert_enriches_null_fields(self, video_repo):
        """Test upsert fills NULL fields without overwriting existing."""
        # Initial insert with minimal data
        video_repo.insert(
            video_id="enrichme",
            domain="youtube",
            cache_path="/cache/enrich",
        )

        # Upsert with additional metadata
        video_repo.upsert(
            video_id="enrichme",
            domain="youtube",
            cache_path="/cache/enrich",
            title="Enriched Title",
            duration=120.0,
            description="Now with description",
        )

        video = video_repo.get_by_video_id("enrichme")
        assert video["title"] == "Enriched Title"
        assert video["duration"] == 120.0
        assert video["description"] == "Now with description"

    def test_upsert_preserves_existing_data(self, video_repo):
        """Test upsert does not overwrite existing non-NULL values."""
        # Insert with title
        video_repo.insert(
            video_id="preserve",
            domain="youtube",
            cache_path="/cache/preserve",
            title="Original Title",
        )

        # Try to upsert with different title
        video_repo.upsert(
            video_id="preserve",
            domain="youtube",
            cache_path="/cache/preserve",
            title="New Title",
            description="Added description",
        )

        video = video_repo.get_by_video_id("preserve")
        # Original title preserved, description added
        assert video["title"] == "Original Title"
        assert video["description"] == "Added description"

    def test_upsert_returns_existing_uuid(self, video_repo):
        """Test upsert returns the existing UUID for an existing video."""
        uuid1 = video_repo.insert(
            video_id="existing",
            domain="youtube",
            cache_path="/cache/existing",
        )

        uuid2 = video_repo.upsert(
            video_id="existing",
            domain="youtube",
            cache_path="/cache/existing",
            title="Added Title",
        )

        assert uuid1 == uuid2


class TestVideoRepositoryGet:
    """Tests for VideoRepository get methods."""

    def test_get_by_video_id(self, video_repo):
        """Test getting a video by its natural key."""
        video_repo.insert(
            video_id="findme",
            domain="vimeo",
            cache_path="/cache/findme",
            title="Find Me Video",
        )

        video = video_repo.get_by_video_id("findme")
        assert video is not None
        assert video["title"] == "Find Me Video"
        assert video["domain"] == "vimeo"

    def test_get_by_video_id_not_found(self, video_repo):
        """Test get_by_video_id returns None for non-existent video."""
        video = video_repo.get_by_video_id("nonexistent")
        assert video is None

    def test_get_by_uuid(self, video_repo):
        """Test getting a video by its UUID."""
        uuid_ = video_repo.insert(
            video_id="uuid-test",
            domain="youtube",
            cache_path="/cache/uuid-test",
            title="UUID Test",
        )

        video = video_repo.get_by_uuid(uuid_)
        assert video is not None
        assert video["video_id"] == "uuid-test"

    def test_get_by_uuid_not_found(self, video_repo):
        """Test get_by_uuid returns None for non-existent UUID."""
        fake_uuid = str(uuid.uuid4())
        video = video_repo.get_by_uuid(fake_uuid)
        assert video is None

    def test_resolve_path(self, video_repo):
        """Test resolve_path returns cache_path for video_id."""
        video_repo.insert(
            video_id="pathtest",
            domain="youtube",
            cache_path="/cache/path/test",
        )

        path = video_repo.resolve_path("pathtest")
        assert path == "/cache/path/test"

    def test_resolve_path_not_found(self, video_repo):
        """Test resolve_path returns None for non-existent video."""
        path = video_repo.resolve_path("nonexistent")
        assert path is None


class TestVideoRepositoryListAndSearch:
    """Tests for VideoRepository list and search methods."""

    def test_list_all_empty(self, video_repo):
        """Test list_all returns empty list when no videos."""
        videos = video_repo.list_all()
        assert videos == []

    def test_list_all_returns_all_videos(self, video_repo):
        """Test list_all returns all videos."""
        for i in range(3):
            video_repo.insert(
                video_id=f"video{i}",
                domain="youtube",
                cache_path=f"/cache/video{i}",
            )

        videos = video_repo.list_all()
        assert len(videos) == 3

    def test_list_all_ordered_by_created_at(self, video_repo, db):
        """Test list_all returns videos in reverse chronological order."""
        # Insert with explicit timestamps to ensure deterministic ordering
        video_repo.insert(video_id="first", domain="youtube", cache_path="/a")
        # Manually update the first video to have an older timestamp
        db.execute(
            "UPDATE videos SET created_at = datetime('now', '-1 minute') WHERE video_id = 'first'"
        )
        db.commit()
        video_repo.insert(video_id="second", domain="youtube", cache_path="/b")

        videos = video_repo.list_all()
        # Most recent first
        assert videos[0]["video_id"] == "second"
        assert videos[1]["video_id"] == "first"

    def test_search_fts_by_title(self, video_repo):
        """Test FTS search by title."""
        video_repo.insert(
            video_id="searchable1",
            domain="youtube",
            cache_path="/cache/s1",
            title="Python Tutorial for Beginners",
        )
        video_repo.insert(
            video_id="searchable2",
            domain="youtube",
            cache_path="/cache/s2",
            title="JavaScript Guide",
        )

        results = video_repo.search_fts("Python")
        assert len(results) == 1
        assert results[0]["video_id"] == "searchable1"

    def test_search_fts_by_description(self, video_repo):
        """Test FTS search by description."""
        video_repo.insert(
            video_id="desc1",
            domain="youtube",
            cache_path="/cache/d1",
            title="Video 1",
            description="Learn about machine learning algorithms",
        )

        results = video_repo.search_fts("machine learning")
        assert len(results) == 1
        assert results[0]["video_id"] == "desc1"

    def test_search_fts_by_channel_name(self, video_repo):
        """Test FTS search by channel name."""
        video_repo.insert(
            video_id="chan1",
            domain="youtube",
            cache_path="/cache/c1",
            title="Some Video",
            channel_name="Computerphile",
        )

        results = video_repo.search_fts("Computerphile")
        assert len(results) == 1
        assert results[0]["video_id"] == "chan1"

    def test_search_fts_no_results(self, video_repo):
        """Test FTS search returns empty list when no matches."""
        video_repo.insert(
            video_id="nomatch",
            domain="youtube",
            cache_path="/cache/nm",
            title="Unrelated Content",
        )

        results = video_repo.search_fts("quantum physics")
        assert results == []


class TestVideoRepositoryDelete:
    """Tests for VideoRepository.delete()."""

    def test_delete_existing_video(self, video_repo):
        """Test deleting an existing video."""
        video_repo.insert(
            video_id="deleteme",
            domain="youtube",
            cache_path="/cache/del",
        )

        result = video_repo.delete("deleteme")
        assert result is True

        video = video_repo.get_by_video_id("deleteme")
        assert video is None

    def test_delete_nonexistent_video(self, video_repo):
        """Test deleting a non-existent video returns False."""
        result = video_repo.delete("nonexistent")
        assert result is False


# ============================================================
# AudioTrackRepository Tests
# ============================================================


class TestAudioTrackRepositoryInsert:
    """Tests for AudioTrackRepository.insert()."""

    def test_insert_audio_track(self, video_repo, audio_repo):
        """Test inserting an audio track."""
        video_uuid = video_repo.insert(
            video_id="audiovid",
            domain="youtube",
            cache_path="/cache/av",
        )

        track_uuid = audio_repo.insert(
            video_uuid=video_uuid,
            format_="mp3",
            file_path="audio.mp3",
            sample_rate=44100,
            channels=2,
            bitrate_kbps=192,
            duration=180.5,
            file_size_bytes=4320000,
        )

        assert track_uuid is not None
        assert len(track_uuid) == 36

    def test_insert_invalid_format_raises(self, video_repo, audio_repo):
        """Test inserting with invalid format raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="badformat",
            domain="youtube",
            cache_path="/cache/bf",
        )

        with pytest.raises(ValueError, match="Invalid audio format"):
            audio_repo.insert(
                video_uuid=video_uuid,
                format_="mp4",  # Invalid
                file_path="audio.mp4",
            )

    def test_insert_all_valid_formats(self, video_repo, audio_repo):
        """Test all valid formats can be inserted."""
        video_uuid = video_repo.insert(
            video_id="allformats",
            domain="youtube",
            cache_path="/cache/af",
        )

        for fmt in AudioTrackRepository.VALID_FORMATS:
            track_uuid = audio_repo.insert(
                video_uuid=video_uuid,
                format_=fmt,
                file_path=f"audio.{fmt}",
            )
            assert track_uuid is not None

    def test_insert_with_nonexistent_video_raises(self, audio_repo):
        """Test inserting track for non-existent video raises IntegrityError."""
        fake_video_uuid = str(uuid.uuid4())

        with pytest.raises(sqlite3.IntegrityError):
            audio_repo.insert(
                video_uuid=fake_video_uuid,
                format_="mp3",
                file_path="audio.mp3",
            )


class TestAudioTrackRepositoryGet:
    """Tests for AudioTrackRepository get methods."""

    def test_get_by_uuid(self, video_repo, audio_repo):
        """Test getting an audio track by UUID."""
        video_uuid = video_repo.insert(
            video_id="getuuid",
            domain="youtube",
            cache_path="/cache/gu",
        )

        track_uuid = audio_repo.insert(
            video_uuid=video_uuid,
            format_="wav",
            file_path="audio.wav",
            duration=60.0,
        )

        track = audio_repo.get_by_uuid(track_uuid)
        assert track is not None
        assert track["format"] == "wav"
        assert track["duration"] == 60.0

    def test_get_by_uuid_not_found(self, audio_repo):
        """Test get_by_uuid returns None for non-existent track."""
        fake_uuid = str(uuid.uuid4())
        track = audio_repo.get_by_uuid(fake_uuid)
        assert track is None

    def test_get_by_video(self, video_repo, audio_repo):
        """Test getting all tracks for a video."""
        video_uuid = video_repo.insert(
            video_id="multitracks",
            domain="youtube",
            cache_path="/cache/mt",
        )

        audio_repo.insert(video_uuid=video_uuid, format_="mp3", file_path="a.mp3")
        audio_repo.insert(video_uuid=video_uuid, format_="wav", file_path="a.wav")
        audio_repo.insert(video_uuid=video_uuid, format_="aac", file_path="a.aac")

        tracks = audio_repo.get_by_video(video_uuid)
        assert len(tracks) == 3
        formats = {t["format"] for t in tracks}
        assert formats == {"mp3", "wav", "aac"}

    def test_get_by_video_empty(self, video_repo, audio_repo):
        """Test get_by_video returns empty list when no tracks."""
        video_uuid = video_repo.insert(
            video_id="notracks",
            domain="youtube",
            cache_path="/cache/nt",
        )

        tracks = audio_repo.get_by_video(video_uuid)
        assert tracks == []

    def test_get_by_video_and_format(self, video_repo, audio_repo):
        """Test getting a specific track by video and format."""
        video_uuid = video_repo.insert(
            video_id="formattest",
            domain="youtube",
            cache_path="/cache/ft",
        )

        audio_repo.insert(video_uuid=video_uuid, format_="mp3", file_path="a.mp3")
        audio_repo.insert(video_uuid=video_uuid, format_="wav", file_path="a.wav")

        track = audio_repo.get_by_video_and_format(video_uuid, "mp3")
        assert track is not None
        assert track["format"] == "mp3"

        track2 = audio_repo.get_by_video_and_format(video_uuid, "flac")
        assert track2 is None


class TestAudioTrackRepositoryDelete:
    """Tests for AudioTrackRepository delete methods."""

    def test_delete_by_uuid(self, video_repo, audio_repo):
        """Test deleting a track by UUID."""
        video_uuid = video_repo.insert(
            video_id="deltrack",
            domain="youtube",
            cache_path="/cache/dt",
        )

        track_uuid = audio_repo.insert(
            video_uuid=video_uuid,
            format_="mp3",
            file_path="audio.mp3",
        )

        result = audio_repo.delete(track_uuid)
        assert result is True

        track = audio_repo.get_by_uuid(track_uuid)
        assert track is None

    def test_delete_nonexistent(self, audio_repo):
        """Test deleting non-existent track returns False."""
        fake_uuid = str(uuid.uuid4())
        result = audio_repo.delete(fake_uuid)
        assert result is False

    def test_delete_by_video(self, video_repo, audio_repo):
        """Test deleting all tracks for a video."""
        video_uuid = video_repo.insert(
            video_id="delall",
            domain="youtube",
            cache_path="/cache/da",
        )

        audio_repo.insert(video_uuid=video_uuid, format_="mp3", file_path="a.mp3")
        audio_repo.insert(video_uuid=video_uuid, format_="wav", file_path="a.wav")

        count = audio_repo.delete_by_video(video_uuid)
        assert count == 2

        tracks = audio_repo.get_by_video(video_uuid)
        assert tracks == []

    def test_cascade_delete_on_video_delete(self, video_repo, audio_repo):
        """Test that deleting a video cascades to audio tracks."""
        video_uuid = video_repo.insert(
            video_id="cascade",
            domain="youtube",
            cache_path="/cache/cas",
        )

        track_uuid = audio_repo.insert(
            video_uuid=video_uuid,
            format_="mp3",
            file_path="audio.mp3",
        )

        video_repo.delete("cascade")

        track = audio_repo.get_by_uuid(track_uuid)
        assert track is None


# ============================================================
# TranscriptionRepository Tests
# ============================================================


class TestTranscriptionRepositoryInsert:
    """Tests for TranscriptionRepository.insert()."""

    def test_insert_transcription(self, video_repo, transcription_repo):
        """Test inserting a transcription."""
        video_uuid = video_repo.insert(
            video_id="transvid",
            domain="youtube",
            cache_path="/cache/tv",
        )

        trans_uuid = transcription_repo.insert(
            video_uuid=video_uuid,
            provider="whisper",
            format_="srt",
            file_path="audio.srt",
            model="small",
            language="en",
            full_text="Hello world, this is a test transcription.",
            word_count=8,
            duration=120.0,
            confidence=0.95,
            file_size_bytes=1024,
            is_primary=True,
        )

        assert trans_uuid is not None
        assert len(trans_uuid) == 36

    def test_insert_invalid_provider_raises(self, video_repo, transcription_repo):
        """Test inserting with invalid provider raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="badprov",
            domain="youtube",
            cache_path="/cache/bp",
        )

        with pytest.raises(ValueError, match="Invalid provider"):
            transcription_repo.insert(
                video_uuid=video_uuid,
                provider="invalid_provider",
                format_="txt",
                file_path="audio.txt",
            )

    def test_insert_invalid_format_raises(self, video_repo, transcription_repo):
        """Test inserting with invalid format raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="badfmt",
            domain="youtube",
            cache_path="/cache/bfm",
        )

        with pytest.raises(ValueError, match="Invalid format"):
            transcription_repo.insert(
                video_uuid=video_uuid,
                provider="whisper",
                format_="json",  # Invalid
                file_path="audio.json",
            )

    def test_insert_all_valid_providers(self, video_repo, transcription_repo):
        """Test all valid providers can be used."""
        video_uuid = video_repo.insert(
            video_id="allprov",
            domain="youtube",
            cache_path="/cache/ap",
        )

        for provider in TranscriptionRepository.VALID_PROVIDERS:
            trans_uuid = transcription_repo.insert(
                video_uuid=video_uuid,
                provider=provider,
                format_="txt",
                file_path=f"audio_{provider}.txt",
            )
            assert trans_uuid is not None

    def test_insert_with_nonexistent_video_raises(self, transcription_repo):
        """Test inserting for non-existent video raises IntegrityError."""
        fake_video_uuid = str(uuid.uuid4())

        with pytest.raises(sqlite3.IntegrityError):
            transcription_repo.insert(
                video_uuid=fake_video_uuid,
                provider="whisper",
                format_="txt",
                file_path="audio.txt",
            )


class TestTranscriptionRepositoryPrimary:
    """Tests for transcription primary designation."""

    def test_get_primary(self, video_repo, transcription_repo):
        """Test getting the primary transcription."""
        video_uuid = video_repo.insert(
            video_id="primary",
            domain="youtube",
            cache_path="/cache/pr",
        )

        transcription_repo.insert(
            video_uuid=video_uuid,
            provider="youtube_subtitles",
            format_="srt",
            file_path="yt.srt",
            is_primary=False,
        )

        transcription_repo.insert(
            video_uuid=video_uuid,
            provider="whisper",
            format_="srt",
            file_path="whisper.srt",
            is_primary=True,
        )

        primary = transcription_repo.get_primary(video_uuid)
        assert primary is not None
        assert primary["provider"] == "whisper"

    def test_get_primary_none(self, video_repo, transcription_repo):
        """Test get_primary returns None when no primary exists."""
        video_uuid = video_repo.insert(
            video_id="noprimary",
            domain="youtube",
            cache_path="/cache/np",
        )

        transcription_repo.insert(
            video_uuid=video_uuid,
            provider="whisper",
            format_="txt",
            file_path="audio.txt",
            is_primary=False,
        )

        primary = transcription_repo.get_primary(video_uuid)
        assert primary is None

    def test_insert_primary_unsets_existing(self, video_repo, transcription_repo):
        """Test inserting a primary transcription unsets the existing primary."""
        video_uuid = video_repo.insert(
            video_id="swappr",
            domain="youtube",
            cache_path="/cache/sp",
        )

        first_uuid = transcription_repo.insert(
            video_uuid=video_uuid,
            provider="youtube_subtitles",
            format_="srt",
            file_path="first.srt",
            is_primary=True,
        )

        second_uuid = transcription_repo.insert(
            video_uuid=video_uuid,
            provider="whisper",
            format_="srt",
            file_path="second.srt",
            is_primary=True,
        )

        # Second should now be primary
        primary = transcription_repo.get_primary(video_uuid)
        assert primary["id"] == second_uuid

        # First should no longer be primary
        first = transcription_repo.get_by_uuid(first_uuid)
        assert first["is_primary"] == 0

    def test_set_primary(self, video_repo, transcription_repo):
        """Test set_primary() changes the primary transcription."""
        video_uuid = video_repo.insert(
            video_id="setpr",
            domain="youtube",
            cache_path="/cache/setp",
        )

        first_uuid = transcription_repo.insert(
            video_uuid=video_uuid,
            provider="youtube_subtitles",
            format_="srt",
            file_path="first.srt",
            is_primary=True,
        )

        second_uuid = transcription_repo.insert(
            video_uuid=video_uuid,
            provider="whisper",
            format_="srt",
            file_path="second.srt",
            is_primary=False,
        )

        result = transcription_repo.set_primary(second_uuid)
        assert result is True

        # Second should now be primary
        primary = transcription_repo.get_primary(video_uuid)
        assert primary["id"] == second_uuid

        # First should no longer be primary
        first = transcription_repo.get_by_uuid(first_uuid)
        assert first["is_primary"] == 0

    def test_set_primary_nonexistent_returns_false(self, transcription_repo):
        """Test set_primary returns False for non-existent transcription."""
        fake_uuid = str(uuid.uuid4())
        result = transcription_repo.set_primary(fake_uuid)
        assert result is False


class TestTranscriptionRepositoryGet:
    """Tests for TranscriptionRepository get methods."""

    def test_get_by_uuid(self, video_repo, transcription_repo):
        """Test getting a transcription by UUID."""
        video_uuid = video_repo.insert(
            video_id="getuuid",
            domain="youtube",
            cache_path="/cache/gu",
        )

        trans_uuid = transcription_repo.insert(
            video_uuid=video_uuid,
            provider="deepgram",
            format_="vtt",
            file_path="audio.vtt",
            full_text="Test content",
        )

        trans = transcription_repo.get_by_uuid(trans_uuid)
        assert trans is not None
        assert trans["provider"] == "deepgram"
        assert trans["full_text"] == "Test content"

    def test_get_by_uuid_not_found(self, transcription_repo):
        """Test get_by_uuid returns None for non-existent transcription."""
        fake_uuid = str(uuid.uuid4())
        trans = transcription_repo.get_by_uuid(fake_uuid)
        assert trans is None

    def test_get_by_video(self, video_repo, transcription_repo):
        """Test getting all transcriptions for a video."""
        video_uuid = video_repo.insert(
            video_id="multitrans",
            domain="youtube",
            cache_path="/cache/mtrans",
        )

        transcription_repo.insert(
            video_uuid=video_uuid,
            provider="youtube_subtitles",
            format_="srt",
            file_path="yt.srt",
            is_primary=False,
        )

        transcription_repo.insert(
            video_uuid=video_uuid,
            provider="whisper",
            format_="txt",
            file_path="w.txt",
            is_primary=True,
        )

        transcriptions = transcription_repo.get_by_video(video_uuid)
        assert len(transcriptions) == 2
        # Primary should be first
        assert transcriptions[0]["is_primary"] == 1
        assert transcriptions[0]["provider"] == "whisper"

    def test_get_by_video_empty(self, video_repo, transcription_repo):
        """Test get_by_video returns empty list when no transcriptions."""
        video_uuid = video_repo.insert(
            video_id="notrans",
            domain="youtube",
            cache_path="/cache/ntrans",
        )

        transcriptions = transcription_repo.get_by_video(video_uuid)
        assert transcriptions == []


class TestTranscriptionRepositoryFTS:
    """Tests for TranscriptionRepository full-text search."""

    def test_search_fts(self, video_repo, transcription_repo):
        """Test FTS search on transcription text."""
        video_uuid = video_repo.insert(
            video_id="searchvid",
            domain="youtube",
            cache_path="/cache/sv",
            title="Search Video",
        )

        transcription_repo.insert(
            video_uuid=video_uuid,
            provider="whisper",
            format_="txt",
            file_path="audio.txt",
            full_text="Today we are going to learn about machine learning and neural networks.",
            is_primary=True,
        )

        results = transcription_repo.search_fts("machine learning")
        assert len(results) == 1
        assert results[0]["video_natural_id"] == "searchvid"
        assert results[0]["video_title"] == "Search Video"

    def test_search_fts_multiple_videos(self, video_repo, transcription_repo):
        """Test FTS search across multiple videos."""
        for i, topic in enumerate(["python programming", "javascript basics", "python decorators"]):
            vid_uuid = video_repo.insert(
                video_id=f"vid{i}",
                domain="youtube",
                cache_path=f"/cache/v{i}",
                title=f"Video {i}",
            )

            transcription_repo.insert(
                video_uuid=vid_uuid,
                provider="whisper",
                format_="txt",
                file_path=f"a{i}.txt",
                full_text=f"This video covers {topic}.",
                is_primary=True,
            )

        results = transcription_repo.search_fts("python")
        assert len(results) == 2
        video_ids = {r["video_natural_id"] for r in results}
        assert video_ids == {"vid0", "vid2"}

    def test_search_fts_no_results(self, video_repo, transcription_repo):
        """Test FTS search returns empty list when no matches."""
        video_uuid = video_repo.insert(
            video_id="nomatch",
            domain="youtube",
            cache_path="/cache/nm",
        )

        transcription_repo.insert(
            video_uuid=video_uuid,
            provider="whisper",
            format_="txt",
            file_path="audio.txt",
            full_text="This is about cooking recipes.",
            is_primary=True,
        )

        results = transcription_repo.search_fts("quantum physics")
        assert results == []

    def test_search_fts_special_characters(self, video_repo, transcription_repo):
        """Test FTS search with special characters doesn't error."""
        video_uuid = video_repo.insert(
            video_id="special",
            domain="youtube",
            cache_path="/cache/spec",
        )

        transcription_repo.insert(
            video_uuid=video_uuid,
            provider="whisper",
            format_="txt",
            file_path="audio.txt",
            full_text="Using C++ and C# for development.",
            is_primary=True,
        )

        # Should not raise even with special chars
        results = transcription_repo.search_fts("C++")
        # May or may not match depending on tokenization
        assert isinstance(results, list)


class TestTranscriptionRepositoryDelete:
    """Tests for TranscriptionRepository delete methods."""

    def test_delete_by_uuid(self, video_repo, transcription_repo):
        """Test deleting a transcription by UUID."""
        video_uuid = video_repo.insert(
            video_id="deltrans",
            domain="youtube",
            cache_path="/cache/dt",
        )

        trans_uuid = transcription_repo.insert(
            video_uuid=video_uuid,
            provider="whisper",
            format_="txt",
            file_path="audio.txt",
        )

        result = transcription_repo.delete(trans_uuid)
        assert result is True

        trans = transcription_repo.get_by_uuid(trans_uuid)
        assert trans is None

    def test_delete_nonexistent(self, transcription_repo):
        """Test deleting non-existent transcription returns False."""
        fake_uuid = str(uuid.uuid4())
        result = transcription_repo.delete(fake_uuid)
        assert result is False

    def test_delete_by_video(self, video_repo, transcription_repo):
        """Test deleting all transcriptions for a video."""
        video_uuid = video_repo.insert(
            video_id="delall",
            domain="youtube",
            cache_path="/cache/da",
        )

        transcription_repo.insert(
            video_uuid=video_uuid,
            provider="whisper",
            format_="txt",
            file_path="w.txt",
        )
        transcription_repo.insert(
            video_uuid=video_uuid,
            provider="deepgram",
            format_="srt",
            file_path="d.srt",
        )

        count = transcription_repo.delete_by_video(video_uuid)
        assert count == 2

        transcriptions = transcription_repo.get_by_video(video_uuid)
        assert transcriptions == []

    def test_cascade_delete_on_video_delete(self, video_repo, transcription_repo):
        """Test that deleting a video cascades to transcriptions."""
        video_uuid = video_repo.insert(
            video_id="cascade",
            domain="youtube",
            cache_path="/cache/cas",
        )

        trans_uuid = transcription_repo.insert(
            video_uuid=video_uuid,
            provider="whisper",
            format_="txt",
            file_path="audio.txt",
        )

        video_repo.delete("cascade")

        trans = transcription_repo.get_by_uuid(trans_uuid)
        assert trans is None


# ============================================================
# Constraint Validation Tests
# ============================================================


class TestConstraintValidation:
    """Tests for CHECK constraint validation."""

    def test_video_invalid_domain_rejected(self, video_repo):
        """Test that invalid domain format is rejected."""
        # Domain must match GLOB '[a-z]*' (lowercase letters)
        with pytest.raises(sqlite3.IntegrityError):
            video_repo.insert(
                video_id="baddomain",
                domain="YouTube",  # Uppercase
                cache_path="/cache/bd",
            )

    def test_video_negative_duration_rejected(self, video_repo):
        """Test that negative duration is rejected."""
        with pytest.raises(sqlite3.IntegrityError):
            video_repo.insert(
                video_id="negdur",
                domain="youtube",
                cache_path="/cache/nd",
                duration=-10.0,
            )

    def test_video_negative_view_count_rejected(self, video_repo):
        """Test that negative view_count is rejected."""
        with pytest.raises(sqlite3.IntegrityError):
            video_repo.insert(
                video_id="negviews",
                domain="youtube",
                cache_path="/cache/nv",
                view_count=-100,
            )

    def test_audio_track_invalid_channels_rejected(self, video_repo, audio_repo):
        """Test that channels > 16 is rejected."""
        video_uuid = video_repo.insert(
            video_id="badchannels",
            domain="youtube",
            cache_path="/cache/bc",
        )

        with pytest.raises(sqlite3.IntegrityError):
            audio_repo.insert(
                video_uuid=video_uuid,
                format_="mp3",
                file_path="audio.mp3",
                channels=32,  # Max is 16
            )

    def test_transcription_invalid_confidence_rejected(self, video_repo, transcription_repo):
        """Test that confidence > 1.0 is rejected."""
        video_uuid = video_repo.insert(
            video_id="badconf",
            domain="youtube",
            cache_path="/cache/bconf",
        )

        with pytest.raises(sqlite3.IntegrityError):
            transcription_repo.insert(
                video_uuid=video_uuid,
                provider="whisper",
                format_="txt",
                file_path="audio.txt",
                confidence=1.5,  # Max is 1.0
            )

    def test_uuid_format_validated(self, video_repo):
        """Test that UUIDs with wrong length are rejected."""
        # The schema checks length(id) = 36
        db = video_repo.db
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO videos (id, video_id, domain, cache_path) VALUES (?, ?, ?, ?)",
                ("short-uuid", "test", "youtube", "/cache"),
            )
            db.commit()
