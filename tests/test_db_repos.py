"""Tests for claudetube database repository classes."""

import sqlite3
import uuid

import pytest

from claudetube.db.connection import Database
from claudetube.db.migrate import run_migrations
from claudetube.db.repos import (
    AudioDescriptionRepository,
    AudioTrackRepository,
    CodeEvolutionRepository,
    EntityRepository,
    FrameRepository,
    NarrativeRepository,
    ObservationRepository,
    QARepository,
    SceneRepository,
    TechnicalContentRepository,
    TranscriptionRepository,
    VideoRepository,
    VisualDescriptionRepository,
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


@pytest.fixture
def scene_repo(db):
    """Create a SceneRepository instance."""
    return SceneRepository(db)


@pytest.fixture
def frame_repo(db):
    """Create a FrameRepository instance."""
    return FrameRepository(db)


@pytest.fixture
def visual_description_repo(db):
    """Create a VisualDescriptionRepository instance."""
    return VisualDescriptionRepository(db)


@pytest.fixture
def technical_content_repo(db):
    """Create a TechnicalContentRepository instance."""
    return TechnicalContentRepository(db)


@pytest.fixture
def audio_description_repo(db):
    """Create an AudioDescriptionRepository instance."""
    return AudioDescriptionRepository(db)


@pytest.fixture
def narrative_repo(db):
    """Create a NarrativeRepository instance."""
    return NarrativeRepository(db)


@pytest.fixture
def code_evolution_repo(db):
    """Create a CodeEvolutionRepository instance."""
    return CodeEvolutionRepository(db)


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


# ============================================================
# SceneRepository Tests
# ============================================================


class TestSceneRepositoryInsert:
    """Tests for SceneRepository.insert()."""

    def test_insert_scene(self, video_repo, scene_repo):
        """Test inserting a scene."""
        video_uuid = video_repo.insert(
            video_id="scenevid",
            domain="youtube",
            cache_path="/cache/sv",
        )

        scene_uuid = scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            start_time=0.0,
            end_time=30.0,
            title="Introduction",
            transcript_text="Welcome to this video about Python.",
            method="transcript",
            relevance_boost=1.0,
        )

        assert scene_uuid is not None
        assert len(scene_uuid) == 36

    def test_insert_minimal_scene(self, video_repo, scene_repo):
        """Test inserting a scene with only required fields."""
        video_uuid = video_repo.insert(
            video_id="minscene",
            domain="youtube",
            cache_path="/cache/ms",
        )

        scene_uuid = scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            start_time=0.0,
            end_time=10.0,
        )

        scene = scene_repo.get_by_uuid(scene_uuid)
        assert scene is not None
        assert scene["title"] is None
        assert scene["transcript_text"] is None
        assert scene["method"] is None
        assert scene["relevance_boost"] == 1.0

    def test_insert_invalid_start_time_raises(self, video_repo, scene_repo):
        """Test that start_time >= end_time raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="badtimes",
            domain="youtube",
            cache_path="/cache/bt",
        )

        with pytest.raises(ValueError, match="start_time.*must be < end_time"):
            scene_repo.insert(
                video_uuid=video_uuid,
                scene_id=0,
                start_time=30.0,
                end_time=10.0,
            )

    def test_insert_equal_times_raises(self, video_repo, scene_repo):
        """Test that start_time == end_time raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="eqtimes",
            domain="youtube",
            cache_path="/cache/et",
        )

        with pytest.raises(ValueError, match="start_time.*must be < end_time"):
            scene_repo.insert(
                video_uuid=video_uuid,
                scene_id=0,
                start_time=10.0,
                end_time=10.0,
            )

    def test_insert_negative_scene_id_raises(self, video_repo, scene_repo):
        """Test that negative scene_id raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="negscene",
            domain="youtube",
            cache_path="/cache/ns",
        )

        with pytest.raises(ValueError, match="scene_id must be >= 0"):
            scene_repo.insert(
                video_uuid=video_uuid,
                scene_id=-1,
                start_time=0.0,
                end_time=10.0,
            )

    def test_insert_invalid_method_raises(self, video_repo, scene_repo):
        """Test that invalid method raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="badmethod",
            domain="youtube",
            cache_path="/cache/bm",
        )

        with pytest.raises(ValueError, match="Invalid method"):
            scene_repo.insert(
                video_uuid=video_uuid,
                scene_id=0,
                start_time=0.0,
                end_time=10.0,
                method="invalid_method",
            )

    def test_insert_all_valid_methods(self, video_repo, scene_repo):
        """Test all valid methods can be inserted."""
        video_uuid = video_repo.insert(
            video_id="allmethods",
            domain="youtube",
            cache_path="/cache/am",
        )

        for i, method in enumerate(SceneRepository.VALID_METHODS):
            scene_uuid = scene_repo.insert(
                video_uuid=video_uuid,
                scene_id=i,
                start_time=float(i * 10),
                end_time=float(i * 10 + 9),
                method=method,
            )
            assert scene_uuid is not None

    def test_insert_duplicate_scene_id_raises(self, video_repo, scene_repo):
        """Test that duplicate (video_uuid, scene_id) raises IntegrityError."""
        video_uuid = video_repo.insert(
            video_id="dupscene",
            domain="youtube",
            cache_path="/cache/ds",
        )

        scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            start_time=0.0,
            end_time=10.0,
        )

        with pytest.raises(sqlite3.IntegrityError):
            scene_repo.insert(
                video_uuid=video_uuid,
                scene_id=0,
                start_time=10.0,
                end_time=20.0,
            )

    def test_insert_negative_relevance_boost_raises(self, video_repo, scene_repo):
        """Test that negative relevance_boost raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="negboost",
            domain="youtube",
            cache_path="/cache/nb",
        )

        with pytest.raises(ValueError, match="relevance_boost must be >= 0"):
            scene_repo.insert(
                video_uuid=video_uuid,
                scene_id=0,
                start_time=0.0,
                end_time=10.0,
                relevance_boost=-1.0,
            )


class TestSceneRepositoryBulkInsert:
    """Tests for SceneRepository.bulk_insert()."""

    def test_bulk_insert_scenes(self, video_repo, scene_repo):
        """Test bulk inserting multiple scenes."""
        video_uuid = video_repo.insert(
            video_id="bulkscenes",
            domain="youtube",
            cache_path="/cache/bs",
        )

        scenes = [
            {"scene_id": 0, "start_time": 0.0, "end_time": 30.0, "title": "Intro"},
            {"scene_id": 1, "start_time": 30.0, "end_time": 60.0, "title": "Main"},
            {"scene_id": 2, "start_time": 60.0, "end_time": 90.0, "title": "Outro"},
        ]

        uuids = scene_repo.bulk_insert(video_uuid, scenes)
        assert len(uuids) == 3
        assert all(len(u) == 36 for u in uuids)

        all_scenes = scene_repo.get_by_video(video_uuid)
        assert len(all_scenes) == 3
        assert all_scenes[0]["title"] == "Intro"
        assert all_scenes[1]["title"] == "Main"
        assert all_scenes[2]["title"] == "Outro"

    def test_bulk_insert_empty_list(self, video_repo, scene_repo):
        """Test bulk insert with empty list returns empty list."""
        video_uuid = video_repo.insert(
            video_id="emptybulk",
            domain="youtube",
            cache_path="/cache/eb",
        )

        uuids = scene_repo.bulk_insert(video_uuid, [])
        assert uuids == []

    def test_bulk_insert_invalid_scene_raises(self, video_repo, scene_repo):
        """Test bulk insert validates all scenes before inserting."""
        video_uuid = video_repo.insert(
            video_id="badbulk",
            domain="youtube",
            cache_path="/cache/bb",
        )

        scenes = [
            {"scene_id": 0, "start_time": 0.0, "end_time": 30.0},
            {"scene_id": 1, "start_time": 60.0, "end_time": 30.0},  # Invalid
        ]

        with pytest.raises(ValueError, match="start_time.*must be < end_time"):
            scene_repo.bulk_insert(video_uuid, scenes)

    def test_bulk_insert_missing_required_field_raises(self, video_repo, scene_repo):
        """Test bulk insert with missing required field raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="missingfield",
            domain="youtube",
            cache_path="/cache/mf",
        )

        scenes = [
            {"scene_id": 0, "start_time": 0.0},  # Missing end_time
        ]

        with pytest.raises(ValueError, match="end_time is required"):
            scene_repo.bulk_insert(video_uuid, scenes)


class TestSceneRepositoryGet:
    """Tests for SceneRepository get methods."""

    def test_get_by_uuid(self, video_repo, scene_repo):
        """Test getting a scene by UUID."""
        video_uuid = video_repo.insert(
            video_id="getuuid",
            domain="youtube",
            cache_path="/cache/gu",
        )

        scene_uuid = scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            start_time=0.0,
            end_time=30.0,
            title="Test Scene",
        )

        scene = scene_repo.get_by_uuid(scene_uuid)
        assert scene is not None
        assert scene["title"] == "Test Scene"

    def test_get_by_uuid_not_found(self, scene_repo):
        """Test get_by_uuid returns None for non-existent scene."""
        fake_uuid = str(uuid.uuid4())
        scene = scene_repo.get_by_uuid(fake_uuid)
        assert scene is None

    def test_get_scene(self, video_repo, scene_repo):
        """Test getting a scene by video_uuid and scene_id."""
        video_uuid = video_repo.insert(
            video_id="getscene",
            domain="youtube",
            cache_path="/cache/gs",
        )

        scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            start_time=0.0,
            end_time=30.0,
        )
        scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=1,
            start_time=30.0,
            end_time=60.0,
            title="Second Scene",
        )

        scene = scene_repo.get_scene(video_uuid, 1)
        assert scene is not None
        assert scene["title"] == "Second Scene"

    def test_get_scene_not_found(self, video_repo, scene_repo):
        """Test get_scene returns None for non-existent scene_id."""
        video_uuid = video_repo.insert(
            video_id="nogetscene",
            domain="youtube",
            cache_path="/cache/ngs",
        )

        scene = scene_repo.get_scene(video_uuid, 99)
        assert scene is None

    def test_get_by_video(self, video_repo, scene_repo):
        """Test getting all scenes for a video."""
        video_uuid = video_repo.insert(
            video_id="multiscenes",
            domain="youtube",
            cache_path="/cache/ms",
        )

        # Insert out of order to verify ordering
        scene_repo.insert(video_uuid=video_uuid, scene_id=2, start_time=60.0, end_time=90.0)
        scene_repo.insert(video_uuid=video_uuid, scene_id=0, start_time=0.0, end_time=30.0)
        scene_repo.insert(video_uuid=video_uuid, scene_id=1, start_time=30.0, end_time=60.0)

        scenes = scene_repo.get_by_video(video_uuid)
        assert len(scenes) == 3
        # Should be ordered by scene_id
        assert scenes[0]["scene_id"] == 0
        assert scenes[1]["scene_id"] == 1
        assert scenes[2]["scene_id"] == 2

    def test_get_by_video_empty(self, video_repo, scene_repo):
        """Test get_by_video returns empty list when no scenes."""
        video_uuid = video_repo.insert(
            video_id="noscenes",
            domain="youtube",
            cache_path="/cache/ns",
        )

        scenes = scene_repo.get_by_video(video_uuid)
        assert scenes == []


class TestSceneRepositoryRelevanceBoost:
    """Tests for SceneRepository.update_relevance_boost()."""

    def test_update_relevance_boost(self, video_repo, scene_repo):
        """Test updating relevance boost for a scene."""
        video_uuid = video_repo.insert(
            video_id="boostvid",
            domain="youtube",
            cache_path="/cache/bv",
        )

        scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            start_time=0.0,
            end_time=30.0,
        )

        result = scene_repo.update_relevance_boost(video_uuid, 0, 2.5)
        assert result is True

        scene = scene_repo.get_scene(video_uuid, 0)
        assert scene["relevance_boost"] == 2.5

    def test_update_relevance_boost_not_found(self, video_repo, scene_repo):
        """Test update_relevance_boost returns False for non-existent scene."""
        video_uuid = video_repo.insert(
            video_id="noboost",
            domain="youtube",
            cache_path="/cache/nbo",
        )

        result = scene_repo.update_relevance_boost(video_uuid, 99, 1.5)
        assert result is False

    def test_update_relevance_boost_negative_raises(self, video_repo, scene_repo):
        """Test that negative boost raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="negboost",
            domain="youtube",
            cache_path="/cache/negb",
        )

        scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            start_time=0.0,
            end_time=30.0,
        )

        with pytest.raises(ValueError, match="relevance_boost must be >= 0"):
            scene_repo.update_relevance_boost(video_uuid, 0, -1.0)


class TestSceneRepositoryFTS:
    """Tests for SceneRepository full-text search."""

    def test_search_fts(self, video_repo, scene_repo):
        """Test FTS search on scene transcript_text."""
        video_uuid = video_repo.insert(
            video_id="ftsvid",
            domain="youtube",
            cache_path="/cache/fts",
            title="FTS Test Video",
        )

        scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            start_time=0.0,
            end_time=30.0,
            transcript_text="Today we are going to learn about machine learning.",
        )

        results = scene_repo.search_fts("machine learning")
        assert len(results) == 1
        assert results[0]["video_natural_id"] == "ftsvid"
        assert results[0]["video_title"] == "FTS Test Video"

    def test_search_fts_multiple_scenes(self, video_repo, scene_repo):
        """Test FTS search across multiple scenes."""
        video_uuid = video_repo.insert(
            video_id="multifts",
            domain="youtube",
            cache_path="/cache/mfts",
        )

        scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            start_time=0.0,
            end_time=30.0,
            transcript_text="Introduction to Python programming.",
        )
        scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=1,
            start_time=30.0,
            end_time=60.0,
            transcript_text="JavaScript for web development.",
        )
        scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=2,
            start_time=60.0,
            end_time=90.0,
            transcript_text="Advanced Python decorators.",
        )

        results = scene_repo.search_fts("Python")
        assert len(results) == 2
        scene_ids = {r["scene_id"] for r in results}
        assert scene_ids == {0, 2}

    def test_search_fts_no_results(self, video_repo, scene_repo):
        """Test FTS search returns empty list when no matches."""
        video_uuid = video_repo.insert(
            video_id="nomatchfts",
            domain="youtube",
            cache_path="/cache/nmfts",
        )

        scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            start_time=0.0,
            end_time=30.0,
            transcript_text="Cooking recipes for beginners.",
        )

        results = scene_repo.search_fts("quantum physics")
        assert results == []


class TestSceneRepositoryDelete:
    """Tests for SceneRepository delete methods."""

    def test_delete_by_uuid(self, video_repo, scene_repo):
        """Test deleting a scene by UUID."""
        video_uuid = video_repo.insert(
            video_id="delscene",
            domain="youtube",
            cache_path="/cache/ds",
        )

        scene_uuid = scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            start_time=0.0,
            end_time=30.0,
        )

        result = scene_repo.delete(scene_uuid)
        assert result is True

        scene = scene_repo.get_by_uuid(scene_uuid)
        assert scene is None

    def test_delete_nonexistent(self, scene_repo):
        """Test deleting non-existent scene returns False."""
        fake_uuid = str(uuid.uuid4())
        result = scene_repo.delete(fake_uuid)
        assert result is False

    def test_delete_by_video(self, video_repo, scene_repo):
        """Test deleting all scenes for a video."""
        video_uuid = video_repo.insert(
            video_id="delallscenes",
            domain="youtube",
            cache_path="/cache/das",
        )

        scene_repo.insert(video_uuid=video_uuid, scene_id=0, start_time=0.0, end_time=30.0)
        scene_repo.insert(video_uuid=video_uuid, scene_id=1, start_time=30.0, end_time=60.0)
        scene_repo.insert(video_uuid=video_uuid, scene_id=2, start_time=60.0, end_time=90.0)

        count = scene_repo.delete_by_video(video_uuid)
        assert count == 3

        scenes = scene_repo.get_by_video(video_uuid)
        assert scenes == []

    def test_cascade_delete_on_video_delete(self, video_repo, scene_repo):
        """Test that deleting a video cascades to scenes."""
        video_uuid = video_repo.insert(
            video_id="cascadescene",
            domain="youtube",
            cache_path="/cache/cs",
        )

        scene_uuid = scene_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            start_time=0.0,
            end_time=30.0,
        )

        video_repo.delete("cascadescene")

        scene = scene_repo.get_by_uuid(scene_uuid)
        assert scene is None


class TestSceneRepositoryCount:
    """Tests for SceneRepository.count_by_video()."""

    def test_count_by_video(self, video_repo, scene_repo):
        """Test counting scenes for a video."""
        video_uuid = video_repo.insert(
            video_id="countvid",
            domain="youtube",
            cache_path="/cache/cv",
        )

        scene_repo.insert(video_uuid=video_uuid, scene_id=0, start_time=0.0, end_time=30.0)
        scene_repo.insert(video_uuid=video_uuid, scene_id=1, start_time=30.0, end_time=60.0)

        count = scene_repo.count_by_video(video_uuid)
        assert count == 2

    def test_count_by_video_empty(self, video_repo, scene_repo):
        """Test count returns 0 for video with no scenes."""
        video_uuid = video_repo.insert(
            video_id="emptycnt",
            domain="youtube",
            cache_path="/cache/ec",
        )

        count = scene_repo.count_by_video(video_uuid)
        assert count == 0


# ============================================================
# FrameRepository Tests
# ============================================================


class TestFrameRepositoryInsert:
    """Tests for FrameRepository.insert()."""

    def test_insert_frame(self, video_repo, frame_repo):
        """Test inserting a frame."""
        video_uuid = video_repo.insert(
            video_id="framevid",
            domain="youtube",
            cache_path="/cache/fv",
        )

        frame_uuid = frame_repo.insert(
            video_uuid=video_uuid,
            timestamp=30.5,
            extraction_type="drill",
            file_path="drill/frame_30.5.jpg",
            scene_id=0,
            quality_tier="low",
            is_thumbnail=False,
            width=480,
            height=270,
            file_size_bytes=25000,
        )

        assert frame_uuid is not None
        assert len(frame_uuid) == 36

    def test_insert_minimal_frame(self, video_repo, frame_repo):
        """Test inserting a frame with only required fields."""
        video_uuid = video_repo.insert(
            video_id="minframe",
            domain="youtube",
            cache_path="/cache/mf",
        )

        frame_uuid = frame_repo.insert(
            video_uuid=video_uuid,
            timestamp=0.0,
            extraction_type="keyframe",
            file_path="keyframes/frame_0.jpg",
        )

        frame = frame_repo.get_by_uuid(frame_uuid)
        assert frame is not None
        assert frame["scene_id"] is None
        assert frame["quality_tier"] is None
        assert frame["is_thumbnail"] == 0

    def test_insert_invalid_extraction_type_raises(self, video_repo, frame_repo):
        """Test that invalid extraction_type raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="badexttype",
            domain="youtube",
            cache_path="/cache/bet",
        )

        with pytest.raises(ValueError, match="Invalid extraction_type"):
            frame_repo.insert(
                video_uuid=video_uuid,
                timestamp=0.0,
                extraction_type="invalid",
                file_path="frame.jpg",
            )

    def test_insert_all_valid_extraction_types(self, video_repo, frame_repo):
        """Test all valid extraction types can be inserted."""
        video_uuid = video_repo.insert(
            video_id="alltypes",
            domain="youtube",
            cache_path="/cache/at",
        )

        for i, ext_type in enumerate(FrameRepository.VALID_EXTRACTION_TYPES):
            frame_uuid = frame_repo.insert(
                video_uuid=video_uuid,
                timestamp=float(i * 10),
                extraction_type=ext_type,
                file_path=f"frame_{ext_type}.jpg",
            )
            assert frame_uuid is not None

    def test_insert_invalid_quality_tier_raises(self, video_repo, frame_repo):
        """Test that invalid quality_tier raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="badquality",
            domain="youtube",
            cache_path="/cache/bq",
        )

        with pytest.raises(ValueError, match="Invalid quality_tier"):
            frame_repo.insert(
                video_uuid=video_uuid,
                timestamp=0.0,
                extraction_type="drill",
                file_path="frame.jpg",
                quality_tier="ultra",  # Invalid
            )

    def test_insert_all_valid_quality_tiers(self, video_repo, frame_repo):
        """Test all valid quality tiers can be inserted."""
        video_uuid = video_repo.insert(
            video_id="allquality",
            domain="youtube",
            cache_path="/cache/aq",
        )

        for i, tier in enumerate(FrameRepository.VALID_QUALITY_TIERS):
            frame_uuid = frame_repo.insert(
                video_uuid=video_uuid,
                timestamp=float(i * 10),
                extraction_type="drill",
                file_path=f"frame_{tier}.jpg",
                quality_tier=tier,
            )
            assert frame_uuid is not None

    def test_insert_negative_timestamp_raises(self, video_repo, frame_repo):
        """Test that negative timestamp raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="negts",
            domain="youtube",
            cache_path="/cache/nts",
        )

        with pytest.raises(ValueError, match="timestamp must be >= 0"):
            frame_repo.insert(
                video_uuid=video_uuid,
                timestamp=-5.0,
                extraction_type="drill",
                file_path="frame.jpg",
            )

    def test_insert_negative_scene_id_raises(self, video_repo, frame_repo):
        """Test that negative scene_id raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="negsceneid",
            domain="youtube",
            cache_path="/cache/nsi",
        )

        with pytest.raises(ValueError, match="scene_id must be >= 0 or None"):
            frame_repo.insert(
                video_uuid=video_uuid,
                timestamp=0.0,
                extraction_type="keyframe",
                file_path="frame.jpg",
                scene_id=-1,
            )


class TestFrameRepositoryGet:
    """Tests for FrameRepository get methods."""

    def test_get_by_uuid(self, video_repo, frame_repo):
        """Test getting a frame by UUID."""
        video_uuid = video_repo.insert(
            video_id="getuuidframe",
            domain="youtube",
            cache_path="/cache/guf",
        )

        frame_uuid = frame_repo.insert(
            video_uuid=video_uuid,
            timestamp=15.0,
            extraction_type="hq",
            file_path="hq/frame_15.jpg",
            width=1280,
        )

        frame = frame_repo.get_by_uuid(frame_uuid)
        assert frame is not None
        assert frame["timestamp"] == 15.0
        assert frame["extraction_type"] == "hq"
        assert frame["width"] == 1280

    def test_get_by_uuid_not_found(self, frame_repo):
        """Test get_by_uuid returns None for non-existent frame."""
        fake_uuid = str(uuid.uuid4())
        frame = frame_repo.get_by_uuid(fake_uuid)
        assert frame is None

    def test_get_by_video(self, video_repo, frame_repo):
        """Test getting all frames for a video."""
        video_uuid = video_repo.insert(
            video_id="allframes",
            domain="youtube",
            cache_path="/cache/af",
        )

        # Insert out of order to verify ordering
        frame_repo.insert(video_uuid=video_uuid, timestamp=30.0, extraction_type="drill", file_path="f30.jpg")
        frame_repo.insert(video_uuid=video_uuid, timestamp=10.0, extraction_type="drill", file_path="f10.jpg")
        frame_repo.insert(video_uuid=video_uuid, timestamp=20.0, extraction_type="drill", file_path="f20.jpg")

        frames = frame_repo.get_by_video(video_uuid)
        assert len(frames) == 3
        # Should be ordered by timestamp
        assert frames[0]["timestamp"] == 10.0
        assert frames[1]["timestamp"] == 20.0
        assert frames[2]["timestamp"] == 30.0

    def test_get_by_video_empty(self, video_repo, frame_repo):
        """Test get_by_video returns empty list when no frames."""
        video_uuid = video_repo.insert(
            video_id="noframes",
            domain="youtube",
            cache_path="/cache/nf",
        )

        frames = frame_repo.get_by_video(video_uuid)
        assert frames == []

    def test_get_by_scene(self, video_repo, frame_repo):
        """Test getting all frames for a specific scene."""
        video_uuid = video_repo.insert(
            video_id="sceneframes",
            domain="youtube",
            cache_path="/cache/sf",
        )

        frame_repo.insert(video_uuid=video_uuid, timestamp=5.0, extraction_type="keyframe", file_path="k0.jpg", scene_id=0)
        frame_repo.insert(video_uuid=video_uuid, timestamp=35.0, extraction_type="keyframe", file_path="k1.jpg", scene_id=1)
        frame_repo.insert(video_uuid=video_uuid, timestamp=45.0, extraction_type="keyframe", file_path="k1b.jpg", scene_id=1)

        frames = frame_repo.get_by_scene(video_uuid, 1)
        assert len(frames) == 2
        assert all(f["scene_id"] == 1 for f in frames)

    def test_get_by_type(self, video_repo, frame_repo):
        """Test getting all frames of a specific type."""
        video_uuid = video_repo.insert(
            video_id="typeframes",
            domain="youtube",
            cache_path="/cache/tf",
        )

        frame_repo.insert(video_uuid=video_uuid, timestamp=0.0, extraction_type="drill", file_path="d1.jpg")
        frame_repo.insert(video_uuid=video_uuid, timestamp=10.0, extraction_type="drill", file_path="d2.jpg")
        frame_repo.insert(video_uuid=video_uuid, timestamp=5.0, extraction_type="hq", file_path="hq1.jpg")

        drill_frames = frame_repo.get_by_type(video_uuid, "drill")
        assert len(drill_frames) == 2
        assert all(f["extraction_type"] == "drill" for f in drill_frames)

        hq_frames = frame_repo.get_by_type(video_uuid, "hq")
        assert len(hq_frames) == 1


class TestFrameRepositoryThumbnail:
    """Tests for FrameRepository thumbnail methods."""

    def test_get_thumbnail(self, video_repo, frame_repo):
        """Test getting the thumbnail frame."""
        video_uuid = video_repo.insert(
            video_id="thumbvid",
            domain="youtube",
            cache_path="/cache/tv",
        )

        frame_repo.insert(
            video_uuid=video_uuid,
            timestamp=0.0,
            extraction_type="drill",
            file_path="normal.jpg",
            is_thumbnail=False,
        )
        frame_repo.insert(
            video_uuid=video_uuid,
            timestamp=15.0,
            extraction_type="thumbnail",
            file_path="thumb.jpg",
            is_thumbnail=True,
        )

        thumb = frame_repo.get_thumbnail(video_uuid)
        assert thumb is not None
        assert thumb["is_thumbnail"] == 1
        assert thumb["file_path"] == "thumb.jpg"

    def test_get_thumbnail_none(self, video_repo, frame_repo):
        """Test get_thumbnail returns None when no thumbnail exists."""
        video_uuid = video_repo.insert(
            video_id="nothumb",
            domain="youtube",
            cache_path="/cache/nt",
        )

        frame_repo.insert(
            video_uuid=video_uuid,
            timestamp=0.0,
            extraction_type="drill",
            file_path="normal.jpg",
            is_thumbnail=False,
        )

        thumb = frame_repo.get_thumbnail(video_uuid)
        assert thumb is None

    def test_set_thumbnail(self, video_repo, frame_repo):
        """Test set_thumbnail() designates a frame as thumbnail."""
        video_uuid = video_repo.insert(
            video_id="setthumb",
            domain="youtube",
            cache_path="/cache/st",
        )

        first_uuid = frame_repo.insert(
            video_uuid=video_uuid,
            timestamp=0.0,
            extraction_type="drill",
            file_path="first.jpg",
            is_thumbnail=True,
        )
        second_uuid = frame_repo.insert(
            video_uuid=video_uuid,
            timestamp=15.0,
            extraction_type="drill",
            file_path="second.jpg",
            is_thumbnail=False,
        )

        result = frame_repo.set_thumbnail(second_uuid)
        assert result is True

        # Second should now be thumbnail
        thumb = frame_repo.get_thumbnail(video_uuid)
        assert thumb["id"] == second_uuid

        # First should no longer be thumbnail
        first = frame_repo.get_by_uuid(first_uuid)
        assert first["is_thumbnail"] == 0

    def test_set_thumbnail_nonexistent_returns_false(self, frame_repo):
        """Test set_thumbnail returns False for non-existent frame."""
        fake_uuid = str(uuid.uuid4())
        result = frame_repo.set_thumbnail(fake_uuid)
        assert result is False


class TestFrameRepositoryKeyframes:
    """Tests for FrameRepository.get_keyframes()."""

    def test_get_keyframes(self, video_repo, frame_repo):
        """Test getting all keyframes for a video."""
        video_uuid = video_repo.insert(
            video_id="keyframevid",
            domain="youtube",
            cache_path="/cache/kfv",
        )

        frame_repo.insert(video_uuid=video_uuid, timestamp=5.0, extraction_type="keyframe", file_path="kf1.jpg", scene_id=0)
        frame_repo.insert(video_uuid=video_uuid, timestamp=35.0, extraction_type="keyframe", file_path="kf2.jpg", scene_id=1)
        frame_repo.insert(video_uuid=video_uuid, timestamp=10.0, extraction_type="drill", file_path="d1.jpg")

        keyframes = frame_repo.get_keyframes(video_uuid)
        assert len(keyframes) == 2
        assert all(f["extraction_type"] == "keyframe" for f in keyframes)

    def test_get_keyframes_by_scene(self, video_repo, frame_repo):
        """Test getting keyframes filtered by scene."""
        video_uuid = video_repo.insert(
            video_id="kfscene",
            domain="youtube",
            cache_path="/cache/kfs",
        )

        frame_repo.insert(video_uuid=video_uuid, timestamp=5.0, extraction_type="keyframe", file_path="kf0.jpg", scene_id=0)
        frame_repo.insert(video_uuid=video_uuid, timestamp=35.0, extraction_type="keyframe", file_path="kf1a.jpg", scene_id=1)
        frame_repo.insert(video_uuid=video_uuid, timestamp=40.0, extraction_type="keyframe", file_path="kf1b.jpg", scene_id=1)

        keyframes = frame_repo.get_keyframes(video_uuid, scene_id=1)
        assert len(keyframes) == 2
        assert all(f["scene_id"] == 1 for f in keyframes)


class TestFrameRepositoryCountByType:
    """Tests for FrameRepository.count_by_type()."""

    def test_count_by_type(self, video_repo, frame_repo):
        """Test counting frames by extraction type."""
        video_uuid = video_repo.insert(
            video_id="counttypes",
            domain="youtube",
            cache_path="/cache/ct",
        )

        frame_repo.insert(video_uuid=video_uuid, timestamp=0.0, extraction_type="drill", file_path="d1.jpg")
        frame_repo.insert(video_uuid=video_uuid, timestamp=10.0, extraction_type="drill", file_path="d2.jpg")
        frame_repo.insert(video_uuid=video_uuid, timestamp=20.0, extraction_type="drill", file_path="d3.jpg")
        frame_repo.insert(video_uuid=video_uuid, timestamp=5.0, extraction_type="hq", file_path="hq1.jpg")
        frame_repo.insert(video_uuid=video_uuid, timestamp=15.0, extraction_type="keyframe", file_path="kf1.jpg")
        frame_repo.insert(video_uuid=video_uuid, timestamp=25.0, extraction_type="keyframe", file_path="kf2.jpg")

        counts = frame_repo.count_by_type(video_uuid)
        assert counts == {"drill": 3, "hq": 1, "keyframe": 2}

    def test_count_by_type_empty(self, video_repo, frame_repo):
        """Test count_by_type returns empty dict for video with no frames."""
        video_uuid = video_repo.insert(
            video_id="emptycount",
            domain="youtube",
            cache_path="/cache/ec",
        )

        counts = frame_repo.count_by_type(video_uuid)
        assert counts == {}


class TestFrameRepositoryDelete:
    """Tests for FrameRepository delete methods."""

    def test_delete_by_uuid(self, video_repo, frame_repo):
        """Test deleting a frame by UUID."""
        video_uuid = video_repo.insert(
            video_id="delframe",
            domain="youtube",
            cache_path="/cache/df",
        )

        frame_uuid = frame_repo.insert(
            video_uuid=video_uuid,
            timestamp=0.0,
            extraction_type="drill",
            file_path="frame.jpg",
        )

        result = frame_repo.delete(frame_uuid)
        assert result is True

        frame = frame_repo.get_by_uuid(frame_uuid)
        assert frame is None

    def test_delete_nonexistent(self, frame_repo):
        """Test deleting non-existent frame returns False."""
        fake_uuid = str(uuid.uuid4())
        result = frame_repo.delete(fake_uuid)
        assert result is False

    def test_delete_by_video(self, video_repo, frame_repo):
        """Test deleting all frames for a video."""
        video_uuid = video_repo.insert(
            video_id="delallframes",
            domain="youtube",
            cache_path="/cache/daf",
        )

        frame_repo.insert(video_uuid=video_uuid, timestamp=0.0, extraction_type="drill", file_path="f1.jpg")
        frame_repo.insert(video_uuid=video_uuid, timestamp=10.0, extraction_type="drill", file_path="f2.jpg")
        frame_repo.insert(video_uuid=video_uuid, timestamp=20.0, extraction_type="hq", file_path="f3.jpg")

        count = frame_repo.delete_by_video(video_uuid)
        assert count == 3

        frames = frame_repo.get_by_video(video_uuid)
        assert frames == []

    def test_delete_by_type(self, video_repo, frame_repo):
        """Test deleting all frames of a specific type."""
        video_uuid = video_repo.insert(
            video_id="deltypeframes",
            domain="youtube",
            cache_path="/cache/dtf",
        )

        frame_repo.insert(video_uuid=video_uuid, timestamp=0.0, extraction_type="drill", file_path="d1.jpg")
        frame_repo.insert(video_uuid=video_uuid, timestamp=10.0, extraction_type="drill", file_path="d2.jpg")
        frame_repo.insert(video_uuid=video_uuid, timestamp=5.0, extraction_type="hq", file_path="hq1.jpg")

        count = frame_repo.delete_by_type(video_uuid, "drill")
        assert count == 2

        remaining = frame_repo.get_by_video(video_uuid)
        assert len(remaining) == 1
        assert remaining[0]["extraction_type"] == "hq"

    def test_cascade_delete_on_video_delete(self, video_repo, frame_repo):
        """Test that deleting a video cascades to frames."""
        video_uuid = video_repo.insert(
            video_id="cascadeframe",
            domain="youtube",
            cache_path="/cache/cf",
        )

        frame_uuid = frame_repo.insert(
            video_uuid=video_uuid,
            timestamp=0.0,
            extraction_type="drill",
            file_path="frame.jpg",
        )

        video_repo.delete("cascadeframe")

        frame = frame_repo.get_by_uuid(frame_uuid)
        assert frame is None


# ============================================================
# VisualDescriptionRepository Tests
# ============================================================


class TestVisualDescriptionRepositoryInsert:
    """Tests for VisualDescriptionRepository.insert()."""

    def test_insert_visual_description(self, video_repo, visual_description_repo):
        """Test inserting a visual description."""
        video_uuid = video_repo.insert(
            video_id="visualvid",
            domain="youtube",
            cache_path="/cache/vv",
        )

        vd_uuid = visual_description_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            description="A person typing on a laptop in a dimly lit room.",
            provider="anthropic",
            file_path="scenes/scene_000/visual.json",
        )

        assert vd_uuid is not None
        assert len(vd_uuid) == 36

    def test_insert_minimal(self, video_repo, visual_description_repo):
        """Test inserting with only required fields."""
        video_uuid = video_repo.insert(
            video_id="minvisual",
            domain="youtube",
            cache_path="/cache/mv",
        )

        vd_uuid = visual_description_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            description="Test description",
        )

        vd = visual_description_repo.get_by_scene(video_uuid, 0)
        assert vd is not None
        assert vd["provider"] is None
        assert vd["file_path"] is None

    def test_insert_negative_scene_id_raises(self, video_repo, visual_description_repo):
        """Test that negative scene_id raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="negsceneid",
            domain="youtube",
            cache_path="/cache/ns",
        )

        with pytest.raises(ValueError, match="scene_id must be >= 0"):
            visual_description_repo.insert(
                video_uuid=video_uuid,
                scene_id=-1,
                description="Test",
            )

    def test_insert_empty_description_raises(self, video_repo, visual_description_repo):
        """Test that empty description raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="emptydesc",
            domain="youtube",
            cache_path="/cache/ed",
        )

        with pytest.raises(ValueError, match="description cannot be empty"):
            visual_description_repo.insert(
                video_uuid=video_uuid,
                scene_id=0,
                description="",
            )

    def test_insert_whitespace_description_raises(self, video_repo, visual_description_repo):
        """Test that whitespace-only description raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="wsdesc",
            domain="youtube",
            cache_path="/cache/ws",
        )

        with pytest.raises(ValueError, match="description cannot be empty"):
            visual_description_repo.insert(
                video_uuid=video_uuid,
                scene_id=0,
                description="   \t\n   ",
            )

    def test_insert_duplicate_scene_raises(self, video_repo, visual_description_repo):
        """Test that duplicate (video_uuid, scene_id) raises IntegrityError."""
        video_uuid = video_repo.insert(
            video_id="dupvd",
            domain="youtube",
            cache_path="/cache/dv",
        )

        visual_description_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            description="First description",
        )

        with pytest.raises(sqlite3.IntegrityError):
            visual_description_repo.insert(
                video_uuid=video_uuid,
                scene_id=0,
                description="Second description",
            )


class TestVisualDescriptionRepositoryGet:
    """Tests for VisualDescriptionRepository get methods."""

    def test_get_by_video(self, video_repo, visual_description_repo):
        """Test getting all visual descriptions for a video."""
        video_uuid = video_repo.insert(
            video_id="getvd",
            domain="youtube",
            cache_path="/cache/gv",
        )

        # Insert out of order
        visual_description_repo.insert(video_uuid=video_uuid, scene_id=2, description="Scene 2")
        visual_description_repo.insert(video_uuid=video_uuid, scene_id=0, description="Scene 0")
        visual_description_repo.insert(video_uuid=video_uuid, scene_id=1, description="Scene 1")

        descriptions = visual_description_repo.get_by_video(video_uuid)
        assert len(descriptions) == 3
        # Should be ordered by scene_id
        assert descriptions[0]["scene_id"] == 0
        assert descriptions[1]["scene_id"] == 1
        assert descriptions[2]["scene_id"] == 2

    def test_get_by_video_empty(self, video_repo, visual_description_repo):
        """Test get_by_video returns empty list when no descriptions."""
        video_uuid = video_repo.insert(
            video_id="novd",
            domain="youtube",
            cache_path="/cache/nv",
        )

        descriptions = visual_description_repo.get_by_video(video_uuid)
        assert descriptions == []

    def test_get_by_scene(self, video_repo, visual_description_repo):
        """Test getting a specific scene's visual description."""
        video_uuid = video_repo.insert(
            video_id="scenevd",
            domain="youtube",
            cache_path="/cache/sv",
        )

        visual_description_repo.insert(video_uuid=video_uuid, scene_id=0, description="Scene 0")
        visual_description_repo.insert(video_uuid=video_uuid, scene_id=1, description="Scene 1")

        vd = visual_description_repo.get_by_scene(video_uuid, 1)
        assert vd is not None
        assert vd["description"] == "Scene 1"

    def test_get_by_scene_not_found(self, video_repo, visual_description_repo):
        """Test get_by_scene returns None for non-existent scene."""
        video_uuid = video_repo.insert(
            video_id="noscene",
            domain="youtube",
            cache_path="/cache/ns",
        )

        vd = visual_description_repo.get_by_scene(video_uuid, 99)
        assert vd is None


class TestVisualDescriptionRepositoryFTS:
    """Tests for VisualDescriptionRepository full-text search."""

    def test_search_fts(self, video_repo, visual_description_repo):
        """Test FTS search on visual descriptions."""
        video_uuid = video_repo.insert(
            video_id="ftsvd",
            domain="youtube",
            cache_path="/cache/fv",
            title="FTS Test Video",
        )

        visual_description_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            description="A developer writing Python code on a laptop.",
        )

        results = visual_description_repo.search_fts("Python")
        assert len(results) == 1
        assert results[0]["video_natural_id"] == "ftsvd"
        assert results[0]["video_title"] == "FTS Test Video"

    def test_search_fts_multiple_scenes(self, video_repo, visual_description_repo):
        """Test FTS search across multiple scenes."""
        video_uuid = video_repo.insert(
            video_id="multifts",
            domain="youtube",
            cache_path="/cache/mf",
        )

        visual_description_repo.insert(video_uuid=video_uuid, scene_id=0, description="A cat sleeping on a couch.")
        visual_description_repo.insert(video_uuid=video_uuid, scene_id=1, description="A dog playing in the yard.")
        visual_description_repo.insert(video_uuid=video_uuid, scene_id=2, description="Another cat climbing a tree.")

        results = visual_description_repo.search_fts("cat")
        assert len(results) == 2
        scene_ids = {r["scene_id"] for r in results}
        assert scene_ids == {0, 2}

    def test_search_fts_no_results(self, video_repo, visual_description_repo):
        """Test FTS search returns empty list when no matches."""
        video_uuid = video_repo.insert(
            video_id="nofts",
            domain="youtube",
            cache_path="/cache/nf",
        )

        visual_description_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            description="A person cooking dinner.",
        )

        results = visual_description_repo.search_fts("programming")
        assert results == []


class TestVisualDescriptionRepositoryDelete:
    """Tests for VisualDescriptionRepository delete methods."""

    def test_delete(self, video_repo, visual_description_repo):
        """Test deleting a visual description by UUID."""
        video_uuid = video_repo.insert(
            video_id="delvd",
            domain="youtube",
            cache_path="/cache/dv",
        )

        vd_uuid = visual_description_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            description="Test",
        )

        result = visual_description_repo.delete(vd_uuid)
        assert result is True

        vd = visual_description_repo.get_by_scene(video_uuid, 0)
        assert vd is None

    def test_delete_nonexistent(self, visual_description_repo):
        """Test deleting non-existent record returns False."""
        fake_uuid = str(uuid.uuid4())
        result = visual_description_repo.delete(fake_uuid)
        assert result is False

    def test_delete_by_video(self, video_repo, visual_description_repo):
        """Test deleting all visual descriptions for a video."""
        video_uuid = video_repo.insert(
            video_id="delallvd",
            domain="youtube",
            cache_path="/cache/da",
        )

        visual_description_repo.insert(video_uuid=video_uuid, scene_id=0, description="Scene 0")
        visual_description_repo.insert(video_uuid=video_uuid, scene_id=1, description="Scene 1")

        count = visual_description_repo.delete_by_video(video_uuid)
        assert count == 2

        descriptions = visual_description_repo.get_by_video(video_uuid)
        assert descriptions == []

    def test_cascade_delete_on_video_delete(self, video_repo, visual_description_repo):
        """Test that deleting a video cascades to visual descriptions."""
        video_uuid = video_repo.insert(
            video_id="cascadevd",
            domain="youtube",
            cache_path="/cache/cv",
        )

        vd_uuid = visual_description_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            description="Test",
        )

        video_repo.delete("cascadevd")

        # Cannot get by UUID since we don't have get_by_uuid, use get_by_scene
        # Actually, let's check count
        count = visual_description_repo.count_by_video(video_uuid)
        assert count == 0


# ============================================================
# TechnicalContentRepository Tests
# ============================================================


class TestTechnicalContentRepositoryInsert:
    """Tests for TechnicalContentRepository.insert()."""

    def test_insert_technical_content(self, video_repo, technical_content_repo):
        """Test inserting technical content."""
        video_uuid = video_repo.insert(
            video_id="techvid",
            domain="youtube",
            cache_path="/cache/tv",
        )

        tc_uuid = technical_content_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            has_code=True,
            has_text=True,
            provider="anthropic",
            ocr_text="def hello_world():\n    print('Hello, World!')",
            code_language="python",
            file_path="scenes/scene_000/technical.json",
        )

        assert tc_uuid is not None
        assert len(tc_uuid) == 36

    def test_insert_minimal(self, video_repo, technical_content_repo):
        """Test inserting with only required fields."""
        video_uuid = video_repo.insert(
            video_id="mintech",
            domain="youtube",
            cache_path="/cache/mt",
        )

        tc_uuid = technical_content_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            has_code=False,
            has_text=False,
        )

        tc = technical_content_repo.get_by_scene(video_uuid, 0)
        assert tc is not None
        assert tc["has_code"] == 0
        assert tc["has_text"] == 0
        assert tc["ocr_text"] is None
        assert tc["code_language"] is None

    def test_insert_negative_scene_id_raises(self, video_repo, technical_content_repo):
        """Test that negative scene_id raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="negscenetech",
            domain="youtube",
            cache_path="/cache/nst",
        )

        with pytest.raises(ValueError, match="scene_id must be >= 0"):
            technical_content_repo.insert(
                video_uuid=video_uuid,
                scene_id=-1,
                has_code=False,
                has_text=False,
            )

    def test_insert_duplicate_scene_raises(self, video_repo, technical_content_repo):
        """Test that duplicate (video_uuid, scene_id) raises IntegrityError."""
        video_uuid = video_repo.insert(
            video_id="duptc",
            domain="youtube",
            cache_path="/cache/dt",
        )

        technical_content_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            has_code=False,
            has_text=False,
        )

        with pytest.raises(sqlite3.IntegrityError):
            technical_content_repo.insert(
                video_uuid=video_uuid,
                scene_id=0,
                has_code=True,
                has_text=True,
            )


class TestTechnicalContentRepositoryGet:
    """Tests for TechnicalContentRepository get methods."""

    def test_get_by_video(self, video_repo, technical_content_repo):
        """Test getting all technical content for a video."""
        video_uuid = video_repo.insert(
            video_id="gettc",
            domain="youtube",
            cache_path="/cache/gt",
        )

        # Insert out of order
        technical_content_repo.insert(video_uuid=video_uuid, scene_id=2, has_code=True, has_text=False)
        technical_content_repo.insert(video_uuid=video_uuid, scene_id=0, has_code=False, has_text=True)
        technical_content_repo.insert(video_uuid=video_uuid, scene_id=1, has_code=True, has_text=True)

        records = technical_content_repo.get_by_video(video_uuid)
        assert len(records) == 3
        # Should be ordered by scene_id
        assert records[0]["scene_id"] == 0
        assert records[1]["scene_id"] == 1
        assert records[2]["scene_id"] == 2

    def test_get_by_scene(self, video_repo, technical_content_repo):
        """Test getting a specific scene's technical content."""
        video_uuid = video_repo.insert(
            video_id="scenetc",
            domain="youtube",
            cache_path="/cache/st",
        )

        technical_content_repo.insert(video_uuid=video_uuid, scene_id=0, has_code=False, has_text=False)
        technical_content_repo.insert(
            video_uuid=video_uuid,
            scene_id=1,
            has_code=True,
            has_text=True,
            code_language="javascript",
        )

        tc = technical_content_repo.get_by_scene(video_uuid, 1)
        assert tc is not None
        assert tc["has_code"] == 1
        assert tc["code_language"] == "javascript"

    def test_get_scenes_with_code(self, video_repo, technical_content_repo):
        """Test getting all scenes with code."""
        video_uuid = video_repo.insert(
            video_id="codevid",
            domain="youtube",
            cache_path="/cache/cv",
        )

        technical_content_repo.insert(video_uuid=video_uuid, scene_id=0, has_code=False, has_text=True)
        technical_content_repo.insert(video_uuid=video_uuid, scene_id=1, has_code=True, has_text=True)
        technical_content_repo.insert(video_uuid=video_uuid, scene_id=2, has_code=True, has_text=False)
        technical_content_repo.insert(video_uuid=video_uuid, scene_id=3, has_code=False, has_text=False)

        code_scenes = technical_content_repo.get_scenes_with_code(video_uuid)
        assert len(code_scenes) == 2
        scene_ids = {r["scene_id"] for r in code_scenes}
        assert scene_ids == {1, 2}

    def test_get_scenes_with_text(self, video_repo, technical_content_repo):
        """Test getting all scenes with text."""
        video_uuid = video_repo.insert(
            video_id="textvid",
            domain="youtube",
            cache_path="/cache/tv2",
        )

        technical_content_repo.insert(video_uuid=video_uuid, scene_id=0, has_code=False, has_text=True)
        technical_content_repo.insert(video_uuid=video_uuid, scene_id=1, has_code=True, has_text=True)
        technical_content_repo.insert(video_uuid=video_uuid, scene_id=2, has_code=True, has_text=False)

        text_scenes = technical_content_repo.get_scenes_with_text(video_uuid)
        assert len(text_scenes) == 2
        scene_ids = {r["scene_id"] for r in text_scenes}
        assert scene_ids == {0, 1}


class TestTechnicalContentRepositoryFTS:
    """Tests for TechnicalContentRepository full-text search."""

    def test_search_fts(self, video_repo, technical_content_repo):
        """Test FTS search on OCR text."""
        video_uuid = video_repo.insert(
            video_id="ftstc",
            domain="youtube",
            cache_path="/cache/ft",
            title="FTS Tech Video",
        )

        technical_content_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            has_code=True,
            has_text=True,
            ocr_text="function authenticate(user, password) { return true; }",
        )

        results = technical_content_repo.search_fts("authenticate")
        assert len(results) == 1
        assert results[0]["video_natural_id"] == "ftstc"
        assert results[0]["video_title"] == "FTS Tech Video"

    def test_search_fts_no_results(self, video_repo, technical_content_repo):
        """Test FTS search returns empty list when no matches."""
        video_uuid = video_repo.insert(
            video_id="noftsc",
            domain="youtube",
            cache_path="/cache/nft",
        )

        technical_content_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            has_code=True,
            has_text=True,
            ocr_text="const x = 5;",
        )

        results = technical_content_repo.search_fts("authentication")
        assert results == []


class TestTechnicalContentRepositoryDelete:
    """Tests for TechnicalContentRepository delete methods."""

    def test_delete(self, video_repo, technical_content_repo):
        """Test deleting technical content by UUID."""
        video_uuid = video_repo.insert(
            video_id="deltc",
            domain="youtube",
            cache_path="/cache/dtc",
        )

        tc_uuid = technical_content_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            has_code=False,
            has_text=False,
        )

        result = technical_content_repo.delete(tc_uuid)
        assert result is True

        tc = technical_content_repo.get_by_scene(video_uuid, 0)
        assert tc is None

    def test_delete_by_video(self, video_repo, technical_content_repo):
        """Test deleting all technical content for a video."""
        video_uuid = video_repo.insert(
            video_id="delalltc",
            domain="youtube",
            cache_path="/cache/datc",
        )

        technical_content_repo.insert(video_uuid=video_uuid, scene_id=0, has_code=False, has_text=False)
        technical_content_repo.insert(video_uuid=video_uuid, scene_id=1, has_code=True, has_text=True)

        count = technical_content_repo.delete_by_video(video_uuid)
        assert count == 2

        records = technical_content_repo.get_by_video(video_uuid)
        assert records == []


# ============================================================
# AudioDescriptionRepository Tests
# ============================================================


class TestAudioDescriptionRepositoryInsert:
    """Tests for AudioDescriptionRepository.insert()."""

    def test_insert_audio_description(self, video_repo, audio_description_repo):
        """Test inserting an audio description."""
        video_uuid = video_repo.insert(
            video_id="advid",
            domain="youtube",
            cache_path="/cache/av",
        )

        ad_uuid = audio_description_repo.insert(
            video_uuid=video_uuid,
            format_="vtt",
            source="generated",
            file_path="audio.ad.vtt",
            provider="anthropic",
        )

        assert ad_uuid is not None
        assert len(ad_uuid) == 36

    def test_insert_all_valid_sources(self, video_repo, audio_description_repo):
        """Test all valid sources can be inserted."""
        video_uuid = video_repo.insert(
            video_id="allsources",
            domain="youtube",
            cache_path="/cache/as",
        )

        for source in AudioDescriptionRepository.VALID_SOURCES:
            ad_uuid = audio_description_repo.insert(
                video_uuid=video_uuid,
                format_="txt",
                source=source,
                file_path=f"audio_{source}.txt",
            )
            assert ad_uuid is not None

    def test_insert_all_valid_formats(self, video_repo, audio_description_repo):
        """Test all valid formats can be inserted."""
        video_uuid = video_repo.insert(
            video_id="allformats",
            domain="youtube",
            cache_path="/cache/af",
        )

        for format_ in AudioDescriptionRepository.VALID_FORMATS:
            ad_uuid = audio_description_repo.insert(
                video_uuid=video_uuid,
                format_=format_,
                source="generated",
                file_path=f"audio.ad.{format_}",
            )
            assert ad_uuid is not None

    def test_insert_invalid_source_raises(self, video_repo, audio_description_repo):
        """Test that invalid source raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="badsrc",
            domain="youtube",
            cache_path="/cache/bs",
        )

        with pytest.raises(ValueError, match="Invalid source"):
            audio_description_repo.insert(
                video_uuid=video_uuid,
                format_="vtt",
                source="invalid_source",
                file_path="audio.vtt",
            )

    def test_insert_invalid_format_raises(self, video_repo, audio_description_repo):
        """Test that invalid format raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="badfmt",
            domain="youtube",
            cache_path="/cache/bf",
        )

        with pytest.raises(ValueError, match="Invalid format"):
            audio_description_repo.insert(
                video_uuid=video_uuid,
                format_="srt",  # Invalid
                source="generated",
                file_path="audio.srt",
            )


class TestAudioDescriptionRepositoryGet:
    """Tests for AudioDescriptionRepository get methods."""

    def test_get_by_video(self, video_repo, audio_description_repo):
        """Test getting all audio descriptions for a video."""
        video_uuid = video_repo.insert(
            video_id="getad",
            domain="youtube",
            cache_path="/cache/ga",
        )

        audio_description_repo.insert(video_uuid=video_uuid, format_="vtt", source="generated", file_path="a.vtt")
        audio_description_repo.insert(video_uuid=video_uuid, format_="txt", source="compiled", file_path="a.txt")

        records = audio_description_repo.get_by_video(video_uuid)
        assert len(records) == 2

    def test_get_by_uuid(self, video_repo, audio_description_repo):
        """Test getting an audio description by UUID."""
        video_uuid = video_repo.insert(
            video_id="getuuidad",
            domain="youtube",
            cache_path="/cache/gua",
        )

        ad_uuid = audio_description_repo.insert(
            video_uuid=video_uuid,
            format_="vtt",
            source="source_track",
            file_path="source.vtt",
        )

        ad = audio_description_repo.get_by_uuid(ad_uuid)
        assert ad is not None
        assert ad["source"] == "source_track"

    def test_get_by_format(self, video_repo, audio_description_repo):
        """Test getting an audio description by format."""
        video_uuid = video_repo.insert(
            video_id="getfmtad",
            domain="youtube",
            cache_path="/cache/gfa",
        )

        audio_description_repo.insert(video_uuid=video_uuid, format_="vtt", source="generated", file_path="a.vtt")
        audio_description_repo.insert(video_uuid=video_uuid, format_="txt", source="generated", file_path="a.txt")

        ad = audio_description_repo.get_by_format(video_uuid, "txt")
        assert ad is not None
        assert ad["file_path"] == "a.txt"

    def test_has_ad_true(self, video_repo, audio_description_repo):
        """Test has_ad returns True when audio description exists."""
        video_uuid = video_repo.insert(
            video_id="hasad",
            domain="youtube",
            cache_path="/cache/ha",
        )

        audio_description_repo.insert(
            video_uuid=video_uuid,
            format_="vtt",
            source="generated",
            file_path="a.vtt",
        )

        assert audio_description_repo.has_ad(video_uuid) is True

    def test_has_ad_false(self, video_repo, audio_description_repo):
        """Test has_ad returns False when no audio description exists."""
        video_uuid = video_repo.insert(
            video_id="noad",
            domain="youtube",
            cache_path="/cache/na",
        )

        assert audio_description_repo.has_ad(video_uuid) is False


class TestAudioDescriptionRepositoryDelete:
    """Tests for AudioDescriptionRepository delete methods."""

    def test_delete(self, video_repo, audio_description_repo):
        """Test deleting an audio description by UUID."""
        video_uuid = video_repo.insert(
            video_id="delad",
            domain="youtube",
            cache_path="/cache/da",
        )

        ad_uuid = audio_description_repo.insert(
            video_uuid=video_uuid,
            format_="vtt",
            source="generated",
            file_path="a.vtt",
        )

        result = audio_description_repo.delete(ad_uuid)
        assert result is True

        ad = audio_description_repo.get_by_uuid(ad_uuid)
        assert ad is None

    def test_delete_by_video(self, video_repo, audio_description_repo):
        """Test deleting all audio descriptions for a video."""
        video_uuid = video_repo.insert(
            video_id="delallad",
            domain="youtube",
            cache_path="/cache/daa",
        )

        audio_description_repo.insert(video_uuid=video_uuid, format_="vtt", source="generated", file_path="a.vtt")
        audio_description_repo.insert(video_uuid=video_uuid, format_="txt", source="compiled", file_path="a.txt")

        count = audio_description_repo.delete_by_video(video_uuid)
        assert count == 2

        records = audio_description_repo.get_by_video(video_uuid)
        assert records == []


# ============================================================
# NarrativeRepository Tests
# ============================================================


class TestNarrativeRepositoryInsert:
    """Tests for NarrativeRepository.insert()."""

    def test_insert_narrative(self, video_repo, narrative_repo):
        """Test inserting a narrative structure."""
        video_uuid = video_repo.insert(
            video_id="narrativevid",
            domain="youtube",
            cache_path="/cache/nv",
        )

        ns_uuid = narrative_repo.insert(
            video_uuid=video_uuid,
            video_type="coding_tutorial",
            section_count=5,
            file_path="structure/narrative.json",
        )

        assert ns_uuid is not None
        assert len(ns_uuid) == 36

    def test_insert_minimal(self, video_repo, narrative_repo):
        """Test inserting with only required fields."""
        video_uuid = video_repo.insert(
            video_id="minnr",
            domain="youtube",
            cache_path="/cache/mn",
        )

        ns_uuid = narrative_repo.insert(video_uuid=video_uuid)

        ns = narrative_repo.get_by_video(video_uuid)
        assert ns is not None
        assert ns["video_type"] is None
        assert ns["section_count"] is None
        assert ns["file_path"] is None

    def test_insert_all_valid_video_types(self, video_repo, narrative_repo):
        """Test all valid video types can be inserted."""
        for i, video_type in enumerate(NarrativeRepository.VALID_VIDEO_TYPES):
            video_uuid = video_repo.insert(
                video_id=f"type{i}",
                domain="youtube",
                cache_path=f"/cache/t{i}",
            )

            ns_uuid = narrative_repo.insert(
                video_uuid=video_uuid,
                video_type=video_type,
            )
            assert ns_uuid is not None

    def test_insert_invalid_video_type_raises(self, video_repo, narrative_repo):
        """Test that invalid video_type raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="badtype",
            domain="youtube",
            cache_path="/cache/bt",
        )

        with pytest.raises(ValueError, match="Invalid video_type"):
            narrative_repo.insert(
                video_uuid=video_uuid,
                video_type="invalid_type",
            )

    def test_insert_negative_section_count_raises(self, video_repo, narrative_repo):
        """Test that negative section_count raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="negsect",
            domain="youtube",
            cache_path="/cache/ns",
        )

        with pytest.raises(ValueError, match="section_count must be >= 0"):
            narrative_repo.insert(
                video_uuid=video_uuid,
                section_count=-1,
            )

    def test_insert_duplicate_video_raises(self, video_repo, narrative_repo):
        """Test that inserting twice for same video raises IntegrityError."""
        video_uuid = video_repo.insert(
            video_id="dupnr",
            domain="youtube",
            cache_path="/cache/dn",
        )

        narrative_repo.insert(video_uuid=video_uuid, video_type="lecture")

        with pytest.raises(sqlite3.IntegrityError):
            narrative_repo.insert(video_uuid=video_uuid, video_type="demo")


class TestNarrativeRepositoryGet:
    """Tests for NarrativeRepository get methods."""

    def test_get_by_video(self, video_repo, narrative_repo):
        """Test getting narrative structure by video UUID."""
        video_uuid = video_repo.insert(
            video_id="getnr",
            domain="youtube",
            cache_path="/cache/gn",
        )

        narrative_repo.insert(
            video_uuid=video_uuid,
            video_type="presentation",
            section_count=3,
        )

        ns = narrative_repo.get_by_video(video_uuid)
        assert ns is not None
        assert ns["video_type"] == "presentation"
        assert ns["section_count"] == 3

    def test_get_by_video_not_found(self, video_repo, narrative_repo):
        """Test get_by_video returns None when not found."""
        video_uuid = video_repo.insert(
            video_id="nonr",
            domain="youtube",
            cache_path="/cache/nn",
        )

        ns = narrative_repo.get_by_video(video_uuid)
        assert ns is None

    def test_get_by_uuid(self, video_repo, narrative_repo):
        """Test getting narrative structure by UUID."""
        video_uuid = video_repo.insert(
            video_id="getuuidnr",
            domain="youtube",
            cache_path="/cache/gun",
        )

        ns_uuid = narrative_repo.insert(video_uuid=video_uuid, video_type="interview")

        ns = narrative_repo.get_by_uuid(ns_uuid)
        assert ns is not None
        assert ns["video_type"] == "interview"

    def test_exists(self, video_repo, narrative_repo):
        """Test exists() method."""
        video_uuid = video_repo.insert(
            video_id="existnr",
            domain="youtube",
            cache_path="/cache/en",
        )

        assert narrative_repo.exists(video_uuid) is False

        narrative_repo.insert(video_uuid=video_uuid)

        assert narrative_repo.exists(video_uuid) is True

    def test_list_by_type(self, video_repo, narrative_repo):
        """Test listing narratives by video type."""
        for i in range(3):
            video_uuid = video_repo.insert(
                video_id=f"tut{i}",
                domain="youtube",
                cache_path=f"/cache/tut{i}",
                title=f"Tutorial {i}",
            )
            narrative_repo.insert(video_uuid=video_uuid, video_type="coding_tutorial")

        video_uuid = video_repo.insert(
            video_id="lecture0",
            domain="youtube",
            cache_path="/cache/lec0",
        )
        narrative_repo.insert(video_uuid=video_uuid, video_type="lecture")

        tutorials = narrative_repo.list_by_type("coding_tutorial")
        assert len(tutorials) == 3


class TestNarrativeRepositoryUpdate:
    """Tests for NarrativeRepository.update()."""

    def test_update(self, video_repo, narrative_repo):
        """Test updating a narrative structure."""
        video_uuid = video_repo.insert(
            video_id="updnr",
            domain="youtube",
            cache_path="/cache/un",
        )

        narrative_repo.insert(video_uuid=video_uuid, video_type="other")

        result = narrative_repo.update(
            video_uuid,
            video_type="demo",
            section_count=4,
        )
        assert result is True

        ns = narrative_repo.get_by_video(video_uuid)
        assert ns["video_type"] == "demo"
        assert ns["section_count"] == 4

    def test_update_not_found(self, narrative_repo):
        """Test update returns False for non-existent record."""
        fake_uuid = str(uuid.uuid4())
        result = narrative_repo.update(fake_uuid, video_type="demo")
        assert result is False


class TestNarrativeRepositoryDelete:
    """Tests for NarrativeRepository.delete()."""

    def test_delete(self, video_repo, narrative_repo):
        """Test deleting a narrative structure."""
        video_uuid = video_repo.insert(
            video_id="delnr",
            domain="youtube",
            cache_path="/cache/dnr",
        )

        narrative_repo.insert(video_uuid=video_uuid, video_type="vlog")

        result = narrative_repo.delete(video_uuid)
        assert result is True

        ns = narrative_repo.get_by_video(video_uuid)
        assert ns is None

    def test_delete_not_found(self, narrative_repo):
        """Test delete returns False for non-existent record."""
        fake_uuid = str(uuid.uuid4())
        result = narrative_repo.delete(fake_uuid)
        assert result is False


# ============================================================
# CodeEvolutionRepository Tests
# ============================================================


class TestCodeEvolutionRepositoryInsert:
    """Tests for CodeEvolutionRepository.insert()."""

    def test_insert_code_evolution(self, video_repo, code_evolution_repo):
        """Test inserting a code evolution record."""
        video_uuid = video_repo.insert(
            video_id="codevid",
            domain="youtube",
            cache_path="/cache/cv",
        )

        ce_uuid = code_evolution_repo.insert(
            video_uuid=video_uuid,
            files_tracked=5,
            total_changes=42,
            file_path="entities/code_evolution.json",
        )

        assert ce_uuid is not None
        assert len(ce_uuid) == 36

    def test_insert_minimal(self, video_repo, code_evolution_repo):
        """Test inserting with only required fields."""
        video_uuid = video_repo.insert(
            video_id="mince",
            domain="youtube",
            cache_path="/cache/mce",
        )

        ce_uuid = code_evolution_repo.insert(video_uuid=video_uuid)

        ce = code_evolution_repo.get_by_video(video_uuid)
        assert ce is not None
        assert ce["files_tracked"] is None
        assert ce["total_changes"] is None
        assert ce["file_path"] is None

    def test_insert_negative_files_tracked_raises(self, video_repo, code_evolution_repo):
        """Test that negative files_tracked raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="negfiles",
            domain="youtube",
            cache_path="/cache/nf",
        )

        with pytest.raises(ValueError, match="files_tracked must be >= 0"):
            code_evolution_repo.insert(
                video_uuid=video_uuid,
                files_tracked=-1,
            )

    def test_insert_negative_total_changes_raises(self, video_repo, code_evolution_repo):
        """Test that negative total_changes raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="negchanges",
            domain="youtube",
            cache_path="/cache/nc",
        )

        with pytest.raises(ValueError, match="total_changes must be >= 0"):
            code_evolution_repo.insert(
                video_uuid=video_uuid,
                total_changes=-1,
            )

    def test_insert_duplicate_video_raises(self, video_repo, code_evolution_repo):
        """Test that inserting twice for same video raises IntegrityError."""
        video_uuid = video_repo.insert(
            video_id="dupce",
            domain="youtube",
            cache_path="/cache/dce",
        )

        code_evolution_repo.insert(video_uuid=video_uuid, files_tracked=1)

        with pytest.raises(sqlite3.IntegrityError):
            code_evolution_repo.insert(video_uuid=video_uuid, files_tracked=2)


class TestCodeEvolutionRepositoryGet:
    """Tests for CodeEvolutionRepository get methods."""

    def test_get_by_video(self, video_repo, code_evolution_repo):
        """Test getting code evolution by video UUID."""
        video_uuid = video_repo.insert(
            video_id="getce",
            domain="youtube",
            cache_path="/cache/gce",
        )

        code_evolution_repo.insert(
            video_uuid=video_uuid,
            files_tracked=3,
            total_changes=15,
        )

        ce = code_evolution_repo.get_by_video(video_uuid)
        assert ce is not None
        assert ce["files_tracked"] == 3
        assert ce["total_changes"] == 15

    def test_get_by_video_not_found(self, video_repo, code_evolution_repo):
        """Test get_by_video returns None when not found."""
        video_uuid = video_repo.insert(
            video_id="noce",
            domain="youtube",
            cache_path="/cache/nce",
        )

        ce = code_evolution_repo.get_by_video(video_uuid)
        assert ce is None

    def test_get_by_uuid(self, video_repo, code_evolution_repo):
        """Test getting code evolution by UUID."""
        video_uuid = video_repo.insert(
            video_id="getuuidce",
            domain="youtube",
            cache_path="/cache/guce",
        )

        ce_uuid = code_evolution_repo.insert(video_uuid=video_uuid, files_tracked=10)

        ce = code_evolution_repo.get_by_uuid(ce_uuid)
        assert ce is not None
        assert ce["files_tracked"] == 10

    def test_exists(self, video_repo, code_evolution_repo):
        """Test exists() method."""
        video_uuid = video_repo.insert(
            video_id="existce",
            domain="youtube",
            cache_path="/cache/ece",
        )

        assert code_evolution_repo.exists(video_uuid) is False

        code_evolution_repo.insert(video_uuid=video_uuid)

        assert code_evolution_repo.exists(video_uuid) is True

    def test_list_all(self, video_repo, code_evolution_repo):
        """Test listing all code evolution records."""
        for i in range(3):
            video_uuid = video_repo.insert(
                video_id=f"list{i}",
                domain="youtube",
                cache_path=f"/cache/list{i}",
                title=f"Video {i}",
            )
            code_evolution_repo.insert(
                video_uuid=video_uuid,
                files_tracked=i + 1,
            )

        all_ce = code_evolution_repo.list_all()
        assert len(all_ce) == 3
        assert all("video_natural_id" in ce for ce in all_ce)
        assert all("video_title" in ce for ce in all_ce)


class TestCodeEvolutionRepositoryUpdate:
    """Tests for CodeEvolutionRepository.update()."""

    def test_update(self, video_repo, code_evolution_repo):
        """Test updating a code evolution record."""
        video_uuid = video_repo.insert(
            video_id="updce",
            domain="youtube",
            cache_path="/cache/uce",
        )

        code_evolution_repo.insert(video_uuid=video_uuid, files_tracked=1)

        result = code_evolution_repo.update(
            video_uuid,
            files_tracked=5,
            total_changes=20,
        )
        assert result is True

        ce = code_evolution_repo.get_by_video(video_uuid)
        assert ce["files_tracked"] == 5
        assert ce["total_changes"] == 20

    def test_update_not_found(self, code_evolution_repo):
        """Test update returns False for non-existent record."""
        fake_uuid = str(uuid.uuid4())
        result = code_evolution_repo.update(fake_uuid, files_tracked=5)
        assert result is False


class TestCodeEvolutionRepositoryDelete:
    """Tests for CodeEvolutionRepository.delete()."""

    def test_delete(self, video_repo, code_evolution_repo):
        """Test deleting a code evolution record."""
        video_uuid = video_repo.insert(
            video_id="delce",
            domain="youtube",
            cache_path="/cache/dce2",
        )

        code_evolution_repo.insert(video_uuid=video_uuid, files_tracked=2)

        result = code_evolution_repo.delete(video_uuid)
        assert result is True

        ce = code_evolution_repo.get_by_video(video_uuid)
        assert ce is None

    def test_delete_not_found(self, code_evolution_repo):
        """Test delete returns False for non-existent record."""
        fake_uuid = str(uuid.uuid4())
        result = code_evolution_repo.delete(fake_uuid)
        assert result is False


# ============================================================
# EntityRepository Tests
# ============================================================


@pytest.fixture
def entity_repo(db):
    """Create an EntityRepository instance."""
    return EntityRepository(db)


class TestEntityRepositoryInsert:
    """Tests for EntityRepository.insert_entity()."""

    def test_insert_entity(self, entity_repo):
        """Test inserting a new entity."""
        uuid_ = entity_repo.insert_entity("Python", "technology")
        assert uuid_ is not None
        assert len(uuid_) == 36

    def test_insert_entity_upsert_returns_existing(self, entity_repo):
        """Test insert_entity returns existing UUID for duplicate."""
        uuid1 = entity_repo.insert_entity("Python", "technology")
        uuid2 = entity_repo.insert_entity("Python", "technology")
        assert uuid1 == uuid2

    def test_insert_entity_different_types_allowed(self, entity_repo):
        """Test same name with different types creates separate entities."""
        uuid1 = entity_repo.insert_entity("Apple", "technology")
        uuid2 = entity_repo.insert_entity("Apple", "organization")
        assert uuid1 != uuid2

    def test_insert_entity_all_valid_types(self, entity_repo):
        """Test all valid entity types can be inserted."""
        types = ["object", "concept", "person", "technology", "organization"]
        for i, etype in enumerate(types):
            uuid_ = entity_repo.insert_entity(f"Entity{i}", etype)
            assert uuid_ is not None

    def test_insert_entity_invalid_type_raises(self, entity_repo):
        """Test invalid entity_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid entity_type"):
            entity_repo.insert_entity("Something", "invalid_type")

    def test_insert_entity_empty_name_raises(self, entity_repo):
        """Test empty name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            entity_repo.insert_entity("", "technology")

    def test_insert_entity_whitespace_name_raises(self, entity_repo):
        """Test whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            entity_repo.insert_entity("   ", "technology")

    def test_insert_entity_strips_whitespace(self, entity_repo):
        """Test name is stripped of leading/trailing whitespace."""
        uuid1 = entity_repo.insert_entity("  Python  ", "technology")
        uuid2 = entity_repo.insert_entity("Python", "technology")
        assert uuid1 == uuid2


class TestEntityRepositoryAppearance:
    """Tests for EntityRepository.insert_appearance()."""

    def test_insert_appearance(self, video_repo, entity_repo):
        """Test inserting an entity appearance."""
        video_uuid = video_repo.insert(
            video_id="testvid",
            domain="youtube",
            cache_path="/cache/tv",
        )
        entity_uuid = entity_repo.insert_entity("Python", "technology")

        appearance_uuid = entity_repo.insert_appearance(
            entity_uuid=entity_uuid,
            video_uuid=video_uuid,
            scene_id=0,
            timestamp=30.5,
            score=0.9,
        )
        assert appearance_uuid is not None
        assert len(appearance_uuid) == 36

    def test_insert_appearance_upsert_returns_existing(self, video_repo, entity_repo):
        """Test duplicate appearance returns existing UUID."""
        video_uuid = video_repo.insert(
            video_id="testvid2",
            domain="youtube",
            cache_path="/cache/tv2",
        )
        entity_uuid = entity_repo.insert_entity("Django", "technology")

        uuid1 = entity_repo.insert_appearance(entity_uuid, video_uuid, 0, 10.0)
        uuid2 = entity_repo.insert_appearance(entity_uuid, video_uuid, 0, 20.0)
        assert uuid1 == uuid2

    def test_insert_appearance_negative_scene_raises(self, entity_repo):
        """Test negative scene_id raises ValueError."""
        with pytest.raises(ValueError, match="scene_id must be >= 0"):
            entity_repo.insert_appearance("fake", "fake", -1, 0.0)

    def test_insert_appearance_negative_timestamp_raises(self, entity_repo):
        """Test negative timestamp raises ValueError."""
        with pytest.raises(ValueError, match="timestamp must be >= 0"):
            entity_repo.insert_appearance("fake", "fake", 0, -1.0)

    def test_insert_appearance_invalid_score_raises(self, entity_repo):
        """Test score out of range raises ValueError."""
        with pytest.raises(ValueError, match="score must be between 0 and 1"):
            entity_repo.insert_appearance("fake", "fake", 0, 0.0, score=1.5)


class TestEntityRepositoryVideoSummary:
    """Tests for EntityRepository.insert_video_summary()."""

    def test_insert_video_summary(self, video_repo, entity_repo):
        """Test inserting a video summary."""
        video_uuid = video_repo.insert(
            video_id="sumvid",
            domain="youtube",
            cache_path="/cache/sv",
        )
        entity_uuid = entity_repo.insert_entity("Machine Learning", "concept")

        summary_uuid = entity_repo.insert_video_summary(
            entity_uuid=entity_uuid,
            video_uuid=video_uuid,
            frequency=5,
            avg_score=0.85,
        )
        assert summary_uuid is not None
        assert len(summary_uuid) == 36

    def test_insert_video_summary_upsert(self, video_repo, entity_repo):
        """Test upserting video summary updates existing."""
        video_uuid = video_repo.insert(
            video_id="upsertvid",
            domain="youtube",
            cache_path="/cache/uv",
        )
        entity_uuid = entity_repo.insert_entity("AI", "concept")

        uuid1 = entity_repo.insert_video_summary(entity_uuid, video_uuid, 3)
        uuid2 = entity_repo.insert_video_summary(entity_uuid, video_uuid, 10, avg_score=0.9)
        assert uuid1 == uuid2

        # Verify the update
        entities = entity_repo.get_video_entities(video_uuid)
        assert len(entities) == 1
        assert entities[0]["frequency"] == 10

    def test_insert_video_summary_invalid_frequency_raises(self, entity_repo):
        """Test frequency < 1 raises ValueError."""
        with pytest.raises(ValueError, match="frequency must be >= 1"):
            entity_repo.insert_video_summary("fake", "fake", 0)

    def test_insert_video_summary_invalid_score_raises(self, entity_repo):
        """Test avg_score out of range raises ValueError."""
        with pytest.raises(ValueError, match="avg_score must be between 0 and 1"):
            entity_repo.insert_video_summary("fake", "fake", 1, avg_score=-0.1)


class TestEntityRepositoryQueries:
    """Tests for EntityRepository query methods."""

    def test_get_by_uuid(self, entity_repo):
        """Test getting entity by UUID."""
        uuid_ = entity_repo.insert_entity("React", "technology")
        entity = entity_repo.get_by_uuid(uuid_)
        assert entity is not None
        assert entity["name"] == "React"
        assert entity["entity_type"] == "technology"

    def test_get_by_uuid_not_found(self, entity_repo):
        """Test get_by_uuid returns None for non-existent."""
        fake_uuid = str(uuid.uuid4())
        entity = entity_repo.get_by_uuid(fake_uuid)
        assert entity is None

    def test_get_by_name_and_type(self, entity_repo):
        """Test getting entity by name and type."""
        entity_repo.insert_entity("JavaScript", "technology")
        entity = entity_repo.get_by_name_and_type("JavaScript", "technology")
        assert entity is not None
        assert entity["name"] == "JavaScript"

    def test_find_by_name(self, entity_repo):
        """Test finding entities by name substring."""
        entity_repo.insert_entity("Python", "technology")
        entity_repo.insert_entity("Python Tutorial", "concept")
        entity_repo.insert_entity("JavaScript", "technology")

        results = entity_repo.find_by_name("python")
        assert len(results) == 2

    def test_get_video_entities(self, video_repo, entity_repo):
        """Test getting all entities for a video."""
        video_uuid = video_repo.insert(
            video_id="getvid",
            domain="youtube",
            cache_path="/cache/gv",
        )
        e1 = entity_repo.insert_entity("Python", "technology")
        e2 = entity_repo.insert_entity("Django", "technology")

        entity_repo.insert_video_summary(e1, video_uuid, 5)
        entity_repo.insert_video_summary(e2, video_uuid, 3)

        entities = entity_repo.get_video_entities(video_uuid)
        assert len(entities) == 2
        # Ordered by frequency desc
        assert entities[0]["name"] == "Python"
        assert entities[0]["frequency"] == 5

    def test_get_entity_videos(self, video_repo, entity_repo):
        """Test getting all videos containing an entity."""
        v1 = video_repo.insert(
            video_id="vid1", domain="youtube", cache_path="/cache/v1", title="Video 1"
        )
        v2 = video_repo.insert(
            video_id="vid2", domain="youtube", cache_path="/cache/v2", title="Video 2"
        )

        entity_uuid = entity_repo.insert_entity("Python", "technology")
        entity_repo.insert_video_summary(entity_uuid, v1, 10)
        entity_repo.insert_video_summary(entity_uuid, v2, 5)

        videos = entity_repo.get_entity_videos("Python")
        assert len(videos) == 2
        # Ordered by frequency desc
        assert videos[0]["video_id"] == "vid1"


class TestEntityRepositoryFindRelated:
    """Tests for EntityRepository.find_related_videos()."""

    def test_find_related_videos(self, video_repo, entity_repo):
        """Test finding videos related to a query."""
        v1 = video_repo.insert(
            video_id="relv1", domain="youtube", cache_path="/c/r1", title="Python Basics"
        )
        v2 = video_repo.insert(
            video_id="relv2", domain="youtube", cache_path="/c/r2", title="JavaScript Guide"
        )

        e1 = entity_repo.insert_entity("Python", "technology")
        e2 = entity_repo.insert_entity("JavaScript", "technology")

        entity_repo.insert_video_summary(e1, v1, 5)
        entity_repo.insert_video_summary(e2, v2, 3)

        results = entity_repo.find_related_videos("python")
        assert len(results) == 1
        assert results[0]["video_id"] == "relv1"
        assert results[0]["matched_term"] == "Python"

    def test_find_related_videos_empty_query(self, entity_repo):
        """Test empty query returns no results."""
        results = entity_repo.find_related_videos("")
        assert len(results) == 0


class TestEntityRepositoryConnections:
    """Tests for EntityRepository.get_connections()."""

    def test_get_connections(self, video_repo, entity_repo):
        """Test finding connected videos via shared entities."""
        v1 = video_repo.insert(
            video_id="connv1", domain="youtube", cache_path="/c/cv1"
        )
        v2 = video_repo.insert(
            video_id="connv2", domain="youtube", cache_path="/c/cv2"
        )
        v3 = video_repo.insert(
            video_id="connv3", domain="youtube", cache_path="/c/cv3"
        )

        e1 = entity_repo.insert_entity("Python", "technology")
        e2 = entity_repo.insert_entity("JavaScript", "technology")

        # v1 and v2 share Python
        entity_repo.insert_video_summary(e1, v1, 5)
        entity_repo.insert_video_summary(e1, v2, 3)
        # v3 has only JavaScript
        entity_repo.insert_video_summary(e2, v3, 2)

        connections = entity_repo.get_connections(v1)
        assert v2 in connections
        assert v3 not in connections
        assert v1 not in connections  # Should not include self

    def test_get_connections_no_connections(self, video_repo, entity_repo):
        """Test video with no connections returns empty list."""
        video_uuid = video_repo.insert(
            video_id="lonely", domain="youtube", cache_path="/c/lonely"
        )
        entity_uuid = entity_repo.insert_entity("Unique", "concept")
        entity_repo.insert_video_summary(entity_uuid, video_uuid, 1)

        connections = entity_repo.get_connections(video_uuid)
        assert connections == []


class TestEntityRepositoryDelete:
    """Tests for EntityRepository delete methods."""

    def test_delete_entity(self, entity_repo):
        """Test deleting an entity."""
        uuid_ = entity_repo.insert_entity("ToDelete", "concept")
        result = entity_repo.delete_entity(uuid_)
        assert result is True

        entity = entity_repo.get_by_uuid(uuid_)
        assert entity is None

    def test_delete_entity_not_found(self, entity_repo):
        """Test delete returns False for non-existent entity."""
        fake_uuid = str(uuid.uuid4())
        result = entity_repo.delete_entity(fake_uuid)
        assert result is False

    def test_delete_video_entities(self, video_repo, entity_repo):
        """Test deleting all entity data for a video."""
        video_uuid = video_repo.insert(
            video_id="delvid", domain="youtube", cache_path="/c/dv"
        )
        e1 = entity_repo.insert_entity("E1", "concept")
        e2 = entity_repo.insert_entity("E2", "concept")

        entity_repo.insert_appearance(e1, video_uuid, 0, 10.0)
        entity_repo.insert_appearance(e2, video_uuid, 1, 20.0)
        entity_repo.insert_video_summary(e1, video_uuid, 1)
        entity_repo.insert_video_summary(e2, video_uuid, 1)

        count = entity_repo.delete_video_entities(video_uuid)
        assert count == 4  # 2 appearances + 2 summaries

        # Verify deletion
        entities = entity_repo.get_video_entities(video_uuid)
        assert len(entities) == 0


class TestEntityRepositoryStats:
    """Tests for EntityRepository.get_stats()."""

    def test_get_stats(self, video_repo, entity_repo):
        """Test getting entity statistics."""
        video_uuid = video_repo.insert(
            video_id="statvid", domain="youtube", cache_path="/c/stat"
        )
        e1 = entity_repo.insert_entity("Stat1", "concept")

        entity_repo.insert_appearance(e1, video_uuid, 0, 0.0)
        entity_repo.insert_video_summary(e1, video_uuid, 1)

        stats = entity_repo.get_stats()
        assert stats["entity_count"] >= 1
        assert stats["appearance_count"] >= 1
        assert stats["summary_count"] >= 1
        assert stats["video_count"] >= 1


# ============================================================
# QARepository Tests
# ============================================================


@pytest.fixture
def qa_repo(db):
    """Create a QARepository instance."""
    return QARepository(db)


class TestQARepositoryInsert:
    """Tests for QARepository.insert()."""

    def test_insert_qa(self, video_repo, qa_repo):
        """Test inserting a Q&A pair."""
        video_uuid = video_repo.insert(
            video_id="qavid",
            domain="youtube",
            cache_path="/cache/qa",
        )

        qa_uuid = qa_repo.insert(
            video_uuid=video_uuid,
            question="What is Python?",
            answer="A programming language.",
        )
        assert qa_uuid is not None
        assert len(qa_uuid) == 36

    def test_insert_qa_with_scenes(self, video_repo, qa_repo):
        """Test inserting Q&A with scene associations."""
        video_uuid = video_repo.insert(
            video_id="qascene",
            domain="youtube",
            cache_path="/cache/qas",
        )

        qa_uuid = qa_repo.insert(
            video_uuid=video_uuid,
            question="When does the demo start?",
            answer="At 5:30",
            scene_ids=[2, 3, 4],
        )

        qa = qa_repo.get_by_uuid(qa_uuid)
        assert qa["scene_ids"] == [2, 3, 4]

    def test_insert_qa_empty_question_raises(self, qa_repo):
        """Test empty question raises ValueError."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            qa_repo.insert("fake", "", "answer")

    def test_insert_qa_empty_answer_raises(self, qa_repo):
        """Test empty answer raises ValueError."""
        with pytest.raises(ValueError, match="Answer cannot be empty"):
            qa_repo.insert("fake", "question", "")

    def test_insert_qa_negative_scene_raises(self, video_repo, qa_repo):
        """Test negative scene_id raises ValueError."""
        video_uuid = video_repo.insert(
            video_id="qaneg",
            domain="youtube",
            cache_path="/cache/qan",
        )
        with pytest.raises(ValueError, match="scene_id must be >= 0"):
            qa_repo.insert(video_uuid, "Q?", "A", scene_ids=[-1])


class TestQARepositoryGet:
    """Tests for QARepository query methods."""

    def test_get_by_uuid(self, video_repo, qa_repo):
        """Test getting Q&A by UUID."""
        video_uuid = video_repo.insert(
            video_id="getqa",
            domain="youtube",
            cache_path="/cache/gqa",
        )

        qa_uuid = qa_repo.insert(video_uuid, "Q?", "A!")

        qa = qa_repo.get_by_uuid(qa_uuid)
        assert qa is not None
        assert qa["question"] == "Q?"
        assert qa["answer"] == "A!"

    def test_get_by_uuid_not_found(self, qa_repo):
        """Test get_by_uuid returns None for non-existent."""
        fake_uuid = str(uuid.uuid4())
        qa = qa_repo.get_by_uuid(fake_uuid)
        assert qa is None

    def test_get_by_video(self, video_repo, qa_repo):
        """Test getting all Q&A for a video."""
        video_uuid = video_repo.insert(
            video_id="multqa",
            domain="youtube",
            cache_path="/cache/mqa",
        )

        qa_repo.insert(video_uuid, "Q1?", "A1")
        qa_repo.insert(video_uuid, "Q2?", "A2")
        qa_repo.insert(video_uuid, "Q3?", "A3")

        qas = qa_repo.get_by_video(video_uuid)
        assert len(qas) == 3


class TestQARepositoryFTS:
    """Tests for QARepository.search_fts()."""

    def test_search_fts_question(self, video_repo, qa_repo):
        """Test FTS search finds question matches."""
        video_uuid = video_repo.insert(
            video_id="ftsqa1",
            domain="youtube",
            cache_path="/cache/fq1",
            title="Test Video",
        )

        qa_repo.insert(video_uuid, "How do I install Python?", "Use pip")
        qa_repo.insert(video_uuid, "What is JavaScript?", "A language")

        results = qa_repo.search_fts("Python")
        assert len(results) == 1
        assert "Python" in results[0]["question"]

    def test_search_fts_answer(self, video_repo, qa_repo):
        """Test FTS search finds answer matches."""
        video_uuid = video_repo.insert(
            video_id="ftsqa2",
            domain="youtube",
            cache_path="/cache/fq2",
            title="Test Video 2",
        )

        qa_repo.insert(video_uuid, "How to install?", "Use pip install Django")

        results = qa_repo.search_fts("Django")
        assert len(results) == 1
        assert "Django" in results[0]["answer"]


class TestQARepositoryScenes:
    """Tests for QARepository scene association methods."""

    def test_get_for_scene(self, video_repo, qa_repo):
        """Test getting Q&A for a specific scene."""
        video_uuid = video_repo.insert(
            video_id="sceneqa",
            domain="youtube",
            cache_path="/cache/sqa",
        )

        qa_repo.insert(video_uuid, "Q1?", "A1", scene_ids=[0, 1])
        qa_repo.insert(video_uuid, "Q2?", "A2", scene_ids=[1, 2])
        qa_repo.insert(video_uuid, "Q3?", "A3", scene_ids=[3])

        scene1_qas = qa_repo.get_for_scene(video_uuid, 1)
        assert len(scene1_qas) == 2

        scene3_qas = qa_repo.get_for_scene(video_uuid, 3)
        assert len(scene3_qas) == 1

    def test_add_scene_association(self, video_repo, qa_repo):
        """Test adding scene association to existing Q&A."""
        video_uuid = video_repo.insert(
            video_id="addsceneqa",
            domain="youtube",
            cache_path="/cache/asqa",
        )

        qa_uuid = qa_repo.insert(video_uuid, "Q?", "A")
        result = qa_repo.add_scene_association(qa_uuid, 5)
        assert result is True

        qa = qa_repo.get_by_uuid(qa_uuid)
        assert 5 in qa["scene_ids"]

    def test_add_scene_association_duplicate(self, video_repo, qa_repo):
        """Test adding duplicate scene association returns False."""
        video_uuid = video_repo.insert(
            video_id="dupsceneqa",
            domain="youtube",
            cache_path="/cache/dsqa",
        )

        qa_uuid = qa_repo.insert(video_uuid, "Q?", "A", scene_ids=[1])
        result = qa_repo.add_scene_association(qa_uuid, 1)
        assert result is False

    def test_remove_scene_association(self, video_repo, qa_repo):
        """Test removing scene association."""
        video_uuid = video_repo.insert(
            video_id="rmsceneqa",
            domain="youtube",
            cache_path="/cache/rsqa",
        )

        qa_uuid = qa_repo.insert(video_uuid, "Q?", "A", scene_ids=[1, 2, 3])
        result = qa_repo.remove_scene_association(qa_uuid, 2)
        assert result is True

        qa = qa_repo.get_by_uuid(qa_uuid)
        assert 2 not in qa["scene_ids"]


class TestQARepositoryDelete:
    """Tests for QARepository delete methods."""

    def test_delete(self, video_repo, qa_repo):
        """Test deleting a Q&A pair."""
        video_uuid = video_repo.insert(
            video_id="delqa",
            domain="youtube",
            cache_path="/cache/dqa",
        )

        qa_uuid = qa_repo.insert(video_uuid, "Q?", "A")
        result = qa_repo.delete(qa_uuid)
        assert result is True

        qa = qa_repo.get_by_uuid(qa_uuid)
        assert qa is None

    def test_delete_not_found(self, qa_repo):
        """Test delete returns False for non-existent."""
        fake_uuid = str(uuid.uuid4())
        result = qa_repo.delete(fake_uuid)
        assert result is False

    def test_delete_by_video(self, video_repo, qa_repo):
        """Test deleting all Q&A for a video."""
        video_uuid = video_repo.insert(
            video_id="delallqa",
            domain="youtube",
            cache_path="/cache/daqa",
        )

        qa_repo.insert(video_uuid, "Q1?", "A1")
        qa_repo.insert(video_uuid, "Q2?", "A2")

        count = qa_repo.delete_by_video(video_uuid)
        assert count == 2

    def test_count_by_video(self, video_repo, qa_repo):
        """Test counting Q&A pairs for a video."""
        video_uuid = video_repo.insert(
            video_id="cntqa",
            domain="youtube",
            cache_path="/cache/cqa",
        )

        qa_repo.insert(video_uuid, "Q1?", "A1")
        qa_repo.insert(video_uuid, "Q2?", "A2")

        count = qa_repo.count_by_video(video_uuid)
        assert count == 2


# ============================================================
# ObservationRepository Tests
# ============================================================


@pytest.fixture
def observation_repo(db):
    """Create an ObservationRepository instance."""
    return ObservationRepository(db)


class TestObservationRepositoryInsert:
    """Tests for ObservationRepository.insert()."""

    def test_insert_observation(self, video_repo, observation_repo):
        """Test inserting an observation."""
        video_uuid = video_repo.insert(
            video_id="obsvid",
            domain="youtube",
            cache_path="/cache/obs",
        )

        obs_uuid = observation_repo.insert(
            video_uuid=video_uuid,
            scene_id=0,
            obs_type="visual",
            content="Screen shows code editor",
        )
        assert obs_uuid is not None
        assert len(obs_uuid) == 36

    def test_insert_observation_negative_scene_raises(self, observation_repo):
        """Test negative scene_id raises ValueError."""
        with pytest.raises(ValueError, match="scene_id must be >= 0"):
            observation_repo.insert("fake", -1, "note", "content")

    def test_insert_observation_empty_type_raises(self, observation_repo):
        """Test empty type raises ValueError."""
        with pytest.raises(ValueError, match="type cannot be empty"):
            observation_repo.insert("fake", 0, "", "content")

    def test_insert_observation_empty_content_raises(self, observation_repo):
        """Test empty content raises ValueError."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            observation_repo.insert("fake", 0, "note", "")


class TestObservationRepositoryGet:
    """Tests for ObservationRepository query methods."""

    def test_get_by_uuid(self, video_repo, observation_repo):
        """Test getting observation by UUID."""
        video_uuid = video_repo.insert(
            video_id="getobs",
            domain="youtube",
            cache_path="/cache/gobs",
        )

        obs_uuid = observation_repo.insert(video_uuid, 0, "note", "Test observation")

        obs = observation_repo.get_by_uuid(obs_uuid)
        assert obs is not None
        assert obs["type"] == "note"
        assert obs["content"] == "Test observation"

    def test_get_by_uuid_not_found(self, observation_repo):
        """Test get_by_uuid returns None for non-existent."""
        fake_uuid = str(uuid.uuid4())
        obs = observation_repo.get_by_uuid(fake_uuid)
        assert obs is None

    def test_get_by_video(self, video_repo, observation_repo):
        """Test getting all observations for a video."""
        video_uuid = video_repo.insert(
            video_id="multobs",
            domain="youtube",
            cache_path="/cache/mobs",
        )

        observation_repo.insert(video_uuid, 0, "visual", "Obs 1")
        observation_repo.insert(video_uuid, 1, "technical", "Obs 2")
        observation_repo.insert(video_uuid, 2, "note", "Obs 3")

        observations = observation_repo.get_by_video(video_uuid)
        assert len(observations) == 3

    def test_get_by_scene(self, video_repo, observation_repo):
        """Test getting observations for a specific scene."""
        video_uuid = video_repo.insert(
            video_id="sceneobs",
            domain="youtube",
            cache_path="/cache/sobs",
        )

        observation_repo.insert(video_uuid, 0, "visual", "Scene 0 - 1")
        observation_repo.insert(video_uuid, 0, "technical", "Scene 0 - 2")
        observation_repo.insert(video_uuid, 1, "note", "Scene 1")

        scene0_obs = observation_repo.get_by_scene(video_uuid, 0)
        assert len(scene0_obs) == 2

        scene1_obs = observation_repo.get_by_scene(video_uuid, 1)
        assert len(scene1_obs) == 1

    def test_get_by_type(self, video_repo, observation_repo):
        """Test getting observations by type."""
        video_uuid = video_repo.insert(
            video_id="typeobs",
            domain="youtube",
            cache_path="/cache/tobs",
        )

        observation_repo.insert(video_uuid, 0, "visual", "V1")
        observation_repo.insert(video_uuid, 1, "visual", "V2")
        observation_repo.insert(video_uuid, 2, "technical", "T1")

        visual_obs = observation_repo.get_by_type(video_uuid, "visual")
        assert len(visual_obs) == 2


class TestObservationRepositoryDelete:
    """Tests for ObservationRepository delete methods."""

    def test_delete(self, video_repo, observation_repo):
        """Test deleting an observation."""
        video_uuid = video_repo.insert(
            video_id="delobs",
            domain="youtube",
            cache_path="/cache/dobs",
        )

        obs_uuid = observation_repo.insert(video_uuid, 0, "note", "To delete")
        result = observation_repo.delete(obs_uuid)
        assert result is True

        obs = observation_repo.get_by_uuid(obs_uuid)
        assert obs is None

    def test_delete_not_found(self, observation_repo):
        """Test delete returns False for non-existent."""
        fake_uuid = str(uuid.uuid4())
        result = observation_repo.delete(fake_uuid)
        assert result is False

    def test_delete_by_video(self, video_repo, observation_repo):
        """Test deleting all observations for a video."""
        video_uuid = video_repo.insert(
            video_id="delallobs",
            domain="youtube",
            cache_path="/cache/daobs",
        )

        observation_repo.insert(video_uuid, 0, "note", "Obs 1")
        observation_repo.insert(video_uuid, 1, "note", "Obs 2")

        count = observation_repo.delete_by_video(video_uuid)
        assert count == 2

    def test_delete_by_scene(self, video_repo, observation_repo):
        """Test deleting all observations for a scene."""
        video_uuid = video_repo.insert(
            video_id="delscnobs",
            domain="youtube",
            cache_path="/cache/dsobs",
        )

        observation_repo.insert(video_uuid, 0, "note", "S0-1")
        observation_repo.insert(video_uuid, 0, "visual", "S0-2")
        observation_repo.insert(video_uuid, 1, "note", "S1")

        count = observation_repo.delete_by_scene(video_uuid, 0)
        assert count == 2

        remaining = observation_repo.get_by_video(video_uuid)
        assert len(remaining) == 1


class TestObservationRepositoryCount:
    """Tests for ObservationRepository count methods."""

    def test_count_by_video(self, video_repo, observation_repo):
        """Test counting observations for a video."""
        video_uuid = video_repo.insert(
            video_id="cntobs",
            domain="youtube",
            cache_path="/cache/cobs",
        )

        observation_repo.insert(video_uuid, 0, "note", "Obs 1")
        observation_repo.insert(video_uuid, 1, "note", "Obs 2")

        count = observation_repo.count_by_video(video_uuid)
        assert count == 2

    def test_count_by_scene(self, video_repo, observation_repo):
        """Test counting observations for a scene."""
        video_uuid = video_repo.insert(
            video_id="cntscnobs",
            domain="youtube",
            cache_path="/cache/csobs",
        )

        observation_repo.insert(video_uuid, 0, "note", "S0-1")
        observation_repo.insert(video_uuid, 0, "visual", "S0-2")
        observation_repo.insert(video_uuid, 1, "note", "S1")

        count0 = observation_repo.count_by_scene(video_uuid, 0)
        assert count0 == 2

        count1 = observation_repo.count_by_scene(video_uuid, 1)
        assert count1 == 1
