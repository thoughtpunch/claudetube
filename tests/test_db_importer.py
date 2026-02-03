"""Tests for the auto-import module that populates SQLite from JSON caches."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from claudetube.db.connection import Database
from claudetube.db.importer import (
    _discover_video_dirs,
    _extract_timestamp_from_filename,
    _import_video,
    _normalize_entity_type,
    _normalize_method,
    auto_import,
)
from claudetube.db.migrate import run_migrations


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    db = Database(db_path)
    run_migrations(db)
    yield db
    db.close()
    db_path.unlink(missing_ok=True)


@pytest.fixture
def temp_cache():
    """Create a temporary cache directory with mock video data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_mock_video(
    cache_dir: Path,
    video_id: str,
    *,
    with_audio: bool = True,
    with_transcript: bool = True,
    with_thumbnail: bool = False,
    with_scenes: bool = False,
    domain: str = "youtube",
    title: str | None = None,
) -> Path:
    """Create a mock video directory with state.json and optional files."""
    video_dir = cache_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    # Create state.json
    state = {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "domain": domain,
        "title": title or f"Test Video {video_id}",
        "duration": 120.0,
        "duration_string": "2:00",
        "uploader": "Test Channel",
        "channel": "Test Channel",
        "upload_date": "20240101",
        "description": "Test description",
        "language": "en",
        "view_count": 1000,
        "like_count": 100,
        "transcript_complete": with_transcript,
        "transcript_source": "whisper",
        "whisper_model": "tiny",
        "has_thumbnail": with_thumbnail,
    }
    (video_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")

    if with_audio:
        # Create mock audio file
        (video_dir / "audio.mp3").write_bytes(b"fake mp3 data")

    if with_transcript:
        # Create mock transcript files
        (video_dir / "audio.srt").write_text(
            "1\n00:00:00,000 --> 00:00:10,000\nHello world\n"
        )
        (video_dir / "audio.txt").write_text("Hello world")

    if with_thumbnail:
        # Create mock thumbnail
        (video_dir / "thumbnail.jpg").write_bytes(b"fake jpg data")

    if with_scenes:
        # Create scenes directory with scenes.json
        scenes_dir = video_dir / "scenes"
        scenes_dir.mkdir()
        scenes_data = [
            {
                "scene_id": 0,
                "start_time": 0.0,
                "end_time": 60.0,
                "transcript": "First half",
            },
            {
                "scene_id": 1,
                "start_time": 60.0,
                "end_time": 120.0,
                "transcript": "Second half",
            },
        ]
        (scenes_dir / "scenes.json").write_text(
            json.dumps(scenes_data), encoding="utf-8"
        )

    return video_dir


class TestDiscoverVideoDirs:
    """Tests for _discover_video_dirs."""

    def test_empty_cache(self, temp_cache):
        """Empty cache returns no video dirs."""
        dirs = list(_discover_video_dirs(temp_cache))
        assert dirs == []

    def test_flat_structure(self, temp_cache):
        """Discovers videos in flat directory structure."""
        create_mock_video(temp_cache, "video1")
        create_mock_video(temp_cache, "video2")

        dirs = list(_discover_video_dirs(temp_cache))
        assert len(dirs) == 2
        assert {d.name for d in dirs} == {"video1", "video2"}

    def test_hierarchical_structure(self, temp_cache):
        """Discovers videos in hierarchical directory structure."""
        # Create hierarchical path
        video_dir = temp_cache / "youtube" / "channel1" / "playlist1" / "video1"
        video_dir.mkdir(parents=True)
        (video_dir / "state.json").write_text('{"video_id": "video1"}')

        dirs = list(_discover_video_dirs(temp_cache))
        assert len(dirs) == 1
        assert dirs[0].name == "video1"

    def test_skips_playlists(self, temp_cache):
        """Skips playlists directory."""
        create_mock_video(temp_cache, "video1")

        # Create a fake video in playlists directory (should be skipped)
        playlists_dir = temp_cache / "playlists" / "playlist1"
        playlists_dir.mkdir(parents=True)
        (playlists_dir / "state.json").write_text('{"video_id": "fake"}')

        dirs = list(_discover_video_dirs(temp_cache))
        assert len(dirs) == 1
        assert dirs[0].name == "video1"


class TestImportVideo:
    """Tests for _import_video."""

    def test_import_basic_video(self, temp_db, temp_cache):
        """Imports a basic video with state.json."""
        video_dir = create_mock_video(temp_cache, "test123")

        result = _import_video(video_dir, temp_cache, temp_db)
        assert result is True

        # Verify video was inserted
        cursor = temp_db.execute(
            "SELECT * FROM videos WHERE video_id = ?", ("test123",)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row["title"] == "Test Video test123"
        assert row["domain"] == "youtube"
        assert row["duration"] == 120.0

    def test_import_video_with_audio(self, temp_db, temp_cache):
        """Imports video with audio track."""
        video_dir = create_mock_video(temp_cache, "test123", with_audio=True)

        _import_video(video_dir, temp_cache, temp_db)

        # Get video UUID
        cursor = temp_db.execute(
            "SELECT id FROM videos WHERE video_id = ?", ("test123",)
        )
        video_uuid = cursor.fetchone()["id"]

        # Verify audio track
        cursor = temp_db.execute(
            "SELECT * FROM audio_tracks WHERE video_id = ?", (video_uuid,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row["format"] == "mp3"

    def test_import_video_with_transcript(self, temp_db, temp_cache):
        """Imports video with transcription."""
        video_dir = create_mock_video(
            temp_cache, "test123", with_audio=True, with_transcript=True
        )

        _import_video(video_dir, temp_cache, temp_db)

        # Get video UUID
        cursor = temp_db.execute(
            "SELECT id FROM videos WHERE video_id = ?", ("test123",)
        )
        video_uuid = cursor.fetchone()["id"]

        # Verify transcription
        cursor = temp_db.execute(
            "SELECT * FROM transcriptions WHERE video_id = ?", (video_uuid,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row["provider"] == "whisper"
        assert row["is_primary"] == 1
        assert row["full_text"] == "Hello world"

    def test_import_video_with_thumbnail(self, temp_db, temp_cache):
        """Imports video with thumbnail."""
        video_dir = create_mock_video(temp_cache, "test123", with_thumbnail=True)

        _import_video(video_dir, temp_cache, temp_db)

        # Get video UUID
        cursor = temp_db.execute(
            "SELECT id FROM videos WHERE video_id = ?", ("test123",)
        )
        video_uuid = cursor.fetchone()["id"]

        # Verify thumbnail frame
        cursor = temp_db.execute(
            "SELECT * FROM frames WHERE video_id = ? AND is_thumbnail = 1",
            (video_uuid,),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row["extraction_type"] == "thumbnail"

    def test_import_video_with_scenes(self, temp_db, temp_cache):
        """Imports video with scenes."""
        video_dir = create_mock_video(temp_cache, "test123", with_scenes=True)

        _import_video(video_dir, temp_cache, temp_db)

        # Get video UUID
        cursor = temp_db.execute(
            "SELECT id FROM videos WHERE video_id = ?", ("test123",)
        )
        video_uuid = cursor.fetchone()["id"]

        # Verify scenes
        cursor = temp_db.execute(
            "SELECT * FROM scenes WHERE video_id = ? ORDER BY scene_id", (video_uuid,)
        )
        scenes = cursor.fetchall()
        assert len(scenes) == 2
        assert scenes[0]["scene_id"] == 0
        assert scenes[0]["start_time"] == 0.0
        assert scenes[0]["end_time"] == 60.0

    def test_import_missing_state_json(self, temp_db, temp_cache):
        """Returns False if state.json is missing."""
        video_dir = temp_cache / "test123"
        video_dir.mkdir()

        result = _import_video(video_dir, temp_cache, temp_db)
        assert result is False

    def test_import_idempotent(self, temp_db, temp_cache):
        """Running import twice doesn't create duplicates."""
        video_dir = create_mock_video(temp_cache, "test123")

        # Import twice
        _import_video(video_dir, temp_cache, temp_db)
        _import_video(video_dir, temp_cache, temp_db)

        # Should only have one video
        cursor = temp_db.execute("SELECT COUNT(*) as cnt FROM videos")
        assert cursor.fetchone()["cnt"] == 1


class TestAutoImport:
    """Tests for the main auto_import function."""

    def test_auto_import_empty_cache(self, temp_db, temp_cache):
        """Auto-import on empty cache returns 0."""
        count = auto_import(temp_cache, temp_db)
        assert count == 0

    def test_auto_import_multiple_videos(self, temp_db, temp_cache):
        """Auto-imports multiple videos."""
        create_mock_video(temp_cache, "video1")
        create_mock_video(temp_cache, "video2")
        create_mock_video(temp_cache, "video3")

        count = auto_import(temp_cache, temp_db)
        assert count == 3

        # Verify all videos exist
        cursor = temp_db.execute("SELECT COUNT(*) as cnt FROM videos")
        assert cursor.fetchone()["cnt"] == 3

    def test_auto_import_nonexistent_cache(self, temp_db):
        """Auto-import on nonexistent cache returns 0."""
        count = auto_import(Path("/nonexistent/path"), temp_db)
        assert count == 0

    def test_auto_import_skips_malformed(self, temp_db, temp_cache):
        """Auto-import skips malformed videos but continues."""
        create_mock_video(temp_cache, "good_video")

        # Create malformed video (invalid JSON)
        bad_video = temp_cache / "bad_video"
        bad_video.mkdir()
        (bad_video / "state.json").write_text("not valid json")

        count = auto_import(temp_cache, temp_db)
        assert count == 1  # Only the good video


class TestHelpers:
    """Tests for helper functions."""

    def test_normalize_method(self):
        """Test method normalization."""
        assert _normalize_method("transcript") == "transcript"
        assert _normalize_method("VISUAL") == "visual"
        assert _normalize_method("chapters") == "chapters"
        assert _normalize_method("chapter") == "chapters"
        assert _normalize_method("text_based") == "transcript"
        assert _normalize_method("video_analysis") == "visual"
        assert _normalize_method(None) is None
        assert _normalize_method("unknown") is None

    def test_normalize_entity_type(self):
        """Test entity type normalization."""
        assert _normalize_entity_type("concept") == "concept"
        assert _normalize_entity_type("PERSON") == "person"
        assert _normalize_entity_type("people") == "person"
        assert _normalize_entity_type("tech") == "technology"
        assert _normalize_entity_type("library") == "technology"
        assert _normalize_entity_type("company") == "organization"
        assert _normalize_entity_type("unknown") == "concept"

    def test_extract_timestamp_from_filename(self):
        """Test timestamp extraction from filenames."""
        assert _extract_timestamp_from_filename("frame_123.5") == 123.5
        assert _extract_timestamp_from_filename("123.5") == 123.5
        assert _extract_timestamp_from_filename("frame_0001") == 1.0
        assert _extract_timestamp_from_filename("no_number") == 0.0


class TestFTSPopulation:
    """Tests that verify FTS tables are populated during import."""

    def test_transcript_fts_populated(self, temp_db, temp_cache):
        """Transcription full_text is indexed in FTS."""
        create_mock_video(
            temp_cache,
            "test123",
            with_transcript=True,
            title="FTS Test Video",
        )

        auto_import(temp_cache, temp_db)

        # Search for content
        cursor = temp_db.execute(
            """
            SELECT t.*, v.title
            FROM transcriptions_fts
            JOIN transcriptions t ON transcriptions_fts.rowid = t.rowid
            JOIN videos v ON t.video_id = v.id
            WHERE transcriptions_fts MATCH '"Hello"'
            """
        )
        results = cursor.fetchall()
        assert len(results) == 1
        assert results[0]["title"] == "FTS Test Video"

    def test_video_fts_populated(self, temp_db, temp_cache):
        """Video title/description is indexed in FTS."""
        create_mock_video(temp_cache, "test123", title="Unique Test Title")

        auto_import(temp_cache, temp_db)

        # Search for title
        cursor = temp_db.execute(
            """
            SELECT v.*
            FROM videos_fts
            JOIN videos v ON videos_fts.rowid = v.rowid
            WHERE videos_fts MATCH '"Unique"'
            """
        )
        results = cursor.fetchall()
        assert len(results) == 1
        assert results[0]["video_id"] == "test123"
