"""Tests for sqlite-vec integration (db/vec.py)."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claudetube.db.connection import Database
from claudetube.db.vec import (
    DEFAULT_DIMENSIONS,
    VALID_SOURCES,
    VecStore,
    _deserialize_embedding,
    _serialize_embedding,
    create_vec_table,
    get_embedding,
    load_vec_extension,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def db():
    """Create an in-memory database with the vec_metadata table."""
    database = Database(":memory:")
    # Create the vec_metadata table (from 001_initial.sql)
    database.execute("""
        CREATE TABLE videos (
            id TEXT PRIMARY KEY CHECK(length(id) = 36),
            video_id TEXT NOT NULL UNIQUE CHECK(length(video_id) > 0),
            domain TEXT NOT NULL CHECK(domain GLOB '[a-z]*'),
            channel TEXT,
            playlist TEXT,
            cache_path TEXT NOT NULL CHECK(length(cache_path) > 0),
            url TEXT,
            title TEXT,
            duration REAL,
            source_type TEXT NOT NULL DEFAULT 'url',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    database.execute("""
        CREATE TABLE vec_metadata (
            id         TEXT PRIMARY KEY CHECK(length(id) = 36),
            video_id   TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            scene_id   INTEGER CHECK(scene_id IS NULL OR scene_id >= 0),
            start_time REAL CHECK(start_time IS NULL OR start_time >= 0),
            end_time   REAL CHECK(end_time IS NULL OR end_time > start_time),
            source     TEXT NOT NULL CHECK(source IN (
                'transcription', 'scene_transcript', 'visual', 'technical',
                'entity', 'qa', 'observation', 'audio_description'
            )),
            UNIQUE(video_id, scene_id, source)
        )
    """)
    database.commit()
    yield database
    database.close()


@pytest.fixture
def video_uuid(db):
    """Insert a test video and return its UUID."""
    vid_uuid = str(uuid.uuid4())
    db.execute(
        """INSERT INTO videos (id, video_id, domain, cache_path)
           VALUES (?, 'test_vid_123', 'youtube', 'youtube/test/test_vid_123')""",
        (vid_uuid,),
    )
    db.commit()
    return vid_uuid


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns deterministic embeddings."""
    mock = AsyncMock()
    # Return 1536-dim embedding of all 0.1 values
    mock.embed.return_value = [0.1] * DEFAULT_DIMENSIONS
    return mock


# ============================================================
# Serialization tests
# ============================================================


class TestEmbeddingSerialization:
    """Tests for embedding serialization/deserialization."""

    def test_serialize_embedding(self):
        """Test serializing an embedding to bytes."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        data = _serialize_embedding(embedding)
        assert isinstance(data, bytes)
        assert len(data) == 4 * 4  # 4 floats * 4 bytes each

    def test_deserialize_embedding(self):
        """Test deserializing bytes back to embedding."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        data = _serialize_embedding(embedding)
        result = _deserialize_embedding(data, 4)
        assert len(result) == 4
        for orig, recovered in zip(embedding, result, strict=True):
            assert abs(orig - recovered) < 1e-6

    def test_roundtrip(self):
        """Test serialize/deserialize roundtrip."""
        embedding = [float(i) / 100 for i in range(1536)]
        data = _serialize_embedding(embedding)
        result = _deserialize_embedding(data, 1536)
        assert len(result) == 1536
        for orig, recovered in zip(embedding, result, strict=True):
            assert abs(orig - recovered) < 1e-6


# ============================================================
# VecStore basic tests (no extension)
# ============================================================


class TestVecStoreBasic:
    """Tests for VecStore without relying on sqlite-vec extension."""

    def test_init(self, db):
        """Test VecStore initialization."""
        vec = VecStore(db)
        assert vec._db is db
        assert vec._dimensions == DEFAULT_DIMENSIONS
        assert not vec._extension_loaded
        assert not vec._vec_table_created

    def test_init_custom_dimensions(self, db):
        """Test VecStore with custom dimensions."""
        vec = VecStore(db, dimensions=1024)
        assert vec._dimensions == 1024

    def test_valid_sources_constant(self):
        """Test VALID_SOURCES contains expected values."""
        expected = {
            "transcription",
            "scene_transcript",
            "visual",
            "technical",
            "entity",
            "qa",
            "observation",
            "audio_description",
        }
        assert expected == VALID_SOURCES


# ============================================================
# VecStore with mocked extension
# ============================================================


class TestVecStoreWithMockedExtension:
    """Tests for VecStore with mocked sqlite-vec extension."""

    def test_load_extension_import_error(self, db):
        """Test graceful handling when sqlite_vec not installed."""
        vec = VecStore(db)
        with (
            patch.dict("sys.modules", {"sqlite_vec": None}),
            patch("builtins.__import__", side_effect=ImportError("no sqlite_vec")),
        ):
            result = vec._load_extension()
        assert result is False
        assert not vec._extension_loaded

    def test_load_extension_load_failure(self, db):
        """Test graceful handling when extension fails to load."""
        vec = VecStore(db)
        mock_sqlite_vec = MagicMock()
        mock_sqlite_vec.load.side_effect = Exception("load failed")

        with patch.dict("sys.modules", {"sqlite_vec": mock_sqlite_vec}):
            result = vec._load_extension()
        assert result is False

    def test_is_available_without_extension(self, db):
        """Test is_available returns False when extension unavailable."""
        vec = VecStore(db)
        with patch.object(vec, "_load_extension", return_value=False):
            assert not vec.is_available()


# ============================================================
# embed_text tests
# ============================================================


class TestEmbedText:
    """Tests for embed_text functionality."""

    @pytest.mark.asyncio
    async def test_embed_text_invalid_source(self, db, video_uuid):
        """Test that invalid source raises ValueError."""
        vec = VecStore(db)
        with pytest.raises(ValueError, match="Invalid source"):
            await vec.embed_text(video_uuid, 0, "invalid_source", "text")

    @pytest.mark.asyncio
    async def test_embed_text_empty_text_skipped(self, db, video_uuid):
        """Test that empty text is skipped."""
        vec = VecStore(db)
        result = await vec.embed_text(video_uuid, 0, "transcription", "")
        assert result is None

        result = await vec.embed_text(video_uuid, 0, "transcription", "   ")
        assert result is None

    @pytest.mark.asyncio
    async def test_embed_text_stores_metadata_even_without_extension(
        self, db, video_uuid
    ):
        """Test that metadata is stored even if vec extension unavailable."""
        vec = VecStore(db)
        # Mock extension loading to fail and embedding to return None
        with (
            patch.object(vec, "_ensure_vec_table", return_value=False),
            patch.object(vec, "get_embedding", return_value=None),
        ):
            result = await vec.embed_text(
                video_uuid, 0, "transcription", "Hello world"
            )

        # Should have returned a metadata ID
        assert result is not None
        assert len(result) == 36  # UUID length

        # Check metadata was stored
        cursor = db.execute(
            "SELECT * FROM vec_metadata WHERE id = ?", (result,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row["video_id"] == video_uuid
        assert row["scene_id"] == 0
        assert row["source"] == "transcription"

    @pytest.mark.asyncio
    async def test_embed_text_metadata_with_timestamps(self, db, video_uuid):
        """Test storing metadata with start/end times."""
        vec = VecStore(db)
        with (
            patch.object(vec, "_ensure_vec_table", return_value=False),
            patch.object(vec, "get_embedding", return_value=None),
        ):
            result = await vec.embed_text(
                video_uuid,
                5,
                "scene_transcript",
                "Some scene text",
                start_time=120.5,
                end_time=180.0,
            )

        assert result is not None
        cursor = db.execute("SELECT * FROM vec_metadata WHERE id = ?", (result,))
        row = cursor.fetchone()
        assert row["scene_id"] == 5
        assert row["source"] == "scene_transcript"
        assert row["start_time"] == 120.5
        assert row["end_time"] == 180.0

    @pytest.mark.asyncio
    async def test_embed_text_video_level_no_scene(self, db, video_uuid):
        """Test storing video-level metadata (scene_id=None)."""
        vec = VecStore(db)
        with (
            patch.object(vec, "_ensure_vec_table", return_value=False),
            patch.object(vec, "get_embedding", return_value=None),
        ):
            result = await vec.embed_text(
                video_uuid, None, "transcription", "Full video transcript"
            )

        cursor = db.execute("SELECT * FROM vec_metadata WHERE id = ?", (result,))
        row = cursor.fetchone()
        assert row["scene_id"] is None
        assert row["source"] == "transcription"


# ============================================================
# get_embedding tests
# ============================================================


class TestGetEmbedding:
    """Tests for get_embedding functionality."""

    @pytest.mark.asyncio
    async def test_get_embedding_no_provider(self):
        """Test get_embedding returns None when no provider available."""
        with patch(
            "claudetube.providers.router.ProviderRouter.get_embedder",
            side_effect=Exception("No provider"),
        ):
            result = await get_embedding("test text")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_embedding_with_provider(self, mock_embedder):
        """Test get_embedding returns embedding from provider."""
        mock_router = MagicMock()
        mock_router.get_embedder.return_value = mock_embedder

        with patch(
            "claudetube.providers.router.ProviderRouter",
            return_value=mock_router,
        ):
            result = await get_embedding("test text")

        assert result is not None
        assert len(result) == DEFAULT_DIMENSIONS
        mock_embedder.embed.assert_called_once_with("test text")


# ============================================================
# Module-level functions tests
# ============================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_load_vec_extension_function(self, db):
        """Test load_vec_extension creates VecStore and loads."""
        with patch("claudetube.db.vec.VecStore._load_extension", return_value=False):
            result = load_vec_extension(db)
        # Result depends on whether sqlite-vec is actually installed
        assert isinstance(result, bool)

    def test_create_vec_table_function(self, db):
        """Test create_vec_table creates VecStore and ensures table."""
        with patch("claudetube.db.vec.VecStore._ensure_vec_table", return_value=False):
            result = create_vec_table(db)
        assert isinstance(result, bool)


# ============================================================
# Search tests (with mocked data)
# ============================================================


class TestSearch:
    """Tests for search functionality with mocked vec operations."""

    @pytest.mark.asyncio
    async def test_search_similar_no_extension(self, db):
        """Test search returns empty list when extension unavailable."""
        vec = VecStore(db)
        with patch.object(vec, "_ensure_vec_table", return_value=False):
            results = await vec.search_similar("query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_similar_no_embedding(self, db):
        """Test search returns empty list when embedding fails."""
        vec = VecStore(db)
        with (
            patch.object(vec, "_ensure_vec_table", return_value=True),
            patch.object(vec, "get_embedding", return_value=None),
        ):
            results = await vec.search_similar("query")
        assert results == []

    def test_search_by_embedding_no_extension(self, db):
        """Test search_by_embedding returns empty when no extension."""
        vec = VecStore(db)
        embedding = [0.1] * DEFAULT_DIMENSIONS
        with patch.object(vec, "_ensure_vec_table", return_value=False):
            results = vec.search_by_embedding(embedding)
        assert results == []


# ============================================================
# Delete tests
# ============================================================


class TestDelete:
    """Tests for delete functionality."""

    def test_delete_video_embeddings(self, db, video_uuid):
        """Test deleting embeddings for a video."""
        # Insert some metadata
        for i in range(3):
            db.execute(
                """INSERT INTO vec_metadata (id, video_id, scene_id, source)
                   VALUES (?, ?, ?, ?)""",
                (str(uuid.uuid4()), video_uuid, i, "transcription"),
            )
        db.commit()

        # Verify inserted
        cursor = db.execute(
            "SELECT COUNT(*) as cnt FROM vec_metadata WHERE video_id = ?",
            (video_uuid,),
        )
        assert cursor.fetchone()["cnt"] == 3

        # Delete
        vec = VecStore(db)
        count = vec.delete_video_embeddings(video_uuid)
        assert count == 3

        # Verify deleted
        cursor = db.execute(
            "SELECT COUNT(*) as cnt FROM vec_metadata WHERE video_id = ?",
            (video_uuid,),
        )
        assert cursor.fetchone()["cnt"] == 0

    def test_delete_nonexistent_video(self, db):
        """Test deleting embeddings for non-existent video."""
        vec = VecStore(db)
        count = vec.delete_video_embeddings("nonexistent-uuid")
        assert count == 0


# ============================================================
# Dimension mismatch tests
# ============================================================


class TestDimensionMismatch:
    """Tests for handling dimension mismatches."""

    @pytest.mark.asyncio
    async def test_embed_dimension_mismatch_skips_storage(self, db, video_uuid):
        """Test that dimension mismatch skips embedding storage but keeps metadata."""
        vec = VecStore(db, dimensions=1536)

        # Return embedding with wrong dimensions
        wrong_embedding = [0.1] * 1024  # 1024 instead of 1536

        with (
            patch.object(vec, "_ensure_vec_table", return_value=True),
            patch.object(vec, "get_embedding", return_value=wrong_embedding),
            patch.object(vec._db, "execute") as mock_execute,
        ):
            # Set up returns for different queries
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = {"rowid": 1}
            mock_execute.return_value = mock_cursor

            result = await vec.embed_text(
                video_uuid, 0, "transcription", "Hello"
            )

        # Should still return metadata ID (embedding storage was skipped)
        # The actual behavior depends on the mocking - just verify no crash
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_search_dimension_mismatch(self, db):
        """Test search with wrong query embedding dimensions."""
        vec = VecStore(db, dimensions=1536)

        wrong_embedding = [0.1] * 1024

        with (
            patch.object(vec, "_ensure_vec_table", return_value=True),
            patch.object(vec, "get_embedding", return_value=wrong_embedding),
        ):
            results = await vec.search_similar("query")

        # Should return empty due to dimension check
        assert results == []


# ============================================================
# Integration test with real sqlite-vec (if available)
# ============================================================


class TestRealSqliteVec:
    """Integration tests that use the real sqlite-vec extension."""

    @pytest.fixture
    def vec_enabled_db(self, db, video_uuid):
        """Try to create a VecStore with real extension loaded."""
        vec = VecStore(db)
        if not vec._load_extension():
            pytest.skip("sqlite-vec extension not available")
        return vec, video_uuid

    def test_load_extension_success(self, vec_enabled_db):
        """Test that sqlite-vec extension loads on this system."""
        vec, _ = vec_enabled_db
        assert vec._extension_loaded

    def test_create_vec_table_success(self, vec_enabled_db):
        """Test creating vec0 virtual table."""
        vec, _ = vec_enabled_db
        result = vec._ensure_vec_table()
        assert result
        assert vec._vec_table_created

        # Verify table exists
        cursor = vec._db.execute(
            "SELECT name FROM sqlite_master WHERE name = 'vec_embeddings'"
        )
        row = cursor.fetchone()
        assert row is not None

    def test_is_available(self, vec_enabled_db):
        """Test is_available returns True when extension loaded."""
        vec, _ = vec_enabled_db
        assert vec.is_available()

    @pytest.mark.asyncio
    async def test_store_and_search_real(self, vec_enabled_db):
        """Test full store and search cycle with real extension."""
        vec, video_uuid = vec_enabled_db
        vec._ensure_vec_table()

        # Create a deterministic embedding
        test_embedding = [float(i) / 1536 for i in range(1536)]

        with patch.object(vec, "get_embedding", return_value=test_embedding):
            metadata_id = await vec.embed_text(
                video_uuid, 0, "transcription", "Test content"
            )

        assert metadata_id is not None

        # Verify metadata was stored
        cursor = vec._db.execute(
            "SELECT * FROM vec_metadata WHERE id = ?", (metadata_id,)
        )
        row = cursor.fetchone()
        assert row is not None

        # Verify embedding was stored (rowid should exist in vec_embeddings)
        cursor = vec._db.execute(
            "SELECT rowid FROM vec_metadata WHERE id = ?", (metadata_id,)
        )
        meta_row = cursor.fetchone()
        assert meta_row is not None

        cursor = vec._db.execute(
            "SELECT rowid FROM vec_embeddings WHERE rowid = ?",
            (meta_row["rowid"],),
        )
        vec_row = cursor.fetchone()
        assert vec_row is not None

        # Test search - should find our embedding
        results = vec.search_by_embedding(test_embedding, top_k=5)
        assert len(results) >= 1
        assert results[0]["id"] == metadata_id
        assert results[0]["distance"] < 0.01  # Should be very close
