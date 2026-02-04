"""sqlite-vec integration for vector similarity search.

This module provides vector embedding storage and search using sqlite-vec.
It integrates with the providers/router.py Embedder protocol for generating
embeddings.

The vec0 virtual table is created at runtime after loading the extension.
If the extension cannot be loaded, all operations degrade gracefully.

IMPORTANT: Vector data is stored in a separate database (claudetube-vectors.db)
from the main metadata database (claudetube.db). This allows the main database
to be opened with standard SQLite tools without requiring the sqlite-vec extension.

Example:
    from claudetube.db import get_vectors_database
    from claudetube.db.vec import VecStore

    vec_db = get_vectors_database()
    vec = VecStore(vec_db)

    # Embed and store
    await vec.embed_text(video_id, scene_id, "scene_transcript", "Hello world")

    # Search
    results = await vec.search_similar("greeting", top_k=5)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claudetube.db.connection import Database

logger = logging.getLogger(__name__)

# Default embedding dimension (OpenAI text-embedding-3-small)
# This can be overridden when creating the vec0 table
DEFAULT_DIMENSIONS = 1536

# Valid source types for vec_metadata (must match schema)
VALID_SOURCES = frozenset(
    {
        "transcription",
        "scene_transcript",
        "visual",
        "technical",
        "entity",
        "qa",
        "observation",
        "audio_description",
    }
)


class VecExtensionError(Exception):
    """Raised when sqlite-vec extension cannot be loaded."""

    pass


class VecStore:
    """Vector storage and search using sqlite-vec.

    Manages the vec0 virtual table and provides methods for storing
    and searching embeddings. Integrates with the providers router
    for generating embeddings.

    The vec0 table is created lazily on first use. If the extension
    cannot be loaded, operations degrade gracefully (log warning,
    return empty results).

    Args:
        db: Database connection to use.
        dimensions: Embedding vector dimensions. Defaults to 1536.
            OpenAI = 1536, Voyage = 1024, etc.
    """

    def __init__(self, db: Database, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        self._db = db
        self._dimensions = dimensions
        self._extension_loaded = False
        self._vec_table_created = False

    def _load_extension(self) -> bool:
        """Load the sqlite-vec extension.

        Returns:
            True if extension was loaded successfully, False otherwise.
        """
        if self._extension_loaded:
            return True

        try:
            import sqlite_vec

            conn = self._db.connection
            # Enable extension loading
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._extension_loaded = True
            logger.debug("sqlite-vec extension loaded successfully")
            return True
        except ImportError:
            logger.warning(
                "sqlite-vec not installed. Vector operations will be disabled. "
                "Install with: pip install sqlite-vec"
            )
            return False
        except Exception as e:
            logger.warning(
                "Failed to load sqlite-vec extension: %s. "
                "Vector operations will be disabled.",
                e,
            )
            return False

    def _ensure_vec_table(self) -> bool:
        """Create the vec0 virtual table if it doesn't exist.

        Returns:
            True if table exists or was created, False if extension unavailable.
        """
        if self._vec_table_created:
            return True

        if not self._load_extension():
            return False

        try:
            # Create vec0 virtual table for storing embeddings
            # The table uses vec_metadata.rowid as the key linking to metadata
            self._db.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                    embedding float[{self._dimensions}]
                )
            """)
            self._db.commit()
            self._vec_table_created = True
            logger.debug(
                "vec_embeddings table created with %d dimensions", self._dimensions
            )
            return True
        except Exception as e:
            logger.warning("Failed to create vec_embeddings table: %s", e)
            return False

    def is_available(self) -> bool:
        """Check if vector operations are available.

        Returns:
            True if sqlite-vec is loaded and vec0 table exists.
        """
        return self._ensure_vec_table()

    async def get_embedding(self, text: str) -> list[float] | None:
        """Get embedding vector from the configured provider.

        Uses the providers/router.py Embedder protocol. If no embedder
        is available, returns None.

        Args:
            text: Text content to embed.

        Returns:
            Embedding vector as list of floats, or None if unavailable.
        """
        try:
            from claudetube.providers.router import NoProviderError, ProviderRouter

            router = ProviderRouter()
            embedder = router.get_embedder()
            return await embedder.embed(text)
        except NoProviderError:
            logger.debug("No embedder available, skipping embedding generation")
            return None
        except Exception as e:
            logger.warning("Failed to generate embedding: %s", e)
            return None

    def _get_embedding_sync(self, text: str) -> list[float] | None:
        """Synchronous wrapper for get_embedding.

        Uses asyncio.run() if not already in an event loop.
        """
        try:
            asyncio.get_running_loop()
            # Already in async context, use nest_asyncio or raise
            # For simplicity, just use asyncio.run in a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.get_embedding(text))
                return future.result()
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.get_embedding(text))

    async def embed_text(
        self,
        video_uuid: str,
        scene_id: int | None,
        source: str,
        text: str,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> str | None:
        """Embed text and store in vec0 + vec_metadata.

        Creates a vec_metadata row and stores the embedding in vec_embeddings.
        If embedding fails (no provider, API error), metadata is still stored
        but no embedding is created.

        Args:
            video_uuid: UUID of the video (videos.id, not video_id).
            scene_id: Scene index (0-based), or None for video-level content.
            source: Content source type (must be in VALID_SOURCES).
            text: Text content to embed.
            start_time: Optional start timestamp in seconds.
            end_time: Optional end timestamp in seconds.

        Returns:
            UUID of the vec_metadata row if successful, None if failed.

        Raises:
            ValueError: If source is not a valid source type.
        """
        if source not in VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'. Must be one of: {sorted(VALID_SOURCES)}"
            )

        if not text or not text.strip():
            logger.debug("Skipping empty text for video %s", video_uuid)
            return None

        # Generate UUID for the metadata row
        metadata_id = str(uuid.uuid4())

        # Insert metadata (even if embedding fails, we track the content)
        try:
            self._db.execute(
                """
                INSERT OR REPLACE INTO vec_metadata
                    (id, video_id, scene_id, start_time, end_time, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (metadata_id, video_uuid, scene_id, start_time, end_time, source),
            )
            self._db.commit()
        except Exception as e:
            logger.warning("Failed to insert vec_metadata: %s", e)
            return None

        # Try to generate and store embedding
        if not self._ensure_vec_table():
            logger.debug("vec table not available, metadata stored without embedding")
            return metadata_id

        embedding = await self.get_embedding(text)
        if embedding is None:
            logger.debug(
                "No embedding generated for video %s scene %s source %s, "
                "metadata stored without embedding",
                video_uuid,
                scene_id,
                source,
            )
            return metadata_id

        # Verify dimension matches
        if len(embedding) != self._dimensions:
            logger.warning(
                "Embedding dimension mismatch: expected %d, got %d. "
                "Skipping embedding storage.",
                self._dimensions,
                len(embedding),
            )
            return metadata_id

        try:
            # Get the rowid of the metadata row we just inserted
            cursor = self._db.execute(
                "SELECT rowid FROM vec_metadata WHERE id = ?", (metadata_id,)
            )
            row = cursor.fetchone()
            if row is None:
                logger.warning("Failed to find vec_metadata row after insert")
                return metadata_id

            rowid = row["rowid"]

            # Insert embedding with matching rowid
            self._db.execute(
                "INSERT INTO vec_embeddings (rowid, embedding) VALUES (?, ?)",
                (rowid, _serialize_embedding(embedding)),
            )
            self._db.commit()
            logger.debug(
                "Stored embedding for video %s scene %s source %s (rowid=%d)",
                video_uuid,
                scene_id,
                source,
                rowid,
            )
        except Exception as e:
            logger.warning("Failed to store embedding: %s", e)

        return metadata_id

    async def search_similar(
        self,
        query: str,
        top_k: int = 10,
        video_uuid: str | None = None,
    ) -> list[dict]:
        """Search for similar content using vector similarity.

        Args:
            query: Query text to search for.
            top_k: Maximum number of results to return.
            video_uuid: Optional video UUID to filter results.

        Returns:
            List of dicts with keys: id, video_id, scene_id, source,
            start_time, end_time, distance. Sorted by distance (closest first).
            Returns empty list if vector operations unavailable.
        """
        if not self._ensure_vec_table():
            return []

        # Generate query embedding
        query_embedding = await self.get_embedding(query)
        if query_embedding is None:
            logger.debug("No query embedding, returning empty results")
            return []

        if len(query_embedding) != self._dimensions:
            logger.warning(
                "Query embedding dimension mismatch: expected %d, got %d",
                self._dimensions,
                len(query_embedding),
            )
            return []

        return self._search_by_embedding(query_embedding, top_k, video_uuid)

    def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 10,
        video_uuid: str | None = None,
    ) -> list[dict]:
        """Search using a pre-computed embedding vector.

        Args:
            embedding: Pre-computed embedding vector.
            top_k: Maximum number of results to return.
            video_uuid: Optional video UUID to filter results.

        Returns:
            List of result dicts sorted by distance.
        """
        return self._search_by_embedding(embedding, top_k, video_uuid)

    def _search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 10,
        video_uuid: str | None = None,
    ) -> list[dict]:
        """Internal search implementation."""
        if not self._ensure_vec_table():
            return []

        try:
            # Build query with optional video filter
            if video_uuid is not None:
                sql = """
                    SELECT
                        m.id,
                        m.video_id,
                        m.scene_id,
                        m.source,
                        m.start_time,
                        m.end_time,
                        v.distance
                    FROM vec_embeddings v
                    JOIN vec_metadata m ON v.rowid = m.rowid
                    WHERE v.embedding MATCH ?
                        AND k = ?
                        AND m.video_id = ?
                    ORDER BY v.distance
                """
                params = (_serialize_embedding(embedding), top_k, video_uuid)
            else:
                sql = """
                    SELECT
                        m.id,
                        m.video_id,
                        m.scene_id,
                        m.source,
                        m.start_time,
                        m.end_time,
                        v.distance
                    FROM vec_embeddings v
                    JOIN vec_metadata m ON v.rowid = m.rowid
                    WHERE v.embedding MATCH ?
                        AND k = ?
                    ORDER BY v.distance
                """
                params = (_serialize_embedding(embedding), top_k)

            cursor = self._db.execute(sql, params)
            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "id": row["id"],
                        "video_id": row["video_id"],
                        "scene_id": row["scene_id"],
                        "source": row["source"],
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                        "distance": row["distance"],
                    }
                )
            return results
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            return []

    def delete_video_embeddings(self, video_uuid: str) -> int:
        """Delete all embeddings for a video.

        Args:
            video_uuid: UUID of the video.

        Returns:
            Number of embeddings deleted.
        """
        try:
            # First, get rowids to delete from vec_embeddings
            cursor = self._db.execute(
                "SELECT rowid FROM vec_metadata WHERE video_id = ?",
                (video_uuid,),
            )
            rowids = [row["rowid"] for row in cursor.fetchall()]

            if not rowids and self._ensure_vec_table():
                # Delete from vec_embeddings
                for rowid in rowids:
                    self._db.execute(
                        "DELETE FROM vec_embeddings WHERE rowid = ?",
                        (rowid,),
                    )

            # Delete metadata (CASCADE would handle this, but explicit is better)
            cursor = self._db.execute(
                "DELETE FROM vec_metadata WHERE video_id = ?",
                (video_uuid,),
            )
            self._db.commit()
            return cursor.rowcount
        except Exception as e:
            logger.warning("Failed to delete video embeddings: %s", e)
            return 0


def _serialize_embedding(embedding: list[float]) -> bytes:
    """Serialize embedding to bytes for sqlite-vec storage.

    sqlite-vec expects embeddings as binary blobs in float32 format.
    """
    import struct

    return struct.pack(f"{len(embedding)}f", *embedding)


def _deserialize_embedding(data: bytes, dimensions: int) -> list[float]:
    """Deserialize embedding from bytes."""
    import struct

    return list(struct.unpack(f"{dimensions}f", data))


def load_vec_extension(db: Database) -> bool:
    """Load the sqlite-vec extension into a database connection.

    Args:
        db: Database connection.

    Returns:
        True if extension loaded successfully, False otherwise.
    """
    vec = VecStore(db)
    return vec._load_extension()


def create_vec_table(db: Database, dimensions: int = DEFAULT_DIMENSIONS) -> bool:
    """Create the vec0 virtual table if it doesn't exist.

    Args:
        db: Database connection.
        dimensions: Embedding vector dimensions.

    Returns:
        True if table exists or was created, False if extension unavailable.
    """
    vec = VecStore(db, dimensions=dimensions)
    return vec._ensure_vec_table()


async def embed_text(
    video_uuid: str,
    scene_id: int | None,
    source: str,
    text: str,
    db: Database | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> str | None:
    """Embed text and store in vec0 + vec_metadata.

    Convenience function that uses the singleton vectors database.

    Args:
        video_uuid: UUID of the video (videos.id from main db).
        scene_id: Scene index, or None for video-level.
        source: Content source type.
        text: Text to embed.
        db: Optional vectors database connection (uses singleton if None).
        start_time: Optional start timestamp.
        end_time: Optional end timestamp.

    Returns:
        UUID of the vec_metadata row, or None if failed.
    """
    if db is None:
        from claudetube.db import get_vectors_database

        db = get_vectors_database()

    vec = VecStore(db)
    return await vec.embed_text(
        video_uuid, scene_id, source, text, start_time, end_time
    )


async def search_similar(
    query_embedding: list[float],
    top_k: int = 10,
    video_uuid: str | None = None,
    db: Database | None = None,
) -> list[dict]:
    """Search for similar embeddings.

    Convenience function that uses the singleton vectors database.

    Args:
        query_embedding: Pre-computed query embedding vector.
        top_k: Maximum results.
        video_uuid: Optional video filter.
        db: Optional vectors database connection (uses singleton if None).

    Returns:
        List of result dicts.
    """
    if db is None:
        from claudetube.db import get_vectors_database

        db = get_vectors_database()

    vec = VecStore(db)
    return vec.search_by_embedding(query_embedding, top_k, video_uuid)


async def get_embedding(text: str) -> list[float] | None:
    """Get embedding vector from the configured provider.

    Convenience function for getting embeddings without a VecStore.

    Args:
        text: Text to embed.

    Returns:
        Embedding vector, or None if unavailable.
    """
    try:
        from claudetube.providers.router import NoProviderError, ProviderRouter

        router = ProviderRouter()
        embedder = router.get_embedder()
        return await embedder.embed(text)
    except NoProviderError:
        logger.debug("No embedder available")
        return None
    except Exception as e:
        logger.warning("Failed to generate embedding: %s", e)
        return None


__all__ = [
    "VecStore",
    "VecExtensionError",
    "load_vec_extension",
    "create_vec_table",
    "embed_text",
    "search_similar",
    "get_embedding",
    "DEFAULT_DIMENSIONS",
    "VALID_SOURCES",
]
