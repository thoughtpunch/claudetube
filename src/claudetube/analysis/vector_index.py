"""
Vector index for semantic scene search using sqlite-vec.

Provides fast similarity search over scene embeddings stored in SQLite.

Architecture: Cheap First, Expensive Last
1. CACHE - Check for existing embeddings in DB
2. BUILD - Create embeddings via provider pattern
3. QUERY - Sub-second KNN similarity search

Config:
- Embeddings stored in vec_embeddings virtual table
- Metadata in vec_metadata table links to video/scene
- Persistent by default (survives restarts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from claudetube.analysis.embeddings import SceneEmbedding
    from claudetube.cache.scenes import SceneBoundary

logger = logging.getLogger(__name__)

# Maximum transcript preview length in metadata
MAX_TRANSCRIPT_PREVIEW = 500


@dataclass
class SearchResult:
    """Single search result from vector index."""

    scene_id: int
    distance: float  # Lower is better (L2/cosine distance)
    start_time: float
    end_time: float
    transcript_preview: str
    visual_description: str
    video_id: str | None = None  # For cross-video search


def _get_db():
    """Get the database instance.

    Returns:
        Database instance or None if unavailable.
    """
    try:
        from claudetube.db import get_database

        return get_database()
    except Exception:
        logger.debug("Database unavailable for vector operations", exc_info=True)
        return None


def _get_vec_store(db=None):
    """Get a VecStore instance.

    Args:
        db: Optional database instance. Uses singleton if None.

    Returns:
        VecStore instance or None if unavailable.
    """
    if db is None:
        db = _get_db()
    if db is None:
        return None

    try:
        from claudetube.db.vec import VecStore

        return VecStore(db)
    except Exception:
        logger.debug("VecStore unavailable", exc_info=True)
        return None


def has_vector_index(cache_dir: Path) -> bool:
    """Check if vector index exists for this video.

    With sqlite-vec, this checks if there are embeddings in the
    vec_embeddings table for the video.

    Args:
        cache_dir: Video cache directory (used to extract video_id).

    Returns:
        True if embeddings exist for this video.
    """
    # Extract video_id from cache_dir
    video_id = cache_dir.name

    db = _get_db()
    if db is None:
        # Fall back to checking for legacy ChromaDB index
        chroma_path = cache_dir / "embeddings" / "chroma"
        return chroma_path.exists() and (chroma_path / "chroma.sqlite3").exists()

    try:
        # Check if video exists in DB
        from claudetube.db.repos.videos import VideoRepository

        video_repo = VideoRepository(db)
        video = video_repo.get_by_video_id(video_id)
        if video is None:
            return False

        # Check if there are embeddings for this video
        cursor = db.execute(
            """
            SELECT COUNT(*) as cnt FROM vec_metadata
            WHERE video_id = ?
            """,
            (video["id"],),
        )
        row = cursor.fetchone()
        return row["cnt"] > 0 if row else False
    except Exception:
        logger.debug("Error checking vector index", exc_info=True)
        return False


def build_scene_index(
    cache_dir: Path,
    scenes: list[dict | SceneBoundary],
    embeddings: list[SceneEmbedding],
    video_id: str | None = None,
) -> int:
    """Create searchable vector index of video scenes.

    Stores scene embeddings in sqlite-vec with metadata for fast retrieval.
    With the auto-embed feature, this may be a no-op since embeddings are
    created at write time.

    Args:
        cache_dir: Video cache directory.
        scenes: List of scene data (dicts or SceneBoundary objects).
        embeddings: List of SceneEmbedding objects from embed_scenes().
        video_id: Optional video ID for metadata.

    Returns:
        Number of scenes indexed.

    Raises:
        ValueError: If scenes and embeddings don't match.
    """
    if not embeddings:
        logger.warning("No embeddings provided, skipping index build")
        return 0

    if video_id is None:
        video_id = cache_dir.name

    db = _get_db()
    if db is None:
        logger.warning("Database unavailable, cannot build vector index")
        return 0

    try:
        from claudetube.db.repos.videos import VideoRepository
        from claudetube.db.vec import VecStore

        video_repo = VideoRepository(db)
        video = video_repo.get_by_video_id(video_id)

        if video is None:
            logger.warning("Video %s not in database, cannot index embeddings", video_id)
            return 0

        video_uuid = video["id"]
        vec_store = VecStore(db)

        if not vec_store.is_available():
            logger.warning("sqlite-vec extension not available")
            return 0

        # Build scene_id -> embedding mapping
        emb_by_id = {e.scene_id: e for e in embeddings}

        # Process each scene
        indexed_count = 0
        for scene in scenes:
            if hasattr(scene, "scene_id"):
                scene_id = scene.scene_id
                start_time = scene.start_time
                end_time = scene.end_time
            else:
                scene_id = scene.get("scene_id", 0)
                start_time = scene.get("start_time", 0)
                end_time = scene.get("end_time", 0)

            if scene_id not in emb_by_id:
                logger.warning("No embedding for scene %d, skipping", scene_id)
                continue

            emb = emb_by_id[scene_id]

            # Store embedding directly without re-computing
            try:
                _store_embedding_direct(
                    db=db,
                    vec_store=vec_store,
                    video_uuid=video_uuid,
                    scene_id=scene_id,
                    embedding=emb.embedding.tolist(),
                    start_time=start_time,
                    end_time=end_time,
                )
                indexed_count += 1
            except Exception as e:
                logger.warning("Failed to store embedding for scene %d: %s", scene_id, e)

        logger.info("Built vector index with %d scenes for video %s", indexed_count, video_id)
        return indexed_count

    except Exception as e:
        logger.warning("Failed to build vector index: %s", e)
        return 0


def _store_embedding_direct(
    db,
    vec_store,
    video_uuid: str,
    scene_id: int,
    embedding: list[float],
    start_time: float | None = None,
    end_time: float | None = None,
) -> str | None:
    """Store a pre-computed embedding directly.

    Args:
        db: Database connection.
        vec_store: VecStore instance.
        video_uuid: Video UUID.
        scene_id: Scene ID.
        embedding: Pre-computed embedding vector.
        start_time: Optional start timestamp.
        end_time: Optional end timestamp.

    Returns:
        UUID of the vec_metadata row, or None if failed.
    """
    import struct
    import uuid

    metadata_id = str(uuid.uuid4())
    source = "scene_transcript"

    try:
        # Delete existing entry if present (for rebuilds)
        db.execute(
            """
            DELETE FROM vec_metadata
            WHERE video_id = ? AND scene_id = ? AND source = ?
            """,
            (video_uuid, scene_id, source),
        )

        # Insert metadata
        db.execute(
            """
            INSERT INTO vec_metadata
                (id, video_id, scene_id, start_time, end_time, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (metadata_id, video_uuid, scene_id, start_time, end_time, source),
        )
        db.commit()

        # Get the rowid
        cursor = db.execute(
            "SELECT rowid FROM vec_metadata WHERE id = ?", (metadata_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None

        rowid = row["rowid"]

        # Serialize and store embedding
        embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

        # Ensure vec_embeddings table exists
        if not vec_store._ensure_vec_table():
            logger.warning("Cannot create vec_embeddings table")
            return metadata_id

        db.execute(
            "INSERT OR REPLACE INTO vec_embeddings (rowid, embedding) VALUES (?, ?)",
            (rowid, embedding_bytes),
        )
        db.commit()

        return metadata_id
    except Exception as e:
        logger.warning("Failed to store embedding: %s", e)
        return None


def load_scene_index(cache_dir: Path):
    """Load existing vector index for a video.

    With sqlite-vec, this just validates that embeddings exist.

    Args:
        cache_dir: Video cache directory.

    Returns:
        True if index exists, None if not found.
    """
    if has_vector_index(cache_dir):
        return True
    return None


def search_scenes(
    cache_dir: Path,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> list[SearchResult]:
    """Search for scenes similar to query embedding.

    Args:
        cache_dir: Video cache directory.
        query_embedding: Query embedding vector (same dim as scene embeddings).
        top_k: Number of results to return.

    Returns:
        List of SearchResult objects, sorted by similarity.

    Raises:
        ValueError: If no index exists.
    """
    video_id = cache_dir.name

    db = _get_db()
    if db is None:
        raise ValueError(
            f"Database unavailable. Cannot search vector index for {cache_dir}."
        )

    try:
        from claudetube.db.repos.videos import VideoRepository

        video_repo = VideoRepository(db)
        video = video_repo.get_by_video_id(video_id)

        if video is None:
            raise ValueError(
                f"No vector index found at {cache_dir}. Run build_scene_index() first."
            )

        video_uuid = video["id"]

        vec_store = _get_vec_store(db)
        if vec_store is None or not vec_store.is_available():
            raise ValueError("sqlite-vec extension not available")

        # Search using pre-computed embedding
        results = vec_store.search_by_embedding(
            query_embedding.tolist(), top_k=top_k, video_uuid=video_uuid
        )

        # Convert to SearchResult objects with scene metadata
        search_results = []
        for result in results:
            scene_id = result.get("scene_id")
            if scene_id is None:
                continue

            # Get scene data for transcript preview
            transcript_preview = ""
            visual_description = ""

            from claudetube.db.repos.scenes import SceneRepository

            scene_repo = SceneRepository(db)
            scene = scene_repo.get_scene(video_uuid, scene_id)
            if scene:
                transcript_preview = (scene.get("transcript_text") or "")[:MAX_TRANSCRIPT_PREVIEW]

            # Get visual description if available
            from claudetube.db.repos.visual_descriptions import (
                VisualDescriptionRepository,
            )

            vis_repo = VisualDescriptionRepository(db)
            vis = vis_repo.get_by_scene(video_uuid, scene_id)
            if vis:
                visual_description = vis.get("description", "")[:MAX_TRANSCRIPT_PREVIEW]

            search_results.append(
                SearchResult(
                    scene_id=scene_id,
                    distance=result.get("distance", 0.0),
                    start_time=result.get("start_time", 0.0),
                    end_time=result.get("end_time", 0.0),
                    transcript_preview=transcript_preview,
                    visual_description=visual_description,
                    video_id=video_id,
                )
            )

        return search_results

    except ValueError:
        raise
    except Exception as e:
        logger.warning("Vector search failed: %s", e)
        raise ValueError(f"No vector index found at {cache_dir}. Run build_scene_index() first.") from e


def search_scenes_by_text(
    cache_dir: Path,
    query_text: str,
    top_k: int = 5,
    model: str | None = None,
) -> list[SearchResult]:
    """Search for scenes using text query.

    Embeds the query text using the provider pattern and searches the vector
    index. Uses the same Embedder providers as scene embedding (Voyage AI,
    local sentence-transformers, etc.).

    Args:
        cache_dir: Video cache directory.
        query_text: Natural language query.
        top_k: Number of results to return.
        model: Embedding model to use (must match index).

    Returns:
        List of SearchResult objects, sorted by similarity.

    Raises:
        ValueError: If no index exists.
    """
    from claudetube.analysis.embeddings import _get_embedder, get_embedding_model

    if model is None:
        model = get_embedding_model()

    # Embed the query using the provider pattern
    embedder = _get_embedder(model)
    embedding_list = embedder.embed_sync(query_text)
    query_embedding = np.array(embedding_list, dtype=np.float32)

    return search_scenes(cache_dir, query_embedding, top_k)


def search_similar_cross_video(
    query_text: str,
    top_k: int = 10,
) -> list[SearchResult]:
    """Search for similar scenes across ALL videos.

    This is a new capability enabled by sqlite-vec.

    Args:
        query_text: Natural language query.
        top_k: Maximum number of results to return.

    Returns:
        List of SearchResult objects from any video, sorted by similarity.
    """
    db = _get_db()
    if db is None:
        return []

    vec_store = _get_vec_store(db)
    if vec_store is None or not vec_store.is_available():
        return []

    try:
        import asyncio

        # Get query embedding
        embedding = asyncio.get_event_loop().run_until_complete(
            vec_store.get_embedding(query_text)
        )
    except RuntimeError:
        try:
            import asyncio
            embedding = asyncio.run(vec_store.get_embedding(query_text))
        except Exception:
            logger.debug("Failed to get query embedding")
            return []

    if embedding is None:
        return []

    # Search without video filter (cross-video)
    results = vec_store.search_by_embedding(embedding, top_k=top_k, video_uuid=None)

    # Convert to SearchResult with video context
    search_results = []
    for result in results:
        video_uuid = result.get("video_id")
        scene_id = result.get("scene_id")

        if video_uuid is None or scene_id is None:
            continue

        # Get video natural ID
        from claudetube.db.repos.videos import VideoRepository

        video_repo = VideoRepository(db)
        video = video_repo.get_by_uuid(video_uuid)
        video_natural_id = video["video_id"] if video else None

        # Get scene data
        transcript_preview = ""
        visual_description = ""

        from claudetube.db.repos.scenes import SceneRepository

        scene_repo = SceneRepository(db)
        scene = scene_repo.get_scene(video_uuid, scene_id)
        if scene:
            transcript_preview = (scene.get("transcript_text") or "")[:MAX_TRANSCRIPT_PREVIEW]

        search_results.append(
            SearchResult(
                scene_id=scene_id,
                distance=result.get("distance", 0.0),
                start_time=result.get("start_time", 0.0),
                end_time=result.get("end_time", 0.0),
                transcript_preview=transcript_preview,
                visual_description=visual_description,
                video_id=video_natural_id,
            )
        )

    return search_results


def delete_scene_index(cache_dir: Path) -> bool:
    """Delete vector index for a video.

    Args:
        cache_dir: Video cache directory.

    Returns:
        True if deleted, False if not found.
    """
    import shutil

    video_id = cache_dir.name

    db = _get_db()
    if db is not None:
        try:
            from claudetube.db.repos.videos import VideoRepository
            from claudetube.db.vec import VecStore

            video_repo = VideoRepository(db)
            video = video_repo.get_by_video_id(video_id)

            if video:
                vec_store = VecStore(db)
                count = vec_store.delete_video_embeddings(video["id"])
                if count > 0:
                    logger.info("Deleted %d embeddings for video %s", count, video_id)
                    return True
        except Exception:
            logger.debug("Failed to delete embeddings from DB", exc_info=True)

    # Also try to delete legacy ChromaDB index
    chroma_path = cache_dir / "embeddings" / "chroma"
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        logger.info("Deleted legacy ChromaDB index at %s", chroma_path)
        return True

    return False


def get_index_stats(cache_dir: Path) -> dict | None:
    """Get statistics about the vector index.

    Args:
        cache_dir: Video cache directory.

    Returns:
        Dict with index stats or None if not found.
    """
    video_id = cache_dir.name

    db = _get_db()
    if db is None:
        return None

    try:
        from claudetube.db.repos.videos import VideoRepository

        video_repo = VideoRepository(db)
        video = video_repo.get_by_video_id(video_id)

        if video is None:
            return None

        video_uuid = video["id"]

        # Count embeddings for this video
        cursor = db.execute(
            """
            SELECT COUNT(*) as cnt FROM vec_metadata
            WHERE video_id = ?
            """,
            (video_uuid,),
        )
        row = cursor.fetchone()
        count = row["cnt"] if row else 0

        if count == 0:
            return None

        return {
            "num_scenes": count,
            "video_id": video_id,
            "storage": "sqlite-vec",
            "path": "SQLite database",
        }

    except Exception:
        logger.debug("Failed to get index stats", exc_info=True)
        return None
