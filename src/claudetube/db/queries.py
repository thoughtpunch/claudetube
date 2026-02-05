"""SQL-based query functions for read operations.

Provides high-level query functions that use SQLite/FTS5 for efficient reads,
with filesystem fallback when the database is unavailable.

This module implements the read-side of the dual-write architecture:
- Primary path: SQL queries for speed
- Fallback path: JSON/filesystem scanning when DB unavailable

All functions are designed to be drop-in replacements for existing
file-scanning operations in cache/manager.py, cache/knowledge_graph.py,
cache/enrichment.py, and analysis/search.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database

logger = logging.getLogger(__name__)


def _get_db() -> Database | None:
    """Get the database instance, or None if unavailable.

    Uses lazy import to avoid import-time database initialization.
    Returns None if the database module is unavailable or fails.
    """
    try:
        from claudetube.db import get_database

        return get_database()
    except Exception:
        logger.debug("Database unavailable for queries", exc_info=True)
        return None


# ============================================================
# VIDEO LISTING (replaces CacheManager.list_cached_videos)
# ============================================================


def list_cached_videos_sql() -> list[dict[str, Any]] | None:
    """List all cached videos from SQLite.

    Queries the video_processing_status VIEW for efficient access to
    derived processing state (scene_count, frame_count, etc.).

    Returns:
        List of video dicts with metadata and processing status,
        or None if database is unavailable.
    """
    db = _get_db()
    if db is None:
        return None

    try:
        cursor = db.execute("""
            SELECT
                v.video_id,
                v.title,
                v.duration_string,
                v.cache_path,
                v.domain,
                v.channel_name,
                vps.has_primary_transcript as transcript_complete,
                vps.transcript_provider as transcript_source,
                vps.scene_count,
                vps.frame_count,
                vps.qa_count
            FROM videos v
            JOIN video_processing_status vps ON v.id = vps.id
            ORDER BY v.created_at DESC
        """)
        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "video_id": row["video_id"],
                    "title": row["title"],
                    "duration_string": row["duration_string"],
                    "transcript_complete": bool(row["transcript_complete"]),
                    "transcript_source": row["transcript_source"],
                    "cache_dir": row["cache_path"],
                    "scene_count": row["scene_count"],
                    "frame_count": row["frame_count"],
                    "qa_count": row["qa_count"],
                }
            )
        logger.debug("Listed %d videos from SQL", len(results))
        return results
    except Exception:
        logger.debug("SQL video listing failed", exc_info=True)
        return None


# ============================================================
# KNOWLEDGE GRAPH (replaces VideoKnowledgeGraph methods)
# ============================================================


def find_related_videos_sql(query: str) -> list[dict[str, Any]] | None:
    """Find videos related to a topic using SQL.

    Uses case-insensitive LIKE matching on entity names in
    entity_video_summary JOIN entities JOIN videos.

    Replaces VideoKnowledgeGraph.find_related_videos().

    Args:
        query: Search query (case-insensitive substring match).

    Returns:
        List of matching video dicts with entity context,
        or None if database is unavailable or has no entity data.
    """
    db = _get_db()
    if db is None:
        return None

    try:
        from claudetube.db.repos.entities import EntityRepository

        repo = EntityRepository(db)

        # Check if there's any entity data at all - if not, fall back
        stats = repo.get_stats()
        if stats["entity_count"] == 0:
            # No entities in DB, use fallback
            return None

        results = repo.find_related_videos(query)

        # Convert to the format expected by VideoKnowledgeGraph
        matches = []
        for row in results:
            matches.append(
                {
                    "video_id": row["video_id"],
                    "video_title": row["video_title"],
                    "match_type": row["match_type"],
                    "matched": row["matched_term"],
                }
            )
        logger.debug("Found %d related videos for '%s' via SQL", len(matches), query)
        return matches
    except Exception:
        logger.debug("SQL find_related_videos failed", exc_info=True)
        return None


def get_video_connections_sql(video_id: str) -> list[str] | None:
    """Get videos sharing entities with a specific video using SQL.

    Uses JOIN on entity_video_summary to find videos with shared entities.

    Replaces VideoKnowledgeGraph.get_video_connections().

    Args:
        video_id: Natural key (e.g., YouTube video ID).

    Returns:
        List of connected video IDs (natural keys),
        or None if database is unavailable or video not in DB.
    """
    db = _get_db()
    if db is None:
        return None

    try:
        from claudetube.db.repos.entities import EntityRepository
        from claudetube.db.repos.videos import VideoRepository

        video_repo = VideoRepository(db)
        entity_repo = EntityRepository(db)

        # First get the video UUID
        video = video_repo.get_by_video_id(video_id)
        if video is None:
            # Video not in DB, use fallback
            return None

        video_uuid = video["id"]

        # Get connected video UUIDs
        connected_uuids = entity_repo.get_connections(video_uuid)

        # Convert UUIDs back to natural video IDs
        connected_ids = []
        for uuid in connected_uuids:
            connected_video = video_repo.get_by_uuid(uuid)
            if connected_video:
                connected_ids.append(connected_video["video_id"])

        logger.debug(
            "Found %d connections for %s via SQL", len(connected_ids), video_id
        )
        return connected_ids
    except Exception:
        logger.debug("SQL get_video_connections failed", exc_info=True)
        return None


# ============================================================
# Q&A SEARCH (replaces search_cached_qa)
# ============================================================


def search_qa_fts(video_id: str, query: str) -> list[dict[str, Any]] | None:
    """Search Q&A history using FTS5.

    Uses qa_fts virtual table for full-text search across
    question and answer fields.

    Replaces VideoMemory.search_qa_history() for faster search.

    Args:
        video_id: Natural key (e.g., YouTube video ID).
        query: Search query string.

    Returns:
        List of matching Q&A dicts with question, answer, scenes,
        or None if database is unavailable or video not in DB.
    """
    db = _get_db()
    if db is None:
        return None

    try:
        from claudetube.db.repos.qa import QARepository
        from claudetube.db.repos.videos import VideoRepository

        video_repo = VideoRepository(db)
        qa_repo = QARepository(db)

        # Get video UUID
        video = video_repo.get_by_video_id(video_id)
        if video is None:
            # Video not in DB, use fallback
            return None

        video_uuid = video["id"]

        # Search using FTS
        results = qa_repo.search_fts(query)

        # Filter to this video and convert format
        matches = []
        for row in results:
            if row["video_id"] == video_uuid:
                matches.append(
                    {
                        "question": row["question"],
                        "answer": row["answer"],
                        "scenes": row.get("scene_ids", []),
                        "timestamp": row.get("created_at", ""),
                    }
                )

        logger.debug(
            "Found %d Q&A matches for '%s' in %s via FTS", len(matches), query, video_id
        )
        return matches
    except Exception:
        logger.debug("FTS Q&A search failed", exc_info=True)
        return None


def search_qa_fts_cross_video(
    query: str, limit: int = 20
) -> list[dict[str, Any]] | None:
    """Search Q&A history across ALL videos using FTS5.

    Uses qa_fts virtual table for full-text search across all videos.
    This is a new capability enabled by the SQLite index.

    Args:
        query: Search query string.
        limit: Maximum results to return (default 20).

    Returns:
        List of matching Q&A dicts with video context,
        or None if database is unavailable.
    """
    db = _get_db()
    if db is None:
        return None

    try:
        from claudetube.db.repos.qa import QARepository

        qa_repo = QARepository(db)

        # Search using FTS across all videos
        results = qa_repo.search_fts(query)

        # Convert format and limit
        matches = []
        for row in results[:limit]:
            matches.append(
                {
                    "video_id": row.get("video_natural_id"),
                    "video_title": row.get("video_title"),
                    "question": row["question"],
                    "answer": row["answer"],
                    "scenes": row.get("scene_ids", []),
                    "timestamp": row.get("created_at", ""),
                }
            )

        logger.debug(
            "Found %d Q&A matches for '%s' across all videos via FTS",
            len(matches),
            query,
        )
        return matches
    except Exception:
        logger.debug("Cross-video FTS Q&A search failed", exc_info=True)
        return None


# ============================================================
# TRANSCRIPT SEARCH (replaces _search_transcript_text)
# ============================================================


def search_transcripts_fts(
    video_id: str,
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]] | None:
    """Search scene transcripts using FTS5.

    Uses scenes_fts virtual table for full-text search on
    transcript_text field.

    Replaces _search_transcript_text() for per-video search.

    Args:
        video_id: Natural key (e.g., YouTube video ID).
        query: Search query string.
        top_k: Maximum results (default 5).

    Returns:
        List of matching scene dicts with relevance scores,
        or None if database is unavailable or video not in DB.
    """
    db = _get_db()
    if db is None:
        return None

    try:
        from claudetube.db.repos.scenes import SceneRepository
        from claudetube.db.repos.videos import VideoRepository

        video_repo = VideoRepository(db)
        scene_repo = SceneRepository(db)

        # Get video UUID
        video = video_repo.get_by_video_id(video_id)
        if video is None:
            # Video not in DB, use fallback
            return None

        video_uuid = video["id"]

        # Search using FTS
        results = scene_repo.search_fts(query)

        # Filter to this video
        matches = []
        for row in results:
            if row["video_id"] == video_uuid:
                # Convert FTS5 rank to a 0-1 relevance score
                # FTS5 rank is negative (lower = better match)
                # Typical range is -20 to 0 for good matches
                rank = row.get("rank", 0)
                relevance = max(0.0, min(1.0, 1.0 + (rank / 20.0)))

                matches.append(
                    {
                        "scene_id": row["scene_id"],
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                        "transcript_text": row.get("transcript_text", ""),
                        "relevance": relevance,
                        "match_type": "fts",
                    }
                )

                if len(matches) >= top_k:
                    break

        logger.debug(
            "Found %d transcript matches for '%s' in %s via FTS",
            len(matches),
            query,
            video_id,
        )
        return matches
    except Exception:
        logger.debug("FTS transcript search failed", exc_info=True)
        return None


def search_transcripts_fts_cross_video(
    query: str,
    top_k: int = 10,
) -> list[dict[str, Any]] | None:
    """Search scene transcripts across ALL videos using FTS5.

    Uses scenes_fts virtual table for cross-video transcript search.
    This is a new capability enabled by the SQLite index.

    Args:
        query: Search query string.
        top_k: Maximum results (default 10).

    Returns:
        List of matching scene dicts with video context,
        or None if database is unavailable.
    """
    db = _get_db()
    if db is None:
        return None

    try:
        from claudetube.db.repos.scenes import SceneRepository

        scene_repo = SceneRepository(db)

        # Search using FTS across all videos
        results = scene_repo.search_fts(query)

        # Convert format and limit
        matches = []
        for row in results[:top_k]:
            # Convert FTS5 rank to relevance
            rank = row.get("rank", 0)
            relevance = max(0.0, min(1.0, 1.0 + (rank / 20.0)))

            matches.append(
                {
                    "video_id": row.get("video_natural_id"),
                    "video_title": row.get("video_title"),
                    "scene_id": row["scene_id"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "transcript_text": row.get("transcript_text", ""),
                    "relevance": relevance,
                    "match_type": "fts",
                }
            )

        logger.debug(
            "Found %d transcript matches for '%s' across all videos via FTS",
            len(matches),
            query,
        )
        return matches
    except Exception:
        logger.debug("Cross-video FTS transcript search failed", exc_info=True)
        return None


def search_transcripts_fts_multi_video(
    video_ids: list[str],
    query: str,
    top_k: int = 10,
) -> list[dict[str, Any]] | None:
    """Search scene transcripts across a specific set of videos using FTS5.

    Used for playlist-scoped search where we want to search only within
    videos that belong to a specific playlist.

    Args:
        video_ids: List of natural video IDs to search within.
        query: Search query string.
        top_k: Maximum results (default 10).

    Returns:
        List of matching scene dicts with video context,
        or None if database is unavailable.
    """
    db = _get_db()
    if db is None:
        return None

    if not video_ids:
        return []

    try:
        from claudetube.db.repos.scenes import SceneRepository

        scene_repo = SceneRepository(db)

        # Build placeholders for video IDs
        placeholders = ",".join("?" * len(video_ids))

        # Escape query for FTS5
        escaped_query = scene_repo._escape_fts_query(query)

        # Search using FTS across specified videos
        cursor = db.execute(
            f"""
            SELECT
                s.*,
                v.video_id as video_natural_id,
                v.title as video_title,
                rank
            FROM scenes_fts
            JOIN scenes s ON scenes_fts.rowid = s.rowid
            JOIN videos v ON s.video_id = v.id
            WHERE scenes_fts MATCH ?
              AND v.video_id IN ({placeholders})
            ORDER BY rank
            LIMIT ?
            """,
            (escaped_query, *video_ids, top_k),
        )

        # Convert format
        matches = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            # Convert FTS5 rank to relevance
            rank = row_dict.get("rank", 0)
            relevance = max(0.0, min(1.0, 1.0 + (rank / 20.0)))

            matches.append(
                {
                    "video_id": row_dict.get("video_natural_id"),
                    "video_title": row_dict.get("video_title"),
                    "scene_id": row_dict["scene_id"],
                    "start_time": row_dict["start_time"],
                    "end_time": row_dict["end_time"],
                    "transcript_text": row_dict.get("transcript_text", ""),
                    "relevance": relevance,
                    "match_type": "fts",
                }
            )

        logger.debug(
            "Found %d transcript matches for '%s' in %d videos via FTS",
            len(matches),
            query,
            len(video_ids),
        )
        return matches
    except Exception:
        logger.debug("Multi-video FTS transcript search failed", exc_info=True)
        return None


# ============================================================
# PROCESSING STATE (replaces file existence checks)
# ============================================================


def get_processing_status(video_id: str) -> dict[str, Any] | None:
    """Get processing status for a video from SQL.

    Queries pipeline_steps and the video_processing_status VIEW
    to determine what has been processed.

    Replaces scattered file existence checks.

    Args:
        video_id: Natural key (e.g., YouTube video ID).

    Returns:
        Dict with processing status flags,
        or None if database is unavailable.
    """
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

        # Query video_processing_status VIEW
        cursor = db.execute(
            "SELECT * FROM video_processing_status WHERE id = ?",
            (video_uuid,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        status = {
            "video_id": row["video_id"],
            "title": row["title"],
            "has_audio": row["audio_track_count"] > 0,
            "has_transcript": row["transcription_count"] > 0,
            "has_primary_transcript": bool(row["has_primary_transcript"]),
            "transcript_source": row["transcript_provider"],
            "has_scenes": row["scene_count"] > 0,
            "scene_count": row["scene_count"],
            "has_frames": row["frame_count"] > 0,
            "frame_count": row["frame_count"],
            "has_keyframes": row["keyframe_count"] > 0,
            "has_thumbnail": bool(row["has_thumbnail"]),
            "has_visual_descriptions": row["visual_description_count"] > 0,
            "has_technical_content": row["technical_content_count"] > 0,
            "has_narrative": bool(row["has_narrative"]),
            "has_code_evolution": bool(row["has_code_evolution"]),
            "has_audio_description": bool(row["has_audio_description"]),
            "entity_count": row["entity_count"],
            "qa_count": row["qa_count"],
            "completed_steps": row["completed_steps"],
            "failed_steps": row["failed_steps"],
            "running_steps": row["running_steps"],
        }

        logger.debug("Got processing status for %s from SQL", video_id)
        return status
    except Exception:
        logger.debug("SQL processing status query failed", exc_info=True)
        return None


def is_step_complete(
    video_id: str,
    step_type: str,
    scene_id: int | None = None,
) -> bool | None:
    """Check if a processing step is complete using SQL.

    Queries pipeline_steps table to check status.

    Replaces file existence checks for determining processing state.

    Args:
        video_id: Natural key (e.g., YouTube video ID).
        step_type: Type of processing step (download, transcribe, etc.).
        scene_id: Optional scene ID for per-scene steps.

    Returns:
        True if completed, False if not completed,
        None if database is unavailable.
    """
    db = _get_db()
    if db is None:
        return None

    try:
        from claudetube.db.repos.pipeline import PipelineRepository
        from claudetube.db.repos.videos import VideoRepository

        video_repo = VideoRepository(db)
        pipeline_repo = PipelineRepository(db)

        # Get video UUID
        video = video_repo.get_by_video_id(video_id)
        if video is None:
            return False

        video_uuid = video["id"]

        # Check pipeline step
        result = pipeline_repo.is_step_complete(video_uuid, step_type, scene_id)
        logger.debug(
            "Step %s complete for %s (scene %s): %s",
            step_type,
            video_id,
            scene_id,
            result,
        )
        return result
    except Exception:
        logger.debug("SQL step completion check failed", exc_info=True)
        return None


def get_incomplete_scenes(video_id: str, step_type: str) -> list[int] | None:
    """Get scene IDs that don't have a completed step of the given type.

    Queries pipeline_steps to find scenes without completed processing.

    Args:
        video_id: Natural key (e.g., YouTube video ID).
        step_type: Type of processing step.

    Returns:
        List of incomplete scene_ids,
        or None if database is unavailable.
    """
    db = _get_db()
    if db is None:
        return None

    try:
        from claudetube.db.repos.pipeline import PipelineRepository
        from claudetube.db.repos.videos import VideoRepository

        video_repo = VideoRepository(db)
        pipeline_repo = PipelineRepository(db)

        # Get video UUID
        video = video_repo.get_by_video_id(video_id)
        if video is None:
            return []

        video_uuid = video["id"]

        # Get incomplete scenes
        result = pipeline_repo.get_incomplete_scenes(video_uuid, step_type)
        logger.debug(
            "Found %d incomplete scenes for %s step %s",
            len(result),
            video_id,
            step_type,
        )
        return result
    except Exception:
        logger.debug("SQL incomplete scenes query failed", exc_info=True)
        return None


# ============================================================
# ENTITY STATS (for knowledge graph)
# ============================================================


def get_knowledge_graph_stats_sql() -> dict[str, Any] | None:
    """Get statistics about the knowledge graph using SQL.

    Returns counts from entities and entity_video_summary tables.

    Replaces VideoKnowledgeGraph.get_stats().

    Returns:
        Dict with entity_count, video_count, etc.,
        or None if database is unavailable or has no entity data.
    """
    db = _get_db()
    if db is None:
        return None

    try:
        from claudetube.db.repos.entities import EntityRepository

        repo = EntityRepository(db)
        stats = repo.get_stats()

        # If no entities in DB, fall back to in-memory
        if stats["entity_count"] == 0 and stats["video_count"] == 0:
            return None

        # Add path info for compatibility
        stats["graph_path"] = "SQLite database"

        logger.debug("Got knowledge graph stats from SQL: %s", stats)
        return stats
    except Exception:
        logger.debug("SQL knowledge graph stats failed", exc_info=True)
        return None


# ============================================================
# VIDEO METADATA (queryable fields from SQLite)
# ============================================================


def get_video_metadata(video_id: str) -> dict[str, Any] | None:
    """Get queryable video metadata from SQLite.

    SQLite is the source of truth for queryable metadata fields (description,
    view_count, like_count). This function retrieves them.

    Args:
        video_id: Natural key (e.g., YouTube video ID).

    Returns:
        Dict with queryable metadata, or None if video not found or DB unavailable.
    """
    db = _get_db()
    if db is None:
        return None

    try:
        cursor = db.execute(
            """
            SELECT
                description,
                view_count,
                like_count
            FROM videos
            WHERE video_id = ?
            """,
            (video_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return {
            "description": row["description"],
            "view_count": row["view_count"],
            "like_count": row["like_count"],
        }
    except Exception:
        logger.debug("Failed to get video metadata from SQL", exc_info=True)
        return None


def get_video_tags(video_id: str) -> list[str] | None:
    """Get video tags from SQLite.

    SQLite is the source of truth for tags. This function retrieves them.

    Args:
        video_id: Natural key (e.g., YouTube video ID).

    Returns:
        List of tag strings, or None if video not found or DB unavailable.
    """
    db = _get_db()
    if db is None:
        return None

    try:
        from claudetube.db.repos.videos import VideoRepository

        repo = VideoRepository(db)
        video = repo.get_by_video_id(video_id)
        if video is None:
            return None

        video_uuid = video["id"]

        cursor = db.execute(
            "SELECT tag FROM video_tags WHERE video_id = ? ORDER BY tag",
            (video_uuid,),
        )
        return [row["tag"] for row in cursor.fetchall()]
    except Exception:
        logger.debug("Failed to get video tags from SQL", exc_info=True)
        return None


def get_full_video_metadata(video_id: str) -> dict[str, Any] | None:
    """Get full video metadata from SQLite.

    Combines all video table fields including queryable metadata.
    Useful for MCP tools that need complete video info.

    Args:
        video_id: Natural key (e.g., YouTube video ID).

    Returns:
        Full video dict from database, or None if not found.
    """
    db = _get_db()
    if db is None:
        return None

    try:
        from claudetube.db.repos.videos import VideoRepository

        repo = VideoRepository(db)
        video = repo.get_by_video_id(video_id)
        if video is None:
            return None

        # Also get tags
        video_uuid = video["id"]
        cursor = db.execute(
            "SELECT tag FROM video_tags WHERE video_id = ?",
            (video_uuid,),
        )
        tags = [row["tag"] for row in cursor.fetchall()]
        video["tags"] = tags

        return video
    except Exception:
        logger.debug("Failed to get full video metadata from SQL", exc_info=True)
        return None
