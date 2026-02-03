"""Scene repository for CRUD operations on the scenes table.

Manages scene segmentation data for videos with support for full-text
search on transcript text.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database


class SceneRepository:
    """Repository for scene operations.

    Scenes represent temporal segments of a video with optional transcript
    text. Each video can have multiple scenes, identified by a sequential
    scene_id starting from 0. The combination (video_uuid, scene_id) is unique.

    Full-text search is supported via the scenes_fts virtual table.
    """

    # Valid segmentation methods per schema CHECK constraint
    VALID_METHODS = frozenset(["transcript", "visual", "hybrid", "chapters"])

    def __init__(self, db: Database) -> None:
        """Initialize with a Database instance.

        Args:
            db: Database connection wrapper.
        """
        self.db = db

    def insert(
        self,
        video_uuid: str,
        scene_id: int,
        start_time: float,
        end_time: float,
        *,
        title: str | None = None,
        transcript_text: str | None = None,
        method: str | None = None,
        relevance_boost: float = 1.0,
    ) -> str:
        """Insert a new scene record.

        Args:
            video_uuid: UUID of the parent video record.
            scene_id: Sequential scene identifier (0-indexed).
            start_time: Scene start time in seconds.
            end_time: Scene end time in seconds.
            title: Optional scene title (e.g., from chapter).
            transcript_text: Transcript text for this scene segment.
            method: Segmentation method (transcript, visual, hybrid, chapters).
            relevance_boost: Relevance boost multiplier (default: 1.0).

        Returns:
            The generated UUID for the new scene.

        Raises:
            ValueError: If start_time >= end_time, method is invalid, or scene_id < 0.
            sqlite3.IntegrityError: If video_uuid doesn't exist or (video_uuid, scene_id) already exists.
        """
        if scene_id < 0:
            msg = f"scene_id must be >= 0, got {scene_id}"
            raise ValueError(msg)

        if start_time >= end_time:
            msg = f"start_time ({start_time}) must be < end_time ({end_time})"
            raise ValueError(msg)

        if method is not None and method not in self.VALID_METHODS:
            msg = (
                f"Invalid method: {method}. Must be one of {sorted(self.VALID_METHODS)}"
            )
            raise ValueError(msg)

        if relevance_boost < 0:
            msg = f"relevance_boost must be >= 0, got {relevance_boost}"
            raise ValueError(msg)

        new_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO scenes (
                id, video_id, scene_id, start_time, end_time,
                title, transcript_text, method, relevance_boost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                video_uuid,
                scene_id,
                start_time,
                end_time,
                title,
                transcript_text,
                method,
                relevance_boost,
            ),
        )
        self.db.commit()
        return new_id

    def bulk_insert(
        self,
        video_uuid: str,
        scenes: list[dict[str, Any]],
    ) -> list[str]:
        """Efficiently insert multiple scenes for a video.

        Each scene dict should contain:
        - scene_id (required): Sequential scene identifier
        - start_time (required): Scene start time in seconds
        - end_time (required): Scene end time in seconds
        - title (optional): Scene title
        - transcript_text (optional): Transcript text for this segment
        - method (optional): Segmentation method

        Args:
            video_uuid: UUID of the parent video record.
            scenes: List of scene dicts to insert.

        Returns:
            List of generated UUIDs for the new scenes.

        Raises:
            ValueError: If any scene has invalid data.
            sqlite3.IntegrityError: If video_uuid doesn't exist or duplicates exist.
        """
        if not scenes:
            return []

        # Validate all scenes first
        for scene in scenes:
            scene_id = scene.get("scene_id")
            start_time = scene.get("start_time")
            end_time = scene.get("end_time")
            method = scene.get("method")

            if scene_id is None:
                msg = "scene_id is required"
                raise ValueError(msg)
            if scene_id < 0:
                msg = f"scene_id must be >= 0, got {scene_id}"
                raise ValueError(msg)
            if start_time is None:
                msg = "start_time is required"
                raise ValueError(msg)
            if end_time is None:
                msg = "end_time is required"
                raise ValueError(msg)
            if start_time >= end_time:
                msg = f"start_time ({start_time}) must be < end_time ({end_time}) for scene {scene_id}"
                raise ValueError(msg)
            if method is not None and method not in self.VALID_METHODS:
                msg = f"Invalid method: {method}. Must be one of {sorted(self.VALID_METHODS)}"
                raise ValueError(msg)

        # Prepare rows for bulk insert
        uuids = []
        rows = []
        for scene in scenes:
            new_id = str(uuid.uuid4())
            uuids.append(new_id)
            rows.append(
                (
                    new_id,
                    video_uuid,
                    scene["scene_id"],
                    scene["start_time"],
                    scene["end_time"],
                    scene.get("title"),
                    scene.get("transcript_text"),
                    scene.get("method"),
                    scene.get("relevance_boost", 1.0),
                )
            )

        self.db.executemany(
            """
            INSERT INTO scenes (
                id, video_id, scene_id, start_time, end_time,
                title, transcript_text, method, relevance_boost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.db.commit()
        return uuids

    def get_by_uuid(self, uuid_: str) -> dict[str, Any] | None:
        """Get a scene by its UUID.

        Args:
            uuid_: The UUID primary key.

        Returns:
            Dict with scene data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM scenes WHERE id = ?",
            (uuid_,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_scene(
        self,
        video_uuid: str,
        scene_id: int,
    ) -> dict[str, Any] | None:
        """Get a specific scene by video UUID and scene_id.

        Args:
            video_uuid: UUID of the parent video.
            scene_id: The scene identifier (0-indexed).

        Returns:
            Dict with scene data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM scenes WHERE video_id = ? AND scene_id = ?",
            (video_uuid, scene_id),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_video(self, video_uuid: str) -> list[dict[str, Any]]:
        """Get all scenes for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            List of dicts with scene data, ordered by scene_id.
        """
        cursor = self.db.execute(
            "SELECT * FROM scenes WHERE video_id = ? ORDER BY scene_id",
            (video_uuid,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_relevance_boost(
        self,
        video_uuid: str,
        scene_id: int,
        boost: float,
    ) -> bool:
        """Update the relevance boost for a scene.

        Args:
            video_uuid: UUID of the parent video.
            scene_id: The scene identifier.
            boost: New relevance boost value (must be >= 0).

        Returns:
            True if the scene was updated, False if not found.

        Raises:
            ValueError: If boost is negative.
        """
        if boost < 0:
            msg = f"relevance_boost must be >= 0, got {boost}"
            raise ValueError(msg)

        cursor = self.db.execute(
            "UPDATE scenes SET relevance_boost = ? WHERE video_id = ? AND scene_id = ?",
            (boost, video_uuid, scene_id),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def search_fts(self, query: str) -> list[dict[str, Any]]:
        """Search scenes using full-text search on transcript_text.

        Searches via the scenes_fts virtual table. Results include
        video context (video_id natural key, title).

        Args:
            query: Search query string. Supports FTS5 syntax.

        Returns:
            List of matching scenes with video context, ordered by relevance.
        """
        escaped_query = self._escape_fts_query(query)

        cursor = self.db.execute(
            """
            SELECT
                s.*,
                v.video_id as video_natural_id,
                v.title as video_title,
                v.domain as video_domain,
                rank
            FROM scenes_fts
            JOIN scenes s ON scenes_fts.rowid = s.rowid
            JOIN videos v ON s.video_id = v.id
            WHERE scenes_fts MATCH ?
            ORDER BY rank
            """,
            (escaped_query,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete(self, scene_uuid: str) -> bool:
        """Delete a scene by its UUID.

        Args:
            scene_uuid: UUID of the scene to delete.

        Returns:
            True if a scene was deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM scenes WHERE id = ?",
            (scene_uuid,),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def delete_by_video(self, video_uuid: str) -> int:
        """Delete all scenes for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Number of scenes deleted.
        """
        cursor = self.db.execute(
            "DELETE FROM scenes WHERE video_id = ?",
            (video_uuid,),
        )
        self.db.commit()
        return cursor.rowcount

    def count_by_video(self, video_uuid: str) -> int:
        """Count scenes for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Number of scenes for the video.
        """
        cursor = self.db.execute(
            "SELECT COUNT(*) as cnt FROM scenes WHERE video_id = ?",
            (video_uuid,),
        )
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    def _escape_fts_query(self, query: str) -> str:
        """Escape a query string for FTS5 MATCH syntax.

        Wraps terms in double quotes to prevent FTS5 syntax errors
        from special characters.

        Args:
            query: Raw search query.

        Returns:
            Escaped query safe for FTS5 MATCH.
        """
        terms = query.split()
        if not terms:
            return '""'
        escaped_terms = ['"' + term.replace('"', '""') + '"' for term in terms]
        return " ".join(escaped_terms)
