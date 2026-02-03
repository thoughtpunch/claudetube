"""Visual description repository for CRUD operations on the visual_descriptions table.

Manages AI-generated visual descriptions for video scenes with support for
full-text search on description content.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database


class VisualDescriptionRepository:
    """Repository for visual description operations.

    Visual descriptions are AI-generated scene descriptions that capture
    what's visually happening in each scene. Each video can have one
    visual description per scene (UNIQUE on video_uuid, scene_id).

    Full-text search is supported via the visual_fts virtual table.
    """

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
        description: str,
        *,
        provider: str | None = None,
        file_path: str | None = None,
    ) -> str:
        """Insert a new visual description.

        Args:
            video_uuid: UUID of the parent video record.
            scene_id: The scene identifier (0-indexed).
            description: The visual description text.
            provider: Optional provider name (e.g., 'anthropic', 'openai').
            file_path: Optional path to the JSON file.

        Returns:
            The generated UUID for the new visual description.

        Raises:
            ValueError: If scene_id < 0 or description is empty.
            sqlite3.IntegrityError: If video_uuid doesn't exist or
                (video_uuid, scene_id) already exists.
        """
        if scene_id < 0:
            msg = f"scene_id must be >= 0, got {scene_id}"
            raise ValueError(msg)

        if not description or not description.strip():
            msg = "description cannot be empty"
            raise ValueError(msg)

        new_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO visual_descriptions (
                id, video_id, scene_id, description, provider, file_path
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (new_id, video_uuid, scene_id, description, provider, file_path),
        )
        self.db.commit()
        return new_id

    def get_by_video(self, video_uuid: str) -> list[dict[str, Any]]:
        """Get all visual descriptions for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            List of dicts with visual description data, ordered by scene_id.
        """
        cursor = self.db.execute(
            "SELECT * FROM visual_descriptions WHERE video_id = ? ORDER BY scene_id",
            (video_uuid,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_by_scene(
        self,
        video_uuid: str,
        scene_id: int,
    ) -> dict[str, Any] | None:
        """Get the visual description for a specific scene.

        Args:
            video_uuid: UUID of the parent video.
            scene_id: The scene identifier (0-indexed).

        Returns:
            Dict with visual description data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM visual_descriptions WHERE video_id = ? AND scene_id = ?",
            (video_uuid, scene_id),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def search_fts(self, query: str) -> list[dict[str, Any]]:
        """Search visual descriptions using full-text search.

        Searches via the visual_fts virtual table. Results include
        video context (video_id natural key, title).

        Args:
            query: Search query string. Supports FTS5 syntax.

        Returns:
            List of matching visual descriptions with video context,
            ordered by relevance.
        """
        escaped_query = self._escape_fts_query(query)

        cursor = self.db.execute(
            """
            SELECT
                vd.*,
                v.video_id as video_natural_id,
                v.title as video_title,
                v.domain as video_domain,
                rank
            FROM visual_fts
            JOIN visual_descriptions vd ON visual_fts.rowid = vd.rowid
            JOIN videos v ON vd.video_id = v.id
            WHERE visual_fts MATCH ?
            ORDER BY rank
            """,
            (escaped_query,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete(self, visual_uuid: str) -> bool:
        """Delete a visual description by its UUID.

        Args:
            visual_uuid: UUID of the visual description to delete.

        Returns:
            True if a record was deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM visual_descriptions WHERE id = ?",
            (visual_uuid,),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def delete_by_video(self, video_uuid: str) -> int:
        """Delete all visual descriptions for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Number of records deleted.
        """
        cursor = self.db.execute(
            "DELETE FROM visual_descriptions WHERE video_id = ?",
            (video_uuid,),
        )
        self.db.commit()
        return cursor.rowcount

    def count_by_video(self, video_uuid: str) -> int:
        """Count visual descriptions for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Number of visual descriptions for the video.
        """
        cursor = self.db.execute(
            "SELECT COUNT(*) as cnt FROM visual_descriptions WHERE video_id = ?",
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
