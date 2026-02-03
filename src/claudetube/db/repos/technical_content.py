"""Technical content repository for CRUD operations on the technical_content table.

Manages per-scene OCR text and code detection data with support for
full-text search on OCR text.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database


class TechnicalContentRepository:
    """Repository for technical content operations.

    Technical content captures OCR text, code detection, and programming
    language identification for each scene. Each video can have one
    technical content record per scene (UNIQUE on video_uuid, scene_id).

    Full-text search is supported via the technical_fts virtual table
    on the ocr_text column.
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
        has_code: bool,
        has_text: bool,
        *,
        provider: str | None = None,
        ocr_text: str | None = None,
        code_language: str | None = None,
        file_path: str | None = None,
    ) -> str:
        """Insert a new technical content record.

        Args:
            video_uuid: UUID of the parent video record.
            scene_id: The scene identifier (0-indexed).
            has_code: Whether code was detected in the scene.
            has_text: Whether text was detected in the scene.
            provider: Optional provider name (e.g., 'anthropic', 'openai').
            ocr_text: Extracted OCR text for FTS search.
            code_language: Detected programming language.
            file_path: Optional path to the JSON file.

        Returns:
            The generated UUID for the new technical content record.

        Raises:
            ValueError: If scene_id < 0.
            sqlite3.IntegrityError: If video_uuid doesn't exist or
                (video_uuid, scene_id) already exists.
        """
        if scene_id < 0:
            msg = f"scene_id must be >= 0, got {scene_id}"
            raise ValueError(msg)

        new_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO technical_content (
                id, video_id, scene_id, provider, has_code, has_text,
                ocr_text, code_language, file_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                video_uuid,
                scene_id,
                provider,
                1 if has_code else 0,
                1 if has_text else 0,
                ocr_text,
                code_language,
                file_path,
            ),
        )
        self.db.commit()
        return new_id

    def get_by_video(self, video_uuid: str) -> list[dict[str, Any]]:
        """Get all technical content records for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            List of dicts with technical content data, ordered by scene_id.
        """
        cursor = self.db.execute(
            "SELECT * FROM technical_content WHERE video_id = ? ORDER BY scene_id",
            (video_uuid,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_by_scene(
        self,
        video_uuid: str,
        scene_id: int,
    ) -> dict[str, Any] | None:
        """Get the technical content for a specific scene.

        Args:
            video_uuid: UUID of the parent video.
            scene_id: The scene identifier (0-indexed).

        Returns:
            Dict with technical content data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM technical_content WHERE video_id = ? AND scene_id = ?",
            (video_uuid, scene_id),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def search_fts(self, query: str) -> list[dict[str, Any]]:
        """Search technical content using full-text search on OCR text.

        Searches via the technical_fts virtual table. Results include
        video context (video_id natural key, title).

        Args:
            query: Search query string. Supports FTS5 syntax.

        Returns:
            List of matching technical content records with video context,
            ordered by relevance.
        """
        escaped_query = self._escape_fts_query(query)

        cursor = self.db.execute(
            """
            SELECT
                tc.*,
                v.video_id as video_natural_id,
                v.title as video_title,
                v.domain as video_domain,
                rank
            FROM technical_fts
            JOIN technical_content tc ON technical_fts.rowid = tc.rowid
            JOIN videos v ON tc.video_id = v.id
            WHERE technical_fts MATCH ?
            ORDER BY rank
            """,
            (escaped_query,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_scenes_with_code(self, video_uuid: str) -> list[dict[str, Any]]:
        """Get all scenes that contain code.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            List of technical content records where has_code = 1.
        """
        cursor = self.db.execute(
            """
            SELECT * FROM technical_content
            WHERE video_id = ? AND has_code = 1
            ORDER BY scene_id
            """,
            (video_uuid,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_scenes_with_text(self, video_uuid: str) -> list[dict[str, Any]]:
        """Get all scenes that contain text.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            List of technical content records where has_text = 1.
        """
        cursor = self.db.execute(
            """
            SELECT * FROM technical_content
            WHERE video_id = ? AND has_text = 1
            ORDER BY scene_id
            """,
            (video_uuid,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete(self, technical_uuid: str) -> bool:
        """Delete a technical content record by its UUID.

        Args:
            technical_uuid: UUID of the technical content to delete.

        Returns:
            True if a record was deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM technical_content WHERE id = ?",
            (technical_uuid,),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def delete_by_video(self, video_uuid: str) -> int:
        """Delete all technical content for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Number of records deleted.
        """
        cursor = self.db.execute(
            "DELETE FROM technical_content WHERE video_id = ?",
            (video_uuid,),
        )
        self.db.commit()
        return cursor.rowcount

    def count_by_video(self, video_uuid: str) -> int:
        """Count technical content records for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Number of technical content records for the video.
        """
        cursor = self.db.execute(
            "SELECT COUNT(*) as cnt FROM technical_content WHERE video_id = ?",
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
