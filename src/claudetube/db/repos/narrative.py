"""Narrative repository for CRUD operations on the narrative_structures table.

Manages narrative structure analysis results including video type classification
and section counting.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database


class NarrativeRepository:
    """Repository for narrative structure operations.

    Narrative structures capture video type classification and section analysis.
    Each video can have at most one narrative structure record
    (UNIQUE on video_uuid).

    Valid video types:
    - coding_tutorial, lecture, demo, presentation, interview,
      review, vlog, documentary, music_video, other
    """

    # Valid video_type values per schema CHECK constraint
    VALID_VIDEO_TYPES = frozenset(
        [
            "coding_tutorial",
            "lecture",
            "demo",
            "presentation",
            "interview",
            "review",
            "vlog",
            "documentary",
            "music_video",
            "other",
        ]
    )

    def __init__(self, db: Database) -> None:
        """Initialize with a Database instance.

        Args:
            db: Database connection wrapper.
        """
        self.db = db

    def insert(
        self,
        video_uuid: str,
        *,
        video_type: str | None = None,
        section_count: int | None = None,
        file_path: str | None = None,
    ) -> str:
        """Insert a new narrative structure record.

        Args:
            video_uuid: UUID of the parent video record.
            video_type: Optional video type classification.
            section_count: Optional number of sections detected.
            file_path: Optional path to the JSON file.

        Returns:
            The generated UUID for the new narrative structure.

        Raises:
            ValueError: If video_type is invalid or section_count is negative.
            sqlite3.IntegrityError: If video_uuid doesn't exist or already has
                a narrative structure.
        """
        if video_type is not None and video_type not in self.VALID_VIDEO_TYPES:
            msg = f"Invalid video_type: {video_type}. Must be one of {sorted(self.VALID_VIDEO_TYPES)}"
            raise ValueError(msg)

        if section_count is not None and section_count < 0:
            msg = f"section_count must be >= 0, got {section_count}"
            raise ValueError(msg)

        new_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO narrative_structures (
                id, video_id, video_type, section_count, file_path
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (new_id, video_uuid, video_type, section_count, file_path),
        )
        self.db.commit()
        return new_id

    def get_by_video(self, video_uuid: str) -> dict[str, Any] | None:
        """Get the narrative structure for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Dict with narrative structure data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM narrative_structures WHERE video_id = ?",
            (video_uuid,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_uuid(self, narrative_uuid: str) -> dict[str, Any] | None:
        """Get a narrative structure by its UUID.

        Args:
            narrative_uuid: UUID of the narrative structure.

        Returns:
            Dict with narrative structure data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM narrative_structures WHERE id = ?",
            (narrative_uuid,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def exists(self, video_uuid: str) -> bool:
        """Check if a video has a narrative structure.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            True if the video has a narrative structure.
        """
        cursor = self.db.execute(
            "SELECT EXISTS(SELECT 1 FROM narrative_structures WHERE video_id = ?) as has_narrative",
            (video_uuid,),
        )
        row = cursor.fetchone()
        return bool(row["has_narrative"]) if row else False

    def update(
        self,
        video_uuid: str,
        *,
        video_type: str | None = None,
        section_count: int | None = None,
        file_path: str | None = None,
    ) -> bool:
        """Update an existing narrative structure.

        Only updates fields that are explicitly provided (not None).

        Args:
            video_uuid: UUID of the parent video.
            video_type: New video type classification.
            section_count: New section count.
            file_path: New file path.

        Returns:
            True if a record was updated, False if not found.

        Raises:
            ValueError: If video_type is invalid or section_count is negative.
        """
        if video_type is not None and video_type not in self.VALID_VIDEO_TYPES:
            msg = f"Invalid video_type: {video_type}. Must be one of {sorted(self.VALID_VIDEO_TYPES)}"
            raise ValueError(msg)

        if section_count is not None and section_count < 0:
            msg = f"section_count must be >= 0, got {section_count}"
            raise ValueError(msg)

        # Build update dynamically for provided fields
        update_fields = []
        update_values = []

        if video_type is not None:
            update_fields.append("video_type = ?")
            update_values.append(video_type)
        if section_count is not None:
            update_fields.append("section_count = ?")
            update_values.append(section_count)
        if file_path is not None:
            update_fields.append("file_path = ?")
            update_values.append(file_path)

        if not update_fields:
            return False

        update_values.append(video_uuid)
        sql = f"UPDATE narrative_structures SET {', '.join(update_fields)} WHERE video_id = ?"

        cursor = self.db.execute(sql, tuple(update_values))
        self.db.commit()
        return cursor.rowcount > 0

    def delete(self, video_uuid: str) -> bool:
        """Delete the narrative structure for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            True if a record was deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM narrative_structures WHERE video_id = ?",
            (video_uuid,),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def list_by_type(self, video_type: str) -> list[dict[str, Any]]:
        """List all narrative structures with a specific video type.

        Args:
            video_type: The video type to filter by.

        Returns:
            List of narrative structures with the given video type.

        Raises:
            ValueError: If video_type is invalid.
        """
        if video_type not in self.VALID_VIDEO_TYPES:
            msg = f"Invalid video_type: {video_type}. Must be one of {sorted(self.VALID_VIDEO_TYPES)}"
            raise ValueError(msg)

        cursor = self.db.execute(
            """
            SELECT ns.*, v.video_id as video_natural_id, v.title as video_title
            FROM narrative_structures ns
            JOIN videos v ON ns.video_id = v.id
            WHERE ns.video_type = ?
            ORDER BY ns.created_at DESC
            """,
            (video_type,),
        )
        return [dict(row) for row in cursor.fetchall()]
