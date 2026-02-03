"""Code evolution repository for CRUD operations on the code_evolutions table.

Manages code evolution tracking data for coding tutorials and live coding videos.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database


class CodeEvolutionRepository:
    """Repository for code evolution operations.

    Code evolution tracks how code changes across scenes in coding tutorials
    and live coding videos. Each video can have at most one code evolution
    record (UNIQUE on video_uuid).
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
        *,
        files_tracked: int | None = None,
        total_changes: int | None = None,
        file_path: str | None = None,
    ) -> str:
        """Insert a new code evolution record.

        Args:
            video_uuid: UUID of the parent video record.
            files_tracked: Optional number of files tracked.
            total_changes: Optional total number of code changes.
            file_path: Optional path to the JSON file.

        Returns:
            The generated UUID for the new code evolution record.

        Raises:
            ValueError: If files_tracked or total_changes is negative.
            sqlite3.IntegrityError: If video_uuid doesn't exist or already has
                a code evolution record.
        """
        if files_tracked is not None and files_tracked < 0:
            msg = f"files_tracked must be >= 0, got {files_tracked}"
            raise ValueError(msg)

        if total_changes is not None and total_changes < 0:
            msg = f"total_changes must be >= 0, got {total_changes}"
            raise ValueError(msg)

        new_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO code_evolutions (
                id, video_id, files_tracked, total_changes, file_path
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (new_id, video_uuid, files_tracked, total_changes, file_path),
        )
        self.db.commit()
        return new_id

    def get_by_video(self, video_uuid: str) -> dict[str, Any] | None:
        """Get the code evolution for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Dict with code evolution data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM code_evolutions WHERE video_id = ?",
            (video_uuid,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_uuid(self, evolution_uuid: str) -> dict[str, Any] | None:
        """Get a code evolution by its UUID.

        Args:
            evolution_uuid: UUID of the code evolution.

        Returns:
            Dict with code evolution data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM code_evolutions WHERE id = ?",
            (evolution_uuid,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def exists(self, video_uuid: str) -> bool:
        """Check if a video has code evolution data.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            True if the video has code evolution data.
        """
        cursor = self.db.execute(
            "SELECT EXISTS(SELECT 1 FROM code_evolutions WHERE video_id = ?) as has_evolution",
            (video_uuid,),
        )
        row = cursor.fetchone()
        return bool(row["has_evolution"]) if row else False

    def update(
        self,
        video_uuid: str,
        *,
        files_tracked: int | None = None,
        total_changes: int | None = None,
        file_path: str | None = None,
    ) -> bool:
        """Update an existing code evolution record.

        Only updates fields that are explicitly provided (not None).

        Args:
            video_uuid: UUID of the parent video.
            files_tracked: New number of files tracked.
            total_changes: New total changes count.
            file_path: New file path.

        Returns:
            True if a record was updated, False if not found.

        Raises:
            ValueError: If files_tracked or total_changes is negative.
        """
        if files_tracked is not None and files_tracked < 0:
            msg = f"files_tracked must be >= 0, got {files_tracked}"
            raise ValueError(msg)

        if total_changes is not None and total_changes < 0:
            msg = f"total_changes must be >= 0, got {total_changes}"
            raise ValueError(msg)

        # Build update dynamically for provided fields
        update_fields = []
        update_values = []

        if files_tracked is not None:
            update_fields.append("files_tracked = ?")
            update_values.append(files_tracked)
        if total_changes is not None:
            update_fields.append("total_changes = ?")
            update_values.append(total_changes)
        if file_path is not None:
            update_fields.append("file_path = ?")
            update_values.append(file_path)

        if not update_fields:
            return False

        update_values.append(video_uuid)
        sql = (
            f"UPDATE code_evolutions SET {', '.join(update_fields)} WHERE video_id = ?"
        )

        cursor = self.db.execute(sql, tuple(update_values))
        self.db.commit()
        return cursor.rowcount > 0

    def delete(self, video_uuid: str) -> bool:
        """Delete the code evolution for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            True if a record was deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM code_evolutions WHERE video_id = ?",
            (video_uuid,),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def list_all(self) -> list[dict[str, Any]]:
        """List all code evolution records with video context.

        Returns:
            List of code evolution records with video natural_id and title.
        """
        cursor = self.db.execute(
            """
            SELECT ce.*, v.video_id as video_natural_id, v.title as video_title
            FROM code_evolutions ce
            JOIN videos v ON ce.video_id = v.id
            ORDER BY ce.created_at DESC
            """,
        )
        return [dict(row) for row in cursor.fetchall()]
