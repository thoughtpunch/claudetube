"""Observation repository for CRUD operations on the observations table.

Manages observations recorded during video analysis for progressive learning.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database


class ObservationRepository:
    """Repository for observation operations.

    Observations are notes and insights recorded during video analysis,
    tied to specific scenes. They support progressive learning by
    allowing Claude to record what it discovers during video analysis.
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
        obs_type: str,
        content: str,
    ) -> str:
        """Insert a new observation.

        Args:
            video_uuid: UUID of the video this observation is about.
            scene_id: Scene index this observation relates to (0-based).
            obs_type: Type of observation (e.g., "visual", "technical", "note").
            content: The observation content.

        Returns:
            The generated UUID for the new observation.

        Raises:
            ValueError: If scene_id < 0, or obs_type/content is empty.
            sqlite3.IntegrityError: If video_uuid doesn't exist.
        """
        if scene_id < 0:
            msg = f"scene_id must be >= 0, got {scene_id}"
            raise ValueError(msg)

        if not obs_type or not obs_type.strip():
            msg = "Observation type cannot be empty"
            raise ValueError(msg)

        if not content or not content.strip():
            msg = "Observation content cannot be empty"
            raise ValueError(msg)

        obs_type = obs_type.strip()
        content = content.strip()

        new_id = str(uuid.uuid4())

        self.db.execute(
            """
            INSERT INTO observations (id, video_id, scene_id, type, content)
            VALUES (?, ?, ?, ?, ?)
            """,
            (new_id, video_uuid, scene_id, obs_type, content),
        )
        self.db.commit()
        return new_id

    def get_by_uuid(self, uuid_: str) -> dict[str, Any] | None:
        """Get an observation by its UUID.

        Args:
            uuid_: The UUID primary key.

        Returns:
            Dict with observation data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM observations WHERE id = ?",
            (uuid_,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_video(self, video_uuid: str) -> list[dict[str, Any]]:
        """Get all observations for a video.

        Args:
            video_uuid: UUID of the video.

        Returns:
            List of observation dicts, ordered by scene_id then created_at.
        """
        cursor = self.db.execute(
            """
            SELECT * FROM observations
            WHERE video_id = ?
            ORDER BY scene_id, created_at
            """,
            (video_uuid,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_by_scene(self, video_uuid: str, scene_id: int) -> list[dict[str, Any]]:
        """Get all observations for a specific scene.

        Args:
            video_uuid: UUID of the video.
            scene_id: The scene identifier (0-indexed).

        Returns:
            List of observation dicts for this scene.
        """
        cursor = self.db.execute(
            """
            SELECT * FROM observations
            WHERE video_id = ? AND scene_id = ?
            ORDER BY created_at
            """,
            (video_uuid, scene_id),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_by_type(
        self,
        video_uuid: str,
        obs_type: str,
    ) -> list[dict[str, Any]]:
        """Get all observations of a specific type for a video.

        Args:
            video_uuid: UUID of the video.
            obs_type: Type of observations to retrieve.

        Returns:
            List of matching observation dicts.
        """
        cursor = self.db.execute(
            """
            SELECT * FROM observations
            WHERE video_id = ? AND type = ?
            ORDER BY scene_id, created_at
            """,
            (video_uuid, obs_type),
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete(self, obs_uuid: str) -> bool:
        """Delete an observation by its UUID.

        Args:
            obs_uuid: UUID of the observation to delete.

        Returns:
            True if an observation was deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM observations WHERE id = ?",
            (obs_uuid,),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def delete_by_video(self, video_uuid: str) -> int:
        """Delete all observations for a video.

        Args:
            video_uuid: UUID of the video.

        Returns:
            Number of observations deleted.
        """
        cursor = self.db.execute(
            "DELETE FROM observations WHERE video_id = ?",
            (video_uuid,),
        )
        self.db.commit()
        return cursor.rowcount

    def delete_by_scene(self, video_uuid: str, scene_id: int) -> int:
        """Delete all observations for a specific scene.

        Args:
            video_uuid: UUID of the video.
            scene_id: The scene identifier.

        Returns:
            Number of observations deleted.
        """
        cursor = self.db.execute(
            "DELETE FROM observations WHERE video_id = ? AND scene_id = ?",
            (video_uuid, scene_id),
        )
        self.db.commit()
        return cursor.rowcount

    def count_by_video(self, video_uuid: str) -> int:
        """Count observations for a video.

        Args:
            video_uuid: UUID of the video.

        Returns:
            Number of observations.
        """
        cursor = self.db.execute(
            "SELECT COUNT(*) as cnt FROM observations WHERE video_id = ?",
            (video_uuid,),
        )
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    def count_by_scene(self, video_uuid: str, scene_id: int) -> int:
        """Count observations for a specific scene.

        Args:
            video_uuid: UUID of the video.
            scene_id: The scene identifier.

        Returns:
            Number of observations for this scene.
        """
        cursor = self.db.execute(
            """
            SELECT COUNT(*) as cnt FROM observations
            WHERE video_id = ? AND scene_id = ?
            """,
            (video_uuid, scene_id),
        )
        row = cursor.fetchone()
        return row["cnt"] if row else 0
