"""Pipeline repository for CRUD operations on the pipeline_steps table.

Manages processing state tracking for all video processing operations.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database


# Valid step types per schema CHECK constraint
VALID_STEP_TYPES = frozenset([
    "download",
    "audio_extract",
    "transcribe",
    "scene_detect",
    "keyframe_extract",
    "visual_analyze",
    "entity_extract",
    "deep_analyze",
    "focus_analyze",
    "narrative_detect",
    "change_detect",
    "code_track",
    "people_track",
    "ad_generate",
    "knowledge_index",
    "embed",
])

# Valid statuses per schema CHECK constraint
VALID_STATUSES = frozenset(["pending", "running", "completed", "failed", "skipped"])


class PipelineRepository:
    """Repository for pipeline step operations.

    Pipeline steps track the processing state for all video operations.
    This is the single source of truth for "what has been processed?"
    replacing scattered boolean flags on videos and scenes tables.

    Each step tracks: what was done, by whom (provider/model), when,
    with what config, and any error messages on failure.
    """

    def __init__(self, db: Database) -> None:
        """Initialize with a Database instance.

        Args:
            db: Database connection wrapper.
        """
        self.db = db

    def record_step(
        self,
        video_uuid: str,
        step_type: str,
        status: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        scene_id: int | None = None,
        config: str | None = None,
        error_message: str | None = None,
    ) -> str:
        """Record a new pipeline step.

        Args:
            video_uuid: UUID of the video.
            step_type: Type of processing step (must be valid enum value).
            status: Current status (must be valid enum value).
            provider: Optional provider name (e.g., 'whisper', 'anthropic').
            model: Optional model name (e.g., 'small', 'claude-3').
            scene_id: Optional scene ID if step is per-scene.
            config: Optional JSON config string for step-specific params.
            error_message: Optional error message (for failed status).

        Returns:
            The generated UUID for the new pipeline step.

        Raises:
            ValueError: If step_type or status is invalid, or scene_id < 0.
        """
        if step_type not in VALID_STEP_TYPES:
            msg = f"Invalid step_type: {step_type}. Must be one of {sorted(VALID_STEP_TYPES)}"
            raise ValueError(msg)

        if status not in VALID_STATUSES:
            msg = f"Invalid status: {status}. Must be one of {sorted(VALID_STATUSES)}"
            raise ValueError(msg)

        if scene_id is not None and scene_id < 0:
            msg = f"scene_id must be >= 0, got {scene_id}"
            raise ValueError(msg)

        new_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        # Set started_at if status is running
        started_at = now if status == "running" else None
        # Set completed_at if status is completed/failed/skipped
        completed_at = now if status in ("completed", "failed", "skipped") else None

        self.db.execute(
            """
            INSERT INTO pipeline_steps (
                id, video_id, step_type, status, provider, model,
                scene_id, config, error_message, started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                video_uuid,
                step_type,
                status,
                provider,
                model,
                scene_id,
                config,
                error_message,
                started_at,
                completed_at,
            ),
        )
        self.db.commit()
        return new_id

    def update_step(
        self,
        step_uuid: str,
        status: str,
        *,
        error_message: str | None = None,
        completed_at: str | None = None,
    ) -> bool:
        """Update an existing pipeline step.

        Args:
            step_uuid: UUID of the step to update.
            status: New status value.
            error_message: Optional error message (for failed status).
            completed_at: Optional completion timestamp. Auto-set if not provided
                         and status is completed/failed/skipped.

        Returns:
            True if step was updated, False if not found.

        Raises:
            ValueError: If status is invalid.
        """
        if status not in VALID_STATUSES:
            msg = f"Invalid status: {status}. Must be one of {sorted(VALID_STATUSES)}"
            raise ValueError(msg)

        # Auto-set timestamps
        now = datetime.now().isoformat()
        if completed_at is None and status in ("completed", "failed", "skipped"):
            completed_at = now

        # Build update query
        if status == "running":
            cursor = self.db.execute(
                """
                UPDATE pipeline_steps
                SET status = ?, started_at = COALESCE(started_at, ?), error_message = ?
                WHERE id = ?
                """,
                (status, now, error_message, step_uuid),
            )
        else:
            cursor = self.db.execute(
                """
                UPDATE pipeline_steps
                SET status = ?, error_message = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, error_message, completed_at, step_uuid),
            )

        self.db.commit()
        return cursor.rowcount > 0

    def get_by_uuid(self, uuid_: str) -> dict[str, Any] | None:
        """Get a pipeline step by its UUID.

        Args:
            uuid_: The UUID primary key.

        Returns:
            Dict with step data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM pipeline_steps WHERE id = ?",
            (uuid_,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_steps(self, video_uuid: str) -> list[dict[str, Any]]:
        """Get all pipeline steps for a video.

        Args:
            video_uuid: UUID of the video.

        Returns:
            List of step dicts, ordered by created_at.
        """
        cursor = self.db.execute(
            """
            SELECT * FROM pipeline_steps
            WHERE video_id = ?
            ORDER BY created_at
            """,
            (video_uuid,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_step(
        self,
        video_uuid: str,
        step_type: str,
        scene_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Get the latest step matching criteria.

        Args:
            video_uuid: UUID of the video.
            step_type: Type of processing step.
            scene_id: Optional scene ID to filter by.

        Returns:
            Dict with step data, or None if not found.
        """
        if scene_id is not None:
            cursor = self.db.execute(
                """
                SELECT * FROM pipeline_steps
                WHERE video_id = ? AND step_type = ? AND scene_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (video_uuid, step_type, scene_id),
            )
        else:
            cursor = self.db.execute(
                """
                SELECT * FROM pipeline_steps
                WHERE video_id = ? AND step_type = ? AND scene_id IS NULL
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (video_uuid, step_type),
            )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_failed(self, video_uuid: str | None = None) -> list[dict[str, Any]]:
        """Get all failed pipeline steps.

        Args:
            video_uuid: Optional UUID to filter by video.

        Returns:
            List of failed step dicts.
        """
        if video_uuid is not None:
            cursor = self.db.execute(
                """
                SELECT * FROM pipeline_steps
                WHERE video_id = ? AND status = 'failed'
                ORDER BY created_at DESC
                """,
                (video_uuid,),
            )
        else:
            cursor = self.db.execute(
                """
                SELECT * FROM pipeline_steps
                WHERE status = 'failed'
                ORDER BY created_at DESC
                """
            )
        return [dict(row) for row in cursor.fetchall()]

    def get_incomplete_scenes(
        self,
        video_uuid: str,
        step_type: str,
    ) -> list[int]:
        """Get scene IDs that don't have a completed step of the given type.

        This is useful for finding which scenes still need processing.

        Args:
            video_uuid: UUID of the video.
            step_type: Type of processing step.

        Returns:
            List of scene_ids without completed step.
        """
        # Get all scene IDs from the scenes table
        cursor = self.db.execute(
            """
            SELECT s.scene_id
            FROM scenes s
            WHERE s.video_id = ?
            AND NOT EXISTS (
                SELECT 1 FROM pipeline_steps p
                WHERE p.video_id = s.video_id
                AND p.scene_id = s.scene_id
                AND p.step_type = ?
                AND p.status = 'completed'
            )
            ORDER BY s.scene_id
            """,
            (video_uuid, step_type),
        )
        return [row["scene_id"] for row in cursor.fetchall()]

    def is_step_complete(
        self,
        video_uuid: str,
        step_type: str,
        scene_id: int | None = None,
    ) -> bool:
        """Check if a step has been completed.

        Args:
            video_uuid: UUID of the video.
            step_type: Type of processing step.
            scene_id: Optional scene ID for per-scene steps.

        Returns:
            True if completed step exists, False otherwise.
        """
        if scene_id is not None:
            cursor = self.db.execute(
                """
                SELECT 1 FROM pipeline_steps
                WHERE video_id = ? AND step_type = ? AND scene_id = ?
                AND status = 'completed'
                LIMIT 1
                """,
                (video_uuid, step_type, scene_id),
            )
        else:
            cursor = self.db.execute(
                """
                SELECT 1 FROM pipeline_steps
                WHERE video_id = ? AND step_type = ? AND scene_id IS NULL
                AND status = 'completed'
                LIMIT 1
                """,
                (video_uuid, step_type),
            )
        return cursor.fetchone() is not None

    def get_running(self, video_uuid: str | None = None) -> list[dict[str, Any]]:
        """Get all running pipeline steps.

        Args:
            video_uuid: Optional UUID to filter by video.

        Returns:
            List of running step dicts.
        """
        if video_uuid is not None:
            cursor = self.db.execute(
                """
                SELECT * FROM pipeline_steps
                WHERE video_id = ? AND status = 'running'
                ORDER BY started_at
                """,
                (video_uuid,),
            )
        else:
            cursor = self.db.execute(
                """
                SELECT * FROM pipeline_steps
                WHERE status = 'running'
                ORDER BY started_at
                """
            )
        return [dict(row) for row in cursor.fetchall()]

    def get_pending(self, video_uuid: str | None = None) -> list[dict[str, Any]]:
        """Get all pending pipeline steps.

        Args:
            video_uuid: Optional UUID to filter by video.

        Returns:
            List of pending step dicts.
        """
        if video_uuid is not None:
            cursor = self.db.execute(
                """
                SELECT * FROM pipeline_steps
                WHERE video_id = ? AND status = 'pending'
                ORDER BY created_at
                """,
                (video_uuid,),
            )
        else:
            cursor = self.db.execute(
                """
                SELECT * FROM pipeline_steps
                WHERE status = 'pending'
                ORDER BY created_at
                """
            )
        return [dict(row) for row in cursor.fetchall()]

    def delete_step(self, step_uuid: str) -> bool:
        """Delete a pipeline step by its UUID.

        Args:
            step_uuid: UUID of the step to delete.

        Returns:
            True if deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM pipeline_steps WHERE id = ?",
            (step_uuid,),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def delete_steps(self, video_uuid: str) -> int:
        """Delete all pipeline steps for a video.

        Args:
            video_uuid: UUID of the video.

        Returns:
            Number of steps deleted.
        """
        cursor = self.db.execute(
            "DELETE FROM pipeline_steps WHERE video_id = ?",
            (video_uuid,),
        )
        self.db.commit()
        return cursor.rowcount

    def count_by_status(self, video_uuid: str) -> dict[str, int]:
        """Count steps by status for a video.

        Args:
            video_uuid: UUID of the video.

        Returns:
            Dict mapping status to count.
        """
        cursor = self.db.execute(
            """
            SELECT status, COUNT(*) as cnt
            FROM pipeline_steps
            WHERE video_id = ?
            GROUP BY status
            """,
            (video_uuid,),
        )
        return {row["status"]: row["cnt"] for row in cursor.fetchall()}
