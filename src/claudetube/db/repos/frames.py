"""Frame repository for CRUD operations on the frames table.

Manages extracted frames from videos including thumbnails, keyframes,
drill (quick) frames, and high-quality frames.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database


class FrameRepository:
    """Repository for frame operations.

    Frames are extracted images from videos at specific timestamps.
    Types include:
    - drill: Quick/low-quality frames for rapid scanning
    - hq: High-quality frames for text/code reading
    - keyframe: Representative frames per scene
    - thumbnail: Video thumbnail image

    The is_thumbnail flag is orthogonal to extraction_type - any frame
    can be marked as the thumbnail. scene_id is NULL for video-level
    extractions (drill, hq without scene context).
    """

    # Valid extraction types per schema CHECK constraint
    VALID_EXTRACTION_TYPES = frozenset(["drill", "hq", "keyframe", "thumbnail"])

    # Valid quality tiers per schema CHECK constraint
    VALID_QUALITY_TIERS = frozenset(["lowest", "low", "medium", "high", "highest"])

    def __init__(self, db: Database) -> None:
        """Initialize with a Database instance.

        Args:
            db: Database connection wrapper.
        """
        self.db = db

    def insert(
        self,
        video_uuid: str,
        timestamp: float,
        extraction_type: str,
        file_path: str,
        *,
        scene_id: int | None = None,
        quality_tier: str | None = None,
        is_thumbnail: bool = False,
        width: int | None = None,
        height: int | None = None,
        file_size_bytes: int | None = None,
    ) -> str:
        """Insert a new frame record.

        Args:
            video_uuid: UUID of the parent video record.
            timestamp: Frame timestamp in seconds.
            extraction_type: Type of extraction (drill, hq, keyframe, thumbnail).
            file_path: Relative path to the frame file in cache.
            scene_id: Scene identifier (NULL for video-level extractions).
            quality_tier: Quality tier (lowest, low, medium, high, highest).
            is_thumbnail: Whether this is the video's thumbnail image.
            width: Frame width in pixels.
            height: Frame height in pixels.
            file_size_bytes: File size in bytes.

        Returns:
            The generated UUID for the new frame.

        Raises:
            ValueError: If extraction_type or quality_tier is invalid.
            sqlite3.IntegrityError: If video_uuid doesn't exist or constraints violated.
        """
        if extraction_type not in self.VALID_EXTRACTION_TYPES:
            msg = f"Invalid extraction_type: {extraction_type}. Must be one of {sorted(self.VALID_EXTRACTION_TYPES)}"
            raise ValueError(msg)

        if quality_tier is not None and quality_tier not in self.VALID_QUALITY_TIERS:
            msg = f"Invalid quality_tier: {quality_tier}. Must be one of {sorted(self.VALID_QUALITY_TIERS)}"
            raise ValueError(msg)

        if timestamp < 0:
            msg = f"timestamp must be >= 0, got {timestamp}"
            raise ValueError(msg)

        if scene_id is not None and scene_id < 0:
            msg = f"scene_id must be >= 0 or None, got {scene_id}"
            raise ValueError(msg)

        new_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO frames (
                id, video_id, scene_id, timestamp, extraction_type,
                quality_tier, is_thumbnail, width, height, file_size_bytes, file_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                video_uuid,
                scene_id,
                timestamp,
                extraction_type,
                quality_tier,
                1 if is_thumbnail else 0,
                width,
                height,
                file_size_bytes,
                file_path,
            ),
        )
        self.db.commit()
        return new_id

    def get_by_uuid(self, uuid_: str) -> dict[str, Any] | None:
        """Get a frame by its UUID.

        Args:
            uuid_: The UUID primary key.

        Returns:
            Dict with frame data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM frames WHERE id = ?",
            (uuid_,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_video(self, video_uuid: str) -> list[dict[str, Any]]:
        """Get all frames for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            List of dicts with frame data, ordered by timestamp.
        """
        cursor = self.db.execute(
            "SELECT * FROM frames WHERE video_id = ? ORDER BY timestamp",
            (video_uuid,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_by_scene(
        self,
        video_uuid: str,
        scene_id: int,
    ) -> list[dict[str, Any]]:
        """Get all frames for a specific scene.

        Args:
            video_uuid: UUID of the parent video.
            scene_id: The scene identifier.

        Returns:
            List of dicts with frame data, ordered by timestamp.
        """
        cursor = self.db.execute(
            "SELECT * FROM frames WHERE video_id = ? AND scene_id = ? ORDER BY timestamp",
            (video_uuid, scene_id),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_by_type(
        self,
        video_uuid: str,
        extraction_type: str,
    ) -> list[dict[str, Any]]:
        """Get all frames of a specific extraction type for a video.

        Args:
            video_uuid: UUID of the parent video.
            extraction_type: The extraction type to filter by.

        Returns:
            List of dicts with frame data, ordered by timestamp.
        """
        cursor = self.db.execute(
            "SELECT * FROM frames WHERE video_id = ? AND extraction_type = ? ORDER BY timestamp",
            (video_uuid, extraction_type),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_thumbnail(self, video_uuid: str) -> dict[str, Any] | None:
        """Get the thumbnail frame for a video.

        Returns the frame marked with is_thumbnail=1. A video should
        have at most one thumbnail.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Dict with frame data, or None if no thumbnail exists.
        """
        cursor = self.db.execute(
            "SELECT * FROM frames WHERE video_id = ? AND is_thumbnail = 1",
            (video_uuid,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_keyframes(
        self,
        video_uuid: str,
        scene_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get keyframes for a video, optionally filtered by scene.

        Keyframes are representative frames per scene (extraction_type='keyframe').

        Args:
            video_uuid: UUID of the parent video.
            scene_id: Optional scene identifier to filter by.

        Returns:
            List of dicts with frame data, ordered by timestamp.
        """
        if scene_id is not None:
            cursor = self.db.execute(
                """
                SELECT * FROM frames
                WHERE video_id = ? AND extraction_type = 'keyframe' AND scene_id = ?
                ORDER BY timestamp
                """,
                (video_uuid, scene_id),
            )
        else:
            cursor = self.db.execute(
                """
                SELECT * FROM frames
                WHERE video_id = ? AND extraction_type = 'keyframe'
                ORDER BY timestamp
                """,
                (video_uuid,),
            )
        return [dict(row) for row in cursor.fetchall()]

    def count_by_type(self, video_uuid: str) -> dict[str, int]:
        """Count frames by extraction type for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Dict mapping extraction_type to count.
        """
        cursor = self.db.execute(
            """
            SELECT extraction_type, COUNT(*) as cnt
            FROM frames
            WHERE video_id = ?
            GROUP BY extraction_type
            """,
            (video_uuid,),
        )
        return {row["extraction_type"]: row["cnt"] for row in cursor.fetchall()}

    def set_thumbnail(self, frame_uuid: str) -> bool:
        """Set a frame as the video's thumbnail.

        This will unset any existing thumbnail for the same video
        and set the specified frame as the thumbnail.

        Args:
            frame_uuid: UUID of the frame to make the thumbnail.

        Returns:
            True if successful, False if frame not found.
        """
        # Get the frame to find its video
        frame = self.get_by_uuid(frame_uuid)
        if frame is None:
            return False

        video_uuid = frame["video_id"]

        # Unset any existing thumbnail for this video
        self.db.execute(
            "UPDATE frames SET is_thumbnail = 0 WHERE video_id = ? AND is_thumbnail = 1",
            (video_uuid,),
        )

        # Set the new thumbnail
        self.db.execute(
            "UPDATE frames SET is_thumbnail = 1 WHERE id = ?",
            (frame_uuid,),
        )
        self.db.commit()
        return True

    def delete(self, frame_uuid: str) -> bool:
        """Delete a frame by its UUID.

        Args:
            frame_uuid: UUID of the frame to delete.

        Returns:
            True if a frame was deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM frames WHERE id = ?",
            (frame_uuid,),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def delete_by_video(self, video_uuid: str) -> int:
        """Delete all frames for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Number of frames deleted.
        """
        cursor = self.db.execute(
            "DELETE FROM frames WHERE video_id = ?",
            (video_uuid,),
        )
        self.db.commit()
        return cursor.rowcount

    def delete_by_type(
        self,
        video_uuid: str,
        extraction_type: str,
    ) -> int:
        """Delete all frames of a specific type for a video.

        Args:
            video_uuid: UUID of the parent video.
            extraction_type: The extraction type to delete.

        Returns:
            Number of frames deleted.
        """
        cursor = self.db.execute(
            "DELETE FROM frames WHERE video_id = ? AND extraction_type = ?",
            (video_uuid, extraction_type),
        )
        self.db.commit()
        return cursor.rowcount
