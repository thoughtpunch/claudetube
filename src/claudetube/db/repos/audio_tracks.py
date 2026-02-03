"""Audio track repository for CRUD operations on the audio_tracks table.

Manages extracted audio files associated with videos.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database


class AudioTrackRepository:
    """Repository for audio track operations.

    Audio tracks are extracted audio files (mp3, wav, etc.) from videos.
    Each video can have multiple audio tracks in different formats.
    """

    # Valid audio formats per schema CHECK constraint
    VALID_FORMATS = frozenset(["mp3", "wav", "aac", "m4a", "opus", "flac", "ogg"])

    def __init__(self, db: Database) -> None:
        """Initialize with a Database instance.

        Args:
            db: Database connection wrapper.
        """
        self.db = db

    def insert(
        self,
        video_uuid: str,
        format_: str,
        file_path: str,
        *,
        sample_rate: int | None = None,
        channels: int | None = None,
        bitrate_kbps: int | None = None,
        duration: float | None = None,
        file_size_bytes: int | None = None,
    ) -> str:
        """Insert a new audio track record.

        Args:
            video_uuid: UUID of the parent video record.
            format_: Audio format (mp3, wav, aac, m4a, opus, flac, ogg).
            file_path: Relative path to the audio file in cache.
            sample_rate: Audio sample rate in Hz.
            channels: Number of audio channels (1=mono, 2=stereo).
            bitrate_kbps: Bitrate in kilobits per second.
            duration: Duration in seconds.
            file_size_bytes: File size in bytes.

        Returns:
            The generated UUID for the new audio track.

        Raises:
            ValueError: If format is not valid.
            sqlite3.IntegrityError: If video_uuid doesn't exist or constraints violated.
        """
        if format_ not in self.VALID_FORMATS:
            msg = f"Invalid audio format: {format_}. Must be one of {sorted(self.VALID_FORMATS)}"
            raise ValueError(msg)

        new_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO audio_tracks (
                id, video_id, format, sample_rate, channels,
                bitrate_kbps, duration, file_size_bytes, file_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                video_uuid,
                format_,
                sample_rate,
                channels,
                bitrate_kbps,
                duration,
                file_size_bytes,
                file_path,
            ),
        )
        self.db.commit()
        return new_id

    def get_by_uuid(self, uuid_: str) -> dict[str, Any] | None:
        """Get an audio track by its UUID.

        Args:
            uuid_: The UUID primary key.

        Returns:
            Dict with audio track data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM audio_tracks WHERE id = ?",
            (uuid_,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_video(self, video_uuid: str) -> list[dict[str, Any]]:
        """Get all audio tracks for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            List of dicts with audio track data, ordered by created_at.
        """
        cursor = self.db.execute(
            "SELECT * FROM audio_tracks WHERE video_id = ? ORDER BY created_at",
            (video_uuid,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_by_video_and_format(
        self, video_uuid: str, format_: str
    ) -> dict[str, Any] | None:
        """Get a specific audio track by video and format.

        Args:
            video_uuid: UUID of the parent video.
            format_: Audio format to find.

        Returns:
            Dict with audio track data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM audio_tracks WHERE video_id = ? AND format = ?",
            (video_uuid, format_),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def delete(self, track_uuid: str) -> bool:
        """Delete an audio track by its UUID.

        Args:
            track_uuid: UUID of the audio track to delete.

        Returns:
            True if a track was deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM audio_tracks WHERE id = ?",
            (track_uuid,),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def delete_by_video(self, video_uuid: str) -> int:
        """Delete all audio tracks for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Number of tracks deleted.
        """
        cursor = self.db.execute(
            "DELETE FROM audio_tracks WHERE video_id = ?",
            (video_uuid,),
        )
        self.db.commit()
        return cursor.rowcount
