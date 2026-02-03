"""Audio description repository for CRUD operations on the audio_descriptions table.

Manages audio descriptions for accessibility, tracking format, source, and provider.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database


class AudioDescriptionRepository:
    """Repository for audio description operations.

    Audio descriptions provide accessibility information for video content.
    Each video can have multiple audio descriptions (e.g., different formats,
    from different sources).

    Valid sources:
    - 'generated': AI-generated descriptions
    - 'source_track': Existing audio description track from the video
    - 'compiled': Compiled from scene visual descriptions
    """

    # Valid source values per schema CHECK constraint
    VALID_SOURCES = frozenset(["generated", "source_track", "compiled"])

    # Valid format values per schema CHECK constraint
    VALID_FORMATS = frozenset(["vtt", "txt"])

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
        source: str,
        file_path: str,
        *,
        provider: str | None = None,
    ) -> str:
        """Insert a new audio description record.

        Args:
            video_uuid: UUID of the parent video record.
            format_: Format of the audio description ('vtt' or 'txt').
            source: Source type ('generated', 'source_track', 'compiled').
            file_path: Path to the audio description file.
            provider: Optional provider name (e.g., 'anthropic', 'openai').

        Returns:
            The generated UUID for the new audio description.

        Raises:
            ValueError: If format_ or source is invalid.
            sqlite3.IntegrityError: If video_uuid doesn't exist.
        """
        if format_ not in self.VALID_FORMATS:
            msg = f"Invalid format: {format_}. Must be one of {sorted(self.VALID_FORMATS)}"
            raise ValueError(msg)

        if source not in self.VALID_SOURCES:
            msg = f"Invalid source: {source}. Must be one of {sorted(self.VALID_SOURCES)}"
            raise ValueError(msg)

        new_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO audio_descriptions (
                id, video_id, format, source, provider, file_path
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (new_id, video_uuid, format_, source, provider, file_path),
        )
        self.db.commit()
        return new_id

    def get_by_video(self, video_uuid: str) -> list[dict[str, Any]]:
        """Get all audio descriptions for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            List of dicts with audio description data, ordered by created_at.
        """
        cursor = self.db.execute(
            "SELECT * FROM audio_descriptions WHERE video_id = ? ORDER BY created_at",
            (video_uuid,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_by_uuid(self, ad_uuid: str) -> dict[str, Any] | None:
        """Get an audio description by its UUID.

        Args:
            ad_uuid: UUID of the audio description.

        Returns:
            Dict with audio description data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM audio_descriptions WHERE id = ?",
            (ad_uuid,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_format(
        self,
        video_uuid: str,
        format_: str,
    ) -> dict[str, Any] | None:
        """Get an audio description by video and format.

        Args:
            video_uuid: UUID of the parent video.
            format_: Format to look for ('vtt' or 'txt').

        Returns:
            Dict with audio description data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM audio_descriptions WHERE video_id = ? AND format = ?",
            (video_uuid, format_),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def has_ad(self, video_uuid: str) -> bool:
        """Check if a video has any audio description.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            True if the video has at least one audio description.
        """
        cursor = self.db.execute(
            "SELECT EXISTS(SELECT 1 FROM audio_descriptions WHERE video_id = ?) as has_ad",
            (video_uuid,),
        )
        row = cursor.fetchone()
        return bool(row["has_ad"]) if row else False

    def delete(self, ad_uuid: str) -> bool:
        """Delete an audio description by its UUID.

        Args:
            ad_uuid: UUID of the audio description to delete.

        Returns:
            True if a record was deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM audio_descriptions WHERE id = ?",
            (ad_uuid,),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def delete_by_video(self, video_uuid: str) -> int:
        """Delete all audio descriptions for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Number of records deleted.
        """
        cursor = self.db.execute(
            "DELETE FROM audio_descriptions WHERE video_id = ?",
            (video_uuid,),
        )
        self.db.commit()
        return cursor.rowcount

    def count_by_video(self, video_uuid: str) -> int:
        """Count audio descriptions for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Number of audio descriptions for the video.
        """
        cursor = self.db.execute(
            "SELECT COUNT(*) as cnt FROM audio_descriptions WHERE video_id = ?",
            (video_uuid,),
        )
        row = cursor.fetchone()
        return row["cnt"] if row else 0
