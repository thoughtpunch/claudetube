"""Transcription repository for CRUD operations on the transcriptions table.

Manages transcription records with support for multiple providers,
primary designation, and full-text search.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database


class TranscriptionRepository:
    """Repository for transcription operations.

    A video can have multiple transcriptions from different providers
    (YouTube subtitles, Whisper, Deepgram, etc.). One transcription
    per video is marked as is_primary for default use.

    Full-text search is supported via the transcriptions_fts virtual table.
    """

    # Valid providers per schema CHECK constraint
    VALID_PROVIDERS = frozenset(
        [
            "youtube_subtitles",
            "whisper",
            "deepgram",
            "openai",
            "manual",
        ]
    )

    # Valid formats per schema CHECK constraint
    VALID_FORMATS = frozenset(["srt", "txt", "vtt"])

    def __init__(self, db: Database) -> None:
        """Initialize with a Database instance.

        Args:
            db: Database connection wrapper.
        """
        self.db = db

    def insert(
        self,
        video_uuid: str,
        provider: str,
        format_: str,
        file_path: str,
        *,
        audio_track_id: str | None = None,
        model: str | None = None,
        language: str | None = None,
        full_text: str | None = None,
        word_count: int | None = None,
        duration: float | None = None,
        confidence: float | None = None,
        file_size_bytes: int | None = None,
        is_primary: bool = False,
    ) -> str:
        """Insert a new transcription record.

        Args:
            video_uuid: UUID of the parent video record.
            provider: Transcription provider (youtube_subtitles, whisper, etc.).
            format_: Transcript format (srt, txt, vtt).
            file_path: Relative path to the transcript file in cache.
            audio_track_id: Optional UUID of the source audio track.
            model: Model used (e.g., 'small', 'medium' for Whisper).
            language: Language code (e.g., 'en', 'es').
            full_text: Complete transcript text for FTS indexing.
            word_count: Number of words in the transcript.
            duration: Duration covered by the transcript in seconds.
            confidence: Confidence score (0.0 to 1.0).
            file_size_bytes: File size in bytes.
            is_primary: Whether this is the primary transcript for the video.

        Returns:
            The generated UUID for the new transcription.

        Raises:
            ValueError: If provider or format is not valid.
            sqlite3.IntegrityError: If video_uuid doesn't exist or constraints violated.
        """
        if provider not in self.VALID_PROVIDERS:
            msg = f"Invalid provider: {provider}. Must be one of {sorted(self.VALID_PROVIDERS)}"
            raise ValueError(msg)

        if format_ not in self.VALID_FORMATS:
            msg = f"Invalid format: {format_}. Must be one of {sorted(self.VALID_FORMATS)}"
            raise ValueError(msg)

        new_id = str(uuid.uuid4())

        # If this is primary, unset any existing primary first
        if is_primary:
            self._unset_primary(video_uuid)

        self.db.execute(
            """
            INSERT INTO transcriptions (
                id, video_id, audio_track_id, provider, model, language,
                format, full_text, word_count, duration, confidence,
                file_path, file_size_bytes, is_primary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                video_uuid,
                audio_track_id,
                provider,
                model,
                language,
                format_,
                full_text,
                word_count,
                duration,
                confidence,
                file_path,
                file_size_bytes,
                1 if is_primary else 0,
            ),
        )
        self.db.commit()
        return new_id

    def get_by_uuid(self, uuid_: str) -> dict[str, Any] | None:
        """Get a transcription by its UUID.

        Args:
            uuid_: The UUID primary key.

        Returns:
            Dict with transcription data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM transcriptions WHERE id = ?",
            (uuid_,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_primary(self, video_uuid: str) -> dict[str, Any] | None:
        """Get the primary transcription for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Dict with transcription data, or None if no primary exists.
        """
        cursor = self.db.execute(
            "SELECT * FROM transcriptions WHERE video_id = ? AND is_primary = 1",
            (video_uuid,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_video(self, video_uuid: str) -> list[dict[str, Any]]:
        """Get all transcriptions for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            List of dicts with transcription data, primary first, then by created_at.
        """
        cursor = self.db.execute(
            """
            SELECT * FROM transcriptions
            WHERE video_id = ?
            ORDER BY is_primary DESC, created_at
            """,
            (video_uuid,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def set_primary(self, transcription_uuid: str) -> bool:
        """Set a transcription as the primary for its video.

        This will unset any existing primary transcription for the same video
        and set the specified transcription as primary.

        Args:
            transcription_uuid: UUID of the transcription to make primary.

        Returns:
            True if successful, False if transcription not found.
        """
        # Get the transcription to find its video
        transcription = self.get_by_uuid(transcription_uuid)
        if transcription is None:
            return False

        video_uuid = transcription["video_id"]

        # Unset any existing primary for this video
        self._unset_primary(video_uuid)

        # Set the new primary
        self.db.execute(
            "UPDATE transcriptions SET is_primary = 1 WHERE id = ?",
            (transcription_uuid,),
        )
        self.db.commit()
        return True

    def search_fts(self, query: str) -> list[dict[str, Any]]:
        """Search transcriptions using full-text search.

        Searches the full_text field via the transcriptions_fts virtual table.
        Results include video context (video_id natural key, title).

        Args:
            query: Search query string. Supports FTS5 syntax.

        Returns:
            List of matching transcriptions with video context, ordered by relevance.
        """
        escaped_query = self._escape_fts_query(query)

        cursor = self.db.execute(
            """
            SELECT
                t.*,
                v.video_id as video_natural_id,
                v.title as video_title,
                v.domain as video_domain,
                rank
            FROM transcriptions_fts
            JOIN transcriptions t ON transcriptions_fts.rowid = t.rowid
            JOIN videos v ON t.video_id = v.id
            WHERE transcriptions_fts MATCH ?
            ORDER BY rank
            """,
            (escaped_query,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete(self, transcription_uuid: str) -> bool:
        """Delete a transcription by its UUID.

        Args:
            transcription_uuid: UUID of the transcription to delete.

        Returns:
            True if a transcription was deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM transcriptions WHERE id = ?",
            (transcription_uuid,),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def delete_by_video(self, video_uuid: str) -> int:
        """Delete all transcriptions for a video.

        Args:
            video_uuid: UUID of the parent video.

        Returns:
            Number of transcriptions deleted.
        """
        cursor = self.db.execute(
            "DELETE FROM transcriptions WHERE video_id = ?",
            (video_uuid,),
        )
        self.db.commit()
        return cursor.rowcount

    def _unset_primary(self, video_uuid: str) -> None:
        """Unset the primary flag for all transcriptions of a video.

        Args:
            video_uuid: UUID of the video.
        """
        self.db.execute(
            "UPDATE transcriptions SET is_primary = 0 WHERE video_id = ? AND is_primary = 1",
            (video_uuid,),
        )

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
