"""Playlist repository for CRUD operations on playlist-related tables.

Manages playlists and playlist_videos (membership) tables.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database


# Valid playlist types per schema CHECK constraint
VALID_PLAYLIST_TYPES = frozenset(["course", "series", "conference", "collection"])


class PlaylistRepository:
    """Repository for playlist operations.

    Playlists represent collections of videos (e.g., YouTube playlists,
    course series). Each playlist can contain multiple videos via the
    playlist_videos junction table, which tracks position for ordering.
    """

    def __init__(self, db: Database) -> None:
        """Initialize with a Database instance.

        Args:
            db: Database connection wrapper.
        """
        self.db = db

    def insert(
        self,
        playlist_id: str,
        domain: str,
        *,
        channel: str | None = None,
        title: str | None = None,
        description: str | None = None,
        url: str | None = None,
        video_count: int | None = None,
        playlist_type: str | None = None,
    ) -> str:
        """Insert a new playlist record.

        Args:
            playlist_id: Natural key (e.g., YouTube playlist ID).
            domain: Playlist source domain (e.g., 'youtube', 'vimeo').
            channel: Optional channel identifier.
            title: Playlist title.
            description: Playlist description.
            url: Original playlist URL.
            video_count: Number of videos in playlist.
            playlist_type: One of: course, series, conference, collection.

        Returns:
            The generated UUID for the new playlist record.

        Raises:
            ValueError: If playlist_type is invalid.
            sqlite3.IntegrityError: If playlist_id already exists.
        """
        if playlist_type is not None and playlist_type not in VALID_PLAYLIST_TYPES:
            msg = f"Invalid playlist_type: {playlist_type}. Must be one of {sorted(VALID_PLAYLIST_TYPES)}"
            raise ValueError(msg)

        new_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO playlists (
                id, playlist_id, domain, channel, title,
                description, url, video_count, playlist_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                playlist_id,
                domain,
                channel,
                title,
                description,
                url,
                video_count,
                playlist_type,
            ),
        )
        self.db.commit()
        return new_id

    def upsert(
        self,
        playlist_id: str,
        domain: str,
        **metadata: Any,
    ) -> str:
        """Insert or update a playlist, filling NULL fields without overwriting.

        Args:
            playlist_id: Natural key (e.g., YouTube playlist ID).
            domain: Playlist source domain.
            **metadata: Optional fields (channel, title, description, etc.).

        Returns:
            The UUID for the playlist (existing or newly generated).
        """
        existing = self.get_by_playlist_id(playlist_id)

        if existing is None:
            return self.insert(playlist_id, domain, **metadata)

        existing_uuid = existing["id"]
        update_fields = []
        update_values = []

        enrichable = [
            "channel",
            "title",
            "description",
            "url",
            "video_count",
            "playlist_type",
        ]

        for field in enrichable:
            if field in metadata and metadata[field] is not None:
                # Validate playlist_type if provided
                if field == "playlist_type" and metadata[field] not in VALID_PLAYLIST_TYPES:
                    msg = f"Invalid playlist_type: {metadata[field]}"
                    raise ValueError(msg)
                update_fields.append(f"{field} = COALESCE({field}, ?)")
                update_values.append(metadata[field])

        if update_fields:
            update_fields.append("updated_at = datetime('now')")
            sql = f"UPDATE playlists SET {', '.join(update_fields)} WHERE id = ?"
            update_values.append(existing_uuid)
            self.db.execute(sql, tuple(update_values))
            self.db.commit()

        return existing_uuid

    def get_by_playlist_id(self, playlist_id: str) -> dict[str, Any] | None:
        """Get a playlist by its natural key.

        Args:
            playlist_id: The natural key (e.g., YouTube playlist ID).

        Returns:
            Dict with playlist data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM playlists WHERE playlist_id = ?",
            (playlist_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_uuid(self, uuid_: str) -> dict[str, Any] | None:
        """Get a playlist by its UUID primary key.

        Args:
            uuid_: The UUID primary key.

        Returns:
            Dict with playlist data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM playlists WHERE id = ?",
            (uuid_,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_all(self) -> list[dict[str, Any]]:
        """List all playlists.

        Returns:
            List of dicts with playlist data, ordered by created_at desc.
        """
        cursor = self.db.execute(
            "SELECT * FROM playlists ORDER BY created_at DESC"
        )
        return [dict(row) for row in cursor.fetchall()]

    def add_video(
        self,
        playlist_uuid: str,
        video_uuid: str,
        position: int = 0,
    ) -> str:
        """Add a video to a playlist.

        Uses INSERT OR IGNORE to handle duplicate (playlist, video) pairs.

        Args:
            playlist_uuid: UUID of the playlist.
            video_uuid: UUID of the video to add.
            position: Position in playlist (0-based).

        Returns:
            UUID of the playlist_videos record (new or existing).

        Raises:
            ValueError: If position is negative.
        """
        if position < 0:
            msg = f"position must be >= 0, got {position}"
            raise ValueError(msg)

        new_id = str(uuid.uuid4())

        # Try to insert (will be ignored if duplicate)
        self.db.execute(
            """
            INSERT OR IGNORE INTO playlist_videos (id, playlist_id, video_id, position)
            VALUES (?, ?, ?, ?)
            """,
            (new_id, playlist_uuid, video_uuid, position),
        )
        self.db.commit()

        # Retrieve the ID (new or existing)
        cursor = self.db.execute(
            """
            SELECT id FROM playlist_videos
            WHERE playlist_id = ? AND video_id = ?
            """,
            (playlist_uuid, video_uuid),
        )
        row = cursor.fetchone()
        return row["id"]

    def remove_video(self, playlist_uuid: str, video_uuid: str) -> bool:
        """Remove a video from a playlist.

        Args:
            playlist_uuid: UUID of the playlist.
            video_uuid: UUID of the video to remove.

        Returns:
            True if removed, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM playlist_videos WHERE playlist_id = ? AND video_id = ?",
            (playlist_uuid, video_uuid),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def update_video_position(
        self,
        playlist_uuid: str,
        video_uuid: str,
        position: int,
    ) -> bool:
        """Update a video's position in a playlist.

        Args:
            playlist_uuid: UUID of the playlist.
            video_uuid: UUID of the video.
            position: New position (0-based).

        Returns:
            True if updated, False if not found.

        Raises:
            ValueError: If position is negative.
        """
        if position < 0:
            msg = f"position must be >= 0, got {position}"
            raise ValueError(msg)

        cursor = self.db.execute(
            """
            UPDATE playlist_videos SET position = ?
            WHERE playlist_id = ? AND video_id = ?
            """,
            (position, playlist_uuid, video_uuid),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def get_videos(self, playlist_uuid: str) -> list[dict[str, Any]]:
        """Get all videos in a playlist.

        Args:
            playlist_uuid: UUID of the playlist.

        Returns:
            List of video dicts ordered by position.
        """
        cursor = self.db.execute(
            """
            SELECT
                v.*,
                pv.position
            FROM playlist_videos pv
            JOIN videos v ON pv.video_id = v.id
            WHERE pv.playlist_id = ?
            ORDER BY pv.position
            """,
            (playlist_uuid,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_playlists_for_video(self, video_uuid: str) -> list[dict[str, Any]]:
        """Get all playlists containing a video.

        Args:
            video_uuid: UUID of the video.

        Returns:
            List of playlist dicts.
        """
        cursor = self.db.execute(
            """
            SELECT
                p.*,
                pv.position
            FROM playlist_videos pv
            JOIN playlists p ON pv.playlist_id = p.id
            WHERE pv.video_id = ?
            ORDER BY p.title
            """,
            (video_uuid,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def count_videos(self, playlist_uuid: str) -> int:
        """Count videos in a playlist.

        Args:
            playlist_uuid: UUID of the playlist.

        Returns:
            Number of videos in the playlist.
        """
        cursor = self.db.execute(
            "SELECT COUNT(*) as cnt FROM playlist_videos WHERE playlist_id = ?",
            (playlist_uuid,),
        )
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    def delete(self, playlist_uuid: str) -> bool:
        """Delete a playlist by its UUID.

        Note: Due to ON DELETE CASCADE, this removes all
        playlist_videos associations.

        Args:
            playlist_uuid: UUID of the playlist to delete.

        Returns:
            True if a playlist was deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM playlists WHERE id = ?",
            (playlist_uuid,),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def delete_by_playlist_id(self, playlist_id: str) -> bool:
        """Delete a playlist by its natural key.

        Args:
            playlist_id: The natural key (e.g., YouTube playlist ID).

        Returns:
            True if a playlist was deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM playlists WHERE playlist_id = ?",
            (playlist_id,),
        )
        self.db.commit()
        return cursor.rowcount > 0
