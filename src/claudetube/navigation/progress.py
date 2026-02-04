"""
Persistent progress tracking for playlists.

Stores user progress (watched videos, bookmarks, timestamps) in progress.json
within each playlist's cache directory.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from claudetube.config.loader import get_cache_dir

logger = logging.getLogger(__name__)


@dataclass
class Bookmark:
    """A bookmark within a video."""

    video_id: str
    timestamp: float
    note: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class PlaylistProgress:
    """Persistent progress tracking for a playlist.

    Tracks which videos have been watched, current position, and bookmarks.
    Progress is saved to progress.json in the playlist cache directory.
    """

    playlist_id: str
    watched_videos: list[str] = field(default_factory=list)
    watch_times: dict[str, float] = field(default_factory=dict)
    current_video: str | None = None
    current_timestamp: float | None = None
    bookmarks: list[dict] = field(default_factory=list)
    last_accessed: float = field(default_factory=time.time)

    @classmethod
    def get_progress_path(
        cls, playlist_id: str, cache_base: Path | None = None
    ) -> Path:
        """Get the path to the progress.json file for a playlist.

        Args:
            playlist_id: Playlist ID.
            cache_base: Optional cache base directory.

        Returns:
            Path to progress.json file.
        """
        cache_base = cache_base or get_cache_dir()
        return cache_base / "playlists" / playlist_id / "progress.json"

    @classmethod
    def load(cls, playlist_id: str, cache_base: Path | None = None) -> PlaylistProgress:
        """Load progress from cache, creating new if not exists.

        Args:
            playlist_id: Playlist ID.
            cache_base: Optional cache base directory.

        Returns:
            PlaylistProgress instance.
        """
        path = cls.get_progress_path(playlist_id, cache_base)

        if path.exists():
            try:
                data = json.loads(path.read_text())
                return cls(
                    playlist_id=data.get("playlist_id", playlist_id),
                    watched_videos=data.get("watched_videos", []),
                    watch_times=data.get("watch_times", {}),
                    current_video=data.get("current_video"),
                    current_timestamp=data.get("current_timestamp"),
                    bookmarks=data.get("bookmarks", []),
                    last_accessed=data.get("last_accessed", time.time()),
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load progress for {playlist_id}: {e}")

        return cls(playlist_id=playlist_id)

    def save(self, cache_base: Path | None = None) -> Path:
        """Persist progress to cache using atomic write.

        Args:
            cache_base: Optional cache base directory.

        Returns:
            Path to saved progress.json file.
        """
        path = self.get_progress_path(self.playlist_id, cache_base)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.last_accessed = time.time()
        data = asdict(self)

        # Atomic write: write to temp file, then rename
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".json", prefix="progress_", dir=path.parent
        )
        try:
            with open(temp_fd, "w") as f:
                json.dump(data, f, indent=2)
            Path(temp_path).rename(path)
        except Exception:
            # Clean up temp file on failure
            Path(temp_path).unlink(missing_ok=True)
            raise

        logger.debug(f"Saved progress for {self.playlist_id}: {path}")
        return path

    def mark_watched(
        self,
        video_id: str,
        timestamp: float | None = None,
        set_current: bool = True,
    ) -> None:
        """Mark a video as watched.

        Args:
            video_id: Video ID to mark as watched.
            timestamp: Optional timestamp when watched (defaults to now).
            set_current: Whether to set this as the current video.
        """
        if video_id not in self.watched_videos:
            self.watched_videos.append(video_id)

        self.watch_times[video_id] = timestamp or time.time()

        if set_current:
            self.current_video = video_id
            self.current_timestamp = None  # Reset position in new video

    def is_watched(self, video_id: str) -> bool:
        """Check if a video has been watched.

        Args:
            video_id: Video ID to check.

        Returns:
            True if video has been watched.
        """
        return video_id in self.watched_videos

    def add_bookmark(
        self,
        video_id: str,
        timestamp: float,
        note: str = "",
    ) -> None:
        """Add a bookmark.

        Args:
            video_id: Video ID to bookmark.
            timestamp: Timestamp in seconds.
            note: Optional note for the bookmark.
        """
        bookmark = {
            "video_id": video_id,
            "timestamp": timestamp,
            "note": note,
            "created_at": time.time(),
        }
        self.bookmarks.append(bookmark)

    def get_bookmarks_for_video(self, video_id: str) -> list[dict]:
        """Get all bookmarks for a video.

        Args:
            video_id: Video ID.

        Returns:
            List of bookmarks for the video.
        """
        return [b for b in self.bookmarks if b.get("video_id") == video_id]

    def update_position(self, video_id: str, timestamp: float) -> None:
        """Update current position in a video.

        Args:
            video_id: Video ID.
            timestamp: Current timestamp in seconds.
        """
        self.current_video = video_id
        self.current_timestamp = timestamp

    def get_completion_stats(self, total_videos: int) -> dict:
        """Get completion statistics.

        Args:
            total_videos: Total number of videos in the playlist.

        Returns:
            Dict with completion statistics.
        """
        completed = len(self.watched_videos)
        return {
            "completed": completed,
            "total": total_videos,
            "percentage": round(completed / total_videos * 100, 1)
            if total_videos
            else 0,
            "remaining": total_videos - completed,
        }
