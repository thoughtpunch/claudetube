"""
Playlist context for session-level navigation state.

Combines playlist metadata with progress tracking for navigation operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from claudetube.config.loader import get_cache_dir
from claudetube.navigation.progress import PlaylistProgress
from claudetube.operations.playlist import load_playlist_metadata

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PlaylistContext:
    """Active playlist context for navigation.

    Combines playlist metadata with progress tracking for a navigation session.
    """

    playlist_id: str
    title: str = ""
    playlist_type: str = "collection"  # course, series, conference, collection
    total_videos: int = 0
    videos: list[dict] = field(default_factory=list)

    # Progress tracking
    progress: PlaylistProgress = field(default_factory=lambda: PlaylistProgress(""))

    # Computed navigation state
    current_position: int | None = None

    @classmethod
    def load(
        cls, playlist_id: str, cache_base: Path | None = None
    ) -> PlaylistContext | None:
        """Load playlist context from cache.

        Args:
            playlist_id: Playlist ID.
            cache_base: Optional cache base directory.

        Returns:
            PlaylistContext instance or None if playlist not cached.
        """
        cache_base = cache_base or get_cache_dir()

        # Load playlist metadata
        metadata = load_playlist_metadata(playlist_id, cache_base)
        if metadata is None:
            logger.warning(f"Playlist not cached: {playlist_id}")
            return None

        # Load progress
        progress = PlaylistProgress.load(playlist_id, cache_base)

        # Build context
        videos = metadata.get("videos", [])
        context = cls(
            playlist_id=playlist_id,
            title=metadata.get("title", ""),
            playlist_type=metadata.get("inferred_type", "collection"),
            total_videos=len(videos),
            videos=videos,
            progress=progress,
        )

        # Compute current position if we have a current video
        if progress.current_video:
            context.current_position = context._get_video_position(
                progress.current_video
            )

        return context

    def _get_video_position(self, video_id: str) -> int | None:
        """Get 0-indexed position of a video in the playlist.

        Args:
            video_id: Video ID.

        Returns:
            Position (0-indexed) or None if not found.
        """
        for i, v in enumerate(self.videos):
            if v.get("video_id") == video_id:
                return i
        return None

    def get_video_at_position(self, position: int) -> dict | None:
        """Get video metadata at a position.

        Args:
            position: 0-indexed position.

        Returns:
            Video metadata dict or None if position out of range.
        """
        if 0 <= position < len(self.videos):
            return self.videos[position]
        return None

    def get_video_by_id(self, video_id: str) -> dict | None:
        """Get video metadata by ID.

        Args:
            video_id: Video ID.

        Returns:
            Video metadata dict or None if not found.
        """
        for v in self.videos:
            if v.get("video_id") == video_id:
                return v
        return None

    @property
    def current_video_id(self) -> str | None:
        """Get current video ID."""
        return self.progress.current_video

    @property
    def next_video(self) -> dict | None:
        """Get next video in sequence.

        Returns:
            Video metadata for next video, or None if at end.
        """
        if self.current_position is None:
            # No current video, return first
            return self.get_video_at_position(0)

        next_pos = self.current_position + 1
        if next_pos < len(self.videos):
            return self.get_video_at_position(next_pos)
        return None

    @property
    def previous_video(self) -> dict | None:
        """Get previous video in sequence.

        Returns:
            Video metadata for previous video, or None if at start.
        """
        if self.current_position is None or self.current_position <= 0:
            return None
        return self.get_video_at_position(self.current_position - 1)

    def get_progress_summary(self) -> dict:
        """Get human-readable progress summary.

        Returns:
            Dict with progress information.
        """
        stats = self.progress.get_completion_stats(self.total_videos)
        return {
            **stats,
            "current_position": self.current_position,
            "current_video": self.progress.current_video,
            "display": f"Video {(self.current_position or 0) + 1} of {self.total_videos}",
        }

    def mark_watched(
        self, video_id: str, save: bool = True, cache_base: Path | None = None
    ) -> None:
        """Mark a video as watched and update position.

        Args:
            video_id: Video ID to mark.
            save: Whether to persist progress immediately.
            cache_base: Optional cache base directory.
        """
        self.progress.mark_watched(video_id)
        self.current_position = self._get_video_position(video_id)

        if save:
            self.progress.save(cache_base)

    def get_unwatched_videos(self) -> list[dict]:
        """Get list of unwatched videos.

        Returns:
            List of video metadata dicts for unwatched videos.
        """
        return [
            v
            for v in self.videos
            if v.get("video_id") not in self.progress.watched_videos
        ]

    def get_watched_videos(self) -> list[dict]:
        """Get list of watched videos in watch order.

        Returns:
            List of video metadata dicts for watched videos.
        """
        # Return in watch order based on watch_times
        watched = [
            (v, self.progress.watch_times.get(v.get("video_id"), 0))
            for v in self.videos
            if v.get("video_id") in self.progress.watched_videos
        ]
        watched.sort(key=lambda x: x[1])
        return [v for v, _ in watched]
