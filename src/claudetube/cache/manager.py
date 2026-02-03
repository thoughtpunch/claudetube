"""
Cache manager for video processing.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from claudetube.cache import memory as memory_cache
from claudetube.cache import scenes as scene_cache
from claudetube.cache.storage import (
    cache_local_file,
    check_cached_source,
    load_state,
    save_state,
)
from claudetube.config.loader import get_cache_dir
from claudetube.models.video_file import VideoFile

if TYPE_CHECKING:
    from pathlib import Path

    from claudetube.models.state import VideoState
    from claudetube.models.video_path import VideoPath

logger = logging.getLogger(__name__)

# Session-level cache for video_id -> cache_path resolution
_resolution_cache: dict[str, str] = {}


class CacheManager:
    """Manages video cache directory and state."""

    def __init__(self, cache_base: Path | None = None):
        self.cache_base = cache_base or get_cache_dir()

    def get_cache_dir(self, video_id_or_path: str | VideoPath) -> Path:
        """Get cache directory for a video.

        Accepts either:
        - VideoPath: returns cache_base / video_path.relative_path() (hierarchical)
        - str (bare video_id): resolves via SQLite -> flat path -> glob fallback

        Resolution chain for bare video_id:
        1. Session cache (fast in-memory lookup)
        2. SQLite lookup (O(1) via idx_videos_video_id)
        3. Flat path check (cache_base / video_id / state.json)
        4. Glob scan (expensive, only when SQLite has no record)

        Args:
            video_id_or_path: Either a VideoPath object or a bare video_id string.

        Returns:
            Path to the video's cache directory.
        """

        from claudetube.models.video_path import VideoPath

        # Handle VideoPath directly (hierarchical path)
        if isinstance(video_id_or_path, VideoPath):
            return self.cache_base / video_id_or_path.relative_path()

        # Handle bare video_id string
        video_id = video_id_or_path
        return self._resolve_video_id(video_id)

    def _resolve_video_id(self, video_id: str) -> Path:
        """Resolve a bare video_id to its cache directory path.

        Resolution chain:
        1. Session cache (fast)
        2. SQLite lookup (fast, O(1))
        3. Flat path check (fast)
        4. Glob scan (slow, last resort)

        Args:
            video_id: Natural key (e.g., YouTube video ID).

        Returns:
            Path to the video's cache directory.
        """

        # 1. Check session cache
        if video_id in _resolution_cache:
            return self.cache_base / _resolution_cache[video_id]

        # 2. Try SQLite lookup
        cache_path = self._resolve_via_sqlite(video_id)
        if cache_path:
            _resolution_cache[video_id] = cache_path
            return self.cache_base / cache_path

        # 3. Check flat legacy path
        flat_path = self.cache_base / video_id
        if (flat_path / "state.json").exists():
            _resolution_cache[video_id] = video_id
            return flat_path

        # 4. Glob scan (expensive, last resort for legacy hierarchical paths)
        glob_path = self._resolve_via_glob(video_id)
        if glob_path:
            _resolution_cache[video_id] = glob_path
            return self.cache_base / glob_path

        # No existing cache found - return flat path for new videos
        # (caller will create it, hierarchical path requires VideoPath)
        return flat_path

    def _resolve_via_sqlite(self, video_id: str) -> str | None:
        """Resolve video_id to cache_path via SQLite.

        Args:
            video_id: Natural key (e.g., YouTube video ID).

        Returns:
            Relative cache_path string, or None if not found.
        """
        try:
            from claudetube.db import get_database
            from claudetube.db.repos.videos import VideoRepository

            db = get_database()
            if db is None:
                return None

            repo = VideoRepository(db)
            cache_path = repo.resolve_path(video_id)
            if cache_path:
                logger.debug("Resolved %s via SQLite: %s", video_id, cache_path)
            return cache_path
        except Exception:
            logger.debug("SQLite resolution failed for %s", video_id, exc_info=True)
            return None

    def _resolve_via_glob(self, video_id: str) -> str | None:
        """Resolve video_id to cache_path via glob scan.

        Searches for **/**/**/{video_id}/state.json in the cache hierarchy.
        This is expensive but handles legacy hierarchical paths not in SQLite.

        Args:
            video_id: Natural key (e.g., YouTube video ID).

        Returns:
            Relative cache_path string, or None if not found.
        """
        try:
            # Search for domain/channel/playlist/video_id/state.json
            pattern = f"**/{video_id}/state.json"
            matches = list(self.cache_base.glob(pattern))
            if matches:
                # Use the first match, relative to cache_base
                cache_dir = matches[0].parent
                rel_path = cache_dir.relative_to(self.cache_base)
                logger.debug("Resolved %s via glob: %s", video_id, rel_path)
                return str(rel_path)
        except Exception:
            logger.debug("Glob resolution failed for %s", video_id, exc_info=True)
        return None

    def get_cache_dir_for_path(self, video_path: VideoPath) -> Path:
        """Get cache directory for a VideoPath (hierarchical).

        This is the preferred method for new videos. Use this when you have
        a VideoPath object constructed from URL parsing or yt-dlp metadata.

        Args:
            video_path: VideoPath object with domain/channel/playlist/video_id.

        Returns:
            Path to the hierarchical cache directory.
        """
        return self.cache_base / video_path.relative_path()

    def get_state_file(self, video_id: str) -> Path:
        """Get path to state.json for a video."""
        return self.get_cache_dir(video_id) / "state.json"

    def is_cached(self, video_id: str) -> bool:
        """Check if a video is cached."""
        return self.get_state_file(video_id).exists()

    def is_transcript_complete(self, video_id: str) -> bool:
        """Check if video has a completed transcript."""
        state = self.get_state(video_id)
        return state is not None and state.transcript_complete

    def get_state(self, video_id: str) -> VideoState | None:
        """Get cached state for a video."""
        return load_state(self.get_state_file(video_id))

    def save_state(self, video_id: str, state: VideoState) -> None:
        """Save state for a video."""
        cache_dir = self.get_cache_dir(video_id)
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_state(state, self.get_state_file(video_id))

    def get_video_file(self, video_id: str) -> VideoFile | None:
        """Get a VideoFile for a cached video.

        Returns None if video is not cached.
        """
        return VideoFile.from_cache(video_id, self.cache_base)

    def ensure_cache_dir(self, video_id: str) -> Path:
        """Ensure cache directory exists and return it."""
        cache_dir = self.get_cache_dir(video_id)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def get_audio_path(self, video_id: str) -> Path:
        """Get path to audio file (may not exist)."""
        return self.get_cache_dir(video_id) / "audio.mp3"

    def get_transcript_paths(self, video_id: str) -> tuple[Path, Path]:
        """Get paths to SRT and TXT transcripts (may not exist)."""
        cache_dir = self.get_cache_dir(video_id)
        return cache_dir / "audio.srt", cache_dir / "audio.txt"

    def get_thumbnail_path(self, video_id: str) -> Path:
        """Get path to thumbnail (may not exist)."""
        return self.get_cache_dir(video_id) / "thumbnail.jpg"

    def list_cached_videos(self) -> list[dict]:
        """List all cached videos with basic metadata.

        Uses SQL query for speed when available, falls back to filesystem
        scanning if database is unavailable.
        """
        # Try SQL first (faster for large caches)
        try:
            from claudetube.db.queries import list_cached_videos_sql

            sql_result = list_cached_videos_sql()
            if sql_result is not None:
                return sql_result
        except Exception:
            pass

        # Fallback: scan filesystem
        return self._list_cached_videos_filesystem()

    def _list_cached_videos_filesystem(self) -> list[dict]:
        """List cached videos by scanning the filesystem (fallback).

        Scans both flat paths (cache_base/video_id/) and hierarchical paths
        (cache_base/domain/channel/playlist/video_id/).
        """
        videos = []
        seen_ids: set[str] = set()

        if not self.cache_base.exists():
            return videos

        # Scan flat paths first (cache_base/*/state.json)
        for state_file in sorted(self.cache_base.glob("*/state.json")):
            try:
                state = json.loads(state_file.read_text())
                video_id = state.get("video_id", state_file.parent.name)
                if video_id not in seen_ids:
                    seen_ids.add(video_id)
                    videos.append(
                        {
                            "video_id": video_id,
                            "title": state.get("title"),
                            "duration_string": state.get("duration_string"),
                            "transcript_complete": state.get(
                                "transcript_complete", False
                            ),
                            "transcript_source": state.get("transcript_source"),
                            "cache_dir": str(state_file.parent),
                        }
                    )
            except (json.JSONDecodeError, OSError):
                continue

        # Scan hierarchical paths (cache_base/domain/channel/playlist/video_id/state.json)
        # Pattern: 4 levels deep - domain/channel_or_no_channel/playlist_or_no_playlist/video_id
        for state_file in sorted(self.cache_base.glob("*/*/*/*/state.json")):
            try:
                state = json.loads(state_file.read_text())
                video_id = state.get("video_id", state_file.parent.name)
                if video_id not in seen_ids:
                    seen_ids.add(video_id)
                    videos.append(
                        {
                            "video_id": video_id,
                            "title": state.get("title"),
                            "duration_string": state.get("duration_string"),
                            "transcript_complete": state.get(
                                "transcript_complete", False
                            ),
                            "transcript_source": state.get("transcript_source"),
                            "cache_dir": str(state_file.parent),
                        }
                    )
            except (json.JSONDecodeError, OSError):
                continue

        return videos

    def clear(self, video_id: str) -> bool:
        """Remove a video from cache.

        Returns True if video was removed, False if not found.
        """
        import shutil

        cache_dir = self.get_cache_dir(video_id)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            return True
        return False

    # Audio Description methods

    def get_ad_paths(self, video_id: str) -> tuple[Path, Path]:
        """Return (vtt_path, txt_path) for audio descriptions."""
        cache_dir = self.get_cache_dir(video_id)
        return (cache_dir / "audio.ad.vtt", cache_dir / "audio.ad.txt")

    def has_ad(self, video_id: str) -> bool:
        """Check if audio descriptions exist in cache."""
        vtt, txt = self.get_ad_paths(video_id)
        return vtt.exists() or txt.exists()

    # Local file methods

    def cache_local_file(
        self,
        video_id: str,
        source: Path,
        copy: bool = False,
    ) -> tuple[Path, str]:
        """Cache a local file by symlink (default) or copy.

        Args:
            video_id: Video ID for cache directory
            source: Absolute path to the source file
            copy: If True, copy the file; if False (default), create symlink

        Returns:
            Tuple of (dest_path, cache_mode) where cache_mode is "symlink" or "copy"
        """
        cache_dir = self.ensure_cache_dir(video_id)
        return cache_local_file(source, cache_dir, copy=copy)

    def get_video_path(self, video_id: str) -> Path | None:
        """Get path to the cached video file, if it exists.

        Checks state.json for a cached_file entry and verifies the file
        exists on disk. Works for both local and remote videos.

        Args:
            video_id: Video ID

        Returns:
            Path to the video file, or None if unavailable.
        """
        state = self.get_state(video_id)
        if not state or not state.cached_file:
            return None
        path = self.get_cache_dir(video_id) / state.cached_file
        return path if path.exists() else None

    def get_source_path(self, video_id: str) -> Path | None:
        """Get path to the cached source file for a local video.

        Returns None if video is not cached or is not a local file.
        """
        state = self.get_state(video_id)
        if not state or state.source_type != "local" or not state.cached_file:
            return None
        return self.get_cache_dir(video_id) / state.cached_file

    def check_source_valid(self, video_id: str) -> tuple[bool, str | None]:
        """Check if the cached source file is valid.

        For symlinks, checks if the target still exists.

        Returns:
            Tuple of (is_valid, warning_message) where warning_message is None if valid
        """
        state = self.get_state(video_id)
        if not state or state.source_type != "local":
            return True, None  # Not a local file, nothing to check
        return check_cached_source(self.get_cache_dir(video_id), state.cached_file)

    # Scene-related methods

    def get_scenes_dir(self, video_id: str) -> Path:
        """Get scenes directory for a video (creates if needed)."""
        return scene_cache.get_scenes_dir(self.get_cache_dir(video_id))

    def get_scene_dir(self, video_id: str, scene_id: int) -> Path:
        """Get directory for a specific scene (creates if needed)."""
        return scene_cache.get_scene_dir(self.get_cache_dir(video_id), scene_id)

    def get_keyframes_dir(self, video_id: str, scene_id: int) -> Path:
        """Get keyframes directory for a scene (creates if needed)."""
        return scene_cache.get_keyframes_dir(self.get_cache_dir(video_id), scene_id)

    def has_scenes(self, video_id: str) -> bool:
        """Check if scenes have been processed for this video."""
        return scene_cache.has_scenes(self.get_cache_dir(video_id))

    def get_scene_status(self, video_id: str, scene_id: int) -> scene_cache.SceneStatus:
        """Get processing status for a specific scene."""
        return scene_cache.get_scene_status(self.get_cache_dir(video_id), scene_id)

    def load_scenes_data(self, video_id: str) -> scene_cache.ScenesData | None:
        """Load scenes.json data for a video."""
        return scene_cache.load_scenes_data(self.get_cache_dir(video_id))

    def save_scenes_data(self, video_id: str, data: scene_cache.ScenesData) -> None:
        """Save scenes.json data for a video."""
        scene_cache.save_scenes_data(self.get_cache_dir(video_id), data)

    def list_scene_keyframes(self, video_id: str, scene_id: int) -> list[Path]:
        """List all keyframe images for a scene."""
        return scene_cache.list_scene_keyframes(self.get_cache_dir(video_id), scene_id)

    def get_all_scene_statuses(
        self, video_id: str
    ) -> dict[int, scene_cache.SceneStatus]:
        """Get processing status for all scenes."""
        return scene_cache.get_all_scene_statuses(self.get_cache_dir(video_id))

    # Memory-related methods

    def get_memory_dir(self, video_id: str) -> Path:
        """Get memory directory for a video (creates if needed)."""
        return memory_cache.get_memory_dir(self.get_cache_dir(video_id))

    def has_memory(self, video_id: str) -> bool:
        """Check if memory data exists for this video."""
        return memory_cache.has_memory(self.get_cache_dir(video_id))

    def get_video_memory(self, video_id: str) -> memory_cache.VideoMemory:
        """Get VideoMemory instance for a video.

        Args:
            video_id: Video identifier

        Returns:
            VideoMemory instance for managing observations and Q&A
        """
        cache_dir = self.ensure_cache_dir(video_id)
        return memory_cache.VideoMemory(video_id, cache_dir)
