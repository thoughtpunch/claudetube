"""
Cache manager for video processing.
"""

from __future__ import annotations

import json
from pathlib import Path

from claudetube.cache import memory as memory_cache
from claudetube.cache import scenes as scene_cache
from claudetube.cache.storage import (
    cache_local_file,
    check_cached_source,
    load_state,
    save_state,
)
from claudetube.config.loader import get_cache_dir
from claudetube.models.state import VideoState
from claudetube.models.video_file import VideoFile


class CacheManager:
    """Manages video cache directory and state."""

    def __init__(self, cache_base: Path | None = None):
        self.cache_base = cache_base or get_cache_dir()

    def get_cache_dir(self, video_id: str) -> Path:
        """Get cache directory for a video."""
        return self.cache_base / video_id

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
        """List all cached videos with basic metadata."""
        videos = []
        if not self.cache_base.exists():
            return videos

        for state_file in sorted(self.cache_base.glob("*/state.json")):
            try:
                state = json.loads(state_file.read_text())
                videos.append({
                    "video_id": state.get("video_id", state_file.parent.name),
                    "title": state.get("title"),
                    "duration_string": state.get("duration_string"),
                    "transcript_complete": state.get("transcript_complete", False),
                    "transcript_source": state.get("transcript_source"),
                    "cache_dir": str(state_file.parent),
                })
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

    def get_scene_status(
        self, video_id: str, scene_id: int
    ) -> scene_cache.SceneStatus:
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
        return scene_cache.list_scene_keyframes(
            self.get_cache_dir(video_id), scene_id
        )

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
