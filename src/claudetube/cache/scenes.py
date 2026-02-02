"""
Scene cache management utilities.

Provides helper functions for managing scene directories and data within
the video cache structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class SceneStatus:
    """Status of scene processing for a single scene."""

    keyframes: bool = False
    visual: bool = False
    technical: bool = False

    def is_complete(self) -> bool:
        """Check if all processing is complete for this scene."""
        return self.keyframes and self.visual and self.technical

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "keyframes": self.keyframes,
            "visual": self.visual,
            "technical": self.technical,
        }


@dataclass
class SceneBoundary:
    """Scene boundary information."""

    scene_id: int
    start_time: float  # seconds
    end_time: float  # seconds
    title: str | None = None  # Optional scene title/description
    transcript_segment: str | None = None  # DEPRECATED: Use transcript_text instead
    transcript: list[dict] = field(
        default_factory=list
    )  # Individual segments with timestamps
    transcript_text: str = ""  # Joined transcript text for this scene

    def duration(self) -> float:
        """Get scene duration in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "scene_id": self.scene_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "title": self.title,
        }
        # Include transcript data if present
        if self.transcript:
            result["transcript"] = self.transcript
            result["transcript_text"] = self.transcript_text
        # Backwards compat: also include transcript_segment if set
        if self.transcript_segment:
            result["transcript_segment"] = self.transcript_segment
        return result

    @classmethod
    def from_dict(cls, data: dict) -> SceneBoundary:
        """Create from dictionary."""
        return cls(
            scene_id=data["scene_id"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            title=data.get("title"),
            transcript_segment=data.get("transcript_segment"),
            transcript=data.get("transcript", []),
            transcript_text=data.get("transcript_text", ""),
        )


@dataclass
class ScenesData:
    """Container for all scene data for a video."""

    video_id: str
    method: str  # "transcript", "visual", "hybrid"
    scenes: list[SceneBoundary] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "video_id": self.video_id,
            "method": self.method,
            "scene_count": len(self.scenes),
            "scenes": [s.to_dict() for s in self.scenes],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ScenesData:
        """Create from dictionary."""
        return cls(
            video_id=data["video_id"],
            method=data["method"],
            scenes=[SceneBoundary.from_dict(s) for s in data.get("scenes", [])],
        )


def get_scenes_dir(cache_dir: Path) -> Path:
    """Get the scenes directory for a video cache.

    Args:
        cache_dir: Video cache directory (e.g., ~/.claude/video_cache/{video_id}/)

    Returns:
        Path to scenes/ directory (created if needed)
    """
    scenes_dir = cache_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    return scenes_dir


def get_scene_dir(cache_dir: Path, scene_id: int) -> Path:
    """Get directory for a specific scene.

    Args:
        cache_dir: Video cache directory
        scene_id: Scene index (0-based)

    Returns:
        Path to scene_{NNN}/ directory (created if needed)
    """
    scene_dir = cache_dir / "scenes" / f"scene_{scene_id:03d}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    return scene_dir


def get_keyframes_dir(cache_dir: Path, scene_id: int) -> Path:
    """Get keyframes directory for a specific scene.

    Args:
        cache_dir: Video cache directory
        scene_id: Scene index (0-based)

    Returns:
        Path to scene_{NNN}/keyframes/ directory (created if needed)
    """
    kf_dir = get_scene_dir(cache_dir, scene_id) / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)
    return kf_dir


def get_scenes_json_path(cache_dir: Path) -> Path:
    """Get path to scenes.json file.

    Args:
        cache_dir: Video cache directory

    Returns:
        Path to scenes/scenes.json
    """
    return get_scenes_dir(cache_dir) / "scenes.json"


def get_visual_json_path(cache_dir: Path, scene_id: int) -> Path:
    """Get path to visual.json for a scene.

    Args:
        cache_dir: Video cache directory
        scene_id: Scene index (0-based)

    Returns:
        Path to scene_{NNN}/visual.json
    """
    return get_scene_dir(cache_dir, scene_id) / "visual.json"


def get_technical_json_path(cache_dir: Path, scene_id: int) -> Path:
    """Get path to technical.json for a scene.

    Args:
        cache_dir: Video cache directory
        scene_id: Scene index (0-based)

    Returns:
        Path to scene_{NNN}/technical.json
    """
    return get_scene_dir(cache_dir, scene_id) / "technical.json"


def get_entities_json_path(cache_dir: Path, scene_id: int) -> Path:
    """Get path to entities.json for a scene.

    Args:
        cache_dir: Video cache directory
        scene_id: Scene index (0-based)

    Returns:
        Path to scene_{NNN}/entities.json
    """
    return get_scene_dir(cache_dir, scene_id) / "entities.json"


def has_scenes(cache_dir: Path) -> bool:
    """Check if scenes have been processed for this video.

    Args:
        cache_dir: Video cache directory

    Returns:
        True if scenes/scenes.json exists
    """
    return get_scenes_json_path(cache_dir).exists()


def get_scene_status(cache_dir: Path, scene_id: int) -> SceneStatus:
    """Get processing status for a specific scene.

    Args:
        cache_dir: Video cache directory
        scene_id: Scene index (0-based)

    Returns:
        SceneStatus with flags for each processing stage
    """
    scene_dir = cache_dir / "scenes" / f"scene_{scene_id:03d}"
    keyframes_dir = scene_dir / "keyframes"

    return SceneStatus(
        keyframes=keyframes_dir.exists() and any(keyframes_dir.iterdir())
        if keyframes_dir.exists()
        else False,
        visual=(scene_dir / "visual.json").exists(),
        technical=(scene_dir / "technical.json").exists(),
    )


def load_scenes_data(cache_dir: Path) -> ScenesData | None:
    """Load scenes.json data.

    Args:
        cache_dir: Video cache directory

    Returns:
        ScenesData or None if not found
    """
    scenes_json = get_scenes_json_path(cache_dir)
    if not scenes_json.exists():
        return None

    try:
        data = json.loads(scenes_json.read_text())
        return ScenesData.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        return None


def save_scenes_data(cache_dir: Path, data: ScenesData) -> None:
    """Save scenes.json data.

    Args:
        cache_dir: Video cache directory
        data: ScenesData to save
    """
    scenes_json = get_scenes_json_path(cache_dir)
    scenes_json.write_text(json.dumps(data.to_dict(), indent=2))


def list_scene_keyframes(cache_dir: Path, scene_id: int) -> list[Path]:
    """List all keyframe images for a scene.

    Args:
        cache_dir: Video cache directory
        scene_id: Scene index (0-based)

    Returns:
        Sorted list of keyframe paths
    """
    keyframes_dir = cache_dir / "scenes" / f"scene_{scene_id:03d}" / "keyframes"
    if not keyframes_dir.exists():
        return []

    return sorted(keyframes_dir.glob("kf_*.jpg"))


def get_all_scene_statuses(cache_dir: Path) -> dict[int, SceneStatus]:
    """Get processing status for all scenes.

    Args:
        cache_dir: Video cache directory

    Returns:
        Dict mapping scene_id to SceneStatus
    """
    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return {}

    return {
        scene.scene_id: get_scene_status(cache_dir, scene.scene_id)
        for scene in scenes_data.scenes
    }
