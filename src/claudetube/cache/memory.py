"""
Video memory management for storing observations and Q&A history.

Provides persistent memory of what the agent has learned about a video,
allowing context to be retrieved when revisiting scenes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class Observation:
    """An observation made about a scene."""

    scene_id: int
    type: str  # 'code_explanation', 'person_identified', 'error_found', etc.
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "scene_id": self.scene_id,
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Observation:
        """Create from dictionary."""
        return cls(
            scene_id=data["scene_id"],
            type=data["type"],
            content=data["content"],
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class QAPair:
    """A question-answer pair with associated scenes."""

    question: str
    answer: str
    relevant_scenes: list[int] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "scenes": self.relevant_scenes,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> QAPair:
        """Create from dictionary."""
        return cls(
            question=data["question"],
            answer=data["answer"],
            relevant_scenes=data.get("scenes", []),
            timestamp=data.get("timestamp", ""),
        )


class VideoMemory:
    """Persistent memory of what the agent has learned about a video.

    Stores observations about scenes and Q&A pairs, persisting across sessions.
    Data is stored in:
      - memory/observations.json: Scene-indexed observations
      - memory/qa_history.json: Q&A pairs with scene references

    Example usage:
        memory = VideoMemory(video_id, cache_dir)

        # Record an observation about a scene
        memory.record_observation(
            scene_id=5,
            obs_type='bug_identified',
            content='Off-by-one error in the loop at line 42'
        )

        # Record a Q&A pair
        memory.record_qa(
            question='What bug was fixed?',
            answer='An off-by-one error in the loop',
            scenes=[5, 8, 12]
        )

        # Later, when revisiting scene 5
        context = memory.get_context_for_scene(5)
        # Returns previous observations and related Q&A
    """

    def __init__(self, video_id: str, cache_dir: Path):
        """Initialize video memory.

        Args:
            video_id: Video identifier
            cache_dir: Video cache directory (e.g., ~/.claude/video_cache/{video_id}/)
        """
        self.video_id = video_id
        self.memory_dir = cache_dir / "memory"
        self.memory_dir.mkdir(exist_ok=True)

        self.observations_file = self.memory_dir / "observations.json"
        self.qa_file = self.memory_dir / "qa_history.json"

        self._observations = self._load_observations()
        self._qa_history = self._load_qa()

    def _load_observations(self) -> dict[str, list[dict]]:
        """Load observations from disk."""
        if self.observations_file.exists():
            try:
                return json.loads(self.observations_file.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _load_qa(self) -> list[dict]:
        """Load Q&A history from disk."""
        if self.qa_file.exists():
            try:
                return json.loads(self.qa_file.read_text())
            except json.JSONDecodeError:
                return []
        return []

    def _save_observations(self) -> None:
        """Save observations to disk and sync to SQLite."""
        self.observations_file.write_text(json.dumps(self._observations, indent=2))

        # Note: Individual observation sync is handled in record_observation()

    def _save_qa(self) -> None:
        """Save Q&A history to disk."""
        self.qa_file.write_text(json.dumps(self._qa_history, indent=2))

        # Note: Individual Q&A sync is handled in record_qa()

    def record_observation(self, scene_id: int, obs_type: str, content: str) -> None:
        """Record something the agent noticed about a scene.

        Args:
            scene_id: Scene index (0-based)
            obs_type: Type of observation (e.g., 'code_explanation', 'bug_identified')
            content: Description of what was observed
        """
        key = str(scene_id)
        if key not in self._observations:
            self._observations[key] = []

        self._observations[key].append(
            {
                "type": obs_type,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self._save_observations()

        # Dual-write: sync observation to SQLite (fire-and-forget)
        try:
            from claudetube.db.sync import get_video_uuid, sync_observation

            video_uuid = get_video_uuid(self.video_id)
            if video_uuid:
                sync_observation(
                    video_uuid=video_uuid,
                    scene_id=scene_id,
                    obs_type=obs_type,
                    content=content,
                )
        except Exception:
            # Fire-and-forget: don't disrupt JSON writes
            pass

    def record_qa(self, question: str, answer: str, scenes: list[int]) -> None:
        """Cache a Q&A pair for future reference.

        Args:
            question: The question that was asked
            answer: The answer that was given
            scenes: List of scene IDs relevant to this Q&A
        """
        self._qa_history.append(
            {
                "question": question,
                "answer": answer,
                "scenes": scenes,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self._save_qa()

        # Dual-write: sync Q&A to SQLite (fire-and-forget)
        try:
            from claudetube.db.sync import get_video_uuid, sync_qa

            video_uuid = get_video_uuid(self.video_id)
            if video_uuid:
                sync_qa(
                    video_uuid=video_uuid,
                    question=question,
                    answer=answer,
                    scene_ids=scenes if scenes else None,
                )
        except Exception:
            # Fire-and-forget: don't disrupt JSON writes
            pass

    def get_observations(self, scene_id: int) -> list[dict]:
        """Get all observations for a specific scene.

        Args:
            scene_id: Scene index (0-based)

        Returns:
            List of observation dicts with type, content, and timestamp
        """
        return self._observations.get(str(scene_id), [])

    def get_all_observations(self) -> dict[str, list[dict]]:
        """Get all observations for all scenes.

        Returns:
            Dict mapping scene_id (as string) to list of observations
        """
        return self._observations.copy()

    def get_context_for_scene(self, scene_id: int) -> dict:
        """Get everything learned about a scene.

        Args:
            scene_id: Scene index (0-based)

        Returns:
            Dict with 'observations' and 'related_qa' keys
        """
        return {
            "observations": self._observations.get(str(scene_id), []),
            "related_qa": [
                qa for qa in self._qa_history if scene_id in qa.get("scenes", [])
            ],
        }

    def search_qa_history(self, query: str) -> list[dict]:
        """Find relevant past Q&A pairs by keyword search.

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching Q&A dicts
        """
        query_lower = query.lower()
        return [
            qa
            for qa in self._qa_history
            if query_lower in qa["question"].lower()
            or query_lower in qa["answer"].lower()
        ]

    def get_qa_history(self) -> list[dict]:
        """Get all Q&A pairs.

        Returns:
            List of all Q&A dicts
        """
        return self._qa_history.copy()

    def clear(self) -> None:
        """Clear all memory data."""
        self._observations = {}
        self._qa_history = []
        if self.observations_file.exists():
            self.observations_file.unlink()
        if self.qa_file.exists():
            self.qa_file.unlink()

    @property
    def observation_count(self) -> int:
        """Get total number of observations."""
        return sum(len(obs) for obs in self._observations.values())

    @property
    def qa_count(self) -> int:
        """Get total number of Q&A pairs."""
        return len(self._qa_history)


def get_memory_dir(cache_dir: Path) -> Path:
    """Get the memory directory for a video cache.

    Args:
        cache_dir: Video cache directory

    Returns:
        Path to memory/ directory (created if needed)
    """
    memory_dir = cache_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir


def has_memory(cache_dir: Path) -> bool:
    """Check if memory data exists for this video.

    Args:
        cache_dir: Video cache directory

    Returns:
        True if either observations.json or qa_history.json exists
    """
    memory_dir = cache_dir / "memory"
    if not memory_dir.exists():
        return False
    return (memory_dir / "observations.json").exists() or (
        memory_dir / "qa_history.json"
    ).exists()
