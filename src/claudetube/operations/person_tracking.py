"""
Person tracking across video scenes.

Identifies distinct people across scenes and tracks their appearances with timestamps.
Follows the "Cheap First, Expensive Last" principle:
1. CACHE - Return instantly if people.json already exists
2. VISUAL - Use visual transcript data (already generated via Claude Haiku)
3. COMPUTE - Only run face detection when visual data unavailable
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from claudetube.cache.manager import CacheManager

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

from claudetube.cache.scenes import (
    SceneBoundary,
    get_visual_json_path,
    list_scene_keyframes,
    load_scenes_data,
)
from claudetube.config.loader import get_cache_dir
from claudetube.utils.logging import log_timed

logger = logging.getLogger(__name__)


@dataclass
class PersonAppearance:
    """A single appearance of a person in a scene."""

    scene_id: int
    timestamp: float  # Seconds from video start
    action: str | None = None  # What the person is doing
    confidence: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scene_id": self.scene_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PersonAppearance:
        """Create from dictionary."""
        return cls(
            scene_id=data["scene_id"],
            timestamp=data["timestamp"],
            action=data.get("action"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class PersonTrack:
    """Track of a single person across the video."""

    person_id: str
    description: str  # e.g., "man in blue shirt"
    appearances: list[PersonAppearance] = field(default_factory=list)
    encoding: list[float] | None = None  # Face encoding for matching (not serialized)

    @property
    def total_screen_time(self) -> float:
        """Estimate screen time in seconds (1 appearance = 1 second assumed)."""
        return float(len(self.appearances))

    @property
    def scene_count(self) -> int:
        """Number of unique scenes this person appears in."""
        return len({a.scene_id for a in self.appearances})

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "person_id": self.person_id,
            "description": self.description,
            "appearances": [a.to_dict() for a in self.appearances],
            "total_screen_time": self.total_screen_time,
            "scene_count": self.scene_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PersonTrack:
        """Create from dictionary."""
        return cls(
            person_id=data["person_id"],
            description=data.get("description", ""),
            appearances=[PersonAppearance.from_dict(a) for a in data.get("appearances", [])],
        )


@dataclass
class PeopleTrackingData:
    """Container for all person tracking data for a video."""

    video_id: str
    method: str  # "visual_transcript", "face_recognition", "hybrid"
    people: list[PersonTrack] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_id": self.video_id,
            "method": self.method,
            "people_count": len(self.people),
            "people": {p.person_id: p.to_dict() for p in self.people},
        }

    @classmethod
    def from_dict(cls, data: dict) -> PeopleTrackingData:
        """Create from dictionary."""
        people_dict = data.get("people", {})
        people = [PersonTrack.from_dict(p) for p in people_dict.values()]
        return cls(
            video_id=data["video_id"],
            method=data.get("method", "unknown"),
            people=people,
        )


def get_people_json_path(cache_dir: Path) -> Path:
    """Get path to entities/people.json for a video.

    Args:
        cache_dir: Video cache directory

    Returns:
        Path to entities/people.json
    """
    entities_dir = cache_dir / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    return entities_dir / "people.json"


def _extract_action_from_visual(visual_data: dict, person_desc: str) -> str | None:
    """Extract what action a person is doing from visual description.

    Args:
        visual_data: Visual transcript data for a scene
        person_desc: Description of the person (e.g., "man in blue shirt")

    Returns:
        Action string or None
    """
    actions = visual_data.get("actions", [])
    if actions:
        # Return first action - could be smarter about matching person to action
        return actions[0]

    # Fallback: check description for action keywords
    description = visual_data.get("description", "").lower()
    action_keywords = [
        "typing",
        "speaking",
        "talking",
        "pointing",
        "writing",
        "presenting",
        "demonstrating",
        "explaining",
        "coding",
        "reading",
        "watching",
        "listening",
    ]

    for action in action_keywords:
        if action in description:
            return action

    return "present"


def _track_from_visual_transcripts(
    scenes: list[SceneBoundary],
    cache_dir: Path,
) -> PeopleTrackingData:
    """Track people using existing visual transcript data.

    This is the CHEAP path - uses already-generated visual descriptions
    to identify and track people without running face detection.

    Args:
        scenes: List of scene boundaries
        cache_dir: Video cache directory

    Returns:
        PeopleTrackingData with tracked people
    """
    # Map description -> PersonTrack
    people_by_desc: dict[str, PersonTrack] = {}
    person_counter = 0

    for scene in scenes:
        visual_path = get_visual_json_path(cache_dir, scene.scene_id)
        if not visual_path.exists():
            continue

        try:
            visual_data = json.loads(visual_path.read_text())
        except json.JSONDecodeError:
            continue

        # Extract people from visual description
        people_in_scene = visual_data.get("people", [])
        scene_timestamp = scene.start_time + scene.duration() / 2  # Middle of scene

        for person_desc in people_in_scene:
            # Normalize description for matching
            normalized = person_desc.lower().strip()

            if normalized not in people_by_desc:
                person_id = f"person_{person_counter}"
                person_counter += 1
                people_by_desc[normalized] = PersonTrack(
                    person_id=person_id,
                    description=person_desc,
                )

            track = people_by_desc[normalized]
            action = _extract_action_from_visual(visual_data, person_desc)

            track.appearances.append(
                PersonAppearance(
                    scene_id=scene.scene_id,
                    timestamp=scene_timestamp,
                    action=action,
                    confidence=0.8,  # Visual transcript based
                )
            )

    return PeopleTrackingData(
        video_id="",  # Will be set by caller
        method="visual_transcript",
        people=list(people_by_desc.values()),
    )


def _is_face_recognition_available() -> bool:
    """Check if face_recognition library is available."""
    try:
        import face_recognition  # noqa: F401

        return True
    except ImportError:
        return False


def _track_with_face_recognition(
    scenes: list[SceneBoundary],
    video_id: str,
    cache_dir: Path,
    tolerance: float = 0.6,
) -> PeopleTrackingData:
    """Track people using face_recognition library.

    This is the EXPENSIVE path - runs face detection on keyframes.
    Only used when visual transcripts don't have people data.

    Args:
        scenes: List of scene boundaries
        video_id: Video ID
        cache_dir: Video cache directory
        tolerance: Face matching tolerance (lower = stricter)

    Returns:
        PeopleTrackingData with tracked people
    """
    try:
        import face_recognition
        import numpy as np
    except ImportError:
        logger.error("face_recognition not installed. Run: pip install face_recognition")
        return PeopleTrackingData(
            video_id=video_id,
            method="face_recognition",
            people=[],
        )

    known_encodings: list[np.ndarray] = []
    known_tracks: list[PersonTrack] = []

    for scene in scenes:
        keyframes = list_scene_keyframes(cache_dir, scene.scene_id)
        if not keyframes:
            continue

        scene_timestamp = scene.start_time + scene.duration() / 2

        for kf_path in keyframes:
            try:
                image = face_recognition.load_image_file(str(kf_path))
                encodings = face_recognition.face_encodings(image)
            except Exception as e:
                logger.warning(f"Failed to process keyframe {kf_path}: {e}")
                continue

            for encoding in encodings:
                # Try to match with known faces
                person_id = None
                track = None

                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, encoding)
                    min_idx = int(np.argmin(distances))

                    if distances[min_idx] < tolerance:
                        track = known_tracks[min_idx]
                        person_id = track.person_id

                if track is None:
                    # New person
                    person_id = f"person_{len(known_tracks)}"
                    track = PersonTrack(
                        person_id=person_id,
                        description=f"Person {len(known_tracks) + 1}",
                        encoding=encoding.tolist(),
                    )
                    known_encodings.append(encoding)
                    known_tracks.append(track)

                # Get action from visual transcript if available
                action = "present"
                visual_path = get_visual_json_path(cache_dir, scene.scene_id)
                if visual_path.exists():
                    try:
                        visual_data = json.loads(visual_path.read_text())
                        action = _extract_action_from_visual(visual_data, track.description) or "present"
                    except json.JSONDecodeError:
                        pass

                track.appearances.append(
                    PersonAppearance(
                        scene_id=scene.scene_id,
                        timestamp=scene_timestamp,
                        action=action,
                        confidence=1.0 - (distances[min_idx] if known_encodings and distances[min_idx] < tolerance else 0),
                    )
                )

    return PeopleTrackingData(
        video_id=video_id,
        method="face_recognition",
        people=known_tracks,
    )


def track_people(
    video_id: str,
    force: bool = False,
    use_face_recognition: bool = False,
    output_base: Path | None = None,
) -> dict:
    """Track people across scenes in a video.

    Follows "Cheap First, Expensive Last" principle:
    1. CACHE - Return entities/people.json instantly if exists
    2. VISUAL - Use visual transcript data (already generated)
    3. COMPUTE - Run face_recognition only if requested and visual data missing

    Args:
        video_id: Video ID
        force: Re-generate even if cached
        use_face_recognition: Use face_recognition library (expensive)
        output_base: Cache directory

    Returns:
        Dict with tracking results
    """
    t0 = time.time()
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    if not cache_dir.exists():
        return {"error": "Video not cached. Run process_video first.", "video_id": video_id}

    # 1. CACHE - Return instantly if already exists
    people_path = get_people_json_path(cache_dir)
    if not force and people_path.exists():
        try:
            data = json.loads(people_path.read_text())
            log_timed(f"People tracking: loaded from cache ({data.get('people_count', 0)} people)", t0)
            return data
        except json.JSONDecodeError:
            pass  # Re-generate if cached data is invalid

    # Load scenes data
    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return {"error": "No scenes found. Run get_scenes first.", "video_id": video_id}

    # 2. VISUAL - Try visual transcripts first (cheap)
    log_timed("People tracking: analyzing visual transcripts...", t0)
    tracking_data = _track_from_visual_transcripts(scenes_data.scenes, cache_dir)
    tracking_data.video_id = video_id

    # 3. COMPUTE - Fall back to face recognition if no people found and requested
    if not tracking_data.people and use_face_recognition:
        if _is_face_recognition_available():
            log_timed("People tracking: running face recognition (expensive)...", t0)
            tracking_data = _track_with_face_recognition(
                scenes_data.scenes,
                video_id,
                cache_dir,
            )
        else:
            logger.warning("face_recognition not available, skipping face detection")

    # Save to cache
    result = tracking_data.to_dict()
    people_path.write_text(json.dumps(result, indent=2))

    # Update state.json
    state_file = cache_dir / "state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        state["people_tracking_complete"] = True
        state_file.write_text(json.dumps(state, indent=2))

    log_timed(f"People tracking complete: {len(tracking_data.people)} people tracked", t0)

    return result


def get_people_tracking(
    video_id: str,
    output_base: Path | None = None,
) -> dict:
    """Get cached people tracking data for a video.

    Does NOT generate new tracking - use track_people for that.

    Args:
        video_id: Video ID
        output_base: Cache directory

    Returns:
        Dict with cached people tracking data
    """
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    if not cache_dir.exists():
        return {"error": "Video not cached", "video_id": video_id}

    people_path = get_people_json_path(cache_dir)
    if not people_path.exists():
        return {"error": "No people tracking data. Run track_people first.", "video_id": video_id}

    try:
        data = json.loads(people_path.read_text())
        return data
    except json.JSONDecodeError:
        return {"error": "Invalid people.json", "video_id": video_id}
