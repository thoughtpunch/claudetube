"""
Change detection between consecutive scenes.

Detects what changed between consecutive scenes:
- Visual changes: objects added/removed
- Topic shift: via embedding similarity (cosine distance)
- Content type changes: code → slides, presenter → diagram, etc.

Follows the "Cheap First, Expensive Last" principle:
1. CACHE - Return instantly if structure/changes.json already exists
2. SCENE DATA - Use visual.json and technical.json (already generated)
3. EMBEDDINGS - Use cached scene embeddings for topic shift detection
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from claudetube.cache.manager import CacheManager
from claudetube.cache.scenes import (
    SceneBoundary,
    get_technical_json_path,
    get_visual_json_path,
    load_scenes_data,
)
from claudetube.config.loader import get_cache_dir
from claudetube.utils.logging import log_timed

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VisualChanges:
    """Visual element changes between two scenes."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    persistent: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "added": self.added,
            "removed": self.removed,
            "persistent": self.persistent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VisualChanges:
        """Create from dictionary."""
        return cls(
            added=data.get("added", []),
            removed=data.get("removed", []),
            persistent=data.get("persistent", []),
        )


@dataclass
class SceneChange:
    """Change information between two consecutive scenes."""

    scene_a_id: int
    scene_b_id: int
    visual_changes: VisualChanges
    topic_shift_score: float  # 0 = same topic, 1 = completely different
    content_type_change: bool
    content_type_from: str | None = None
    content_type_to: str | None = None

    @property
    def is_major_transition(self) -> bool:
        """Check if this is a major transition (significant topic shift or content change)."""
        return self.topic_shift_score > 0.5 or self.content_type_change

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scene_a_id": self.scene_a_id,
            "scene_b_id": self.scene_b_id,
            "visual_changes": self.visual_changes.to_dict(),
            "topic_shift_score": self.topic_shift_score,
            "content_type_change": self.content_type_change,
            "content_type_from": self.content_type_from,
            "content_type_to": self.content_type_to,
            "is_major_transition": self.is_major_transition,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SceneChange:
        """Create from dictionary."""
        return cls(
            scene_a_id=data["scene_a_id"],
            scene_b_id=data["scene_b_id"],
            visual_changes=VisualChanges.from_dict(data.get("visual_changes", {})),
            topic_shift_score=data.get("topic_shift_score", 0.0),
            content_type_change=data.get("content_type_change", False),
            content_type_from=data.get("content_type_from"),
            content_type_to=data.get("content_type_to"),
        )


@dataclass
class ChangesData:
    """Container for all scene change data for a video."""

    video_id: str
    changes: list[SceneChange] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        major_transitions = [
            c.scene_b_id for c in self.changes if c.is_major_transition
        ]
        avg_topic_shift = (
            sum(c.topic_shift_score for c in self.changes) / len(self.changes)
            if self.changes
            else 0.0
        )

        return {
            "video_id": self.video_id,
            "changes": [c.to_dict() for c in self.changes],
            "summary": {
                "major_transitions": major_transitions,
                "major_transition_count": len(major_transitions),
                "avg_topic_shift": round(avg_topic_shift, 3),
                "content_type_changes": sum(
                    1 for c in self.changes if c.content_type_change
                ),
                "total_changes": len(self.changes),
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChangesData:
        """Create from dictionary."""
        return cls(
            video_id=data["video_id"],
            changes=[SceneChange.from_dict(c) for c in data.get("changes", [])],
            summary=data.get("summary", {}),
        )


def get_changes_json_path(cache_dir: Path) -> Path:
    """Get path to structure/changes.json for a video.

    Args:
        cache_dir: Video cache directory

    Returns:
        Path to structure/changes.json
    """
    structure_dir = cache_dir / "structure"
    structure_dir.mkdir(parents=True, exist_ok=True)
    return structure_dir / "changes.json"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity (0 to 1)
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def _word_overlap_similarity(text_a: str, text_b: str) -> float:
    """Compute word overlap similarity between two texts.

    Uses Jaccard similarity on word sets as a simple fallback
    when embeddings are not available.

    Args:
        text_a: First text
        text_b: Second text

    Returns:
        Similarity score (0 to 1)
    """
    if not text_a or not text_b:
        return 0.0

    # Simple word tokenization and normalization
    import re

    def tokenize(text: str) -> set[str]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        # Filter common stop words
        stop_words = {
            "the",
            "and",
            "for",
            "that",
            "this",
            "with",
            "are",
            "was",
            "were",
            "you",
            "have",
            "has",
            "had",
            "can",
            "but",
            "not",
            "what",
            "all",
            "when",
            "from",
            "they",
            "will",
            "would",
            "there",
            "their",
            "which",
            "about",
            "just",
            "like",
            "into",
            "your",
            "also",
            "been",
            "more",
            "some",
            "then",
            "them",
        }
        return {w for w in words if w not in stop_words}

    words_a = tokenize(text_a)
    words_b = tokenize(text_b)

    if not words_a or not words_b:
        return 0.0

    # Jaccard similarity
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)

    return intersection / union if union > 0 else 0.0


def _infer_content_type_from_transcript(transcript: str) -> str:
    """Infer content type from transcript text using keyword analysis.

    This is a fallback when visual analysis is not available.

    Args:
        transcript: Transcript text for the scene

    Returns:
        Inferred content type or 'unknown'
    """
    if not transcript:
        return "unknown"

    text = transcript.lower()

    # Code/programming indicators
    code_keywords = [
        "function",
        "variable",
        "class",
        "method",
        "import",
        "define",
        "syntax",
        "code",
        "programming",
        "python",
        "javascript",
        "compile",
        "debug",
        "error",
        "bug",
        "stack",
        "array",
        "loop",
        "algorithm",
    ]
    if sum(1 for kw in code_keywords if kw in text) >= 3:
        return "code"

    # Math/diagram indicators
    math_keywords = [
        "equation",
        "formula",
        "matrix",
        "vector",
        "derivative",
        "integral",
        "function",
        "graph",
        "plot",
        "axis",
        "coordinate",
        "sigmoid",
        "neuron",
        "layer",
        "weight",
        "gradient",
        "backpropagation",
    ]
    if sum(1 for kw in math_keywords if kw in text) >= 3:
        return "diagram"

    # Presentation/slides indicators
    slide_keywords = [
        "slide",
        "next slide",
        "presentation",
        "bullet point",
        "summary",
        "agenda",
        "outline",
        "takeaway",
        "key point",
    ]
    if sum(1 for kw in slide_keywords if kw in text) >= 2:
        return "slides"

    # Interview/conversation indicators
    interview_keywords = [
        "interview",
        "guest",
        "welcome",
        "today we have",
        "joining us",
        "question",
        "tell us about",
        "what do you think",
        "in your experience",
    ]
    if sum(1 for kw in interview_keywords if kw in text) >= 2:
        return "interview"

    # Demo/screencast indicators
    demo_keywords = [
        "click",
        "button",
        "screen",
        "window",
        "browser",
        "website",
        "demo",
        "showing",
        "let me show",
        "as you can see",
    ]
    if sum(1 for kw in demo_keywords if kw in text) >= 2:
        return "screencast"

    # Default to presenter (talking head) if substantial speech content
    word_count = len(text.split())
    if word_count > 50:
        return "presenter"

    return "unknown"


def _load_visual_data(cache_dir: Path, scene_id: int) -> dict | None:
    """Load visual.json for a scene.

    Args:
        cache_dir: Video cache directory
        scene_id: Scene ID

    Returns:
        Visual data dict or None
    """
    visual_path = get_visual_json_path(cache_dir, scene_id)
    if not visual_path.exists():
        return None

    try:
        return json.loads(visual_path.read_text())
    except json.JSONDecodeError:
        return None


def _load_technical_data(cache_dir: Path, scene_id: int) -> dict | None:
    """Load technical.json for a scene.

    Args:
        cache_dir: Video cache directory
        scene_id: Scene ID

    Returns:
        Technical data dict or None
    """
    technical_path = get_technical_json_path(cache_dir, scene_id)
    if not technical_path.exists():
        return None

    try:
        return json.loads(technical_path.read_text())
    except json.JSONDecodeError:
        return None


def _extract_objects(visual_data: dict | None, technical_data: dict | None) -> set[str]:
    """Extract all objects/elements from scene data.

    Combines:
    - visual.json: objects, people, text descriptions
    - technical.json: content_type, detected elements

    Args:
        visual_data: Visual data dict
        technical_data: Technical data dict

    Returns:
        Set of object/element strings
    """
    objects: set[str] = set()

    if visual_data:
        # Objects from visual description
        for obj in visual_data.get("objects", []):
            if isinstance(obj, str):
                objects.add(obj.lower())
            elif isinstance(obj, dict):
                name = obj.get("name", obj.get("label", ""))
                if name:
                    objects.add(name.lower())

        # People (normalized)
        for person in visual_data.get("people", []):
            objects.add(f"person:{person.lower()}")

        # Key elements mentioned in description
        description = visual_data.get("description", "")
        # Extract content type indicators from description
        content_indicators = [
            "code",
            "terminal",
            "editor",
            "slides",
            "presentation",
            "diagram",
            "chart",
            "graph",
            "table",
            "screen share",
            "browser",
            "video",
            "animation",
        ]
        for indicator in content_indicators:
            if indicator in description.lower():
                objects.add(f"content:{indicator}")

    if technical_data:
        # Content type
        content_type = technical_data.get("content_type", "")
        if content_type:
            objects.add(f"content_type:{content_type.lower()}")

        # Check frames for code blocks
        for frame in technical_data.get("frames", []):
            if frame.get("code_blocks"):
                objects.add("content:code")
            if frame.get("content_type"):
                objects.add(f"frame_type:{frame.get('content_type').lower()}")

    return objects


def _get_content_type(
    visual_data: dict | None,
    technical_data: dict | None,
    transcript: str | None = None,
) -> str | None:
    """Extract primary content type from scene data.

    Follows priority: technical.json > visual.json > transcript inference

    Args:
        visual_data: Visual data dict
        technical_data: Technical data dict
        transcript: Optional transcript text for fallback inference

    Returns:
        Content type string or 'unknown'
    """
    # Technical data is most reliable for content type
    if technical_data:
        content_type = technical_data.get("content_type")
        if content_type and content_type != "unknown":
            return content_type

        # Check frames for consistent content type
        frame_types = []
        for frame in technical_data.get("frames", []):
            ft = frame.get("content_type")
            if ft and ft != "unknown":
                frame_types.append(ft)
        if frame_types:
            # Return most common
            from collections import Counter

            return Counter(frame_types).most_common(1)[0][0]

    # Fallback to visual data
    if visual_data:
        description = visual_data.get("description", "").lower()

        # Heuristic content type detection from visual description
        if any(
            kw in description
            for kw in ["code", "editor", "terminal", "function", "class"]
        ):
            return "code"
        if any(
            kw in description
            for kw in ["slide", "presentation", "powerpoint", "keynote"]
        ):
            return "slides"
        if any(kw in description for kw in ["diagram", "chart", "graph", "flowchart"]):
            return "diagram"
        if any(
            kw in description for kw in ["person", "presenter", "speaker", "talking"]
        ):
            return "presenter"
        if any(kw in description for kw in ["screen", "browser", "website", "app"]):
            return "screencast"

    # Fallback to transcript-based inference
    if transcript:
        return _infer_content_type_from_transcript(transcript)

    return "unknown"


def _detect_changes_between_scenes(
    scene_a: SceneBoundary,
    scene_b: SceneBoundary,
    cache_dir: Path,
    embeddings: dict[int, np.ndarray] | None = None,
) -> SceneChange:
    """Detect changes between two consecutive scenes.

    Uses a tiered approach (cheap first, expensive last):
    1. Try embedding similarity for topic shift (most accurate)
    2. Fall back to transcript word overlap similarity
    3. Use transcript-based content type inference if no visual data

    Args:
        scene_a: First scene (earlier)
        scene_b: Second scene (later)
        cache_dir: Video cache directory
        embeddings: Optional dict of scene_id -> embedding vector

    Returns:
        SceneChange with detected changes
    """
    # Load scene data
    visual_a = _load_visual_data(cache_dir, scene_a.scene_id)
    visual_b = _load_visual_data(cache_dir, scene_b.scene_id)
    technical_a = _load_technical_data(cache_dir, scene_a.scene_id)
    technical_b = _load_technical_data(cache_dir, scene_b.scene_id)

    # Get transcript text from scene boundaries
    transcript_a = scene_a.transcript_text or scene_a.transcript_segment or ""
    transcript_b = scene_b.transcript_text or scene_b.transcript_segment or ""

    # 1. Visual element changes
    objects_a = _extract_objects(visual_a, technical_a)
    objects_b = _extract_objects(visual_b, technical_b)

    visual_changes = VisualChanges(
        added=sorted(objects_b - objects_a),
        removed=sorted(objects_a - objects_b),
        persistent=sorted(objects_a & objects_b),
    )

    # 2. Content type change (with transcript fallback)
    content_type_a = _get_content_type(visual_a, technical_a, transcript_a)
    content_type_b = _get_content_type(visual_b, technical_b, transcript_b)

    # Only consider it a content type change if both types are known
    content_type_change = (
        content_type_a != content_type_b
        and content_type_a != "unknown"
        and content_type_b != "unknown"
    )

    # 3. Topic shift via embedding similarity OR transcript word overlap
    topic_shift_score = 0.0

    if embeddings:
        emb_a = embeddings.get(scene_a.scene_id)
        emb_b = embeddings.get(scene_b.scene_id)

        if emb_a is not None and emb_b is not None:
            similarity = _cosine_similarity(emb_a, emb_b)
            # Convert similarity to shift score (0 = same, 1 = different)
            topic_shift_score = 1.0 - max(0.0, min(1.0, similarity))

    # Fallback: use transcript word overlap if no embeddings or embedding missing
    if topic_shift_score == 0.0 and transcript_a and transcript_b:
        similarity = _word_overlap_similarity(transcript_a, transcript_b)
        # Convert similarity to shift score
        # Scale up since word overlap typically produces lower similarity values
        topic_shift_score = 1.0 - max(0.0, min(1.0, similarity * 1.5))

    return SceneChange(
        scene_a_id=scene_a.scene_id,
        scene_b_id=scene_b.scene_id,
        visual_changes=visual_changes,
        topic_shift_score=round(topic_shift_score, 3),
        content_type_change=content_type_change,
        content_type_from=content_type_a,
        content_type_to=content_type_b,
    )


def _load_embeddings_dict(cache_dir: Path) -> dict[int, np.ndarray] | None:
    """Load scene embeddings as a dict mapping scene_id to embedding.

    Args:
        cache_dir: Video cache directory

    Returns:
        Dict of scene_id -> embedding, or None if not available
    """
    emb_path = cache_dir / "embeddings" / "scene_embeddings.npy"
    ids_path = cache_dir / "embeddings" / "scene_ids.json"

    if not emb_path.exists() or not ids_path.exists():
        return None

    try:
        embeddings = np.load(emb_path)
        scene_ids = json.loads(ids_path.read_text())

        return {
            scene_id: embeddings[i]
            for i, scene_id in enumerate(scene_ids)
            if i < len(embeddings)
        }
    except Exception as e:
        logger.warning(f"Failed to load embeddings: {e}")
        return None


def detect_scene_changes(
    video_id: str,
    force: bool = False,
    output_base: Path | None = None,
) -> dict:
    """Detect changes between consecutive scenes in a video.

    Follows "Cheap First, Expensive Last" principle:
    1. CACHE - Return structure/changes.json instantly if exists
    2. SCENE DATA - Use visual.json and technical.json (already generated)
    3. EMBEDDINGS - Use cached embeddings for topic shift (if available)

    Args:
        video_id: Video ID
        force: Re-generate even if cached
        output_base: Cache directory

    Returns:
        Dict with change detection results
    """
    t0 = time.time()
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    if not cache_dir.exists():
        return {
            "error": "Video not cached. Run process_video first.",
            "video_id": video_id,
        }

    # 1. CACHE - Return instantly if already exists
    changes_path = get_changes_json_path(cache_dir)
    if not force and changes_path.exists():
        try:
            data = json.loads(changes_path.read_text())
            log_timed(
                f"Scene changes: loaded from cache ({data.get('summary', {}).get('total_changes', 0)} changes)",
                t0,
            )
            return data
        except json.JSONDecodeError:
            pass  # Re-generate if cached data is invalid

    # Load scenes data
    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return {"error": "No scenes found. Run get_scenes first.", "video_id": video_id}

    if len(scenes_data.scenes) < 2:
        return {
            "video_id": video_id,
            "changes": [],
            "summary": {
                "major_transitions": [],
                "major_transition_count": 0,
                "avg_topic_shift": 0.0,
                "content_type_changes": 0,
                "total_changes": 0,
            },
            "message": "Video has fewer than 2 scenes, no changes to detect",
        }

    # 2. Load embeddings if available (for semantic topic shift detection)
    log_timed("Scene changes: loading embeddings...", t0)
    embeddings = _load_embeddings_dict(cache_dir)

    if embeddings:
        logger.info(
            f"Loaded {len(embeddings)} scene embeddings for topic shift detection"
        )
    else:
        logger.info(
            "No embeddings available, falling back to transcript word overlap for topic shift"
        )

    # 3. Detect changes between consecutive scenes
    log_timed("Scene changes: detecting changes between scenes...", t0)
    changes = []

    for i in range(len(scenes_data.scenes) - 1):
        scene_a = scenes_data.scenes[i]
        scene_b = scenes_data.scenes[i + 1]

        change = _detect_changes_between_scenes(scene_a, scene_b, cache_dir, embeddings)
        changes.append(change)

    # Build result
    changes_data = ChangesData(
        video_id=video_id,
        changes=changes,
    )

    result = changes_data.to_dict()

    # Save to cache
    changes_path.write_text(json.dumps(result, indent=2))

    # Update state.json
    state_file = cache_dir / "state.json"
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            state["scene_changes_complete"] = True
            state_file.write_text(json.dumps(state, indent=2))
        except (json.JSONDecodeError, OSError):
            pass

    log_timed(
        f"Scene changes complete: {len(changes)} changes detected, "
        f"{result['summary']['major_transition_count']} major transitions",
        t0,
    )

    return result


def get_scene_changes(
    video_id: str,
    output_base: Path | None = None,
) -> dict:
    """Get cached scene change data for a video.

    Does NOT generate new detection - use detect_scene_changes for that.

    Args:
        video_id: Video ID
        output_base: Cache directory

    Returns:
        Dict with cached scene change data
    """
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    if not cache_dir.exists():
        return {"error": "Video not cached", "video_id": video_id}

    changes_path = get_changes_json_path(cache_dir)
    if not changes_path.exists():
        return {
            "error": "No scene changes data. Run detect_scene_changes first.",
            "video_id": video_id,
        }

    try:
        data = json.loads(changes_path.read_text())
        return data
    except json.JSONDecodeError:
        return {"error": "Invalid changes.json", "video_id": video_id}


def get_major_transitions(
    video_id: str,
    output_base: Path | None = None,
) -> list[int]:
    """Get scene IDs of major transitions.

    A major transition is a scene where:
    - Topic shift score > 0.5, OR
    - Content type changed

    Args:
        video_id: Video ID
        output_base: Cache directory

    Returns:
        List of scene IDs that represent major transitions
    """
    data = get_scene_changes(video_id, output_base)

    if "error" in data:
        return []

    return data.get("summary", {}).get("major_transitions", [])
