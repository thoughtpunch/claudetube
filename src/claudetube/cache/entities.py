"""
Entity tracking for objects and concepts across scenes.

Tracks:
- Objects: Physical items appearing/disappearing visually (from visual.json)
- Concepts: Key terms mentioned in transcript (using TF-IDF)

Follows the "Cheap First, Expensive Last" architecture:
1. CACHE - Return cached entities/*.json instantly if available
2. CHEAP - Extract from already-processed scene data (visual.json, transcript)
3. NO NEW COMPUTE - Never triggers expensive operations (visual analysis, transcription)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Default number of top concepts to track
DEFAULT_TOP_CONCEPTS = 20

# Minimum TF-IDF score to include a concept
MIN_TFIDF_SCORE = 0.1


@dataclass
class ObjectAppearance:
    """A single appearance of an object in a scene."""

    scene_id: int
    timestamp: float  # Start of scene

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"scene_id": self.scene_id, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, data: dict) -> ObjectAppearance:
        """Create from dictionary."""
        return cls(scene_id=data["scene_id"], timestamp=data["timestamp"])


@dataclass
class TrackedObject:
    """An object tracked across multiple scenes."""

    name: str
    appearances: list[ObjectAppearance] = field(default_factory=list)

    @property
    def first_seen(self) -> float:
        """Timestamp when object first appears."""
        return self.appearances[0].timestamp if self.appearances else 0.0

    @property
    def last_seen(self) -> float:
        """Timestamp when object last appears."""
        return self.appearances[-1].timestamp if self.appearances else 0.0

    @property
    def frequency(self) -> int:
        """Number of scenes object appears in."""
        return len(self.appearances)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "appearances": [a.to_dict() for a in self.appearances],
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrackedObject:
        """Create from dictionary."""
        obj = cls(name=data["name"])
        obj.appearances = [
            ObjectAppearance.from_dict(a) for a in data.get("appearances", [])
        ]
        return obj


@dataclass
class ConceptMention:
    """A mention of a concept in a scene."""

    scene_id: int
    timestamp: float
    score: float  # TF-IDF relevance score

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "scene_id": self.scene_id,
            "timestamp": self.timestamp,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConceptMention:
        """Create from dictionary."""
        return cls(
            scene_id=data["scene_id"], timestamp=data["timestamp"], score=data["score"]
        )


@dataclass
class TrackedConcept:
    """A concept tracked across multiple scenes."""

    term: str
    mentions: list[ConceptMention] = field(default_factory=list)

    @property
    def first_mention(self) -> float:
        """Timestamp when concept first mentioned."""
        return self.mentions[0].timestamp if self.mentions else 0.0

    @property
    def frequency(self) -> int:
        """Number of scenes concept appears in."""
        return len(self.mentions)

    @property
    def avg_score(self) -> float:
        """Average TF-IDF score across mentions."""
        if not self.mentions:
            return 0.0
        return sum(m.score for m in self.mentions) / len(self.mentions)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "term": self.term,
            "mentions": [m.to_dict() for m in self.mentions],
            "first_mention": self.first_mention,
            "frequency": self.frequency,
            "avg_score": round(self.avg_score, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrackedConcept:
        """Create from dictionary."""
        concept = cls(term=data["term"])
        concept.mentions = [
            ConceptMention.from_dict(m) for m in data.get("mentions", [])
        ]
        return concept


def get_entities_dir(cache_dir: Path) -> Path:
    """Get the entities directory for a video cache.

    Args:
        cache_dir: Video cache directory (e.g., ~/.claude/video_cache/{video_id}/)

    Returns:
        Path to entities/ directory (created if needed)
    """
    entities_dir = cache_dir / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    return entities_dir


def get_objects_json_path(cache_dir: Path) -> Path:
    """Get path to entities/objects.json."""
    return get_entities_dir(cache_dir) / "objects.json"


def get_concepts_json_path(cache_dir: Path) -> Path:
    """Get path to entities/concepts.json."""
    return get_entities_dir(cache_dir) / "concepts.json"


def load_objects(cache_dir: Path) -> dict[str, TrackedObject] | None:
    """Load cached objects.json.

    Returns:
        Dict mapping object name to TrackedObject, or None if not cached.
    """
    path = get_objects_json_path(cache_dir)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        return {
            name: TrackedObject.from_dict(obj)
            for name, obj in data.get("objects", {}).items()
        }
    except (json.JSONDecodeError, KeyError):
        return None


def load_concepts(cache_dir: Path) -> dict[str, TrackedConcept] | None:
    """Load cached concepts.json.

    Returns:
        Dict mapping term to TrackedConcept, or None if not cached.
    """
    path = get_concepts_json_path(cache_dir)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        return {
            term: TrackedConcept.from_dict(concept)
            for term, concept in data.get("concepts", {}).items()
        }
    except (json.JSONDecodeError, KeyError):
        return None


def save_objects(
    cache_dir: Path, objects: dict[str, TrackedObject], video_id: str
) -> None:
    """Save objects to entities/objects.json and sync to SQLite."""
    path = get_objects_json_path(cache_dir)
    data = {
        "video_id": video_id,
        "object_count": len(objects),
        "objects": {name: obj.to_dict() for name, obj in objects.items()},
    }
    path.write_text(json.dumps(data, indent=2))

    # Dual-write: sync entities to SQLite (fire-and-forget)
    try:
        from claudetube.db.sync import (
            get_video_uuid,
            sync_entity,
            sync_entity_appearance,
            sync_entity_video_summary,
        )

        video_uuid = get_video_uuid(video_id)
        if video_uuid:
            for name, obj in objects.items():
                # Insert entity (uses INSERT OR IGNORE for dedup)
                entity_uuid = sync_entity(name, "object")
                if entity_uuid:
                    # Insert appearances for each scene
                    for appearance in obj.appearances:
                        sync_entity_appearance(
                            entity_uuid=entity_uuid,
                            video_uuid=video_uuid,
                            scene_id=appearance.scene_id,
                            timestamp=appearance.timestamp,
                        )
                    # Insert video summary
                    sync_entity_video_summary(
                        entity_uuid=entity_uuid,
                        video_uuid=video_uuid,
                        frequency=obj.frequency,
                    )
    except Exception:
        # Fire-and-forget: don't disrupt JSON writes
        pass


def save_concepts(
    cache_dir: Path, concepts: dict[str, TrackedConcept], video_id: str
) -> None:
    """Save concepts to entities/concepts.json and sync to SQLite."""
    path = get_concepts_json_path(cache_dir)
    data = {
        "video_id": video_id,
        "concept_count": len(concepts),
        "concepts": {term: concept.to_dict() for term, concept in concepts.items()},
    }
    path.write_text(json.dumps(data, indent=2))

    # Dual-write: sync entities to SQLite (fire-and-forget)
    try:
        from claudetube.db.sync import (
            get_video_uuid,
            sync_entity,
            sync_entity_appearance,
            sync_entity_video_summary,
        )

        video_uuid = get_video_uuid(video_id)
        if video_uuid:
            for term, concept in concepts.items():
                # Insert entity (uses INSERT OR IGNORE for dedup)
                entity_uuid = sync_entity(term, "concept")
                if entity_uuid:
                    # Insert appearances for each mention
                    for mention in concept.mentions:
                        sync_entity_appearance(
                            entity_uuid=entity_uuid,
                            video_uuid=video_uuid,
                            scene_id=mention.scene_id,
                            timestamp=mention.timestamp,
                            score=mention.score,
                        )
                    # Insert video summary with avg_score
                    sync_entity_video_summary(
                        entity_uuid=entity_uuid,
                        video_uuid=video_uuid,
                        frequency=concept.frequency,
                        avg_score=concept.avg_score if concept.avg_score > 0 else None,
                    )
    except Exception:
        # Fire-and-forget: don't disrupt JSON writes
        pass


def track_objects_from_scenes(scenes: list[dict]) -> dict[str, TrackedObject]:
    """Track physical objects across scenes from visual.json data.

    Extracts objects from the 'objects' field of each scene's visual data
    and tracks their appearances across the video.

    Args:
        scenes: List of scene dicts, each containing:
            - scene_id: int
            - start_time: float (seconds)
            - visual: dict with 'objects' list (optional)

    Returns:
        Dict mapping normalized object name to TrackedObject.

    Example:
        >>> scenes = [
        ...     {"scene_id": 0, "start_time": 0.0, "visual": {"objects": ["laptop", "Whiteboard"]}},
        ...     {"scene_id": 1, "start_time": 30.0, "visual": {"objects": ["laptop", "code editor"]}},
        ... ]
        >>> objects = track_objects_from_scenes(scenes)
        >>> objects["laptop"].frequency
        2
        >>> objects["whiteboard"].first_seen
        0.0
    """
    object_appearances: dict[str, list[ObjectAppearance]] = defaultdict(list)

    for scene in scenes:
        scene_id = scene.get("scene_id", 0)
        start_time = scene.get("start_time", 0.0)

        # Get objects from visual data
        visual = scene.get("visual", {})
        detected = visual.get("objects", [])

        for obj in detected:
            # Normalize: lowercase, strip whitespace
            normalized = obj.lower().strip()
            if normalized:
                object_appearances[normalized].append(
                    ObjectAppearance(scene_id=scene_id, timestamp=start_time)
                )

    # Convert to TrackedObject
    return {
        name: TrackedObject(name=name, appearances=appearances)
        for name, appearances in object_appearances.items()
    }


def track_concepts_from_scenes(
    scenes: list[dict],
    top_n: int = DEFAULT_TOP_CONCEPTS,
    min_score: float = MIN_TFIDF_SCORE,
) -> dict[str, TrackedConcept]:
    """Extract and track key concepts from transcript using TF-IDF.

    Uses scikit-learn's TfidfVectorizer to identify important terms
    per scene, then tracks their mentions across the video.

    Args:
        scenes: List of scene dicts, each containing:
            - scene_id: int
            - start_time: float (seconds)
            - transcript_text: str (scene transcript)
        top_n: Maximum number of concepts to track. Default 20.
        min_score: Minimum TF-IDF score to include. Default 0.1.

    Returns:
        Dict mapping term to TrackedConcept, sorted by frequency.

    Example:
        >>> scenes = [
        ...     {"scene_id": 0, "start_time": 0.0, "transcript_text": "Python programming language"},
        ...     {"scene_id": 1, "start_time": 30.0, "transcript_text": "Machine learning neural networks"},
        ... ]
        >>> concepts = track_concepts_from_scenes(scenes, top_n=5)
        >>> "python" in concepts or "machine learning" in concepts
        True
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        logger.warning("scikit-learn not installed. Cannot track concepts.")
        return {}

    # Extract texts from scenes
    texts = []
    scene_info = []  # Track scene_id and start_time for each text

    for scene in scenes:
        text = scene.get("transcript_text", "")
        if not text:
            # Fallback to transcript_segment for older data
            text = scene.get("transcript_segment", "")

        if text.strip():
            texts.append(text)
            scene_info.append(
                {
                    "scene_id": scene.get("scene_id", 0),
                    "start_time": scene.get("start_time", 0.0),
                }
            )

    if len(texts) < 2:
        logger.info("Not enough transcript data for concept tracking")
        return {}

    # Build TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),  # Unigrams and bigrams
        max_features=100,
        min_df=1,  # Include terms appearing in at least 1 document
    )

    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        # Empty vocabulary
        return {}

    feature_names = vectorizer.get_feature_names_out()
    concept_mentions: dict[str, list[ConceptMention]] = defaultdict(list)

    # Extract top terms per scene
    for scene_idx, info in enumerate(scene_info):
        scene_tfidf = tfidf[scene_idx].toarray()[0]

        # Get indices sorted by TF-IDF score descending
        top_indices = scene_tfidf.argsort()[::-1]

        for idx in top_indices[:10]:  # Top 10 terms per scene
            score = float(scene_tfidf[idx])
            if score >= min_score:
                term = feature_names[idx]
                concept_mentions[term].append(
                    ConceptMention(
                        scene_id=info["scene_id"],
                        timestamp=info["start_time"],
                        score=score,
                    )
                )

    # Sort by frequency and take top_n
    sorted_concepts = sorted(
        concept_mentions.items(), key=lambda x: len(x[1]), reverse=True
    )[:top_n]

    return {
        term: TrackedConcept(term=term, mentions=mentions)
        for term, mentions in sorted_concepts
    }


def track_entities(
    video_id: str,
    cache_dir: Path,
    force: bool = False,
) -> dict:
    """Track objects and concepts for a video.

    Main entry point for entity tracking. Loads scene data and visual.json
    files, then tracks objects and concepts across the video.

    Follows "Cheap First, Expensive Last":
    1. Return cached entities instantly if available
    2. Load from already-processed visual.json and scenes.json
    3. Never triggers new visual analysis or transcription

    Args:
        video_id: Video ID
        cache_dir: Video cache directory
        force: Re-process even if cached

    Returns:
        Dict with tracking results:
        - video_id: str
        - objects: dict of tracked objects
        - concepts: dict of tracked concepts
        - error: str if failed
    """
    from claudetube.cache.scenes import load_scenes_data

    # 1. CACHE - Return instantly if both files exist
    if not force:
        cached_objects = load_objects(cache_dir)
        cached_concepts = load_concepts(cache_dir)
        if cached_objects is not None and cached_concepts is not None:
            logger.info(f"Loaded cached entities for {video_id}")
            return {
                "video_id": video_id,
                "objects": {
                    name: obj.to_dict() for name, obj in cached_objects.items()
                },
                "concepts": {term: c.to_dict() for term, c in cached_concepts.items()},
                "from_cache": True,
            }

    # 2. Load scenes data
    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return {"video_id": video_id, "error": "No scenes found. Run get_scenes first."}

    # 3. Build scene list with visual data
    scenes_with_visual = []
    for scene in scenes_data.scenes:
        scene_dict = {
            "scene_id": scene.scene_id,
            "start_time": scene.start_time,
            "transcript_text": scene.transcript_text,
        }

        # Load visual.json if available
        visual_path = (
            cache_dir / "scenes" / f"scene_{scene.scene_id:03d}" / "visual.json"
        )
        if visual_path.exists():
            try:
                visual_data = json.loads(visual_path.read_text())
                scene_dict["visual"] = {
                    "objects": visual_data.get("objects", []),
                    "people": visual_data.get("people", []),
                    "text_on_screen": visual_data.get("text_on_screen", []),
                }
            except json.JSONDecodeError:
                pass

        scenes_with_visual.append(scene_dict)

    # 4. Track objects (from visual.json)
    objects = track_objects_from_scenes(scenes_with_visual)
    logger.info(f"Tracked {len(objects)} objects for {video_id}")

    # 5. Track concepts (from transcript)
    concepts = track_concepts_from_scenes(scenes_with_visual)
    logger.info(f"Tracked {len(concepts)} concepts for {video_id}")

    # 6. Save to cache
    save_objects(cache_dir, objects, video_id)
    save_concepts(cache_dir, concepts, video_id)
    logger.info(f"Saved entities to {get_entities_dir(cache_dir)}")

    # Record pipeline step (fire-and-forget)
    try:
        from claudetube.db.sync import record_pipeline_step

        record_pipeline_step(
            video_id,
            step_type="entity_extract",
            status="completed",
        )
    except Exception:
        pass

    return {
        "video_id": video_id,
        "objects": {name: obj.to_dict() for name, obj in objects.items()},
        "concepts": {term: c.to_dict() for term, c in concepts.items()},
        "from_cache": False,
    }
