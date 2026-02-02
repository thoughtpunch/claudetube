"""
Narrative structure detection for videos.

Detects the high-level structure of a video by clustering scenes into
sections and classifying the overall video type. Uses cached scene
embeddings for topic similarity clustering.

Follows the "Cheap First, Expensive Last" principle:
1. CACHE - Return instantly if structure/narrative.json already exists
2. SCENE DATA - Use transcript text and visual/technical data (already generated)
3. EMBEDDINGS - Use cached scene embeddings for topic clustering
4. HEURISTICS - Fallback to transcript-only heuristics if no embeddings
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
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


# Emphasis phrases that indicate important content
_EMPHASIS_PHRASES = frozenset(
    {
        "important",
        "key point",
        "remember",
        "crucial",
        "essential",
        "pay attention",
        "note that",
        "in summary",
        "to summarize",
        "in conclusion",
        "let me show",
        "let's look at",
    }
)

# Intro phrases that suggest introductory content
_INTRO_PHRASES = frozenset(
    {
        "welcome",
        "hello",
        "hey everyone",
        "today we",
        "in this video",
        "i'm going to",
        "we're going to",
        "let's get started",
        "introduction",
        "overview",
        "what we'll cover",
    }
)

# Conclusion phrases
_CONCLUSION_PHRASES = frozenset(
    {
        "in conclusion",
        "to summarize",
        "to wrap up",
        "that's it",
        "thanks for watching",
        "see you",
        "goodbye",
        "subscribe",
        "like and subscribe",
        "next time",
        "in the next",
    }
)


@dataclass
class Section:
    """A narrative section grouping consecutive scenes."""

    section_id: int
    label: str  # "introduction", "main_content", "conclusion", "transition"
    start_time: float
    end_time: float
    scene_ids: list[int] = field(default_factory=list)
    summary: str = ""

    def duration(self) -> float:
        """Get section duration in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "section_id": self.section_id,
            "label": self.label,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "scene_ids": self.scene_ids,
            "summary": self.summary,
            "duration": round(self.duration(), 1),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Section:
        """Create from dictionary."""
        return cls(
            section_id=data["section_id"],
            label=data["label"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            scene_ids=data.get("scene_ids", []),
            summary=data.get("summary", ""),
        )


@dataclass
class NarrativeStructure:
    """Complete narrative structure for a video."""

    video_id: str
    video_type: str  # "coding_tutorial", "lecture", "demo", "interview", "tutorial"
    sections: list[Section] = field(default_factory=list)
    cluster_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_id": self.video_id,
            "video_type": self.video_type,
            "sections": [s.to_dict() for s in self.sections],
            "cluster_count": self.cluster_count,
            "summary": {
                "section_count": len(self.sections),
                "video_type": self.video_type,
                "section_labels": [s.label for s in self.sections],
                "total_duration": round(sum(s.duration() for s in self.sections), 1),
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> NarrativeStructure:
        """Create from dictionary."""
        return cls(
            video_id=data["video_id"],
            video_type=data["video_type"],
            sections=[Section.from_dict(s) for s in data.get("sections", [])],
            cluster_count=data.get("cluster_count", 0),
        )


def get_narrative_json_path(cache_dir: Path) -> Path:
    """Get path to structure/narrative.json for a video.

    Args:
        cache_dir: Video cache directory

    Returns:
        Path to structure/narrative.json
    """
    structure_dir = cache_dir / "structure"
    structure_dir.mkdir(parents=True, exist_ok=True)
    return structure_dir / "narrative.json"


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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity (-1 to 1)
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def _find_optimal_clusters(
    similarity_matrix: np.ndarray,
    max_k: int,
) -> int:
    """Find optimal number of clusters using silhouette score.

    Uses AgglomerativeClustering with ward linkage on precomputed
    distance matrix derived from similarity scores.

    Args:
        similarity_matrix: NxN cosine similarity matrix.
        max_k: Maximum number of clusters to try.

    Returns:
        Optimal number of clusters (2 to max_k).
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    # Convert similarity to distance
    distance_matrix = 1.0 - np.clip(similarity_matrix, 0.0, 1.0)

    best_score = -1.0
    best_k = 2

    for k in range(2, max_k + 1):
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distance_matrix)

        # Silhouette score needs at least 2 unique labels
        if len(set(labels)) < 2:
            continue

        score = silhouette_score(distance_matrix, labels, metric="precomputed")

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _cluster_scenes_with_embeddings(
    scenes: list[SceneBoundary],
    embeddings: dict[int, np.ndarray],
) -> list[int]:
    """Cluster scenes by embedding similarity preserving temporal order.

    Uses AgglomerativeClustering with optimal k selection via
    silhouette score. Only clusters scenes that have embeddings.

    Args:
        scenes: List of scenes (in temporal order).
        embeddings: Dict of scene_id -> embedding vector.

    Returns:
        List of cluster labels (one per scene). -1 for scenes without embeddings.
    """
    from sklearn.cluster import AgglomerativeClustering

    # Build embedding matrix for scenes that have embeddings
    scene_indices = []  # indices into scenes list
    emb_list = []

    for i, scene in enumerate(scenes):
        emb = embeddings.get(scene.scene_id)
        if emb is not None:
            scene_indices.append(i)
            emb_list.append(emb)

    if len(emb_list) < 3:
        # Not enough scenes for meaningful clustering
        return [0] * len(scenes)

    emb_array = np.stack(emb_list)

    # Compute similarity matrix
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = emb_array / norms
    similarity_matrix = normalized @ normalized.T

    # Find optimal k
    max_k = min(8, len(emb_list) // 2)
    max_k = max(2, max_k)
    optimal_k = _find_optimal_clusters(similarity_matrix, max_k)

    # Cluster
    distance_matrix = 1.0 - np.clip(similarity_matrix, 0.0, 1.0)
    clustering = AgglomerativeClustering(
        n_clusters=optimal_k,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distance_matrix)

    # Map back to full scene list
    result = [-1] * len(scenes)
    for idx, scene_idx in enumerate(scene_indices):
        result[scene_idx] = int(labels[idx])

    # Fill gaps (scenes without embeddings inherit neighbor's label)
    for i in range(len(result)):
        if result[i] == -1:
            # Look forward then backward
            for j in range(i + 1, len(result)):
                if result[j] != -1:
                    result[i] = result[j]
                    break
            if result[i] == -1:
                for j in range(i - 1, -1, -1):
                    if result[j] != -1:
                        result[i] = result[j]
                        break

    return result


def _cluster_scenes_by_transcript(
    scenes: list[SceneBoundary],
) -> list[int]:
    """Fallback clustering using transcript text similarity.

    When embeddings are not available, uses simple word overlap
    between consecutive scenes to detect topic boundaries.

    Args:
        scenes: List of scenes (in temporal order).

    Returns:
        List of cluster labels (one per scene).
    """
    if len(scenes) < 3:
        return [0] * len(scenes)

    # Compute word overlap between consecutive scenes
    shifts = []
    for i in range(len(scenes) - 1):
        words_a = set(scenes[i].transcript_text.lower().split())
        words_b = set(scenes[i + 1].transcript_text.lower().split())

        if not words_a or not words_b:
            shifts.append(1.0)
            continue

        overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
        shifts.append(1.0 - overlap)

    # Find significant shifts (above mean + 0.5 * std)
    if not shifts:
        return [0] * len(scenes)

    mean_shift = np.mean(shifts)
    std_shift = np.std(shifts)
    threshold = mean_shift + 0.5 * std_shift

    # Assign cluster labels based on detected boundaries
    labels = [0]
    current_label = 0
    for shift in shifts:
        if shift > threshold:
            current_label += 1
        labels.append(current_label)

    return labels


def _build_sections(
    scenes: list[SceneBoundary],
    labels: list[int],
    cache_dir: Path,
) -> list[Section]:
    """Group consecutively-labeled scenes into sections.

    Preserves temporal order: even if two distant scene groups share
    a cluster label, they form separate sections.

    Args:
        scenes: List of scenes in temporal order.
        labels: Cluster label per scene.
        cache_dir: Video cache directory for loading scene data.

    Returns:
        List of Section objects in temporal order.
    """
    if not scenes:
        return []

    sections: list[Section] = []
    current_label = labels[0]
    current_scenes = [scenes[0]]

    for scene, label in zip(scenes[1:], labels[1:], strict=True):
        if label != current_label:
            # New section
            section = _create_section(current_scenes, len(sections), cache_dir)
            sections.append(section)
            current_scenes = [scene]
            current_label = label
        else:
            current_scenes.append(scene)

    # Final section
    section = _create_section(current_scenes, len(sections), cache_dir)
    sections.append(section)

    # Label sections (intro/main/conclusion)
    _label_sections(sections, scenes)

    return sections


def _create_section(
    section_scenes: list[SceneBoundary],
    section_id: int,
    cache_dir: Path,
) -> Section:
    """Create a Section from a group of scenes.

    Args:
        section_scenes: Scenes belonging to this section.
        section_id: Section index.
        cache_dir: Video cache directory.

    Returns:
        Section with summary generated from scene transcripts.
    """
    scene_ids = [s.scene_id for s in section_scenes]
    start_time = section_scenes[0].start_time
    end_time = section_scenes[-1].end_time

    # Build summary from transcript text
    all_text = " ".join(s.transcript_text for s in section_scenes if s.transcript_text)

    # Truncate to reasonable length for summary
    summary = all_text[:300].strip()
    if len(all_text) > 300:
        summary += "..."

    return Section(
        section_id=section_id,
        label="main_content",  # Default; _label_sections overrides
        start_time=start_time,
        end_time=end_time,
        scene_ids=scene_ids,
        summary=summary,
    )


def _label_sections(
    sections: list[Section],
    scenes: list[SceneBoundary],
) -> None:
    """Label sections as introduction, main_content, conclusion, or transition.

    Uses heuristics based on position, duration, and transcript content.
    Modifies sections in place.

    Args:
        sections: List of sections to label.
        scenes: All scenes for context.
    """
    if not sections:
        return

    total_duration = sum(s.duration() for s in sections)

    for i, section in enumerate(sections):
        text = section.summary.lower()

        # Check for intro phrases in first section
        if i == 0:
            has_intro_phrases = any(p in text for p in _INTRO_PHRASES)
            is_short = section.duration() < total_duration * 0.2
            if has_intro_phrases or is_short:
                section.label = "introduction"
                continue

        # Check for conclusion phrases in last section
        if i == len(sections) - 1:
            has_conclusion_phrases = any(p in text for p in _CONCLUSION_PHRASES)
            is_short = section.duration() < total_duration * 0.15
            if has_conclusion_phrases or is_short:
                section.label = "conclusion"
                continue

        # Check for very short transitional sections
        if len(section.scene_ids) == 1 and section.duration() < 15:
            section.label = "transition"
            continue

        # Default: main_content
        section.label = "main_content"


def _get_content_type(cache_dir: Path, scene_id: int) -> str | None:
    """Extract content type for a scene from cached data.

    Args:
        cache_dir: Video cache directory.
        scene_id: Scene ID.

    Returns:
        Content type string or None.
    """
    technical_path = get_technical_json_path(cache_dir, scene_id)
    if technical_path.exists():
        try:
            data = json.loads(technical_path.read_text())
            ct = data.get("content_type")
            if ct:
                return ct
            # Check frames
            frame_types = []
            for frame in data.get("frames", []):
                ft = frame.get("content_type")
                if ft:
                    frame_types.append(ft)
            if frame_types:
                return Counter(frame_types).most_common(1)[0][0]
        except (json.JSONDecodeError, OSError):
            pass

    visual_path = get_visual_json_path(cache_dir, scene_id)
    if visual_path.exists():
        try:
            data = json.loads(visual_path.read_text())
            description = data.get("description", "").lower()
            if any(kw in description for kw in ("code", "editor", "terminal")):
                return "code"
            if any(kw in description for kw in ("slide", "presentation")):
                return "slides"
            if any(kw in description for kw in ("person", "speaker", "talking")):
                return "talking_head"
        except (json.JSONDecodeError, OSError):
            pass

    return None


def classify_video_type(
    scenes: list[SceneBoundary],
    sections: list[Section],
    cache_dir: Path,
) -> str:
    """Classify video type from content patterns and structure.

    Args:
        scenes: All scenes.
        sections: Detected narrative sections.
        cache_dir: Video cache directory.

    Returns:
        Video type string: "coding_tutorial", "lecture", "demo",
        "interview", "tutorial", "presentation", "screencast".
    """
    # Count content types across all scenes
    content_types: list[str] = []
    for scene in scenes:
        ct = _get_content_type(cache_dir, scene.scene_id)
        if ct:
            content_types.append(ct)

    type_counts = Counter(content_types)
    total = len(scenes)

    if total == 0:
        return "unknown"

    # Classify based on content type distribution
    code_ratio = type_counts.get("code", 0) / total
    slides_ratio = type_counts.get("slides", 0) / total
    talking_head_ratio = type_counts.get("talking_head", 0) / total
    terminal_ratio = type_counts.get("terminal", 0) / total

    if code_ratio > 0.3 or (code_ratio + terminal_ratio) > 0.4:
        return "coding_tutorial"
    if slides_ratio > 0.5:
        return "lecture"
    if talking_head_ratio > 0.7:
        return "interview"
    if slides_ratio > 0.3 and talking_head_ratio > 0.2:
        return "presentation"
    if len(sections) > 5:
        return "tutorial"

    # Transcript-based heuristics
    all_text = " ".join(s.transcript_text.lower() for s in scenes if s.transcript_text)
    if "step" in all_text and ("follow" in all_text or "next" in all_text):
        return "tutorial"
    if "demo" in all_text or "let me show" in all_text:
        return "demo"

    return "tutorial"


def detect_narrative_structure(
    video_id: str,
    force: bool = False,
    output_base: Path | None = None,
) -> dict:
    """Detect the narrative structure of a video.

    Clusters scenes into sections and classifies the overall video type.

    Follows "Cheap First, Expensive Last" principle:
    1. CACHE - Return structure/narrative.json instantly if exists
    2. EMBEDDINGS - Use cached embeddings for clustering (if available)
    3. TRANSCRIPT - Fallback to transcript-based heuristics

    Args:
        video_id: Video ID.
        force: Re-generate even if cached.
        output_base: Cache directory override.

    Returns:
        Dict with narrative structure results.
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
    narrative_path = get_narrative_json_path(cache_dir)
    if not force and narrative_path.exists():
        try:
            data = json.loads(narrative_path.read_text())
            log_timed(
                f"Narrative structure: loaded from cache "
                f"({data.get('summary', {}).get('section_count', 0)} sections)",
                t0,
            )
            return data
        except json.JSONDecodeError:
            pass  # Re-generate if cached data is invalid

    # Load scenes data
    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return {
            "error": "No scenes found. Run get_scenes first.",
            "video_id": video_id,
        }

    scenes = scenes_data.scenes

    if len(scenes) < 2:
        # Single scene video
        structure = NarrativeStructure(
            video_id=video_id,
            video_type=classify_video_type(scenes, [], cache_dir),
            sections=[
                Section(
                    section_id=0,
                    label="main_content",
                    start_time=scenes[0].start_time if scenes else 0.0,
                    end_time=scenes[0].end_time if scenes else 0.0,
                    scene_ids=[s.scene_id for s in scenes],
                    summary=scenes[0].transcript_text[:300] if scenes else "",
                )
            ],
            cluster_count=1,
        )
        result = structure.to_dict()
        narrative_path.write_text(json.dumps(result, indent=2))
        log_timed("Narrative structure: single scene video", t0)
        return result

    # 2. Try embedding-based clustering
    log_timed("Narrative structure: loading embeddings...", t0)
    embeddings = _load_embeddings_dict(cache_dir)

    if embeddings and len(embeddings) >= 3:
        log_timed(
            f"Narrative structure: clustering {len(scenes)} scenes "
            f"with {len(embeddings)} embeddings...",
            t0,
        )
        labels = _cluster_scenes_with_embeddings(scenes, embeddings)
    else:
        # 3. Fallback to transcript-based clustering
        log_timed(
            "Narrative structure: no embeddings available, "
            "using transcript heuristics...",
            t0,
        )
        labels = _cluster_scenes_by_transcript(scenes)

    cluster_count = len(set(labels))

    # Build sections from clusters
    sections = _build_sections(scenes, labels, cache_dir)

    # Classify video type
    video_type = classify_video_type(scenes, sections, cache_dir)

    structure = NarrativeStructure(
        video_id=video_id,
        video_type=video_type,
        sections=sections,
        cluster_count=cluster_count,
    )

    result = structure.to_dict()

    # Save to cache
    narrative_path.write_text(json.dumps(result, indent=2))

    # Update state.json
    state_file = cache_dir / "state.json"
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            state["narrative_structure_complete"] = True
            state_file.write_text(json.dumps(state, indent=2))
        except (json.JSONDecodeError, OSError):
            pass

    log_timed(
        f"Narrative structure complete: {len(sections)} sections, "
        f"type={video_type}, clusters={cluster_count}",
        t0,
    )

    return result


def get_narrative_structure(
    video_id: str,
    output_base: Path | None = None,
) -> dict:
    """Get cached narrative structure for a video.

    Does NOT generate new analysis - use detect_narrative_structure for that.

    Args:
        video_id: Video ID.
        output_base: Cache directory override.

    Returns:
        Dict with cached narrative structure data.
    """
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    if not cache_dir.exists():
        return {"error": "Video not cached", "video_id": video_id}

    narrative_path = get_narrative_json_path(cache_dir)
    if not narrative_path.exists():
        return {
            "error": "No narrative structure data. Run detect_narrative_structure first.",
            "video_id": video_id,
        }

    try:
        return json.loads(narrative_path.read_text())
    except json.JSONDecodeError:
        return {"error": "Invalid narrative.json", "video_id": video_id}
