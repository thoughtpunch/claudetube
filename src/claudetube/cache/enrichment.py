"""
Interaction-driven cache enrichment.

Automatically enriches the cache when Claude examines frames or answers questions.
Records observations, updates scene metadata, and boosts relevance for examined scenes.

This module implements progressive learning - subsequent queries benefit from prior analysis.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from claudetube.cache.memory import VideoMemory
from claudetube.cache.scenes import load_scenes_data

if TYPE_CHECKING:
    from pathlib import Path


def find_scene_at_timestamp(
    cache_dir: Path,
    timestamp: float,
) -> int | None:
    """Find the scene containing a specific timestamp.

    Args:
        cache_dir: Video cache directory.
        timestamp: Time in seconds.

    Returns:
        Scene ID (0-based) or None if no scene contains the timestamp.
    """
    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data or not scenes_data.scenes:
        return None

    for scene in scenes_data.scenes:
        if scene.start_time <= timestamp < scene.end_time:
            return scene.scene_id

    # Check if timestamp is past the last scene
    if scenes_data.scenes:
        last_scene = scenes_data.scenes[-1]
        if timestamp >= last_scene.start_time:
            return last_scene.scene_id

    return None


def find_relevant_scenes(
    cache_dir: Path,
    question: str,
    answer: str | None = None,
) -> list[int]:
    """Identify which scenes are relevant to a question and answer.

    Uses simple keyword matching against scene transcripts.
    More sophisticated semantic matching could be added later.

    Args:
        cache_dir: Video cache directory.
        question: The question that was asked.
        answer: Optional answer text (improves relevance detection).

    Returns:
        List of relevant scene IDs (up to 5).
    """
    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data or not scenes_data.scenes:
        return []

    # Extract significant words (4+ chars) from question and answer
    combined_text = question
    if answer:
        combined_text += " " + answer

    # Filter out common words
    stop_words = {
        "what", "when", "where", "which", "that", "this", "these", "those",
        "have", "been", "being", "would", "could", "should", "about", "from",
        "with", "into", "does", "doing", "their", "there", "they", "them",
        "will", "were", "your", "more", "most", "some", "than", "then",
    }

    words = {
        word.lower()
        for word in re.findall(r"\w+", combined_text.lower())
        if len(word) >= 4 and word.lower() not in stop_words
    }

    if not words:
        return []

    # Score each scene by word overlap
    scene_scores: list[tuple[int, float]] = []

    for scene in scenes_data.scenes:
        transcript = scene.transcript_text or ""
        if not transcript:
            continue

        transcript_lower = transcript.lower()
        transcript_words = set(re.findall(r"\w+", transcript_lower))

        # Count matching words
        matches = sum(1 for w in words if w in transcript_words)
        if matches > 0:
            # Normalize by number of query words
            score = matches / len(words)
            scene_scores.append((scene.scene_id, score))

    # Sort by score descending and return top 5
    scene_scores.sort(key=lambda x: x[1], reverse=True)
    return [scene_id for scene_id, _ in scene_scores[:5]]


def get_relevance_boosts(cache_dir: Path) -> dict[str, float]:
    """Load relevance boost values for scenes.

    Args:
        cache_dir: Video cache directory.

    Returns:
        Dict mapping scene_id (as string) to boost multiplier.
    """
    boosts_file = cache_dir / "scenes" / "relevance_boosts.json"
    if not boosts_file.exists():
        return {}

    try:
        return json.loads(boosts_file.read_text())
    except json.JSONDecodeError:
        return {}


def save_relevance_boosts(cache_dir: Path, boosts: dict[str, float]) -> None:
    """Save relevance boost values.

    Args:
        cache_dir: Video cache directory.
        boosts: Dict mapping scene_id (as string) to boost multiplier.
    """
    scenes_dir = cache_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    boosts_file = scenes_dir / "relevance_boosts.json"
    boosts_file.write_text(json.dumps(boosts, indent=2))


def boost_scene_relevance(
    cache_dir: Path,
    scene_id: int,
    boost: float = 0.1,
) -> float:
    """Boost relevance score for a scene.

    Call this when a scene is examined (frames extracted, Q&A about it).
    Boosts accumulate over time, giving frequently-examined scenes higher
    priority in future searches.

    Args:
        cache_dir: Video cache directory.
        scene_id: Scene index (0-based).
        boost: Amount to add to the multiplier (default: 0.1).

    Returns:
        New boost value for the scene.
    """
    boosts = get_relevance_boosts(cache_dir)
    key = str(scene_id)

    current = boosts.get(key, 1.0)
    new_value = current + boost
    boosts[key] = new_value

    save_relevance_boosts(cache_dir, boosts)
    return new_value


def get_boosted_relevance(
    cache_dir: Path,
    scene_id: int,
    base_relevance: float,
) -> float:
    """Apply relevance boost to a search score.

    Args:
        cache_dir: Video cache directory.
        scene_id: Scene index (0-based).
        base_relevance: Original relevance score (0.0 to 1.0).

    Returns:
        Boosted relevance score (may exceed 1.0 for frequently-examined scenes).
    """
    boosts = get_relevance_boosts(cache_dir)
    boost = boosts.get(str(scene_id), 1.0)
    return base_relevance * boost


def record_frame_examination(
    video_id: str,
    cache_dir: Path,
    start_time: float,
    duration: float,
    quality: str = "standard",
) -> dict | None:
    """Record that frames were examined at a specific timestamp.

    This function:
    1. Finds the scene containing the timestamp
    2. Records an observation in VideoMemory
    3. Boosts the scene's relevance for future searches

    Args:
        video_id: Video identifier.
        cache_dir: Video cache directory.
        start_time: Start time in seconds.
        duration: Duration examined in seconds.
        quality: Quality level ("standard", "hq", etc.).

    Returns:
        Dict with scene_id and new_boost, or None if no scene found.
    """
    scene_id = find_scene_at_timestamp(cache_dir, start_time)
    if scene_id is None:
        return None

    # Record observation
    memory = VideoMemory(video_id, cache_dir)
    memory.record_observation(
        scene_id=scene_id,
        obs_type="frames_examined",
        content=f"Examined {quality} frames at {start_time:.1f}s for {duration:.1f}s",
    )

    # Boost relevance
    new_boost = boost_scene_relevance(cache_dir, scene_id, boost=0.1)

    return {
        "scene_id": scene_id,
        "new_boost": new_boost,
    }


def record_qa_interaction(
    video_id: str,
    cache_dir: Path,
    question: str,
    answer: str,
    relevant_scene_ids: list[int] | None = None,
) -> dict:
    """Record a question-answer interaction about the video.

    This function:
    1. Identifies relevant scenes (if not provided)
    2. Records the Q&A in VideoMemory
    3. Boosts relevance for all relevant scenes

    Args:
        video_id: Video identifier.
        cache_dir: Video cache directory.
        question: The question asked.
        answer: The answer given.
        relevant_scene_ids: Optional list of relevant scene IDs.
            If not provided, will be auto-detected from question/answer.

    Returns:
        Dict with question, scenes, and cached status.
    """
    # Find relevant scenes if not provided
    if relevant_scene_ids is None:
        relevant_scene_ids = find_relevant_scenes(cache_dir, question, answer)

    # Record Q&A
    memory = VideoMemory(video_id, cache_dir)
    memory.record_qa(question, answer, relevant_scene_ids)

    # Boost relevance for all relevant scenes
    for scene_id in relevant_scene_ids:
        boost_scene_relevance(cache_dir, scene_id, boost=0.05)

    return {
        "question": question,
        "scenes": relevant_scene_ids,
        "cached": True,
        "qa_count": memory.qa_count,
    }


def search_cached_qa(
    video_id: str,
    cache_dir: Path,
    query: str,
) -> list[dict]:
    """Search for previously answered questions similar to the query.

    This enables "second query faster than first" by returning cached
    answers when the same or similar question is asked again.

    Args:
        video_id: Video identifier.
        cache_dir: Video cache directory.
        query: The question to search for.

    Returns:
        List of matching Q&A dicts from history.
    """
    memory = VideoMemory(video_id, cache_dir)
    return memory.search_qa_history(query)


def get_scene_context(
    video_id: str,
    cache_dir: Path,
    scene_id: int,
) -> dict:
    """Get all learned context for a scene.

    Returns observations, related Q&A, and relevance boost.
    Use this when revisiting a scene to leverage prior analysis.

    Args:
        video_id: Video identifier.
        cache_dir: Video cache directory.
        scene_id: Scene index (0-based).

    Returns:
        Dict with observations, related_qa, and boost.
    """
    memory = VideoMemory(video_id, cache_dir)
    context = memory.get_context_for_scene(scene_id)

    # Add relevance boost info
    boosts = get_relevance_boosts(cache_dir)
    context["boost"] = boosts.get(str(scene_id), 1.0)

    return context


def get_enrichment_stats(cache_dir: Path) -> dict:
    """Get statistics about cache enrichment.

    Args:
        cache_dir: Video cache directory.

    Returns:
        Dict with observation_count, qa_count, boosted_scenes, etc.
    """
    # Memory stats
    memory_dir = cache_dir / "memory"
    observation_count = 0
    qa_count = 0

    if memory_dir.exists():
        obs_file = memory_dir / "observations.json"
        if obs_file.exists():
            try:
                obs_data = json.loads(obs_file.read_text())
                observation_count = sum(len(v) for v in obs_data.values())
            except json.JSONDecodeError:
                pass

        qa_file = memory_dir / "qa_history.json"
        if qa_file.exists():
            try:
                qa_data = json.loads(qa_file.read_text())
                qa_count = len(qa_data)
            except json.JSONDecodeError:
                pass

    # Boost stats
    boosts = get_relevance_boosts(cache_dir)
    boosted_scenes = len([v for v in boosts.values() if v > 1.0])

    return {
        "observation_count": observation_count,
        "qa_count": qa_count,
        "boosted_scenes": boosted_scenes,
        "total_scenes_examined": len(boosts),
        "has_enrichment": observation_count > 0 or qa_count > 0,
    }
