"""
Active video watching operation.

Orchestrates the ActiveVideoWatcher to answer questions about videos
by strategically examining scenes, building hypotheses, and verifying
comprehension before returning answers.

Architecture: Cheap First, Expensive Last
1. CACHE  - Return cached Q&A if similar question was asked before
2. TEXT   - Quick examination uses transcript + cached visual descriptions
3. VISUAL - Deep examination extracts and analyzes frames (expensive)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from claudetube.analysis.comprehension import verify_comprehension
from claudetube.analysis.search import format_timestamp
from claudetube.analysis.watcher import ActiveVideoWatcher
from claudetube.cache.enrichment import record_qa_interaction, search_cached_qa
from claudetube.cache.scenes import has_scenes, load_scenes_data
from claudetube.cache.storage import load_state
from claudetube.config.loader import get_cache_dir
from claudetube.operations.extract_frames import extract_frames
from claudetube.operations.narrative_structure import (
    classify_video_type,
    get_narrative_json_path,
)
from claudetube.parsing.utils import extract_video_id

logger = logging.getLogger(__name__)


def examine_scene_quick(scene: dict, question: str) -> list[dict]:
    """Quick examination: transcript + existing visual description.

    This is the cheap path - no new frame extraction or API calls.

    Args:
        scene: Scene dict with transcript_text, visual, etc.
        question: User's question.

    Returns:
        List of finding dicts with type, description, timestamp, confidence.
    """
    findings = []
    question_lower = question.lower()
    question_words = set(question_lower.split())

    # Check transcript relevance
    transcript = scene.get("transcript_text", "")
    if transcript:
        transcript_lower = transcript.lower()
        transcript_words = set(transcript_lower.split())
        overlap = question_words & transcript_words
        if len(overlap) >= 2 or any(
            phrase in transcript_lower
            for phrase in _extract_key_phrases(question_lower)
        ):
            # Include a preview of the relevant transcript
            preview = transcript[:200].strip()
            if len(transcript) > 200:
                preview += "..."
            findings.append(
                {
                    "type": "transcript_match",
                    "description": f"Transcript mentions relevant content: {preview}",
                    "claim": preview,
                    "timestamp": scene.get("start_time", 0),
                    "scene_id": scene.get("scene_id", 0),
                    "initial_confidence": 0.5,
                }
            )

    # Check cached visual description
    visual = scene.get("visual", {})
    if isinstance(visual, dict):
        visual_desc = visual.get("description", "")
    elif isinstance(visual, str):
        visual_desc = visual
    else:
        visual_desc = ""

    if visual_desc:
        visual_lower = visual_desc.lower()
        visual_words = set(visual_lower.split())
        overlap = question_words & visual_words
        if len(overlap) >= 2:
            findings.append(
                {
                    "type": "visual_match",
                    "description": f"Visual content relevant: {visual_desc[:200]}",
                    "claim": visual_desc[:200],
                    "timestamp": scene.get("start_time", 0),
                    "scene_id": scene.get("scene_id", 0),
                    "initial_confidence": 0.6,
                }
            )

    # Check title relevance
    title = scene.get("title", "")
    if title:
        title_lower = title.lower()
        title_words = set(title_lower.split())
        overlap = question_words & title_words
        if len(overlap) >= 1:
            findings.append(
                {
                    "type": "title_match",
                    "description": f"Scene title relevant: {title}",
                    "claim": title,
                    "timestamp": scene.get("start_time", 0),
                    "scene_id": scene.get("scene_id", 0),
                    "initial_confidence": 0.4,
                }
            )

    return findings


def examine_scene_deep(
    scene: dict,
    question: str,
    video_id: str,
    output_base: Path | None = None,
) -> list[dict]:
    """Deep examination: extract frames and analyze visually.

    This is the expensive path - extracts keyframes and returns paths
    for vision analysis. The actual vision analysis is left to the caller
    or provider system.

    Args:
        scene: Scene dict with start_time, end_time, etc.
        question: User's question.
        video_id: Video ID for frame extraction.
        output_base: Cache directory.

    Returns:
        List of finding dicts with frame paths and analysis.
    """
    findings = []
    base = output_base or get_cache_dir()

    start_time = scene.get("start_time", 0)
    end_time = scene.get("end_time", start_time + 10)
    duration = min(end_time - start_time, 10)  # Cap at 10s per scene

    # Extract keyframes at key moments within the scene
    # Use 3 frames: start, middle, end
    interval = max(1.0, duration / 3)

    try:
        frames = extract_frames(
            video_id,
            start_time=start_time,
            duration=duration,
            interval=interval,
            output_base=base,
            quality="medium",
        )
    except Exception as e:
        logger.warning(
            f"Frame extraction failed for scene {scene.get('scene_id')}: {e}"
        )
        frames = []

    if frames:
        findings.append(
            {
                "type": "deep_analysis",
                "description": (
                    f"Extracted {len(frames)} frames from scene "
                    f"{scene.get('scene_id')} ({format_timestamp(start_time)}-"
                    f"{format_timestamp(end_time)})"
                ),
                "claim": scene.get("transcript_text", "")[:200]
                or "Visual content examined",
                "timestamp": start_time,
                "scene_id": scene.get("scene_id", 0),
                "frame_paths": [str(f) for f in frames],
                "initial_confidence": 0.7,
            }
        )

    # Also include transcript findings from deep examination
    transcript = scene.get("transcript_text", "")
    if transcript:
        findings.append(
            {
                "type": "deep_transcript",
                "description": f"Full transcript examined: {transcript[:300]}",
                "claim": transcript[:300],
                "timestamp": start_time,
                "scene_id": scene.get("scene_id", 0),
                "initial_confidence": 0.6,
            }
        )

    return findings


def watch_video(
    video_id_or_url: str,
    question: str,
    max_iterations: int = 15,
    output_base: Path | None = None,
) -> dict:
    """Actively watch a video to answer a question.

    Uses ActiveVideoWatcher to strategically explore scenes,
    building hypotheses and gathering evidence until confident
    enough to answer.

    Args:
        video_id_or_url: Video ID or URL.
        question: User's question about the video.
        max_iterations: Maximum examination iterations.
        output_base: Cache directory.

    Returns:
        Dict with answer, confidence, evidence, examination log.
    """
    video_id = extract_video_id(video_id_or_url)
    base = output_base or get_cache_dir()
    cache_dir = base / video_id

    if not cache_dir.exists():
        return {
            "error": "Video not cached. Run process_video first.",
            "video_id": video_id,
        }

    # Check for cached Q&A first (cheapest path)
    try:
        cached_qa = search_cached_qa(video_id, cache_dir, question)
        if cached_qa:
            best = cached_qa[0]
            return {
                "video_id": video_id,
                "question": question,
                "answer": best["answer"],
                "confidence": 0.9,
                "evidence": [
                    {"observation": "Previously answered question", "timestamp": None}
                ],
                "source": "cached_qa",
                "scenes_examined": 0,
                "examination_log": [],
                "comprehension_verified": True,
            }
    except Exception:
        pass  # Proceed to active watching

    # Load scenes
    if not has_scenes(cache_dir):
        return {
            "error": "Video has no scene data. Run get_scenes first.",
            "video_id": video_id,
        }

    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data or not scenes_data.scenes:
        return {
            "error": "No scenes found for video.",
            "video_id": video_id,
        }

    # Convert SceneBoundary objects to dicts for the watcher
    scenes = []
    for s in scenes_data.scenes:
        scene_dict = s.to_dict()
        # Enrich with visual descriptions if available
        visual_file = cache_dir / "scenes" / f"scene_{s.scene_id:03d}" / "visual.json"
        if visual_file.exists():
            try:
                visual_data = json.loads(visual_file.read_text())
                if visual_data:
                    scene_dict["visual"] = visual_data
            except (json.JSONDecodeError, OSError):
                pass
        scenes.append(scene_dict)

    # Load video state for duration metadata
    video_duration = 0.0
    state = load_state(cache_dir / "state.json")
    if state and state.duration:
        video_duration = state.duration

    # Auto-detect video_type for attention model weighting
    video_type = _detect_video_type(cache_dir, scenes_data.scenes)

    # Try to compute goal embedding for semantic attention scoring
    goal_embedding = _compute_goal_embedding(question)

    # Create active watcher with attention priority model
    # Scene embeddings are loaded from cache_dir automatically
    watcher = ActiveVideoWatcher(
        video_id=video_id,
        user_goal=question,
        scenes=scenes,
        cache_dir=cache_dir,
        video_duration=video_duration,
        video_type=video_type,
        goal_embedding=goal_embedding,
    )

    # Active exploration loop
    examination_log = []

    for i in range(max_iterations):
        action = watcher.decide_next_action()

        if action.action == "answer":
            break

        # Find the scene to examine
        target_scene = None
        for s in scenes:
            if s.get("scene_id") == action.scene_id:
                target_scene = s
                break

        if target_scene is None:
            logger.warning(f"Scene {action.scene_id} not found, skipping")
            watcher.examined.add(action.scene_id)
            continue

        # Execute the examination
        if action.action == "examine_deep":
            findings = examine_scene_deep(
                target_scene,
                question,
                video_id,
                output_base=base,
            )
        else:
            findings = examine_scene_quick(target_scene, question)

        # Update watcher's understanding
        watcher.update_understanding(action.scene_id, findings)

        examination_log.append(
            {
                "iteration": i,
                "scene_id": action.scene_id,
                "depth": action.action,
                "timestamp": format_timestamp(target_scene.get("start_time", 0)),
                "findings_count": len(findings),
            }
        )

    # Formulate final answer
    answer = watcher.formulate_answer()

    # Verify comprehension
    video_understanding = {
        "scenes": scenes,
        "answer": answer,
    }
    verification = verify_comprehension(video_understanding)

    # Record Q&A for future caching
    try:
        record_qa_interaction(
            video_id,
            cache_dir,
            question,
            answer.get("main_answer", ""),
        )
    except Exception as e:
        logger.debug(f"Failed to record Q&A: {e}")

    return {
        "video_id": video_id,
        "question": question,
        "answer": answer.get("main_answer", "Unable to determine from video content"),
        "confidence": answer.get("confidence", 0.0),
        "evidence": answer.get("evidence", []),
        "alternative_interpretations": answer.get("alternative_interpretations", []),
        "examination_log": examination_log,
        "scenes_examined": answer.get("scenes_examined", 0),
        "comprehension_verified": verification.get("ready_to_answer", False),
    }


def _compute_goal_embedding(question: str):
    """Compute an embedding vector for the user's question.

    Uses the configured Embedder provider. Returns None if embedding
    fails (no API key, provider unavailable, etc.) so attention
    scoring falls back to keyword matching.

    Args:
        question: User's question string.

    Returns:
        numpy.ndarray embedding vector, or None on failure.
    """
    try:
        import numpy as np

        from claudetube.analysis.embeddings import _get_embedder, get_embedding_model

        model = get_embedding_model()
        embedder = _get_embedder(model)
        embedding_list = embedder.embed_sync(question)
        return np.array(embedding_list, dtype=np.float32)
    except Exception as e:
        logger.debug(f"Could not compute goal embedding: {e}")
        return None


def _detect_video_type(cache_dir: Path, scenes: list) -> str:
    """Auto-detect video type from cached narrative structure or scene classification.

    Follows "Cheap First, Expensive Last":
    1. Try loading cached narrative structure (structure/narrative.json)
    2. If not cached, run classify_video_type() on available scenes
    3. Fall back to 'unknown' if detection fails

    Args:
        cache_dir: Video cache directory.
        scenes: List of SceneBoundary objects.

    Returns:
        Video type string (e.g. 'coding_tutorial', 'lecture', 'unknown').
    """
    # 1. Try cached narrative structure (cheapest path)
    try:
        narrative_path = get_narrative_json_path(cache_dir)
        if narrative_path.exists():
            data = json.loads(narrative_path.read_text())
            video_type = data.get("video_type")
            if video_type:
                logger.debug(f"Auto-detected video_type from cache: {video_type}")
                return video_type
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Failed to load cached narrative structure: {e}")

    # 2. Run classify_video_type() on scenes (fast local heuristic)
    try:
        if scenes:
            video_type = classify_video_type(scenes, [], cache_dir)
            logger.debug(f"Auto-detected video_type from scenes: {video_type}")
            return video_type
    except Exception as e:
        logger.debug(f"Failed to classify video type from scenes: {e}")

    # 3. Fall back to 'unknown'
    return "unknown"


def _extract_key_phrases(text: str) -> list[str]:
    """Extract multi-word key phrases from text for matching.

    Args:
        text: Input text (already lowered).

    Returns:
        List of 2-3 word phrases.
    """
    words = text.split()
    phrases = []
    for i in range(len(words) - 1):
        phrases.append(f"{words[i]} {words[i + 1]}")
    for i in range(len(words) - 2):
        phrases.append(f"{words[i]} {words[i + 1]} {words[i + 2]}")
    return phrases
