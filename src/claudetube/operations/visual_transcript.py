"""
Visual transcript generation for scenes.

Generates natural language descriptions of what's happening on screen for each scene.
Follows the "Cheap First, Expensive Last" principle:
1. CACHE - Return instantly if visual.json already exists
2. SKIP - Skip scenes where transcript provides sufficient context
3. COMPUTE - Only generate for scenes that need visual context
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from claudetube.cache.manager import CacheManager

if TYPE_CHECKING:
    from pathlib import Path
from claudetube.cache.scenes import (
    SceneBoundary,
    get_scene_dir,
    get_visual_json_path,
    list_scene_keyframes,
    load_scenes_data,
)
from claudetube.config.loader import get_cache_dir
from claudetube.tools.ffmpeg import FFmpegTool
from claudetube.utils.logging import log_timed

logger = logging.getLogger(__name__)

# Default vision model setting
DEFAULT_VISION_MODEL = "claude"


@dataclass
class VisualDescription:
    """Visual description for a scene."""

    scene_id: int
    description: str
    people: list[str]
    objects: list[str]
    text_on_screen: list[str]
    actions: list[str]
    setting: str | None = None
    keyframe_count: int = 0
    model_used: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scene_id": self.scene_id,
            "description": self.description,
            "people": self.people,
            "objects": self.objects,
            "text_on_screen": self.text_on_screen,
            "actions": self.actions,
            "setting": self.setting,
            "keyframe_count": self.keyframe_count,
            "model_used": self.model_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VisualDescription:
        """Create from dictionary."""
        return cls(
            scene_id=data.get("scene_id", 0),
            description=data.get("description", ""),
            people=data.get("people", []),
            objects=data.get("objects", []),
            text_on_screen=data.get("text_on_screen", []),
            actions=data.get("actions", []),
            setting=data.get("setting"),
            keyframe_count=data.get("keyframe_count", 0),
            model_used=data.get("model_used"),
        )


def get_vision_model() -> str:
    """Get configured vision model.

    Priority:
    1. CLAUDETUBE_VISION_MODEL environment variable
    2. Default: "claude"

    Returns:
        Vision model identifier: "claude", "molmo", or "llava"
    """
    return os.environ.get("CLAUDETUBE_VISION_MODEL", DEFAULT_VISION_MODEL)


def _load_image_base64(path: Path) -> str:
    """Load image file as base64 string."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _select_keyframes_for_scene(
    scene: SceneBoundary,
    video_id: str,
    cache_dir: Path,
    n: int = 3,
) -> list[Path]:
    """Select representative keyframes from a scene.

    Extracts frames at evenly-distributed timestamps within the scene.
    Uses existing keyframes if available, otherwise extracts new ones.

    Args:
        scene: Scene boundary with start/end times
        video_id: Video ID
        cache_dir: Video cache directory
        n: Maximum number of keyframes to select (default: 3)

    Returns:
        List of keyframe paths
    """
    # Check for existing keyframes
    existing = list_scene_keyframes(cache_dir, scene.scene_id)
    if existing:
        # Use up to n existing keyframes, evenly distributed
        if len(existing) <= n:
            return existing
        step = len(existing) // n
        return [existing[i * step] for i in range(n)]

    # Extract new keyframes
    duration = scene.duration()
    if duration < 3:
        # Short scene: just middle frame
        timestamps = [scene.start_time + duration / 2]
    else:
        # Distribute evenly across scene
        timestamps = [scene.start_time + i * duration / (n - 1) for i in range(n)]

    # Extract frames to scene keyframes directory
    keyframes_dir = get_scene_dir(cache_dir, scene.scene_id) / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = FFmpegTool()

    # Get video source path from state.json
    state_file = cache_dir / "state.json"
    if not state_file.exists():
        logger.warning(f"No state.json found for {video_id}")
        return []

    state = json.loads(state_file.read_text())
    video_path = None

    # Try cached file first (for local videos)
    cached_file = state.get("cached_file")
    if cached_file:
        video_path = cache_dir / cached_file
        if not video_path.exists():
            video_path = None

    # For URL videos, we need to download a segment
    if not video_path and state.get("url"):
        from claudetube.operations.download import download_video_segment

        # Download segment covering the scene
        seg_path = cache_dir / f"scene_{scene.scene_id:03d}_segment.mp4"
        if not seg_path.exists():
            download_video_segment(
                url=state["url"],
                output_path=seg_path,
                start_time=max(0, scene.start_time - 1),
                end_time=scene.end_time + 1,
                quality_sort="res:720",  # Medium quality for keyframes
                concurrent_fragments=2,
            )
        if seg_path.exists():
            video_path = seg_path

    if not video_path or not video_path.exists():
        logger.warning(f"No video source available for scene {scene.scene_id}")
        return []

    # Extract frames
    frames = []
    for ts in timestamps:
        # Adjust timestamp for segment videos
        if state.get("url") and not state.get("cached_file"):
            seek_ts = ts - max(0, scene.start_time - 1)
        else:
            seek_ts = ts

        output_path = keyframes_dir / f"kf_{int(ts // 60):02d}_{int(ts % 60):02d}.jpg"
        frame = ffmpeg.extract_frame(
            video_path=video_path,
            output_path=output_path,
            timestamp=seek_ts,
            width=854,  # Medium quality for visual analysis
            jpeg_quality=4,
        )
        if frame:
            frames.append(frame)

    # Clean up segment file if we created one
    if state.get("url") and not state.get("cached_file"):
        seg_path = cache_dir / f"scene_{scene.scene_id:03d}_segment.mp4"
        if seg_path.exists():
            seg_path.unlink()

    return frames


def _should_skip_scene(scene: SceneBoundary) -> bool:
    """Determine if a scene can be skipped for visual transcript.

    Scenes with good transcript coverage may not need visual descriptions.
    This helps avoid unnecessary API calls for talking-head videos.

    Args:
        scene: Scene boundary with transcript data

    Returns:
        True if scene can be skipped, False if visual description needed
    """
    # Skip if scene has substantial transcript (likely talking head)
    transcript_text = scene.transcript_text or scene.transcript_segment or ""

    # If scene has good transcript density, visual may be unnecessary
    duration = scene.duration()
    if duration > 0:
        words = len(transcript_text.split())
        words_per_second = words / duration
        # Typical speech is 2-3 words/second; if we have good coverage, skip
        if words_per_second > 2.0 and duration < 60:
            return True

    return False


def _generate_visual_claude(
    keyframes: list[Path],
    scene: SceneBoundary,
) -> VisualDescription | None:
    """Generate visual description using Claude API.

    Args:
        keyframes: List of keyframe image paths
        scene: Scene boundary for context

    Returns:
        VisualDescription or None if failed
    """
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed. Run: pip install anthropic")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    # Build message content with images
    content = []
    for kf in keyframes:
        image_data = _load_image_base64(kf)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data,
            },
        })

    # Add context from transcript if available
    context = ""
    if scene.transcript_text:
        context = f"\n\nTranscript context: {scene.transcript_text[:500]}"

    content.append({
        "type": "text",
        "text": f"""Describe what is visually happening in these frames from a video.
Focus on: visual actions, people present, objects visible, any text on screen, and the setting/environment.
Be specific and factual. Avoid speculation about things not visible.{context}

Return a JSON object with these keys:
- description: A 1-2 sentence natural language description of what's happening
- people: Array of people descriptions (e.g., ["man in blue shirt", "woman at desk"])
- objects: Array of notable objects (e.g., ["laptop", "whiteboard", "code editor"])
- text_on_screen: Array of any visible text (e.g., ["def main():", "Chapter 3"])
- actions: Array of actions occurring (e.g., ["typing on keyboard", "pointing at screen"])
- setting: Brief description of the environment (e.g., "office", "conference room", "outdoors")

Output only valid JSON, no markdown code blocks.""",
    })

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Fast + cheap for visual analysis
            max_tokens=500,
            messages=[{"role": "user", "content": content}],
        )

        # Parse JSON response
        response_text = response.content[0].text.strip()
        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        data = json.loads(response_text)

        return VisualDescription(
            scene_id=scene.scene_id,
            description=data.get("description", ""),
            people=data.get("people", []),
            objects=data.get("objects", []),
            text_on_screen=data.get("text_on_screen", []),
            actions=data.get("actions", []),
            setting=data.get("setting"),
            keyframe_count=len(keyframes),
            model_used="claude-3-haiku-20240307",
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {e}")
        return None
    except anthropic.APIError as e:
        logger.error(f"Claude API error: {e}")
        return None


def generate_visual_transcript(
    video_id: str,
    scene_id: int | None = None,
    force: bool = False,
    output_base: Path | None = None,
) -> dict:
    """Generate visual transcript for a video's scenes.

    Follows "Cheap First, Expensive Last" principle:
    1. Return cached visual.json instantly if available
    2. Skip scenes with good transcript coverage
    3. Only call vision API when visual context adds value

    Args:
        video_id: Video ID
        scene_id: Optional specific scene ID (None = all scenes)
        force: Re-generate even if cached
        output_base: Cache directory

    Returns:
        Dict with results and any errors
    """
    t0 = time.time()
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    if not cache_dir.exists():
        return {"error": "Video not cached. Run process_video first.", "video_id": video_id}

    # Load scenes data
    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return {"error": "No scenes found. Run get_scenes first.", "video_id": video_id}

    # Determine which scenes to process
    if scene_id is not None:
        scenes = [s for s in scenes_data.scenes if s.scene_id == scene_id]
        if not scenes:
            return {"error": f"Scene {scene_id} not found", "video_id": video_id}
    else:
        scenes = scenes_data.scenes

    results = []
    skipped = []
    errors = []

    for scene in scenes:
        # 1. CACHE - Return instantly if already exists
        visual_path = get_visual_json_path(cache_dir, scene.scene_id)
        if not force and visual_path.exists():
            try:
                data = json.loads(visual_path.read_text())
                results.append(VisualDescription.from_dict(data))
                log_timed(f"Scene {scene.scene_id}: loaded from cache", t0)
                continue
            except (json.JSONDecodeError, KeyError):
                pass  # Re-generate if cached data is invalid

        # 2. SKIP - Check if scene needs visual description
        if not force and _should_skip_scene(scene):
            skipped.append(scene.scene_id)
            log_timed(f"Scene {scene.scene_id}: skipped (good transcript coverage)", t0)
            continue

        # 3. COMPUTE - Generate visual description
        log_timed(f"Scene {scene.scene_id}: extracting keyframes...", t0)
        keyframes = _select_keyframes_for_scene(scene, video_id, cache_dir)

        if not keyframes:
            errors.append({"scene_id": scene.scene_id, "error": "No keyframes available"})
            continue

        log_timed(f"Scene {scene.scene_id}: generating visual description...", t0)
        model = get_vision_model()

        visual_desc = None
        if model == "claude":
            visual_desc = _generate_visual_claude(keyframes, scene)
        else:
            # Placeholder for future local model support
            errors.append({
                "scene_id": scene.scene_id,
                "error": f"Vision model '{model}' not yet supported. Use 'claude'.",
            })
            continue

        if visual_desc:
            # Save to cache
            visual_path.parent.mkdir(parents=True, exist_ok=True)
            visual_path.write_text(json.dumps(visual_desc.to_dict(), indent=2))
            results.append(visual_desc)
            log_timed(f"Scene {scene.scene_id}: visual transcript generated", t0)
        else:
            errors.append({"scene_id": scene.scene_id, "error": "Failed to generate description"})

    # Update state.json if all scenes processed
    if scene_id is None:  # Only update when processing all scenes
        state_file = cache_dir / "state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            all_complete = len(results) + len(skipped) == len(scenes_data.scenes)
            state["visual_transcripts_complete"] = all_complete
            state_file.write_text(json.dumps(state, indent=2))

    log_timed(
        f"Visual transcript complete: {len(results)} generated, {len(skipped)} skipped, {len(errors)} errors",
        t0,
    )

    return {
        "video_id": video_id,
        "generated": len(results),
        "skipped": len(skipped),
        "skipped_scene_ids": skipped,
        "errors": errors,
        "results": [r.to_dict() for r in results],
    }


def get_visual_transcript(
    video_id: str,
    scene_id: int | None = None,
    output_base: Path | None = None,
) -> dict:
    """Get cached visual transcript for a video's scenes.

    Does NOT generate new transcripts - use generate_visual_transcript for that.

    Args:
        video_id: Video ID
        scene_id: Optional specific scene ID (None = all scenes)
        output_base: Cache directory

    Returns:
        Dict with cached visual descriptions
    """
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    if not cache_dir.exists():
        return {"error": "Video not cached", "video_id": video_id}

    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return {"error": "No scenes found", "video_id": video_id}

    if scene_id is not None:
        scenes = [s for s in scenes_data.scenes if s.scene_id == scene_id]
    else:
        scenes = scenes_data.scenes

    results = []
    missing = []

    for scene in scenes:
        visual_path = get_visual_json_path(cache_dir, scene.scene_id)
        if visual_path.exists():
            try:
                data = json.loads(visual_path.read_text())
                results.append(data)
            except json.JSONDecodeError:
                missing.append(scene.scene_id)
        else:
            missing.append(scene.scene_id)

    return {
        "video_id": video_id,
        "count": len(results),
        "missing": missing,
        "results": results,
    }
