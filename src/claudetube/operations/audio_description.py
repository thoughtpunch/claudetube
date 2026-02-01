"""
Audio description compilation from scene analysis data.

Compiles existing visual.json and transcript data into WebVTT audio description
format. This is a compilation task - it reads cached scene analysis results and
formats them, not AI generation.

Follows the "Cheap First, Expensive Last" principle:
1. CACHE - Return instantly if AD files already exist
2. COMPILE - Format existing visual.json + transcript into WebVTT
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from claudetube.cache.manager import CacheManager
from claudetube.cache.scenes import (
    ScenesData,
    get_visual_json_path,
    load_scenes_data,
)
from claudetube.config.loader import get_cache_dir
from claudetube.operations.visual_transcript import VisualDescription
from claudetube.utils.logging import log_timed

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _format_vtt_timestamp(seconds: float) -> str:
    """Format seconds as WebVTT timestamp (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted VTT timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _format_description(
    visual: VisualDescription | None,
    transcript_text: str,
    title: str | None,
) -> str:
    """Format a scene description from visual and transcript data.

    Combines visual description, text on screen, actions, and setting
    into a concise audio description line.

    Args:
        visual: VisualDescription from visual.json (may be None)
        transcript_text: Scene transcript text
        title: Optional scene title (from chapters)

    Returns:
        Formatted description string
    """
    parts = []

    # Scene title/chapter if available
    if title:
        parts.append(title + ".")

    if visual:
        # Primary visual description
        if visual.description:
            parts.append(visual.description)

        # Setting context (only if not already implied by description)
        if visual.setting and visual.setting.lower() not in (visual.description or "").lower():
            parts.append(f"Setting: {visual.setting}.")

        # Text on screen (code, diagrams, etc.) - important for accessibility
        if visual.text_on_screen:
            text_items = ", ".join(visual.text_on_screen[:5])  # Limit to 5 items
            parts.append(f"On screen: {text_items}.")

        # Actions if they add context beyond the description
        if visual.actions and not visual.description:
            action_text = ", ".join(visual.actions[:3])
            parts.append(action_text.capitalize() + ".")
    else:
        # No visual data - use transcript summary as fallback
        if transcript_text:
            # Take first sentence or first 100 chars as a basic description
            first_sentence = transcript_text.split(".")[0].strip()
            if first_sentence and len(first_sentence) > 10:
                parts.append(f"Speaker: {first_sentence[:150]}.")

    return " ".join(parts).strip()


def _compile_vtt(
    scenes_data: ScenesData,
    cache_dir: Path,
) -> tuple[list[str], list[str]]:
    """Compile scene data into VTT lines and plain text lines.

    Args:
        scenes_data: Loaded scenes data with boundaries
        cache_dir: Video cache directory

    Returns:
        Tuple of (vtt_lines, txt_lines)
    """
    vtt_lines = [
        "WEBVTT",
        "Kind: descriptions",
        "Language: en",
        "",
    ]
    txt_lines = []
    cue_index = 0

    for scene in scenes_data.scenes:
        # Load visual.json if it exists
        visual = None
        visual_path = get_visual_json_path(cache_dir, scene.scene_id)
        if visual_path.exists():
            try:
                data = json.loads(visual_path.read_text())
                visual = VisualDescription.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Invalid visual.json for scene {scene.scene_id}")

        # Format description from available data
        description = _format_description(
            visual=visual,
            transcript_text=scene.transcript_text,
            title=scene.title,
        )

        if not description:
            continue

        cue_index += 1

        # VTT cue
        start_ts = _format_vtt_timestamp(scene.start_time)
        end_ts = _format_vtt_timestamp(scene.end_time)
        vtt_lines.append(str(cue_index))
        vtt_lines.append(f"{start_ts} --> {end_ts}")
        vtt_lines.append(description)
        vtt_lines.append("")

        # Plain text line with timestamp
        minutes = int(scene.start_time // 60)
        seconds = int(scene.start_time % 60)
        txt_lines.append(f"[{minutes:02d}:{seconds:02d}] {description}")

    return vtt_lines, txt_lines


def compile_scene_descriptions(
    video_id: str,
    force: bool = False,
    output_base: Path | None = None,
) -> dict:
    """Compile scene analysis into WebVTT audio descriptions.

    Reads existing visual.json files and scene transcript data, then
    compiles them into .ad.vtt and .ad.txt formats. This is a compilation
    step, not AI generation.

    Args:
        video_id: Video ID
        force: Re-compile even if cached AD files exist
        output_base: Cache base directory override

    Returns:
        Dict with compilation results
    """
    t0 = time.time()
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    if not cache_dir.exists():
        return {"error": "Video not cached. Run process_video first.", "video_id": video_id}

    # 1. CACHE - Return instantly if AD already exists
    if not force and cache.has_ad(video_id):
        vtt_path, txt_path = cache.get_ad_paths(video_id)
        log_timed("Audio descriptions loaded from cache", t0)
        return {
            "video_id": video_id,
            "status": "cached",
            "vtt_path": str(vtt_path),
            "txt_path": str(txt_path),
        }

    # Load scenes data
    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return {"error": "No scenes found. Run scene analysis first.", "video_id": video_id}

    if not scenes_data.scenes:
        return {"error": "No scene boundaries found.", "video_id": video_id}

    # 2. COMPILE - Format existing data into VTT
    log_timed(f"Compiling audio descriptions for {len(scenes_data.scenes)} scenes...", t0)
    vtt_lines, txt_lines = _compile_vtt(scenes_data, cache_dir)

    if not txt_lines:
        return {
            "error": "No descriptions could be generated from available scene data.",
            "video_id": video_id,
        }

    # Write output files
    vtt_path, txt_path = cache.get_ad_paths(video_id)

    vtt_path.write_text("\n".join(vtt_lines))
    txt_path.write_text("\n".join(txt_lines))

    # Update state
    state = cache.get_state(video_id)
    if state:
        state.ad_complete = True
        state.ad_source = "scene_compilation"
        cache.save_state(video_id, state)

    log_timed(f"Audio descriptions compiled: {len(txt_lines)} cues", t0)

    return {
        "video_id": video_id,
        "status": "compiled",
        "cue_count": len(txt_lines),
        "vtt_path": str(vtt_path),
        "txt_path": str(txt_path),
        "source": "scene_compilation",
    }


def get_scene_descriptions(
    video_id: str,
    output_base: Path | None = None,
) -> dict:
    """Get cached audio descriptions for a video.

    Does NOT compile new descriptions - use compile_scene_descriptions for that.

    Args:
        video_id: Video ID
        output_base: Cache base directory override

    Returns:
        Dict with cached AD content or error
    """
    cache = CacheManager(output_base or get_cache_dir())

    if not cache.has_ad(video_id):
        return {"error": "No audio descriptions found. Run compile_scene_descriptions first.", "video_id": video_id}

    vtt_path, txt_path = cache.get_ad_paths(video_id)

    result: dict = {"video_id": video_id}

    if vtt_path.exists():
        result["vtt"] = vtt_path.read_text()
        result["vtt_path"] = str(vtt_path)

    if txt_path.exists():
        result["txt"] = txt_path.read_text()
        result["txt_path"] = str(txt_path)

    return result
