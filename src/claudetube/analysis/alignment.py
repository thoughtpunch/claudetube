"""
Transcript-to-scene alignment utilities.

Aligns transcript segments to their containing scenes using midpoint matching.
This enables answering questions like "what did they say when showing X".
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claudetube.cache.scenes import SceneBoundary


def align_transcript_to_scenes(
    transcript_segments: list[dict],
    scenes: list[SceneBoundary],
) -> list[SceneBoundary]:
    """Map transcript segments to their containing scenes.

    Uses midpoint matching: each transcript segment is assigned to the scene
    containing its midpoint. This handles edge cases where a segment spans
    scene boundaries by assigning to the scene with the majority of content.

    Args:
        transcript_segments: List of segment dicts with 'start', 'end', 'text' keys.
            If 'end' is missing, uses 'start' as midpoint.
        scenes: List of SceneBoundary objects with start_time and end_time.

    Returns:
        The same scenes list with transcript and transcript_text populated.

    Example:
        >>> from claudetube.cache.scenes import SceneBoundary
        >>> segments = [
        ...     {"start": 5.0, "end": 10.0, "text": "Hello"},
        ...     {"start": 35.0, "end": 40.0, "text": "World"},
        ... ]
        >>> scenes = [
        ...     SceneBoundary(scene_id=0, start_time=0, end_time=30),
        ...     SceneBoundary(scene_id=1, start_time=30, end_time=60),
        ... ]
        >>> result = align_transcript_to_scenes(segments, scenes)
        >>> result[0].transcript_text
        'Hello'
        >>> result[1].transcript_text
        'World'
    """
    if not scenes:
        return scenes

    # Initialize transcript storage for each scene
    for scene in scenes:
        scene.transcript = []
        scene.transcript_text = ""

    if not transcript_segments:
        return scenes

    # Use binary search for O(n log m) alignment
    scene_starts = [s.start_time for s in scenes]

    for seg in transcript_segments:
        start = seg.get("start", 0)
        end = seg.get("end", start)
        text = seg.get("text", "").strip()

        if not text:
            continue

        # Calculate midpoint for scene assignment
        seg_mid = (start + end) / 2

        # Binary search: find rightmost scene with start_time <= seg_mid
        idx = bisect.bisect_right(scene_starts, seg_mid) - 1

        # Ensure valid index
        if 0 <= idx < len(scenes):
            # Verify midpoint is within scene bounds
            scene = scenes[idx]
            if scene.start_time <= seg_mid < scene.end_time:
                scene.transcript.append(seg)

    # Join transcript text for each scene
    for scene in scenes:
        scene.transcript_text = " ".join(
            seg.get("text", "").strip() for seg in scene.transcript
        )

    return scenes


def align_transcript_to_scenes_simple(
    transcript_segments: list[dict],
    scenes: list[dict],
) -> list[dict]:
    """Map transcript segments to scenes (dict-based version).

    This is a simpler version that works with plain dicts instead of
    SceneBoundary objects. Useful for testing or when working with
    raw JSON data.

    Args:
        transcript_segments: List of segment dicts with 'start', 'end', 'text' keys.
        scenes: List of scene dicts with 'start_time' and 'end_time' keys.

    Returns:
        The same scenes list with 'transcript' and 'transcript_text' added.
    """
    if not scenes:
        return scenes

    # Initialize transcript storage for each scene
    for scene in scenes:
        scene["transcript"] = []
        scene["transcript_text"] = ""

    if not transcript_segments:
        return scenes

    # Use binary search for O(n log m) alignment
    scene_starts = [s["start_time"] for s in scenes]

    for seg in transcript_segments:
        start = seg.get("start", 0)
        end = seg.get("end", start)
        text = seg.get("text", "").strip()

        if not text:
            continue

        # Calculate midpoint for scene assignment
        seg_mid = (start + end) / 2

        # Binary search: find rightmost scene with start_time <= seg_mid
        idx = bisect.bisect_right(scene_starts, seg_mid) - 1

        # Ensure valid index and midpoint is within scene bounds
        if 0 <= idx < len(scenes):
            scene = scenes[idx]
            if scene["start_time"] <= seg_mid < scene["end_time"]:
                scene["transcript"].append(seg)

    # Join transcript text for each scene
    for scene in scenes:
        scene["transcript_text"] = " ".join(
            seg.get("text", "").strip() for seg in scene["transcript"]
        )

    return scenes
