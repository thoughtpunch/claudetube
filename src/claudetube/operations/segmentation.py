"""
Smart video segmentation.

Implements the "Cheap First, Expensive Last" architecture principle.
Tries cheap boundary detection first, only falls back to visual
detection when coverage is insufficient.

Segmentation workflow:
1. CACHE     - Return cached scenes/scenes.json if exists
2. CHAPTERS  - Extract YouTube chapters from metadata (free)
3. TRANSCRIPT - Linguistic, pauses, vocabulary shifts (fast)
4. VISUAL    - PySceneDetect fallback ONLY if needed (expensive)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from claudetube.analysis.alignment import align_transcript_to_scenes
from claudetube.analysis.unified import (
    detect_boundaries_cheap,
    merge_nearby_boundaries,
)
from claudetube.analysis.visual import (
    detect_visual_boundaries,
    should_use_visual_detection,
)
from claudetube.cache.scenes import (
    SceneBoundary,
    ScenesData,
    get_scenes_json_path,
    load_scenes_data,
    save_scenes_data,
)

if TYPE_CHECKING:
    from claudetube.analysis.linguistic import Boundary

logger = logging.getLogger(__name__)

# Minimum chapters to skip visual detection entirely
MIN_CHAPTERS_TO_SKIP_VISUAL = 5


def boundaries_to_segments(
    boundaries: list[Boundary],
    duration: float,
) -> list[SceneBoundary]:
    """Convert a list of boundaries to segments with start/end times.

    Boundaries mark the start of new sections. This function creates
    segments that span from one boundary to the next.

    Args:
        boundaries: List of Boundary tuples sorted by timestamp.
        duration: Total video duration in seconds.

    Returns:
        List of SceneBoundary objects with scene_id, start_time, end_time.

    Example:
        >>> from claudetube.analysis.linguistic import Boundary
        >>> boundaries = [
        ...     Boundary(60.0, "chapter", "Intro", 0.95),
        ...     Boundary(180.0, "chapter", "Setup", 0.95),
        ... ]
        >>> segments = boundaries_to_segments(boundaries, 300.0)
        >>> len(segments)  # 3 segments: 0-60, 60-180, 180-300
        3
        >>> segments[0].start_time, segments[0].end_time
        (0.0, 60.0)
    """
    if not boundaries:
        # No boundaries = single segment spanning entire video
        return [
            SceneBoundary(
                scene_id=0,
                start_time=0.0,
                end_time=duration,
                title=None,
            )
        ]

    # Sort by timestamp
    sorted_b = sorted(boundaries, key=lambda x: x.timestamp)
    segments: list[SceneBoundary] = []

    # First segment: 0 to first boundary
    first_boundary = sorted_b[0]
    if first_boundary.timestamp > 0.5:  # Only add if boundary isn't at very start
        segments.append(
            SceneBoundary(
                scene_id=0,
                start_time=0.0,
                end_time=first_boundary.timestamp,
                title=None,
            )
        )

    # Middle segments: each boundary marks the start of a new segment
    for i, boundary in enumerate(sorted_b):
        # End time is either next boundary or video end
        end_time = sorted_b[i + 1].timestamp if i + 1 < len(sorted_b) else duration

        # Extract title from trigger_text if it's a chapter
        title = None
        if boundary.type == "chapter" or "chapter" in boundary.type:
            # Chapter trigger_text format: "[source] Title"
            trigger = boundary.trigger_text
            title = trigger.split("] ", 1)[1] if "] " in trigger else trigger

        segments.append(
            SceneBoundary(
                scene_id=len(segments),
                start_time=boundary.timestamp,
                end_time=end_time,
                title=title,
            )
        )

    return segments


def segment_video_smart(
    video_id: str,
    video_path: str | Path | None,
    transcript_segments: list[dict] | None,
    video_info: dict | None,
    cache_dir: Path,
    srt_path: str | Path | None = None,
    force: bool = False,
) -> ScenesData:
    """Smart video segmentation - cheap methods first, visual fallback only when needed.

    Implements the architecture principle: Cheap First, Expensive Last.

    Args:
        video_id: Unique video identifier.
        video_path: Path to video file (needed for visual detection fallback).
        transcript_segments: List of segment dicts with 'start' and 'text' keys.
        video_info: yt-dlp metadata dict (for chapter extraction).
        cache_dir: Video cache directory for storing scenes.json.
        srt_path: Path to SRT file for pause detection.
        force: Re-run segmentation even if cached.

    Returns:
        ScenesData with segments and method used.

    Workflow:
        1. Check cache - return immediately if scenes.json exists
        2. Run cheap detection (chapters, linguistic, pauses, vocabulary)
        3. Evaluate coverage - count boundaries, check for chapters
        4. Skip visual if good chapters exist (>=5)
        5. Run visual detection only if coverage is poor
        6. Merge all boundaries and convert to segments
        7. Save to scenes/scenes.json

    Performance:
        - <2s for videos with good YouTube chapters
        - ~1-5s for videos needing transcript analysis
        - ~30-60s if visual fallback is needed (rare)
    """
    # Step 1: Check cache
    if not force:
        cached = load_scenes_data(cache_dir)
        if cached is not None:
            logger.info(f"Using cached scenes for {video_id}")
            return cached

    # Get video duration from metadata
    duration = 0.0
    if video_info:
        duration = video_info.get("duration", 0) or 0

    if duration == 0 and video_path:
        # Try to get duration from ffprobe
        try:
            from claudetube.tools.ffprobe import FFprobeTool

            ffprobe = FFprobeTool()
            metadata = ffprobe.get_metadata(Path(video_path))
            duration = metadata.duration or 0.0
        except Exception:
            pass

    if duration == 0:
        logger.warning(
            f"Could not determine duration for {video_id}, using 3600s default"
        )
        duration = 3600.0

    # Step 2: Always try cheap methods first
    logger.info(f"Running cheap boundary detection for {video_id}")
    cheap_boundaries = detect_boundaries_cheap(
        video_info=video_info,
        transcript_segments=transcript_segments,
        srt_path=srt_path,
    )
    logger.info(f"Cheap detection found {len(cheap_boundaries)} boundaries")

    # Step 3: Evaluate coverage
    has_chapters = any(
        b.type == "chapter" or "chapter" in b.type for b in cheap_boundaries
    )
    chapter_count = sum(
        1 for b in cheap_boundaries if b.type == "chapter" or "chapter" in b.type
    )

    # Step 4: Decide if visual detection is needed
    need_visual = should_use_visual_detection(
        cheap_boundaries,
        duration,
        has_transcript=bool(transcript_segments),
    )

    # Step 5: Skip visual if good chapters exist
    if has_chapters and chapter_count >= MIN_CHAPTERS_TO_SKIP_VISUAL:
        logger.info(
            f"Skipping visual detection - {chapter_count} chapters found (>={MIN_CHAPTERS_TO_SKIP_VISUAL})"
        )
        need_visual = False

    # Step 6: Run visual detection if needed
    all_boundaries = cheap_boundaries
    method = "transcript"

    if need_visual and video_path:
        video_path = Path(video_path)
        if video_path.exists():
            logger.info("Coverage insufficient - running visual detection fallback")
            try:
                visual_boundaries = detect_visual_boundaries(video_path)
                logger.info(
                    f"Visual detection found {len(visual_boundaries)} boundaries"
                )
                # Merge cheap + visual boundaries
                all_boundaries = merge_nearby_boundaries(
                    list(cheap_boundaries) + list(visual_boundaries)
                )
                method = "hybrid"
            except ImportError:
                logger.warning(
                    "PySceneDetect not available, using cheap detection only"
                )
            except Exception as e:
                logger.warning(f"Visual detection failed: {e}")
        else:
            logger.warning(f"Video file not found for visual detection: {video_path}")
    elif need_visual:
        logger.info("Visual detection needed but video_path not provided")

    # Step 7: Convert boundaries to segments
    segments = boundaries_to_segments(all_boundaries, duration)
    logger.info(f"Created {len(segments)} segments using {method} method")

    # Step 8: Align transcript to scenes
    if transcript_segments:
        logger.info("Aligning transcript segments to scenes")
        segments = align_transcript_to_scenes(transcript_segments, segments)
        aligned_count = sum(1 for s in segments if s.transcript)
        logger.info(f"Aligned transcript to {aligned_count}/{len(segments)} scenes")

    # Step 9: Save to cache
    scenes_data = ScenesData(
        video_id=video_id,
        method=method,
        scenes=segments,
    )
    save_scenes_data(cache_dir, scenes_data)
    logger.info(f"Saved scenes to {get_scenes_json_path(cache_dir)}")

    return scenes_data
