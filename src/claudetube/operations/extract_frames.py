"""
Frame extraction operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from claudetube.cache.manager import CacheManager
from claudetube.config.loader import get_cache_dir
from claudetube.config.quality import QUALITY_LADDER, QUALITY_TIERS
from claudetube.operations.download import download_video_segment
from claudetube.tools.ffmpeg import FFmpegTool
from claudetube.utils.logging import log_timed

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Width mapping for quality tiers (used for local file extraction)
QUALITY_WIDTHS: dict[str, int] = {
    "lowest": 480,
    "low": 640,
    "medium": 854,
    "high": 1280,
    "highest": 1920,
}


def extract_frames(
    video_id_or_url: str,
    start_time: float,
    duration: float = 5.0,
    interval: float = 1.0,
    output_base: Path | None = None,
    width: int | None = None,
    quality: str = "lowest",
) -> list[Path]:
    """Extract frames for a specific time range (drill-in feature).

    Use this when Claude needs visual context for a specific part of the video.

    Args:
        video_id_or_url: Video ID or URL
        start_time: Start time in seconds
        duration: Duration to capture (default: 5s)
        interval: Seconds between frames (default: 1s)
        output_base: Cache directory
        width: Frame width (overrides tier default if set)
        quality: Quality tier (lowest/low/medium/high/highest)

    Returns:
        List of frame paths
    """
    import json
    import time

    from claudetube.parsing.utils import extract_video_id

    t0 = time.time()

    # Validate quality
    if quality not in QUALITY_TIERS:
        raise ValueError(
            f"Invalid quality '{quality}'. Must be one of: {', '.join(QUALITY_LADDER)}"
        )

    tier = QUALITY_TIERS[quality]
    effective_width = width or tier["width"]

    cache = CacheManager(output_base or get_cache_dir())
    video_id = extract_video_id(video_id_or_url)
    cache_dir = cache.get_cache_dir(video_id)
    drill_dir = cache_dir / f"drill_{quality}"
    drill_dir.mkdir(parents=True, exist_ok=True)

    # Download only the needed section (+ 2s buffer for keyframes)
    section_start = max(0, start_time - 2)
    section_end = start_time + duration + 2

    seg_name = f"segment_{quality}_{int(start_time)}_{int(start_time + duration)}.mp4"
    video_path = cache_dir / seg_name
    state_file = cache_dir / "state.json"

    if not video_path.exists() and state_file.exists():
        state = json.loads(state_file.read_text())
        url = state.get("url")
        if url:
            log_timed(
                f"Downloading {quality} segment ({section_start}s-{section_end}s)...",
                t0,
            )
            download_video_segment(
                url=url,
                output_path=video_path,
                start_time=section_start,
                end_time=section_end,
                quality_sort=tier["sort"],
                concurrent_fragments=tier["concurrent_fragments"],
            )

    if not video_path.exists():
        log_timed("No video available for drill-in", t0)
        return []

    # Extract frames
    ffmpeg = FFmpegTool()
    log_timed(
        f"Extracting {quality} frames from {start_time}s to {start_time + duration}s...",
        t0,
    )

    frames = ffmpeg.extract_frames_range(
        video_path=video_path,
        output_dir=drill_dir,
        start_time=start_time,
        duration=duration,
        interval=interval,
        width=effective_width,
        jpeg_quality=tier["jpeg_q"],
        seek_offset=section_start,
        prefix="drill",
    )

    # Clean up segment file
    if video_path.exists():
        video_path.unlink()
        log_timed(f"Cleaned up {quality} segment file", t0)

    # Track extraction in state.json
    if state_file.exists():
        state = json.loads(state_file.read_text())
        extractions = state.get("quality_extractions", {})
        extractions[quality] = {
            "start_time": start_time,
            "duration": duration,
            "frames": len(frames),
            "width": effective_width,
        }
        state["quality_extractions"] = extractions
        state_file.write_text(json.dumps(state, indent=2))

    log_timed(f"Drill-in complete ({quality}): {len(frames)} frames", t0)
    return frames


def extract_hq_frames(
    video_id_or_url: str,
    start_time: float,
    duration: float = 5.0,
    interval: float = 1.0,
    output_base: Path | None = None,
    width: int = 1280,
) -> list[Path]:
    """Extract HIGH QUALITY frames for a specific time range.

    Use this when the low-quality drill-in frames aren't clear enough
    (e.g., reading text, code, small UI elements).

    Args:
        video_id_or_url: Video ID or URL
        start_time: Start time in seconds
        duration: Duration to capture (default: 5s)
        interval: Seconds between frames (default: 1s)
        output_base: Cache directory (default: ~/.claude/video_cache)
        width: Frame width (default: 1280 for HD)

    Returns:
        List of frame paths
    """
    import json
    import time

    from claudetube.parsing.utils import extract_video_id

    t0 = time.time()

    cache = CacheManager(output_base or get_cache_dir())
    video_id = extract_video_id(video_id_or_url)
    cache_dir = cache.get_cache_dir(video_id)
    hq_dir = cache_dir / "hq"
    hq_dir.mkdir(parents=True, exist_ok=True)

    state_file = cache_dir / "state.json"

    if not state_file.exists():
        log_timed("No state.json found - run process_video first", t0)
        return []

    state = json.loads(state_file.read_text())
    url = state.get("url")
    if not url:
        log_timed("No URL in state.json", t0)
        return []

    # Download HQ segment
    section_start = max(0, start_time - 2)
    section_end = start_time + duration + 2

    seg_name = f"segment_hq_{int(start_time)}_{int(start_time + duration)}.mp4"
    hq_video_path = cache_dir / seg_name

    if not hq_video_path.exists():
        log_timed(f"Downloading HQ segment ({section_start}s-{section_end}s)...", t0)
        result = download_video_segment(
            url=url,
            output_path=hq_video_path,
            start_time=section_start,
            end_time=section_end,
            quality_sort="res:1080",
            concurrent_fragments=4,
        )
        if not result:
            log_timed("HQ segment download failed", t0)
            return []
        size_mb = hq_video_path.stat().st_size / 1024 / 1024
        log_timed(f"Downloaded HQ segment: {size_mb:.1f}MB", t0)

    # Extract HQ frames
    ffmpeg = FFmpegTool()
    log_timed(
        f"Extracting HQ frames from {start_time}s to {start_time + duration}s...", t0
    )

    frames = ffmpeg.extract_frames_range(
        video_path=hq_video_path,
        output_dir=hq_dir,
        start_time=start_time,
        duration=duration,
        interval=interval,
        width=width,
        jpeg_quality=2,  # High quality JPEG
        seek_offset=section_start,
        prefix="hq",
    )

    # Clean up segment file
    if hq_video_path.exists():
        hq_video_path.unlink()

    log_timed(f"HQ drill-in complete: {len(frames)} frames", t0)
    return frames


def extract_frames_local(
    video_id: str,
    start_time: float,
    duration: float = 5.0,
    interval: float = 1.0,
    quality: str = "lowest",
    output_base: Path | None = None,
) -> list[Path]:
    """Extract frames from a local video file (no download step).

    This is significantly faster than URL-based extraction since it
    operates directly on the cached local file without downloading.

    Args:
        video_id: Video ID (must be a cached local file)
        start_time: Start time in seconds
        duration: Duration to capture (default: 5s)
        interval: Seconds between frames (default: 1s)
        quality: Quality tier (lowest/low/medium/high/highest)
        output_base: Cache directory (default: ~/.claude/video_cache)

    Returns:
        List of extracted frame paths

    Raises:
        ValueError: If quality tier is invalid or video is not a local file
        FileNotFoundError: If video is not cached or source file is missing
    """
    import time

    t0 = time.time()

    # Validate quality tier
    if quality not in QUALITY_WIDTHS:
        raise ValueError(
            f"Invalid quality '{quality}'. Must be one of: {', '.join(QUALITY_LADDER)}"
        )

    width = QUALITY_WIDTHS[quality]
    jpeg_quality = QUALITY_TIERS[quality]["jpeg_q"]

    cache = CacheManager(output_base or get_cache_dir())

    # Get state and validate it's a local file
    state = cache.get_state(video_id)
    if not state:
        raise FileNotFoundError(f"Video not cached: {video_id}")

    if state.source_type != "local":
        raise ValueError(
            f"Video '{video_id}' is not a local file (source_type={state.source_type}). "
            "Use extract_frames() for URL-based videos."
        )

    # Get the source file path
    source_path = cache.get_source_path(video_id)
    if not source_path:
        raise FileNotFoundError(f"No cached source file for video: {video_id}")

    # Check if source is still valid (symlink not broken)
    is_valid, warning = cache.check_source_valid(video_id)
    if not is_valid:
        raise FileNotFoundError(f"Source file unavailable: {warning}")

    # Set up output directory
    cache_dir = cache.get_cache_dir(video_id)
    drill_dir = cache_dir / f"drill_{quality}"
    drill_dir.mkdir(parents=True, exist_ok=True)

    log_timed(
        f"Extracting {quality} frames from local file at {start_time}s...",
        t0,
    )

    # Extract frames directly from source (no download, no segment extraction)
    # Using -ss before -i for fast keyframe-based seeking
    ffmpeg = FFmpegTool()
    frames = ffmpeg.extract_frames_range(
        video_path=source_path,
        output_dir=drill_dir,
        start_time=start_time,
        duration=duration,
        interval=interval,
        width=width,
        jpeg_quality=jpeg_quality,
        seek_offset=0.0,  # No offset needed - seeking directly in source
        prefix="frame",
    )

    # Track extraction in state.json
    import json

    state_file = cache.get_state_file(video_id)
    if state_file.exists():
        state_data = json.loads(state_file.read_text())
        extractions = state_data.get("quality_extractions", {})
        extractions[quality] = {
            "start_time": start_time,
            "duration": duration,
            "frames": len(frames),
            "width": width,
            "local": True,
        }
        state_data["quality_extractions"] = extractions
        state_file.write_text(json.dumps(state_data, indent=2))

    log_timed(f"Local drill-in complete ({quality}): {len(frames)} frames", t0)
    return frames


def extract_hq_frames_local(
    video_id: str,
    start_time: float,
    duration: float = 5.0,
    interval: float = 1.0,
    width: int = 1280,
    output_base: Path | None = None,
) -> list[Path]:
    """Extract HIGH QUALITY frames from a local video file.

    Use this when you need to read text, code, or small UI elements
    from a local video file.

    Args:
        video_id: Video ID (must be a cached local file)
        start_time: Start time in seconds
        duration: Duration to capture (default: 5s)
        interval: Seconds between frames (default: 1s)
        width: Frame width in pixels (default: 1280)
        output_base: Cache directory (default: ~/.claude/video_cache)

    Returns:
        List of extracted frame paths

    Raises:
        ValueError: If video is not a local file
        FileNotFoundError: If video is not cached or source file is missing
    """
    import time

    t0 = time.time()

    cache = CacheManager(output_base or get_cache_dir())

    # Get state and validate it's a local file
    state = cache.get_state(video_id)
    if not state:
        raise FileNotFoundError(f"Video not cached: {video_id}")

    if state.source_type != "local":
        raise ValueError(
            f"Video '{video_id}' is not a local file (source_type={state.source_type}). "
            "Use extract_hq_frames() for URL-based videos."
        )

    # Get the source file path
    source_path = cache.get_source_path(video_id)
    if not source_path:
        raise FileNotFoundError(f"No cached source file for video: {video_id}")

    # Check if source is still valid (symlink not broken)
    is_valid, warning = cache.check_source_valid(video_id)
    if not is_valid:
        raise FileNotFoundError(f"Source file unavailable: {warning}")

    # Set up output directory
    cache_dir = cache.get_cache_dir(video_id)
    hq_dir = cache_dir / "hq"
    hq_dir.mkdir(parents=True, exist_ok=True)

    log_timed(
        f"Extracting HQ frames from local file at {start_time}s (width={width})...",
        t0,
    )

    # Extract HQ frames directly from source
    ffmpeg = FFmpegTool()
    frames = ffmpeg.extract_frames_range(
        video_path=source_path,
        output_dir=hq_dir,
        start_time=start_time,
        duration=duration,
        interval=interval,
        width=width,
        jpeg_quality=2,  # High quality JPEG
        seek_offset=0.0,  # No offset needed - seeking directly in source
        prefix="hq",
    )

    log_timed(f"Local HQ drill-in complete: {len(frames)} frames", t0)
    return frames
