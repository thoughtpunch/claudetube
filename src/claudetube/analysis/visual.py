"""
Visual scene detection using PySceneDetect.

This is the EXPENSIVE FALLBACK - only use when cheap methods fail.
See architecture principle: Cheap First, Expensive Last.

Use cases:
- Videos without YouTube chapters
- Videos without usable timestamps in description
- Videos where transcript analysis finds <3 boundaries for 5+ min content
"""

from __future__ import annotations

import logging
from pathlib import Path

from claudetube.analysis.linguistic import Boundary

logger = logging.getLogger(__name__)

# Minimum boundaries threshold for cheap detection
# If cheap detection finds fewer than this, consider visual fallback
MIN_CHEAP_BOUNDARIES = 3

# Minimum video duration (seconds) to trigger fallback consideration
MIN_DURATION_FOR_FALLBACK = 300  # 5 minutes

# Maximum average segment duration before considering fallback
MAX_AVG_SEGMENT_DURATION = 300  # 5 minutes


def should_use_visual_detection(
    cheap_boundaries: list[Boundary],
    video_duration: float,
    has_transcript: bool = True,
) -> bool:
    """Determine if visual detection fallback should be used.

    Visual detection is expensive and should only run when cheap
    methods provide insufficient coverage.

    Args:
        cheap_boundaries: Boundaries found by cheap detection methods.
        video_duration: Total video duration in seconds.
        has_transcript: Whether a transcript is available.

    Returns:
        True if visual detection should be attempted.

    Examples:
        >>> # Short video - no fallback needed
        >>> should_use_visual_detection([], 60.0)
        False

        >>> # Long video with no boundaries - needs fallback
        >>> should_use_visual_detection([], 600.0)
        True

        >>> # Long video with good coverage - no fallback
        >>> from claudetube.analysis.linguistic import Boundary
        >>> boundaries = [Boundary(0, "ch", "Intro", 0.9) for _ in range(5)]
        >>> should_use_visual_detection(boundaries, 600.0)
        False
    """
    # Don't run visual detection on short videos
    if video_duration < MIN_DURATION_FOR_FALLBACK:
        return False

    # No transcript and no boundaries - definitely need visual
    if not has_transcript and len(cheap_boundaries) < MIN_CHEAP_BOUNDARIES:
        return True

    # Too few boundaries for the video length
    if len(cheap_boundaries) < MIN_CHEAP_BOUNDARIES:
        return True

    # Check average segment duration
    # (video_duration / (boundaries + 1)) gives average segment length
    avg_segment = video_duration / (len(cheap_boundaries) + 1)
    return avg_segment > MAX_AVG_SEGMENT_DURATION


def detect_visual_boundaries(
    video_path: str | Path,
    downscale_factor: int = 2,
    min_scene_len: int = 30,
    adaptive_threshold: float = 3.0,
) -> list[Boundary]:
    """Detect scene boundaries using PySceneDetect's AdaptiveDetector.

    This is an expensive operation that decodes video frames.
    Only call when cheap detection methods are insufficient.

    Args:
        video_path: Path to video file.
        downscale_factor: Factor to reduce resolution (2 = half). Higher = faster.
        min_scene_len: Minimum frames between scene cuts. 30 frames ≈ 1 second.
        adaptive_threshold: Sensitivity threshold. Higher = fewer detections.

    Returns:
        List of Boundary tuples with visual scene information.

    Performance:
        ~30-60 seconds for a 30-minute video at downscale_factor=2.

    Raises:
        ImportError: If scenedetect is not installed.
        FileNotFoundError: If video file doesn't exist.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        from scenedetect import AdaptiveDetector, SceneManager, open_video
    except ImportError as e:
        raise ImportError(
            "PySceneDetect not installed. Run: pip install scenedetect[opencv]"
        ) from e

    logger.info(f"Running visual scene detection on {video_path.name}")

    # Open video with downscaling for performance
    video = open_video(str(video_path))
    if downscale_factor > 1:
        video.set_downscale_factor(downscale_factor)

    # AdaptiveDetector is better for semantic boundaries than ContentDetector
    # ContentDetector catches more visual cuts but less meaningful ones
    scene_manager = SceneManager()
    scene_manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_scene_len=min_scene_len,
        )
    )

    # Process the video
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    logger.info(f"Visual detection found {len(scene_list)} scenes")

    # Convert to Boundary format
    boundaries: list[Boundary] = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        # Skip the first scene start (always at 0)
        if i == 0 and start_time < 1.0:
            continue

        boundaries.append(
            Boundary(
                timestamp=start_time,
                type="visual_scene",
                trigger_text=f"Scene {i + 1} start",
                confidence=0.75,
            )
        )

    return boundaries


def detect_visual_boundaries_fast(
    video_path: str | Path,
    content_threshold: float = 27.0,
    min_scene_len: int = 15,
) -> list[Boundary]:
    """Faster visual detection using ContentDetector.

    ContentDetector is simpler and faster than AdaptiveDetector but
    may catch more visual changes that aren't semantic boundaries.

    Args:
        video_path: Path to video file.
        content_threshold: Threshold for scene changes (0-255). Lower = more sensitive.
        min_scene_len: Minimum frames between scenes.

    Returns:
        List of Boundary tuples.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError as e:
        raise ImportError(
            "PySceneDetect not installed. Run: pip install scenedetect[opencv]"
        ) from e

    logger.info(f"Running fast visual detection on {video_path.name}")

    video = open_video(str(video_path))
    video.set_downscale_factor(2)

    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=content_threshold,
            min_scene_len=min_scene_len,
        )
    )

    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    logger.info(f"Fast visual detection found {len(scene_list)} scenes")

    boundaries: list[Boundary] = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        if i == 0 and start_time < 1.0:
            continue

        boundaries.append(
            Boundary(
                timestamp=start_time,
                type="visual_scene_fast",
                trigger_text=f"Scene {i + 1} start",
                confidence=0.65,  # Lower confidence for fast detection
            )
        )

    return boundaries
