"""
claudetube - Let Claude watch YouTube videos.

Process YouTube videos for Claude Code:
1. Download video (lowest quality for speed)
2. Transcribe audio with faster-whisper
3. Extract frames on-demand when visual context is needed
"""

from claudetube.core import (
    QUALITY_LADDER,
    QUALITY_TIERS,
    VideoResult,
    extract_video_id,
    get_frames_at,
    get_hq_frames_at,
    next_quality,
    process_video,
)

__version__ = "0.2.0"
__all__ = [
    "process_video",
    "get_frames_at",
    "get_hq_frames_at",
    "extract_video_id",
    "VideoResult",
    "QUALITY_TIERS",
    "QUALITY_LADDER",
    "next_quality",
]
