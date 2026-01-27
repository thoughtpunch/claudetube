"""
claudetube - Let Claude watch YouTube videos.

Process YouTube videos for Claude Code:
1. Download video (lowest quality for speed)
2. Transcribe audio with faster-whisper
3. Extract frames on-demand when visual context is needed
"""

from claudetube.core import (
    VideoResult,
    extract_video_id,
    get_frames_at,
    get_hq_frames_at,
    process_video,
)

__version__ = "0.1.0"
__all__ = [
    "process_video",
    "get_frames_at",
    "get_hq_frames_at",
    "extract_video_id",
    "VideoResult",
]
