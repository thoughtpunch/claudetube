"""
claudetube - Let Claude watch online videos.

Process videos from 70+ sites for Claude Code:
1. Download video (lowest quality for speed)
2. Transcribe audio with faster-whisper (or use existing subtitles)
3. Extract frames on-demand when visual context is needed
"""

from claudetube.core import (
    QUALITY_LADDER,
    QUALITY_TIERS,
    VideoResult,
    get_frames_at,
    get_hq_frames_at,
    next_quality,
    process_video,
)
from claudetube.urls import (
    VideoURL,
    extract_playlist_id,
    extract_url_context,
    extract_video_id,
    get_provider_for_url,
    list_supported_providers,
)

__version__ = "0.3.0"
__all__ = [
    # Core functions
    "process_video",
    "get_frames_at",
    "get_hq_frames_at",
    "VideoResult",
    "QUALITY_TIERS",
    "QUALITY_LADDER",
    "next_quality",
    # URL utilities
    "VideoURL",
    "extract_video_id",
    "extract_playlist_id",
    "extract_url_context",
    "get_provider_for_url",
    "list_supported_providers",
]
