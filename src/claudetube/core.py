"""
Core video processing for claudetube.

Downloads videos, transcribes with faster-whisper, and extracts
frames on-demand for visual analysis.

This module re-exports the main API from the refactored submodules
for backward compatibility.
"""

# Re-export main processing functions
# Re-export config
from claudetube.config.quality import (
    QUALITY_LADDER,
    QUALITY_TIERS,
    next_quality,
)

# Re-export models
from claudetube.models.video_file import VideoFile
from claudetube.models.video_result import VideoResult

# Re-export VideoURL for backward compatibility
from claudetube.models.video_url import VideoURL
from claudetube.operations.extract_frames import (
    extract_frames as get_frames_at,
)
from claudetube.operations.extract_frames import (
    extract_hq_frames as get_hq_frames_at,
)
from claudetube.operations.processor import process_video
from claudetube.operations.transcribe import transcribe_video

# Re-export URL parsing for backward compatibility
from claudetube.parsing.utils import (
    extract_playlist_id,
    extract_url_context,
    extract_video_id,
    get_provider_for_url,
)

__all__ = [
    # Main functions
    "process_video",
    "transcribe_video",
    "get_frames_at",
    "get_hq_frames_at",
    # Models
    "VideoResult",
    "VideoFile",
    "VideoURL",
    # Config
    "QUALITY_TIERS",
    "QUALITY_LADDER",
    "next_quality",
    # URL utilities
    "extract_video_id",
    "extract_playlist_id",
    "extract_url_context",
    "get_provider_for_url",
]
