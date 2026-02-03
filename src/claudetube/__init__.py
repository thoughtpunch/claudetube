"""
claudetube - Let Claude watch online videos.

Process videos from 1,500+ sites for Claude Code:
1. Download video (lowest quality for speed)
2. Transcribe audio with faster-whisper (or use existing subtitles)
3. Extract frames on-demand when visual context is needed
"""

# Core processing functions
from claudetube.config.providers import (
    VIDEO_PROVIDERS,
    get_provider_count,
    list_supported_providers,
)

# Config
from claudetube.config.quality import (
    QUALITY_LADDER,
    QUALITY_TIERS,
    next_quality,
)

# Exceptions
from claudetube.exceptions import (
    AgeRestrictedError,
    CacheError,
    ClaudetubeError,
    DownloadError,
    ExtractorError,
    FormatNotAvailableError,
    FrameExtractionError,
    GeoRestrictedError,
    MetadataError,
    NetworkError,
    RateLimitError,
    SubtitleError,
    ToolNotFoundError,
    TranscriptionError,
    VideoUnavailableError,
    YouTubeAuthError,
)
from claudetube.models.local_file import LocalFile, LocalFileError
from claudetube.models.state import VideoState

# Models
from claudetube.models.video_file import VideoFile
from claudetube.models.video_result import VideoResult
from claudetube.models.video_url import VideoURL
from claudetube.operations.extract_frames import (
    extract_frames as get_frames_at,
)
from claudetube.operations.extract_frames import (
    extract_hq_frames as get_hq_frames_at,
)
from claudetube.operations.processor import process_video
from claudetube.operations.transcribe import transcribe_video

# Parsing utilities
from claudetube.parsing.utils import (
    extract_playlist_id,
    extract_url_context,
    extract_video_id,
    get_provider_for_url,
    parse_input,
)

__version__ = "1.0.0rc1"

__all__ = [
    # Core functions
    "process_video",
    "transcribe_video",
    "get_frames_at",
    "get_hq_frames_at",
    # Models
    "VideoFile",
    "VideoResult",
    "VideoURL",
    "LocalFile",
    "LocalFileError",
    "VideoState",
    # Config
    "QUALITY_TIERS",
    "QUALITY_LADDER",
    "next_quality",
    "VIDEO_PROVIDERS",
    "list_supported_providers",
    "get_provider_count",
    # URL utilities
    "extract_video_id",
    "extract_playlist_id",
    "extract_url_context",
    "get_provider_for_url",
    "parse_input",
    # Exceptions
    "ClaudetubeError",
    "DownloadError",
    "YouTubeAuthError",
    "GeoRestrictedError",
    "FormatNotAvailableError",
    "RateLimitError",
    "ExtractorError",
    "NetworkError",
    "VideoUnavailableError",
    "AgeRestrictedError",
    "TranscriptionError",
    "FrameExtractionError",
    "CacheError",
    "ToolNotFoundError",
    "MetadataError",
    "SubtitleError",
]
