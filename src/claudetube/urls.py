"""
URL parsing and video ID extraction for claudetube.

Supports 70+ video providers with site-specific regex patterns,
plus a generic fallback for unknown sites.
Also supports local file paths for offline video processing.

This module re-exports from the refactored submodules for backward compatibility.
"""

# Re-export models
# Re-export config
from claudetube.config.providers import (
    VIDEO_PROVIDERS,
    get_provider_count,
    list_supported_providers,
)
from claudetube.models.local_file import (
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    LocalFile,
    LocalFileError,
    is_local_file,
    is_url,
)
from claudetube.models.video_url import VideoURL

# Re-export parsing utilities
from claudetube.parsing.utils import (
    extract_playlist_id,
    extract_url_context,
    extract_video_id,
    get_provider_for_url,
    parse_input,
)

__all__ = [
    # Models
    "VideoURL",
    "LocalFile",
    "LocalFileError",
    # Constants
    "VIDEO_PROVIDERS",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "SUPPORTED_AUDIO_EXTENSIONS",
    # Functions
    "extract_video_id",
    "extract_playlist_id",
    "extract_url_context",
    "get_provider_for_url",
    "list_supported_providers",
    "get_provider_count",
    "is_local_file",
    "is_url",
    "parse_input",
]
