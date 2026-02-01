"""
Data models for claudetube.

Provides dataclasses and Pydantic models for video processing results,
URL parsing, local files, and cache state.
"""

from claudetube.models.local_file import LocalFile, LocalFileError
from claudetube.models.state import VideoState
from claudetube.models.video_file import VideoFile
from claudetube.models.video_result import VideoResult
from claudetube.models.video_url import VideoURL

__all__ = [
    "VideoFile",
    "VideoResult",
    "VideoURL",
    "LocalFile",
    "LocalFileError",
    "VideoState",
]
