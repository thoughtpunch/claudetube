"""
High-level video processing operations.
"""

from claudetube.operations.download import (
    download_audio,
    download_thumbnail,
    fetch_metadata,
    fetch_subtitles,
)
from claudetube.operations.extract_frames import (
    extract_frames,
    extract_hq_frames,
)
from claudetube.operations.processor import process_video
from claudetube.operations.transcribe import transcribe_audio, transcribe_video

__all__ = [
    "process_video",
    "fetch_metadata",
    "download_audio",
    "download_thumbnail",
    "fetch_subtitles",
    "transcribe_audio",
    "transcribe_video",
    "extract_frames",
    "extract_hq_frames",
]
