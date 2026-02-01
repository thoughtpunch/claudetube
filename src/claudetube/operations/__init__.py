"""
High-level video processing operations.
"""

from claudetube.operations.chapters import (
    extract_youtube_chapters,
    parse_timestamp,
)
from claudetube.operations.download import (
    download_audio,
    download_thumbnail,
    extract_audio_local,
    fetch_metadata,
    fetch_subtitles,
)
from claudetube.operations.extract_frames import (
    extract_frames,
    extract_frames_local,
    extract_hq_frames,
    extract_hq_frames_local,
)
from claudetube.operations.processor import process_local_video, process_video
from claudetube.operations.subtitles import (
    fetch_local_subtitles,
    find_embedded_subtitles,
    find_sidecar_subtitles,
)
from claudetube.operations.segmentation import (
    boundaries_to_segments,
    segment_video_smart,
)
from claudetube.operations.transcribe import transcribe_audio, transcribe_video

__all__ = [
    "process_video",
    "process_local_video",
    "fetch_metadata",
    "download_audio",
    "download_thumbnail",
    "extract_audio_local",
    "fetch_subtitles",
    "transcribe_audio",
    "transcribe_video",
    "extract_frames",
    "extract_frames_local",
    "extract_hq_frames",
    "extract_hq_frames_local",
    "extract_youtube_chapters",
    "parse_timestamp",
    "fetch_local_subtitles",
    "find_embedded_subtitles",
    "find_sidecar_subtitles",
    "boundaries_to_segments",
    "segment_video_smart",
]
