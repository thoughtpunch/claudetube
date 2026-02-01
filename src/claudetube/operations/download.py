"""
Download operations for videos and metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path

from claudetube.tools.yt_dlp import YtDlpTool

logger = logging.getLogger(__name__)

# Singleton tool instance
_yt_dlp: YtDlpTool | None = None


def _get_yt_dlp() -> YtDlpTool:
    """Get or create yt-dlp tool instance."""
    global _yt_dlp
    if _yt_dlp is None:
        _yt_dlp = YtDlpTool()
    return _yt_dlp


def fetch_metadata(url: str, timeout: int = 30) -> dict:
    """Fetch video metadata without downloading.

    Args:
        url: Video URL
        timeout: Request timeout in seconds

    Returns:
        Dict with video metadata

    Raises:
        MetadataError: If fetch fails
    """
    tool = _get_yt_dlp()
    return tool.get_metadata(url, timeout=timeout)


def download_audio(
    url: str,
    output_path: Path,
    quality: str = "64K",
) -> Path:
    """Download audio from video.

    Args:
        url: Video URL
        output_path: Output file path
        quality: Audio quality (e.g., "64K", "128K")

    Returns:
        Path to downloaded audio file

    Raises:
        DownloadError: If download fails
    """
    tool = _get_yt_dlp()
    return tool.download_audio(url, output_path, quality=quality)


def download_thumbnail(
    url: str,
    output_dir: Path,
    timeout: int = 15,
) -> Path | None:
    """Download video thumbnail.

    Args:
        url: Video URL
        output_dir: Output directory
        timeout: Request timeout

    Returns:
        Path to thumbnail file, or None if not available
    """
    tool = _get_yt_dlp()
    return tool.download_thumbnail(url, output_dir, timeout=timeout)


def fetch_subtitles(
    url: str,
    output_dir: Path,
    timeout: int = 30,
) -> dict | None:
    """Fetch subtitles from video source.

    Args:
        url: Video URL
        output_dir: Output directory
        timeout: Request timeout

    Returns:
        Dict with 'srt', 'txt', 'source' keys, or None if not available
    """
    tool = _get_yt_dlp()
    return tool.fetch_subtitles(url, output_dir, timeout=timeout)


def download_video_segment(
    url: str,
    output_path: Path,
    start_time: float,
    end_time: float,
    quality_sort: str = "+res,+size,+br,+fps",
    concurrent_fragments: int = 1,
) -> Path | None:
    """Download a video segment for frame extraction.

    Args:
        url: Video URL
        output_path: Output file path
        start_time: Start time in seconds
        end_time: End time in seconds
        quality_sort: yt-dlp format sort string
        concurrent_fragments: Number of concurrent fragments

    Returns:
        Path to downloaded video, or None if failed
    """
    tool = _get_yt_dlp()
    return tool.download_video_segment(
        url=url,
        output_path=output_path,
        start_time=start_time,
        end_time=end_time,
        quality_sort=quality_sort,
        concurrent_fragments=concurrent_fragments,
    )
