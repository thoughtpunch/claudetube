"""
Download operations for videos and metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path

from claudetube.tools.ffmpeg import FFmpegTool
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


# FFmpeg tool singleton for local file operations
_ffmpeg: FFmpegTool | None = None


def _get_ffmpeg() -> FFmpegTool:
    """Get or create FFmpeg tool instance."""
    global _ffmpeg
    if _ffmpeg is None:
        _ffmpeg = FFmpegTool()
    return _ffmpeg


def extract_audio_local(
    input_path: Path,
    output_dir: Path,
) -> Path:
    """Extract audio from a local video/audio file to MP3.

    Uses FFmpeg directly (not yt-dlp) for local files.
    Optimized for whisper transcription:
    - 16kHz sample rate (whisper native)
    - Mono channel (speech doesn't need stereo)
    - 128kbps quality (plenty for speech)

    Args:
        input_path: Path to local video/audio file
        output_dir: Cache directory to write audio.mp3

    Returns:
        Path to extracted audio.mp3

    Raises:
        RuntimeError: If extraction fails
    """
    output_path = output_dir / "audio.mp3"

    # Cache hit - skip if already extracted
    if output_path.exists():
        logger.debug(f"Audio cache hit: {output_path}")
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)

    tool = _get_ffmpeg()
    result = tool.extract_audio(input_path, output_path)

    if result is None:
        raise RuntimeError(f"Failed to extract audio from {input_path}")

    logger.info(f"Extracted audio: {output_path}")
    return output_path
