"""
Download operations for videos and metadata.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from claudetube.tools.ffmpeg import FFmpegTool
from claudetube.tools.yt_dlp import (
    DownloadProgress,
    ProgressCallback,
    YtDlpTool,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class PlaylistDownloadResult:
    """Result of a playlist batch download operation.

    Attributes:
        playlist_url: Original playlist URL.
        cache_base: Cache base directory used.
        archive_file: Path to the download archive file.
        downloaded_count: Number of videos downloaded in this run.
        skipped_count: Number of videos skipped (already in archive).
        failed_count: Number of videos that failed to download.
        video_ids: List of video IDs successfully downloaded.
        errors: List of error messages for failed downloads.
    """

    playlist_url: str
    cache_base: Path
    archive_file: Path
    downloaded_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    video_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


logger = logging.getLogger(__name__)

# Re-export progress types for callers
__all__ = [
    "DownloadProgress",
    "ProgressCallback",
    "PlaylistDownloadResult",
    "fetch_metadata",
    "download_audio",
    "download_thumbnail",
    "fetch_subtitles",
    "download_video_segment",
    "extract_audio_local",
    "download_playlist",
]

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
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Download audio from video.

    Args:
        url: Video URL
        output_path: Output file path
        quality: Audio quality (e.g., "64K", "128K")
        on_progress: Optional callback for progress updates.
            Receives DownloadProgress objects with status, percent, speed, etc.

    Returns:
        Path to downloaded audio file

    Raises:
        DownloadError: If download fails
    """
    tool = _get_yt_dlp()
    return tool.download_audio(
        url, output_path, quality=quality, on_progress=on_progress
    )


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
    on_progress: ProgressCallback | None = None,
) -> Path | None:
    """Download a video segment for frame extraction.

    Args:
        url: Video URL
        output_path: Output file path
        start_time: Start time in seconds
        end_time: End time in seconds
        quality_sort: yt-dlp format sort string
        concurrent_fragments: Number of concurrent fragments
        on_progress: Optional callback for progress updates.
            Receives DownloadProgress objects with status, percent, speed, etc.

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
        on_progress=on_progress,
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


def download_playlist(
    playlist_url: str,
    cache_base: Path | None = None,
    quality: str = "64K",
    concurrent_fragments: int = 4,
    on_progress: ProgressCallback | None = None,
) -> PlaylistDownloadResult:
    """Download all videos in a playlist using yt-dlp's native batch handling.

    Uses a single yt-dlp command with output templates for efficient batch
    downloads. Leverages --download-archive for resume support and skip
    tracking.

    Benefits over per-video Python loops:
    - Single command downloads all videos
    - yt-dlp handles rate limiting, retries, concurrent downloads
    - Progress tracking across playlist
    - Resume support built-in via --download-archive

    Args:
        playlist_url: URL to a playlist (YouTube, Vimeo, etc.)
        cache_base: Base cache directory. Defaults to get_cache_dir().
        quality: Audio quality (e.g., "64K", "128K"). Default "64K".
        concurrent_fragments: Number of concurrent download fragments. Default 4.
        on_progress: Optional callback for progress updates.
            Receives DownloadProgress objects with status, percent, speed, etc.

    Returns:
        PlaylistDownloadResult with download statistics and video IDs.

    Raises:
        DownloadError: If the playlist download fails completely.
    """
    from pathlib import Path as PathLib

    from claudetube.config.loader import get_cache_dir
    from claudetube.config.output_templates import get_output_path

    if cache_base is None:
        cache_base = get_cache_dir()

    # Ensure cache_base is a Path object
    cache_base = PathLib(cache_base)

    # Archive file for tracking completed downloads (enables resume)
    archive_file = cache_base / "playlists" / "download_archive.txt"
    archive_file.parent.mkdir(parents=True, exist_ok=True)

    # Build output template for audio files
    output_template = get_output_path("audio", cache_base)

    tool = _get_yt_dlp()

    # Build yt-dlp args for playlist batch download
    args = [
        # Output template matching cache structure
        "-o",
        output_template,
        # Audio extraction
        "-f",
        "ba",  # Best audio
        "-x",  # Extract audio
        "--audio-format",
        "mp3",
        "--audio-quality",
        quality,
        # Playlist handling
        "--yes-playlist",  # Ensure playlist is treated as playlist
        # Resume support
        "--download-archive",
        str(archive_file),
        # Performance
        "-N",
        str(concurrent_fragments),
        # Output control
        "--no-warnings",
        # Write metadata alongside audio
        "--write-info-json",
        "--write-thumbnail",
        "--convert-thumbnails",
        "jpg",
        # The playlist URL
        playlist_url,
    ]

    # Add YouTube-specific config if applicable
    if tool._is_youtube_url(playlist_url):
        yt_args = tool._youtube_config_args()
        args = yt_args + args

    # Track results
    result = PlaylistDownloadResult(
        playlist_url=playlist_url,
        cache_base=cache_base,
        archive_file=archive_file,
    )

    # Parse progress to track per-video results
    def _progress_wrapper(progress: DownloadProgress) -> None:
        # Track completed downloads
        if progress.status == "finished" and progress.filename:
            # Extract video_id from filename path
            # Path: cache_base/domain/channel/playlist/video_id/audio.mp3
            try:
                path = PathLib(progress.filename)
                # video_id is parent directory name
                video_id = path.parent.name
                if video_id and video_id not in result.video_ids:
                    result.video_ids.append(video_id)
                    result.downloaded_count += 1
            except Exception:
                pass

        # Pass through to user callback
        if on_progress:
            with contextlib.suppress(Exception):
                on_progress(progress)

    # Run the download
    if on_progress:
        run_result = tool._run_with_progress(args, on_progress=_progress_wrapper)
    else:
        run_result = tool._run(args)

    # Parse output for statistics
    if run_result.stdout:
        for line in run_result.stdout.splitlines():
            # yt-dlp prints "has already been recorded in the archive" for skipped
            if "has already been recorded in the archive" in line:
                result.skipped_count += 1
            # Track errors from stdout
            elif "ERROR:" in line:
                result.errors.append(line.strip())
                result.failed_count += 1

    # Also check stderr for errors
    if run_result.stderr:
        for line in run_result.stderr.splitlines():
            if "ERROR:" in line and line.strip() not in result.errors:
                result.errors.append(line.strip())
                result.failed_count += 1

    # If complete failure, raise exception
    if not run_result.success and result.downloaded_count == 0:
        from claudetube.tools.yt_dlp import (
            parse_yt_dlp_error,
            yt_dlp_error_to_exception,
        )

        stderr = run_result.stderr or "Unknown error"
        parsed_error = parse_yt_dlp_error(stderr)
        raise yt_dlp_error_to_exception(
            parsed_error,
            is_youtube=tool._is_youtube_url(playlist_url),
        )

    logger.info(
        f"Playlist download complete: {result.downloaded_count} downloaded, "
        f"{result.skipped_count} skipped, {result.failed_count} failed"
    )

    return result
