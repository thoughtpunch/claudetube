"""
yt-dlp output templates matching claudetube's hierarchical cache structure.

These templates enable batch downloads with correct path organization:
    {cache_base}/{domain}/{channel}/{playlist}/{video_id}/

Uses yt-dlp's fallback syntax: %(field|default)s for missing fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# Fallback values matching VideoPath sentinel values
NO_CHANNEL = "no_channel"
NO_PLAYLIST = "no_playlist"


@dataclass(frozen=True)
class OutputTemplates:
    """yt-dlp output templates for different content types.

    All templates follow the hierarchical structure:
        {extractor}/{channel|no_channel}/{playlist|no_playlist}/{id}/{filename}

    Attributes:
        audio: Template for audio downloads (mp3, m4a, etc.)
        video: Template for video downloads (mp4, webm, etc.)
        thumbnail: Template for thumbnail downloads (jpg, png, etc.)
        subtitle: Template for subtitle downloads (srt, vtt, etc.)
        infojson: Template for info.json metadata files
    """

    audio: str
    video: str
    thumbnail: str
    subtitle: str
    infojson: str

    @classmethod
    def default(cls) -> OutputTemplates:
        """Create default templates matching claudetube cache structure.

        Returns:
            OutputTemplates with standard cache hierarchy paths.
        """
        # Base path pattern: extractor/channel/playlist/video_id
        # Uses | for fallback to sentinel values when fields are missing
        base = (
            f"%(extractor)s/"
            f"%(channel_id|uploader_id|{NO_CHANNEL})s/"
            f"%(playlist_id|{NO_PLAYLIST})s/"
            f"%(id)s"
        )

        return cls(
            audio=f"{base}/audio.%(ext)s",
            video=f"{base}/video.%(ext)s",
            thumbnail=f"{base}/thumbnail.%(ext)s",
            subtitle=f"{base}/%(id)s.%(ext)s",
            infojson=f"{base}/info.json",
        )


# Default templates instance for easy access
TEMPLATES = OutputTemplates.default()


def get_output_path(
    template_type: str,
    cache_base: Path,
) -> str:
    """Get absolute output path template for yt-dlp.

    Combines cache_base with the relative template to create a full
    output path suitable for yt-dlp's -o option.

    Args:
        template_type: One of 'audio', 'video', 'thumbnail', 'subtitle', 'infojson'
        cache_base: Base cache directory (e.g., ~/.claude/video_cache)

    Returns:
        Absolute path template string for yt-dlp.

    Raises:
        ValueError: If template_type is not recognized.

    Example:
        >>> from pathlib import Path
        >>> get_output_path('audio', Path('/cache'))
        '/cache/%(extractor)s/%(channel_id|uploader_id|no_channel)s/%(playlist_id|no_playlist)s/%(id)s/audio.%(ext)s'
    """
    templates = TEMPLATES

    template_map = {
        "audio": templates.audio,
        "video": templates.video,
        "thumbnail": templates.thumbnail,
        "subtitle": templates.subtitle,
        "infojson": templates.infojson,
    }

    if template_type not in template_map:
        valid = ", ".join(sorted(template_map.keys()))
        raise ValueError(
            f"Unknown template type: {template_type!r}. Valid types: {valid}"
        )

    relative_template = template_map[template_type]
    return str(cache_base / relative_template)


def build_outtmpl_dict(cache_base: Path) -> dict[str, str]:
    """Build a complete outtmpl dictionary for yt-dlp options.

    Creates a dictionary suitable for passing to yt-dlp's 'outtmpl' option,
    with type-specific templates for different output types.

    Args:
        cache_base: Base cache directory.

    Returns:
        Dictionary mapping yt-dlp output types to template strings.

    Example:
        >>> from pathlib import Path
        >>> opts = build_outtmpl_dict(Path('/cache'))
        >>> opts['default']  # doctest: +ELLIPSIS
        '/cache/%(extractor)s/.../video.%(ext)s'
    """
    return {
        "default": get_output_path("video", cache_base),
        "thumbnail": get_output_path("thumbnail", cache_base),
        "subtitle": get_output_path("subtitle", cache_base),
        "infojson": get_output_path("infojson", cache_base),
    }


def build_cli_args(
    cache_base: Path,
    *,
    include_audio: bool = True,
    include_thumbnail: bool = True,
    include_subtitles: bool = True,
    include_infojson: bool = True,
) -> list[str]:
    """Build yt-dlp CLI arguments for multi-path output.

    Generates -P (paths) and -o (output template) arguments that configure
    yt-dlp to download different asset types to their correct locations
    within the cache hierarchy.

    The -P flag sets the base path for each type:
        -P home:CACHE_BASE          # Main output directory
        -P thumbnail:CACHE_BASE     # Thumbnail files
        -P subtitle:CACHE_BASE      # Subtitle files
        -P infojson:CACHE_BASE      # Info JSON files

    The -o flag sets type-specific templates:
        -o "thumbnail:TEMPLATE"     # Where thumbnails go
        -o "subtitle:TEMPLATE"      # Where subtitles go
        -o "infojson:TEMPLATE"      # Where info.json goes

    Args:
        cache_base: Base cache directory (e.g., ~/.claude/video_cache)
        include_audio: Include audio download output template (default)
        include_thumbnail: Include --write-thumbnail and thumbnail output
        include_subtitles: Include --write-subs and subtitle output
        include_infojson: Include --write-info-json and infojson output

    Returns:
        List of CLI arguments for yt-dlp.

    Example:
        >>> from pathlib import Path
        >>> args = build_cli_args(Path('/cache'), include_infojson=False)
        >>> '-P' in args
        True
        >>> '--write-thumbnail' in args
        True
    """
    args: list[str] = []
    cache_str = str(cache_base)

    # Set the base home path
    args.extend(["-P", f"home:{cache_str}"])

    # Audio output template (the main/default output)
    if include_audio:
        args.extend(["-o", TEMPLATES.audio])

    # Thumbnail: set path prefix and type-specific template
    if include_thumbnail:
        args.extend(["-P", f"thumbnail:{cache_str}"])
        args.extend(["-o", f"thumbnail:{TEMPLATES.thumbnail}"])
        args.append("--write-thumbnail")
        args.extend(["--convert-thumbnails", "jpg"])

    # Subtitles: set path prefix and type-specific template
    if include_subtitles:
        args.extend(["-P", f"subtitle:{cache_str}"])
        args.extend(["-o", f"subtitle:{TEMPLATES.subtitle}"])
        args.append("--write-subs")
        args.append("--write-auto-subs")
        args.extend(["--sub-langs", "en.*,en"])
        args.extend(["--convert-subs", "srt"])

    # Info JSON: set path prefix and type-specific template
    if include_infojson:
        args.extend(["-P", f"infojson:{cache_str}"])
        args.extend(["-o", f"infojson:{TEMPLATES.infojson}"])
        args.append("--write-info-json")

    return args


def build_audio_download_args(
    cache_base: Path,
    url: str,
    *,
    quality: str = "64K",
    include_thumbnail: bool = True,
    include_subtitles: bool = True,
    include_infojson: bool = False,
) -> list[str]:
    """Build complete yt-dlp CLI arguments for audio download with assets.

    Combines multi-path output configuration with audio extraction settings.
    This is designed for claudetube's typical workflow: download audio + metadata.

    Args:
        cache_base: Base cache directory
        url: Video URL to download
        quality: Audio quality (e.g., "64K", "128K")
        include_thumbnail: Download thumbnail (default: True)
        include_subtitles: Download subtitles (default: True)
        include_infojson: Write info.json (default: False, use metadata instead)

    Returns:
        Complete list of CLI arguments for yt-dlp audio download.

    Example:
        >>> from pathlib import Path
        >>> args = build_audio_download_args(
        ...     Path('/cache'),
        ...     'https://youtube.com/watch?v=dQw4w9WgXcQ',
        ...     quality='128K',
        ... )
        >>> '-x' in args  # Extract audio
        True
        >>> '--audio-format' in args
        True
    """
    # Start with multi-path output configuration
    args = build_cli_args(
        cache_base,
        include_audio=True,
        include_thumbnail=include_thumbnail,
        include_subtitles=include_subtitles,
        include_infojson=include_infojson,
    )

    # Audio extraction settings
    args.extend(
        [
            "-f",
            "ba",  # Best audio format
            "-x",  # Extract audio
            "--audio-format",
            "mp3",
            "--audio-quality",
            quality,
            "--no-playlist",
            "--no-warnings",
        ]
    )

    # Add the URL last
    args.append(url)

    return args
