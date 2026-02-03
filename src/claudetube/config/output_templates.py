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
        raise ValueError(f"Unknown template type: {template_type!r}. Valid types: {valid}")

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
