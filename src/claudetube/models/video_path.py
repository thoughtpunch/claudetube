"""
VideoPath Pydantic model for hierarchical cache paths.

Organizes videos by domain/channel/playlist/video_id instead of flat video_id directories.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from claudetube.models.local_file import LocalFile

# Prefixes stripped from hostnames before domain extraction
_HOSTNAME_PREFIXES = ("www.", "m.", "mobile.", "clips.", "player.", "music.")

# Filesystem sentinel values for None channel/playlist
_NO_CHANNEL = "no_channel"
_NO_PLAYLIST = "no_playlist"

# Regex for valid sanitized domain: lowercase letter followed by lowercase alphanumeric
_DOMAIN_RE = re.compile(r"^[a-z][a-z0-9]*$")

# Regex for sanitizing channel/playlist path components
_UNSAFE_PATH_RE = re.compile(r"[^\w-]")

# Max length for channel/playlist path components
_PATH_COMPONENT_MAX_LEN = 60


def sanitize_domain(hostname: str) -> str:
    """Extract a clean domain name from a hostname.

    Strips common prefixes (www., m., mobile., clips., player., music.),
    strips TLD(s), lowercases, and removes non-word characters.

    Args:
        hostname: Raw hostname string (e.g. "www.youtube.com", "clips.twitch.tv")

    Returns:
        Clean domain string (e.g. "youtube", "twitch")

    Raises:
        ValueError: If the result is empty after sanitization.

    Examples:
        >>> sanitize_domain("youtube.com")
        'youtube'
        >>> sanitize_domain("clips.twitch.tv")
        'twitch'
        >>> sanitize_domain("m.facebook.com")
        'facebook'
        >>> sanitize_domain("music.youtube.com")
        'youtube'
    """
    h = hostname.strip().lower()

    # Strip common prefixes
    for prefix in _HOSTNAME_PREFIXES:
        if h.startswith(prefix):
            h = h[len(prefix) :]
            break  # Only strip one prefix

    # Strip TLD(s): take the first part before any dots
    parts = h.split(".")
    if len(parts) >= 2:
        h = parts[0]

    # Remove non-word characters
    result = re.sub(r"\W+", "", h)

    if not result:
        raise ValueError(f"Cannot extract domain from hostname: {hostname!r}")

    return result


def _sanitize_path_component(value: str) -> str:
    """Sanitize a string for use as a filesystem path component.

    Replaces unsafe characters with underscores and truncates to 60 chars.
    """
    sanitized = _UNSAFE_PATH_RE.sub("_", value)
    return sanitized[:_PATH_COMPONENT_MAX_LEN]


class VideoPath(BaseModel):
    """Hierarchical path components for a cached video.

    Immutable and strictly validated. Used to construct filesystem paths
    for the video cache hierarchy: domain/channel/playlist/video_id.

    Attributes:
        domain: Sanitized domain name (e.g. "youtube", "twitter", "local").
            Must match ^[a-z][a-z0-9]*$.
        channel: Channel identifier, or None if unknown.
        playlist: Playlist identifier, or None if unknown.
        video_id: Unique video identifier. Must be non-empty.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    domain: str
    channel: str | None = None
    playlist: str | None = None
    video_id: str

    @field_validator("domain")
    @classmethod
    def domain_must_be_lowercase_alpha(cls, v: str) -> str:
        if not _DOMAIN_RE.match(v):
            raise ValueError(
                f"domain must be lowercase alphanumeric starting with a letter, got: {v!r}"
            )
        return v

    @field_validator("video_id")
    @classmethod
    def video_id_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("video_id must be non-empty")
        return v

    @field_validator("channel")
    @classmethod
    def channel_must_be_nonempty_or_none(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError(
                "channel must be non-empty string or None, got empty string"
            )
        return v

    @field_validator("playlist")
    @classmethod
    def playlist_must_be_nonempty_or_none(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError(
                "playlist must be non-empty string or None, got empty string"
            )
        return v

    def relative_path(self) -> Path:
        """Return domain/channel/playlist/video_id as a Path.

        Uses 'no_channel'/'no_playlist' as filesystem placeholders for None.
        """
        return (
            Path(self.domain)
            / (self.channel or _NO_CHANNEL)
            / (self.playlist or _NO_PLAYLIST)
            / self.video_id
        )

    @classmethod
    def from_cache_path(cls, cache_path: str) -> VideoPath:
        """Reconstruct from a relative cache path string.

        Translates filesystem sentinels back to None:
        'no_channel' -> None, 'no_playlist' -> None.

        Args:
            cache_path: Relative path like "youtube/UCxxx/no_playlist/abc"

        Returns:
            VideoPath with sentinels translated back to None.
        """
        parts = Path(cache_path).parts
        if len(parts) < 4:
            raise ValueError(
                f"Cache path must have at least 4 components (domain/channel/playlist/video_id), "
                f"got {len(parts)}: {cache_path!r}"
            )
        domain, channel, playlist, video_id = parts[0], parts[1], parts[2], parts[3]
        return cls(
            domain=domain,
            channel=None if channel == _NO_CHANNEL else channel,
            playlist=None if playlist == _NO_PLAYLIST else playlist,
            video_id=video_id,
        )

    @classmethod
    def from_url(cls, url: str, metadata: dict | None = None) -> VideoPath:
        """Extract path from URL, optionally augmenting with yt-dlp metadata.

        Parses the URL to extract domain (via sanitize_domain) and video_id
        (via existing VideoURL parsing). Optionally augments with channel and
        playlist from a yt-dlp metadata dict.

        Args:
            url: Video URL string.
            metadata: Optional yt-dlp metadata dict with keys like
                'channel_id', 'uploader_id', 'channel', 'playlist_id',
                'playlist_title'.

        Returns:
            VideoPath with extracted components.
        """
        from claudetube.models.video_url import VideoURL

        # Parse URL for video_id and provider data
        parsed_url = VideoURL.parse(url)

        # Extract domain from hostname
        url_parsed = urlparse(parsed_url.url)
        domain = sanitize_domain(url_parsed.netloc)

        # Extract channel from URL provider_data or metadata
        channel = _extract_channel(parsed_url.provider_data, metadata)

        # Extract playlist from URL provider_data or metadata
        playlist = _extract_playlist(parsed_url.provider_data, metadata)

        # Sanitize channel/playlist for filesystem safety
        if channel:
            channel = _sanitize_path_component(channel)
        if playlist:
            playlist = _sanitize_path_component(playlist)

        return cls(
            domain=domain,
            channel=channel or None,
            playlist=playlist or None,
            video_id=parsed_url.video_id,
        )

    @classmethod
    def from_local(cls, local_file: LocalFile) -> VideoPath:
        """Create path for a local file.

        Args:
            local_file: LocalFile instance with a video_id property.

        Returns:
            VideoPath with domain="local" and no channel/playlist.
        """
        return cls(
            domain="local",
            channel=None,
            playlist=None,
            video_id=local_file.video_id,
        )

    def cache_dir(self, cache_base: Path) -> Path:
        """Full absolute cache directory.

        Args:
            cache_base: Base cache directory (e.g. ~/.claude/video_cache/).

        Returns:
            Absolute path: cache_base / relative_path()
        """
        return cache_base / self.relative_path()


def _extract_channel(provider_data: dict | None, metadata: dict | None) -> str | None:
    """Extract channel from provider data and/or yt-dlp metadata.

    Priority:
    1. URL named capture 'channel' from provider_data
    2. yt-dlp 'channel_id'
    3. yt-dlp 'uploader_id'
    4. yt-dlp 'channel' (sanitized)
    """
    # From URL regex captures
    if provider_data and provider_data.get("channel"):
        return provider_data["channel"]

    if not metadata:
        return None

    # From yt-dlp metadata
    for key in ("channel_id", "uploader_id"):
        if metadata.get(key):
            return metadata[key]

    # Fallback: channel display name (sanitized)
    if metadata.get("channel"):
        return _sanitize_path_component(metadata["channel"])

    return None


def _extract_playlist(provider_data: dict | None, metadata: dict | None) -> str | None:
    """Extract playlist from provider data and/or yt-dlp metadata.

    Priority:
    1. URL named capture 'playlist' from provider_data
    2. yt-dlp 'playlist_id'
    3. yt-dlp 'playlist_title' (sanitized)
    """
    # From URL regex captures
    if provider_data and provider_data.get("playlist"):
        return provider_data["playlist"]

    if not metadata:
        return None

    # From yt-dlp metadata
    if metadata.get("playlist_id"):
        return metadata["playlist_id"]

    # Fallback: playlist title (sanitized)
    if metadata.get("playlist_title"):
        return _sanitize_path_component(metadata["playlist_title"])

    return None
