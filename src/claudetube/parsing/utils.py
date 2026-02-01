"""
URL parsing and video ID extraction utilities.
"""

from __future__ import annotations

import re

from claudetube.models.local_file import LocalFile, is_local_file
from claudetube.models.video_url import VideoURL


def extract_video_id(url: str) -> str:
    """Extract video ID from URL.

    Args:
        url: Video URL or ID

    Returns:
        Extracted video ID
    """
    try:
        return VideoURL.parse(url).video_id
    except Exception:
        # Absolute fallback
        clean = re.sub(r"^https?://", "", url)
        clean = re.sub(r"[^\w.-]", "_", clean)
        return clean[:50] if len(clean) > 50 else clean


def extract_playlist_id(url: str) -> str | None:
    """Extract YouTube playlist ID from URL (list= parameter).

    Args:
        url: Video URL

    Returns:
        Playlist ID or None if not found
    """
    match = re.search(r"[?&]list=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def extract_url_context(url: str) -> dict:
    """Extract video ID, playlist ID, and other context from URL.

    Args:
        url: Video URL

    Returns:
        Dict with video_id, playlist_id, provider, provider_data, etc.
    """
    parsed = VideoURL.try_parse(url)
    if parsed:
        return {
            "video_id": parsed.video_id,
            "playlist_id": extract_playlist_id(url),
            "original_url": url,
            "clean_url": re.sub(r"[&?]list=[^&]+", "", url),
            "provider": parsed.provider,
            "provider_data": parsed.provider_data,
        }
    # Fallback
    return {
        "video_id": extract_video_id(url),
        "playlist_id": extract_playlist_id(url),
        "original_url": url,
        "clean_url": re.sub(r"[&?]list=[^&]+", "", url),
        "provider": None,
        "provider_data": {},
    }


def get_provider_for_url(url: str) -> str | None:
    """Get the provider name for a URL, or None if unknown.

    Args:
        url: Video URL

    Returns:
        Provider name (e.g., "YouTube") or None
    """
    parsed = VideoURL.try_parse(url)
    return parsed.provider if parsed else None


def parse_input(input_str: str) -> dict:
    """Parse input string as either a URL or local file.

    This is the main entry point for handling user input that could be
    either a URL or a local file path.

    Args:
        input_str: URL or file path

    Returns:
        dict with keys:
            - type: 'url' or 'local'
            - For URLs: video_id, playlist_id, provider, provider_data, etc.
            - For local files: path, filename, extension, is_video

    Raises:
        ValueError: If input cannot be parsed as either URL or local file
    """
    input_str = input_str.strip()

    # Try local file first (more specific check)
    if is_local_file(input_str):
        local = LocalFile.parse(input_str)
        return {
            "type": "local",
            "path": str(local.path),
            "filename": local.filename,
            "stem": local.stem,
            "extension": local.extension,
            "is_video": local.is_video,
            "original_input": input_str,
        }

    # Try as URL
    url_context = extract_url_context(input_str)
    return {
        "type": "url",
        **url_context,
    }
