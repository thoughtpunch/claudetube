"""
Playlist metadata extraction operations.

Extracts playlist metadata (title, description, videos) without downloading content.
Uses yt-dlp's flat extraction mode for efficiency.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from claudetube.config.loader import get_cache_dir
from claudetube.tools.yt_dlp import YtDlpTool

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def extract_playlist_metadata(playlist_url: str, timeout: int = 60) -> dict:
    """Fetch playlist metadata without downloading videos.

    Args:
        playlist_url: URL to a playlist on any supported site
        timeout: Timeout for metadata fetch in seconds

    Returns:
        Dict with playlist metadata including video list

    Raises:
        MetadataError: If playlist metadata fetch fails
    """
    yt_dlp = YtDlpTool()

    # Use flat extraction to get playlist info without downloading
    result = yt_dlp._run(
        ["--flat-playlist", "--dump-json", "--no-download", playlist_url],
        timeout=timeout,
    )

    if not result.success:
        from claudetube.exceptions import MetadataError

        error_msg = result.stderr.strip() if result.stderr else "Unknown error"
        if "ERROR:" in error_msg:
            error_msg = error_msg.split("ERROR:")[-1].strip()
        raise MetadataError(f"Playlist fetch failed: {error_msg[:500]}")

    # Parse JSON lines output (one per video + playlist header)
    videos = []
    playlist_info = {}

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Playlist-level entry has _type: playlist
        if entry.get("_type") == "playlist":
            playlist_info = entry
        else:
            # Video entry
            videos.append(entry)

    # Extract playlist ID from URL or info
    playlist_id = playlist_info.get("id") or _extract_playlist_id(playlist_url)

    return {
        "playlist_id": playlist_id,
        "title": playlist_info.get("title", ""),
        "description": playlist_info.get("description", ""),
        "channel": playlist_info.get("channel", playlist_info.get("uploader", "")),
        "channel_id": playlist_info.get(
            "channel_id", playlist_info.get("uploader_id", "")
        ),
        "video_count": len(videos),
        "videos": [
            {
                "video_id": v.get("id", ""),
                "title": v.get("title", ""),
                "duration": v.get("duration"),
                "position": idx,
                "url": v.get("url", ""),
            }
            for idx, v in enumerate(videos)
            if v.get("id")  # Skip unavailable videos
        ],
        "inferred_type": classify_playlist_type(playlist_info, videos),
        "url": playlist_url,
    }


def _extract_playlist_id(url: str) -> str:
    """Extract playlist ID from URL."""
    # YouTube playlist
    match = re.search(r"list=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    # Fallback: use URL hash
    import hashlib

    return hashlib.sha256(url.encode()).hexdigest()[:12]


def classify_playlist_type(playlist_info: dict, videos: list[dict]) -> str:
    """Infer playlist type from metadata patterns.

    Returns one of: 'course', 'series', 'conference', 'collection'
    """
    title = playlist_info.get("title", "").lower()
    description = playlist_info.get("description", "").lower()
    video_titles = [v.get("title", "").lower() for v in videos if v.get("title")]

    # Course detection
    course_keywords = [
        "course",
        "tutorial",
        "lesson",
        "learn",
        "bootcamp",
        "workshop",
        "training",
    ]
    if any(kw in title or kw in description for kw in course_keywords):
        return "course"

    # Series detection (numbered episodes)
    numbered_patterns = [
        r"(part|ep|episode|#|chapter|lecture|video)\s*\d+",
        r"^\d+[\.\):\-]",  # Starts with number
        r"\[\d+/\d+\]",  # [1/10] format
    ]
    numbered_count = 0
    for vt in video_titles:
        for pattern in numbered_patterns:
            if re.search(pattern, vt):
                numbered_count += 1
                break

    if numbered_count > len(video_titles) * 0.4:
        return "series"

    # Conference detection
    conference_keywords = [
        "conference",
        "summit",
        "meetup",
        "talks",
        "keynote",
        "pycon",
        "jsconf",
        "devcon",
    ]
    if any(kw in title or kw in description for kw in conference_keywords):
        return "conference"

    return "collection"


def save_playlist_metadata(playlist_data: dict, cache_base: Path | None = None) -> Path:
    """Save playlist metadata to cache.

    Args:
        playlist_data: Playlist metadata dict from extract_playlist_metadata
        cache_base: Cache base directory (defaults to get_cache_dir())

    Returns:
        Path to saved playlist.json
    """
    cache_base = cache_base or get_cache_dir()
    playlist_id = playlist_data["playlist_id"]

    playlist_dir = cache_base / "playlists" / playlist_id
    playlist_dir.mkdir(parents=True, exist_ok=True)

    playlist_file = playlist_dir / "playlist.json"
    playlist_file.write_text(json.dumps(playlist_data, indent=2))

    logger.info(f"Saved playlist metadata: {playlist_file}")
    return playlist_file


def load_playlist_metadata(
    playlist_id: str, cache_base: Path | None = None
) -> dict | None:
    """Load cached playlist metadata.

    Args:
        playlist_id: Playlist ID
        cache_base: Cache base directory

    Returns:
        Playlist data dict or None if not cached
    """
    cache_base = cache_base or get_cache_dir()
    playlist_file = cache_base / "playlists" / playlist_id / "playlist.json"

    if not playlist_file.exists():
        return None

    try:
        return json.loads(playlist_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def list_cached_playlists(cache_base: Path | None = None) -> list[dict]:
    """List all cached playlists.

    Returns:
        List of playlist summaries
    """
    cache_base = cache_base or get_cache_dir()
    playlists_dir = cache_base / "playlists"

    if not playlists_dir.exists():
        return []

    playlists = []
    for playlist_file in sorted(playlists_dir.glob("*/playlist.json")):
        try:
            data = json.loads(playlist_file.read_text())
            playlists.append(
                {
                    "playlist_id": data.get("playlist_id", playlist_file.parent.name),
                    "title": data.get("title"),
                    "video_count": data.get("video_count", 0),
                    "inferred_type": data.get("inferred_type"),
                    "cache_dir": str(playlist_file.parent),
                }
            )
        except (json.JSONDecodeError, OSError):
            continue

    return playlists
