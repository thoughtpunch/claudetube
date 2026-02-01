"""
Chapter extraction operations for video metadata.

Extracts chapter markers from YouTube/yt-dlp metadata and video descriptions.
"""

from __future__ import annotations

import logging
import re

from claudetube.models.chapter import Chapter

logger = logging.getLogger(__name__)


def parse_timestamp(ts: str) -> float:
    """Convert timestamp string to seconds.

    Supports formats:
    - "1:23" (minutes:seconds)
    - "1:23:45" (hours:minutes:seconds)

    Args:
        ts: Timestamp string

    Returns:
        Time in seconds as float
    """
    parts = list(map(int, ts.split(":")))
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(parts[0]) if parts else 0.0


def extract_youtube_chapters(video_info: dict) -> list[Chapter]:
    """Extract chapter markers from yt-dlp video metadata.

    Tries two methods:
    1. Native chapters from yt-dlp metadata (video_info['chapters'])
    2. Parse timestamps from video description

    Args:
        video_info: yt-dlp metadata dict (from get_metadata or --dump-json)

    Returns:
        List of Chapter objects, sorted by start time.
        Empty list if no chapters found.
    """
    chapters: list[Chapter] = []

    # Method 1: Native chapters from yt-dlp
    # These are the highest quality - human-curated by the creator
    if video_info.get("chapters"):
        for ch in video_info["chapters"]:
            chapters.append(
                Chapter(
                    title=ch.get("title", ""),
                    start=ch.get("start_time", 0.0),
                    end=ch.get("end_time"),
                    source="youtube_chapters",
                    confidence=0.95,
                )
            )
        logger.debug(f"Found {len(chapters)} native YouTube chapters")
        return chapters

    # Method 2: Parse from description
    # Look for lines like "0:00 Introduction" or "1:23:45 - Main Topic"
    description = video_info.get("description", "") or ""

    # Pattern matches timestamps at start of line or after newline
    # Handles formats: "0:00 Title", "1:23 - Title", "01:23:45  Title"
    pattern = r"(?:^|\n)\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]?\s*(.+?)(?=\n|$)"

    matches = re.findall(pattern, description)

    for ts_str, title in matches:
        title = title.strip()
        if not title:
            continue

        try:
            start = parse_timestamp(ts_str)
            chapters.append(
                Chapter(
                    title=title,
                    start=start,
                    end=None,  # Will be filled in post-processing
                    source="description_parsed",
                    confidence=0.9,
                )
            )
        except (ValueError, IndexError):
            logger.debug(f"Failed to parse timestamp: {ts_str}")
            continue

    if chapters:
        # Sort by start time
        chapters.sort(key=lambda c: c.start)

        # Fill in end times based on next chapter's start
        duration = video_info.get("duration")
        for i, chapter in enumerate(chapters):
            if i + 1 < len(chapters):
                chapter.end = chapters[i + 1].start
            elif duration:
                chapter.end = float(duration)

        logger.debug(f"Parsed {len(chapters)} chapters from description")

    return chapters
