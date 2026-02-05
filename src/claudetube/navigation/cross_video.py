"""
Cross-video search for playlist navigation.

Provides search functionality across all videos in a playlist:
- Full-text search across all transcripts
- Chapter-level topic mapping
- Result ranking by relevance and position

Architecture: Cheap First, Expensive Last
1. CACHE - Check for indexed chapters first
2. FTS - Use SQLite FTS5 for fast cross-video search
3. FALLBACK - In-memory search if FTS unavailable
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from claudetube.config.loader import get_cache_dir

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PlaylistSearchResult:
    """A search result from cross-video playlist search."""

    video_id: str
    video_title: str
    video_position: int  # Position in playlist (0-indexed)
    scene_id: int | None  # Scene within video, if available
    start_time: float
    end_time: float
    relevance: float  # 0.0 to 1.0, higher is better
    preview: str  # Transcript snippet
    timestamp_str: str  # Human-readable timestamp (MM:SS)
    match_type: str  # "chapter", "transcript", "fts"
    chapter_title: str | None = None  # If match is within a chapter

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "video_position": self.video_position,
            "scene_id": self.scene_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "relevance": self.relevance,
            "preview": self.preview,
            "timestamp_str": self.timestamp_str,
            "match_type": self.match_type,
            "chapter_title": self.chapter_title,
        }


@dataclass
class ChapterMatch:
    """A chapter matching a topic query."""

    video_id: str
    video_title: str
    video_position: int
    chapter_title: str
    chapter_index: int
    start_time: float
    end_time: float | None
    relevance: float
    timestamp_str: str

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "video_position": self.video_position,
            "chapter_title": self.chapter_title,
            "chapter_index": self.chapter_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "relevance": self.relevance,
            "timestamp_str": self.timestamp_str,
        }


@dataclass
class PlaylistChapterIndex:
    """Index of all chapters across a playlist."""

    playlist_id: str
    chapters: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "playlist_id": self.playlist_id,
            "chapter_count": len(self.chapters),
            "chapters": self.chapters,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlaylistChapterIndex:
        """Create from dictionary."""
        return cls(
            playlist_id=data["playlist_id"],
            chapters=data.get("chapters", []),
        )


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format."""
    total_seconds = int(seconds)
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _create_preview(text: str, query: str, max_len: int = 150) -> str:
    """Create a preview snippet centered on the query match."""
    if len(text) <= max_len:
        return text

    query_lower = query.lower()
    text_lower = text.lower()
    pos = text_lower.find(query_lower)

    if pos == -1:
        # Try first query word
        words = re.findall(r"\w+", query_lower)
        for word in words:
            pos = text_lower.find(word)
            if pos != -1:
                break

    if pos == -1:
        return text[:max_len] + "..."

    half_len = max_len // 2
    start = max(0, pos - half_len)
    end = min(len(text), pos + half_len)
    preview = text[start:end]

    if start > 0:
        preview = "..." + preview
    if end < len(text):
        preview = preview + "..."

    return preview


def build_chapter_index(
    playlist_id: str,
    cache_base: Path | None = None,
) -> PlaylistChapterIndex:
    """Build an index of all chapters across a playlist.

    Reads YouTube chapters from each video's state.json metadata and
    creates a unified searchable index.

    Args:
        playlist_id: Playlist ID.
        cache_base: Cache base directory.

    Returns:
        PlaylistChapterIndex with all chapters.
    """
    from claudetube.cache.storage import load_state
    from claudetube.operations.playlist import load_playlist_metadata

    cache_base = cache_base or get_cache_dir()
    playlist = load_playlist_metadata(playlist_id, cache_base)

    if not playlist:
        logger.warning(f"Playlist not found: {playlist_id}")
        return PlaylistChapterIndex(playlist_id=playlist_id)

    all_chapters = []
    videos = playlist.get("videos", [])

    for video in videos:
        video_id = video.get("video_id")
        if not video_id:
            continue

        video_cache = cache_base / video_id
        state_file = video_cache / "state.json"

        if not state_file.exists():
            continue

        state = load_state(state_file)
        if not state:
            continue

        # Get chapters from metadata
        chapters = []
        if state.chapters:
            chapters = state.chapters
        elif state.metadata and "chapters" in state.metadata:
            chapters = state.metadata["chapters"]

        if not chapters:
            continue

        # Add chapters to index
        for i, chapter in enumerate(chapters):
            chapter_title = chapter.get("title", "")
            start_time = chapter.get("start_time", 0)

            # Get end time from next chapter or video duration
            if i + 1 < len(chapters):
                end_time = chapters[i + 1].get("start_time")
            else:
                end_time = state.duration

            all_chapters.append(
                {
                    "video_id": video_id,
                    "video_title": video.get("title", state.title or ""),
                    "video_position": video.get("position", 0),
                    "chapter_title": chapter_title,
                    "chapter_index": i,
                    "start_time": start_time,
                    "end_time": end_time,
                    "timestamp_str": _format_timestamp(start_time),
                }
            )

    index = PlaylistChapterIndex(playlist_id=playlist_id, chapters=all_chapters)
    logger.info(
        f"Built chapter index for {playlist_id}: "
        f"{len(all_chapters)} chapters across {len(videos)} videos"
    )
    return index


def save_chapter_index(
    index: PlaylistChapterIndex,
    cache_base: Path | None = None,
) -> Path:
    """Save chapter index to playlist cache directory.

    Args:
        index: PlaylistChapterIndex to save.
        cache_base: Cache base directory.

    Returns:
        Path to saved chapter_index.json.
    """
    cache_base = cache_base or get_cache_dir()
    playlist_dir = cache_base / "playlists" / index.playlist_id
    playlist_dir.mkdir(parents=True, exist_ok=True)

    index_file = playlist_dir / "chapter_index.json"
    index_file.write_text(json.dumps(index.to_dict(), indent=2))

    logger.info(f"Saved chapter index: {index_file}")
    return index_file


def load_chapter_index(
    playlist_id: str,
    cache_base: Path | None = None,
) -> PlaylistChapterIndex | None:
    """Load cached chapter index for a playlist.

    Args:
        playlist_id: Playlist ID.
        cache_base: Cache base directory.

    Returns:
        PlaylistChapterIndex or None if not cached.
    """
    cache_base = cache_base or get_cache_dir()
    index_file = cache_base / "playlists" / playlist_id / "chapter_index.json"

    if not index_file.exists():
        return None

    try:
        data = json.loads(index_file.read_text())
        return PlaylistChapterIndex.from_dict(data)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load chapter index: {e}")
        return None


def find_chapters_by_topic(
    playlist_id: str,
    query: str,
    top_k: int = 10,
    cache_base: Path | None = None,
) -> list[ChapterMatch]:
    """Find chapters matching a topic query across all videos.

    Uses keyword matching on chapter titles.

    Args:
        playlist_id: Playlist ID.
        query: Topic query (e.g., "authentication", "setup").
        top_k: Maximum number of results.
        cache_base: Cache base directory.

    Returns:
        List of ChapterMatch objects sorted by relevance.
    """
    cache_base = cache_base or get_cache_dir()

    # Try loading cached index first
    index = load_chapter_index(playlist_id, cache_base)
    if not index:
        # Build and cache the index
        index = build_chapter_index(playlist_id, cache_base)
        if index.chapters:
            save_chapter_index(index, cache_base)

    if not index.chapters:
        return []

    # Normalize query for matching
    query_lower = query.lower()
    query_words = set(re.findall(r"\w+", query_lower))

    results: list[tuple[float, ChapterMatch]] = []

    for chapter in index.chapters:
        title = chapter.get("chapter_title", "")
        title_lower = title.lower()
        title_words = set(re.findall(r"\w+", title_lower))

        # Calculate relevance score
        score = 0.0

        # Exact phrase match (highest weight)
        if query_lower in title_lower:
            score += 0.6

        # Word overlap
        if query_words:
            word_overlap = len(query_words & title_words) / len(query_words)
            score += 0.3 * word_overlap

        # Partial word matches
        partial_matches = sum(
            1 for qw in query_words if any(qw in tw for tw in title_words)
        )
        if query_words:
            score += 0.1 * (partial_matches / len(query_words))

        if score > 0:
            results.append(
                (
                    score,
                    ChapterMatch(
                        video_id=chapter["video_id"],
                        video_title=chapter["video_title"],
                        video_position=chapter["video_position"],
                        chapter_title=title,
                        chapter_index=chapter["chapter_index"],
                        start_time=chapter["start_time"],
                        end_time=chapter.get("end_time"),
                        relevance=min(score, 1.0),
                        timestamp_str=chapter["timestamp_str"],
                    ),
                )
            )

    # Sort by score descending, then by video position
    results.sort(key=lambda x: (-x[0], x[1].video_position))

    return [match for _, match in results[:top_k]]


def search_playlist_transcripts(
    playlist_id: str,
    query: str,
    top_k: int = 10,
    cache_base: Path | None = None,
) -> list[PlaylistSearchResult]:
    """Search across all transcripts in a playlist.

    Uses FTS5 for fast search when available, falls back to in-memory search.

    Args:
        playlist_id: Playlist ID.
        query: Search query.
        top_k: Maximum number of results.
        cache_base: Cache base directory.

    Returns:
        List of PlaylistSearchResult objects sorted by relevance.
    """
    from claudetube.operations.playlist import load_playlist_metadata

    cache_base = cache_base or get_cache_dir()
    playlist = load_playlist_metadata(playlist_id, cache_base)

    if not playlist:
        logger.warning(f"Playlist not found: {playlist_id}")
        return []

    videos = playlist.get("videos", [])
    video_ids = [v.get("video_id") for v in videos if v.get("video_id")]

    if not video_ids:
        return []

    # Create video lookup for metadata
    video_info = {
        v["video_id"]: {"title": v.get("title", ""), "position": v.get("position", 0)}
        for v in videos
        if v.get("video_id")
    }

    # Try FTS5 search across these specific videos
    results = _search_playlist_fts(video_ids, query, top_k, video_info)
    if results:
        return results

    # Fallback: in-memory search
    return _search_playlist_memory(video_ids, query, top_k, video_info, cache_base)


def _search_playlist_fts(
    video_ids: list[str],
    query: str,
    top_k: int,
    video_info: dict[str, dict],
) -> list[PlaylistSearchResult]:
    """Search playlist using FTS5."""
    try:
        from claudetube.db.queries import search_transcripts_fts_multi_video

        fts_results = search_transcripts_fts_multi_video(video_ids, query, top_k)
        if fts_results is None:
            return []

        results = []
        for row in fts_results:
            video_id = row.get("video_id")
            info = video_info.get(video_id, {})

            results.append(
                PlaylistSearchResult(
                    video_id=video_id,
                    video_title=info.get("title", ""),
                    video_position=info.get("position", 0),
                    scene_id=row.get("scene_id"),
                    start_time=row.get("start_time", 0),
                    end_time=row.get("end_time", 0),
                    relevance=row.get("relevance", 0.5),
                    preview=_create_preview(
                        row.get("transcript_text", ""), query, max_len=150
                    ),
                    timestamp_str=_format_timestamp(row.get("start_time", 0)),
                    match_type="fts",
                    chapter_title=None,
                )
            )

        # Sort by relevance, then video position
        results.sort(key=lambda r: (-r.relevance, r.video_position))
        return results

    except Exception as e:
        logger.debug(f"Playlist FTS search failed: {e}")
        return []


def _search_playlist_memory(
    video_ids: list[str],
    query: str,
    top_k: int,
    video_info: dict[str, dict],
    cache_base: Path,
) -> list[PlaylistSearchResult]:
    """Search playlist using in-memory transcript scanning."""
    from claudetube.cache.scenes import load_scenes_data

    query_lower = query.lower()
    query_words = set(re.findall(r"\w+", query_lower))

    results: list[tuple[float, PlaylistSearchResult]] = []

    for video_id in video_ids:
        video_cache = cache_base / video_id
        scenes_data = load_scenes_data(video_cache)

        if not scenes_data or not scenes_data.scenes:
            continue

        info = video_info.get(video_id, {})

        for scene in scenes_data.scenes:
            transcript = scene.transcript_text or ""
            if not transcript:
                continue

            transcript_lower = transcript.lower()
            transcript_words = set(re.findall(r"\w+", transcript_lower))

            # Calculate relevance score
            score = 0.0

            # Exact phrase match
            if query_lower in transcript_lower:
                score += 0.5

            # Word overlap
            if query_words:
                word_overlap = len(query_words & transcript_words) / len(query_words)
                score += 0.3 * word_overlap

            # Partial matches
            partial = sum(
                1 for qw in query_words if any(qw in tw for tw in transcript_words)
            )
            if query_words:
                score += 0.2 * (partial / len(query_words))

            if score > 0:
                results.append(
                    (
                        score,
                        PlaylistSearchResult(
                            video_id=video_id,
                            video_title=info.get("title", ""),
                            video_position=info.get("position", 0),
                            scene_id=scene.scene_id,
                            start_time=scene.start_time,
                            end_time=scene.end_time,
                            relevance=min(score, 1.0),
                            preview=_create_preview(transcript, query, max_len=150),
                            timestamp_str=_format_timestamp(scene.start_time),
                            match_type="transcript",
                            chapter_title=scene.title,
                        ),
                    )
                )

    # Sort by score descending, then video position
    results.sort(key=lambda x: (-x[0], x[1].video_position, x[1].start_time))

    return [r for _, r in results[:top_k]]


__all__ = [
    "PlaylistSearchResult",
    "ChapterMatch",
    "PlaylistChapterIndex",
    "build_chapter_index",
    "save_chapter_index",
    "load_chapter_index",
    "find_chapters_by_topic",
    "search_playlist_transcripts",
]
