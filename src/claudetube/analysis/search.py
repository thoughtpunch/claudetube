"""
Temporal grounding search for finding moments in videos.

Implements semantic search over video scenes with a tiered strategy:
1. TEXT - Fast transcript search (instant)
2. EMBEDDINGS - Vector similarity search (if text search fails)

Architecture: Cheap First, Expensive Last
- Text search: <100ms (regex/substring matching)
- Embedding search: <500ms (query embed + ChromaDB lookup)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SearchMoment:
    """A single moment found by temporal search."""

    rank: int
    scene_id: int
    start_time: float
    end_time: float
    relevance: float  # 0.0 to 1.0, higher is better
    preview: str  # Transcript snippet
    timestamp_str: str  # Human-readable timestamp (MM:SS or HH:MM:SS)
    match_type: str  # "text" or "semantic"

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "rank": self.rank,
            "scene_id": self.scene_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "relevance": self.relevance,
            "preview": self.preview,
            "timestamp_str": self.timestamp_str,
            "match_type": self.match_type,
        }


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted timestamp string.
    """
    total_seconds = int(seconds)
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _search_transcript_text(
    cache_dir: Path,
    query: str,
    top_k: int = 5,
) -> list[SearchMoment]:
    """Search scenes by transcript text matching.

    Uses case-insensitive substring and word matching for fast results.

    Args:
        cache_dir: Video cache directory.
        query: Natural language query.
        top_k: Maximum number of results.

    Returns:
        List of SearchMoment objects sorted by relevance.
    """
    from claudetube.cache.scenes import load_scenes_data

    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data or not scenes_data.scenes:
        return []

    # Normalize query for matching
    query_lower = query.lower()
    query_words = set(re.findall(r"\w+", query_lower))

    results: list[tuple[float, SearchMoment]] = []

    for scene in scenes_data.scenes:
        transcript = scene.transcript_text or ""
        if not transcript:
            continue

        transcript_lower = transcript.lower()
        transcript_words = set(re.findall(r"\w+", transcript_lower))

        # Calculate relevance score
        score = 0.0

        # Exact phrase match (highest weight)
        if query_lower in transcript_lower:
            score += 0.5

        # Word overlap (lower weight)
        if query_words:
            word_overlap = len(query_words & transcript_words) / len(query_words)
            score += 0.3 * word_overlap

        # Partial word matches
        partial_matches = sum(
            1 for qw in query_words if any(qw in tw for tw in transcript_words)
        )
        if query_words:
            score += 0.2 * (partial_matches / len(query_words))

        if score > 0:
            # Create preview - try to center on the match
            preview = _create_preview(transcript, query, max_len=150)

            results.append((
                score,
                SearchMoment(
                    rank=0,  # Will be set after sorting
                    scene_id=scene.scene_id,
                    start_time=scene.start_time,
                    end_time=scene.end_time,
                    relevance=min(score, 1.0),  # Cap at 1.0
                    preview=preview,
                    timestamp_str=format_timestamp(scene.start_time),
                    match_type="text",
                ),
            ))

    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)

    # Take top_k and assign ranks
    moments = []
    for i, (_, moment) in enumerate(results[:top_k]):
        moment.rank = i + 1
        moments.append(moment)

    return moments


def _create_preview(transcript: str, query: str, max_len: int = 150) -> str:
    """Create a preview snippet centered on the query match.

    Args:
        transcript: Full transcript text.
        query: Search query.
        max_len: Maximum preview length.

    Returns:
        Preview string with ellipsis if truncated.
    """
    if len(transcript) <= max_len:
        return transcript

    # Find query position (case-insensitive)
    query_lower = query.lower()
    transcript_lower = transcript.lower()
    pos = transcript_lower.find(query_lower)

    if pos == -1:
        # No exact match, try first query word
        words = re.findall(r"\w+", query_lower)
        if words:
            for word in words:
                pos = transcript_lower.find(word)
                if pos != -1:
                    break

    if pos == -1:
        # No match found, return start of transcript
        return transcript[:max_len] + "..."

    # Center the preview around the match
    half_len = max_len // 2
    start = max(0, pos - half_len)
    end = min(len(transcript), pos + half_len)

    preview = transcript[start:end]

    # Add ellipsis if truncated
    if start > 0:
        preview = "..." + preview
    if end < len(transcript):
        preview = preview + "..."

    return preview


def _search_embedding(
    cache_dir: Path,
    query: str,
    top_k: int = 5,
) -> list[SearchMoment]:
    """Search scenes using vector embedding similarity.

    Args:
        cache_dir: Video cache directory.
        query: Natural language query.
        top_k: Maximum number of results.

    Returns:
        List of SearchMoment objects sorted by relevance.
    """
    from claudetube.analysis.vector_index import (
        has_vector_index,
        search_scenes_by_text,
    )

    if not has_vector_index(cache_dir):
        logger.debug(f"No vector index found at {cache_dir}")
        return []

    try:
        results = search_scenes_by_text(cache_dir, query, top_k=top_k)
    except Exception as e:
        logger.warning(f"Embedding search failed: {e}")
        return []

    moments = []
    for i, result in enumerate(results):
        # Convert distance to relevance (assuming L2 distance)
        # Lower distance = higher relevance
        # Typical L2 distances range 0-2 for normalized embeddings
        relevance = max(0.0, 1.0 - (result.distance / 2.0))

        preview = result.transcript_preview
        if not preview and result.visual_description:
            preview = f"[Visual: {result.visual_description[:100]}...]"
        elif not preview:
            preview = "[No transcript]"

        moments.append(SearchMoment(
            rank=i + 1,
            scene_id=result.scene_id,
            start_time=result.start_time,
            end_time=result.end_time,
            relevance=relevance,
            preview=preview[:150] + "..." if len(preview) > 150 else preview,
            timestamp_str=format_timestamp(result.start_time),
            match_type="semantic",
        ))

    return moments


def find_moments(
    video_id: str,
    query: str,
    top_k: int = 5,
    cache_dir: Path | None = None,
    strategy: str = "auto",
) -> list[SearchMoment]:
    """Find scenes matching a natural language query.

    Implements tiered search following "Cheap First, Expensive Last":
    1. TEXT - Fast transcript text matching
    2. SEMANTIC - Vector embedding similarity (if text search has few results)

    Args:
        video_id: Video identifier.
        query: Natural language query (e.g., "when do they fix the bug").
        top_k: Maximum number of results to return (default 5).
        cache_dir: Optional cache directory override.
        strategy: Search strategy - "auto" (default), "text", or "semantic".

    Returns:
        List of SearchMoment objects sorted by relevance.

    Raises:
        ValueError: If video is not indexed.
        FileNotFoundError: If video cache doesn't exist.
    """
    from claudetube.analysis.vector_index import has_vector_index
    from claudetube.cache.scenes import has_scenes
    from claudetube.config.loader import get_cache_dir

    # Resolve cache directory
    if cache_dir is None:
        cache_dir = get_cache_dir() / video_id
    else:
        cache_dir = cache_dir / video_id

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Video {video_id} not found in cache. "
            "Run process_video() first."
        )

    if not has_scenes(cache_dir):
        raise ValueError(
            f"Video {video_id} has no scene data. "
            "Run scene segmentation first."
        )

    logger.info(f"Searching for '{query}' in video {video_id}")

    # Strategy selection
    if strategy == "text":
        return _search_transcript_text(cache_dir, query, top_k)
    elif strategy == "semantic":
        if not has_vector_index(cache_dir):
            raise ValueError(
                f"Video {video_id} has no vector index. "
                "Run build_scene_index() first for semantic search."
            )
        return _search_embedding(cache_dir, query, top_k)

    # Auto strategy: try text first, fall back to semantic
    text_results = _search_transcript_text(cache_dir, query, top_k)

    # If text search found good results (high relevance), return them
    if text_results and text_results[0].relevance >= 0.5:
        logger.info(f"Text search found {len(text_results)} results")
        return text_results

    # Try semantic search if available
    if has_vector_index(cache_dir):
        semantic_results = _search_embedding(cache_dir, query, top_k)
        if semantic_results:
            # Merge results, preferring higher relevance
            all_results = _merge_results(text_results, semantic_results, top_k)
            logger.info(
                f"Combined search found {len(all_results)} results "
                f"(text: {len(text_results)}, semantic: {len(semantic_results)})"
            )
            return all_results

    # Return whatever text results we have
    logger.info(f"Text-only search found {len(text_results)} results")
    return text_results


def _merge_results(
    text_results: list[SearchMoment],
    semantic_results: list[SearchMoment],
    top_k: int,
) -> list[SearchMoment]:
    """Merge text and semantic search results, deduplicating by scene_id.

    Args:
        text_results: Results from text search.
        semantic_results: Results from semantic search.
        top_k: Maximum number of results.

    Returns:
        Merged and deduplicated results.
    """
    # Index by scene_id, preferring higher relevance
    by_scene: dict[int, SearchMoment] = {}

    for moment in text_results + semantic_results:
        existing = by_scene.get(moment.scene_id)
        if existing is None or moment.relevance > existing.relevance:
            by_scene[moment.scene_id] = moment

    # Sort by relevance and take top_k
    sorted_moments = sorted(
        by_scene.values(),
        key=lambda m: m.relevance,
        reverse=True,
    )[:top_k]

    # Reassign ranks
    for i, moment in enumerate(sorted_moments):
        moment.rank = i + 1

    return sorted_moments
