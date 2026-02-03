"""
Temporal grounding search for finding moments in videos.

Implements semantic search over video scenes with a tiered strategy:
1. TEXT - Fast transcript search via FTS5 (instant)
2. EMBEDDINGS - Vector similarity search via sqlite-vec (if text search fails)
3. QUERY EXPANSION - Optional LLM-powered query expansion for better recall

Architecture: Cheap First, Expensive Last
- FTS5 search: <100ms (SQLite full-text search)
- Embedding search: <500ms (query embed + sqlite-vec KNN)
- Query expansion: ~1-2s (LLM call, optional)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from claudetube.providers.base import Reasoner

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
    match_type: str  # "text", "fts", "semantic", "text+semantic", etc.
    video_id: str | None = None  # For cross-video search

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        result = {
            "rank": self.rank,
            "scene_id": self.scene_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "relevance": self.relevance,
            "preview": self.preview,
            "timestamp_str": self.timestamp_str,
            "match_type": self.match_type,
        }
        if self.video_id is not None:
            result["video_id"] = self.video_id
        return result


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


# =============================================================================
# Score Normalization
# =============================================================================


def normalize_fts_score(rank: float, max_rank: float = -20.0) -> float:
    """Normalize FTS5 BM25 rank to 0-1 relevance score.

    FTS5 BM25 rank is negative (more negative = better match).
    Typical range is -20 to 0 for good matches.

    Args:
        rank: FTS5 rank value (negative).
        max_rank: The "best" rank value (most negative). Scores at or
            beyond this are clamped to 1.0.

    Returns:
        Normalized score in range [0.0, 1.0] where higher is better.
    """
    if rank >= 0:
        return 0.0
    # Clamp to max_rank
    clamped = max(rank, max_rank)
    # Linear interpolation: max_rank -> 1.0, 0 -> 0.0
    return min(1.0, -clamped / -max_rank)


def normalize_vec_distance(distance: float, max_distance: float = 2.0) -> float:
    """Normalize sqlite-vec distance to 0-1 relevance score.

    sqlite-vec uses L2 distance by default (lower = closer = better).
    Typical L2 distances range 0-2 for normalized embeddings.

    Args:
        distance: L2 distance (positive, lower is better).
        max_distance: Maximum expected distance. Distances at or beyond
            this are clamped to 0.0 relevance.

    Returns:
        Normalized score in range [0.0, 1.0] where higher is better.
    """
    if distance < 0:
        distance = 0
    if distance >= max_distance:
        return 0.0
    return 1.0 - (distance / max_distance)


# =============================================================================
# FTS5 Search (Text)
# =============================================================================


def _search_fts5(
    video_id: str,
    query: str,
    top_k: int = 5,
) -> list[SearchMoment]:
    """Search scenes using FTS5 full-text search.

    Args:
        video_id: Video identifier.
        query: Natural language query.
        top_k: Maximum number of results.

    Returns:
        List of SearchMoment objects sorted by relevance.
    """
    try:
        from claudetube.db.queries import search_transcripts_fts

        fts_result = search_transcripts_fts(video_id, query, top_k)
        if fts_result is None:
            return []

        moments = []
        for i, row in enumerate(fts_result):
            preview = _create_preview(
                row.get("transcript_text", ""), query, max_len=150
            )
            # FTS5 returns relevance already normalized in queries.py
            relevance = row.get("relevance", 0.5)
            moments.append(
                SearchMoment(
                    rank=i + 1,
                    scene_id=row["scene_id"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    relevance=relevance,
                    preview=preview,
                    timestamp_str=format_timestamp(row["start_time"]),
                    match_type="fts",
                )
            )
        return moments
    except Exception:
        logger.debug("FTS5 search failed", exc_info=True)
        return []


def _search_fts5_cross_video(
    query: str,
    top_k: int = 10,
) -> list[SearchMoment]:
    """Search scenes across ALL videos using FTS5.

    Args:
        query: Natural language query.
        top_k: Maximum number of results.

    Returns:
        List of SearchMoment objects with video context.
    """
    try:
        from claudetube.db.queries import search_transcripts_fts_cross_video

        fts_result = search_transcripts_fts_cross_video(query, top_k)
        if fts_result is None:
            return []

        moments = []
        for i, row in enumerate(fts_result):
            preview = _create_preview(
                row.get("transcript_text", ""), query, max_len=150
            )
            relevance = row.get("relevance", 0.5)
            moments.append(
                SearchMoment(
                    rank=i + 1,
                    scene_id=row["scene_id"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    relevance=relevance,
                    preview=preview,
                    timestamp_str=format_timestamp(row["start_time"]),
                    match_type="fts",
                    video_id=row.get("video_id"),
                )
            )
        return moments
    except Exception:
        logger.debug("Cross-video FTS5 search failed", exc_info=True)
        return []


# =============================================================================
# sqlite-vec Search (Semantic/Vector)
# =============================================================================


def _search_vec(
    video_id: str,
    query: str,
    top_k: int = 5,
    cache_dir: Path | None = None,
) -> list[SearchMoment]:
    """Search scenes using sqlite-vec vector similarity.

    Args:
        video_id: Video identifier.
        query: Natural language query.
        top_k: Maximum number of results.
        cache_dir: Optional cache directory.

    Returns:
        List of SearchMoment objects sorted by similarity.
    """
    try:
        from claudetube.config.loader import get_cache_dir

        if cache_dir is None:
            cache_dir = get_cache_dir()

        video_cache_dir = cache_dir / video_id

        from claudetube.analysis.vector_index import (
            has_vector_index,
            search_scenes_by_text,
        )

        if not has_vector_index(video_cache_dir):
            return []

        results = search_scenes_by_text(video_cache_dir, query, top_k=top_k)

        moments = []
        for i, result in enumerate(results):
            # Normalize distance to relevance
            relevance = normalize_vec_distance(result.distance)

            preview = result.transcript_preview
            if not preview and result.visual_description:
                preview = f"[Visual: {result.visual_description[:100]}...]"
            elif not preview:
                preview = "[No transcript]"

            moments.append(
                SearchMoment(
                    rank=i + 1,
                    scene_id=result.scene_id,
                    start_time=result.start_time,
                    end_time=result.end_time,
                    relevance=relevance,
                    preview=preview[:150] + "..." if len(preview) > 150 else preview,
                    timestamp_str=format_timestamp(result.start_time),
                    match_type="semantic",
                )
            )
        return moments
    except Exception:
        logger.debug("sqlite-vec search failed", exc_info=True)
        return []


def _search_vec_cross_video(
    query: str,
    top_k: int = 10,
) -> list[SearchMoment]:
    """Search scenes across ALL videos using sqlite-vec.

    Args:
        query: Natural language query.
        top_k: Maximum number of results.

    Returns:
        List of SearchMoment objects with video context.
    """
    try:
        from claudetube.analysis.vector_index import search_similar_cross_video

        results = search_similar_cross_video(query, top_k=top_k)

        moments = []
        for i, result in enumerate(results):
            relevance = normalize_vec_distance(result.distance)

            preview = result.transcript_preview
            if not preview:
                preview = "[No transcript]"

            moments.append(
                SearchMoment(
                    rank=i + 1,
                    scene_id=result.scene_id,
                    start_time=result.start_time,
                    end_time=result.end_time,
                    relevance=relevance,
                    preview=preview[:150] + "..." if len(preview) > 150 else preview,
                    timestamp_str=format_timestamp(result.start_time),
                    match_type="semantic",
                    video_id=result.video_id,
                )
            )
        return moments
    except Exception:
        logger.debug("Cross-video vec search failed", exc_info=True)
        return []


# =============================================================================
# Unified Search (FTS5 + sqlite-vec)
# =============================================================================


def unified_search(
    video_id: str,
    query: str,
    top_k: int = 5,
    semantic_weight: float = 0.5,
    cache_dir: Path | None = None,
) -> list[SearchMoment]:
    """Unified search combining FTS5 and sqlite-vec results.

    Searches both FTS5 (keyword matching) and sqlite-vec (semantic similarity),
    then blends scores for results that appear in both.

    Score blending formula:
        final_score = (1 - semantic_weight) * fts_score + semantic_weight * vec_score

    Args:
        video_id: Video identifier.
        query: Natural language query.
        top_k: Maximum number of results.
        semantic_weight: Weight for semantic scores (0.0 to 1.0).
            Text weight is 1 - semantic_weight.
        cache_dir: Optional cache directory.

    Returns:
        List of SearchMoment objects sorted by blended relevance.
    """
    # Get FTS results
    fts_results = _search_fts5(video_id, query, top_k=top_k * 2)

    # Get vec results
    vec_results = _search_vec(video_id, query, top_k=top_k * 2, cache_dir=cache_dir)

    # If no results from either, try fallback in-memory search
    if not fts_results and not vec_results:
        from claudetube.config.loader import get_cache_dir

        if cache_dir is None:
            cache_dir = get_cache_dir()
        video_cache_dir = cache_dir / video_id
        return _search_transcript_text_memory(video_cache_dir, query, top_k)

    # Merge and blend
    return _merge_results(fts_results, vec_results, top_k, semantic_weight)


def unified_search_cross_video(
    query: str,
    top_k: int = 10,
    semantic_weight: float = 0.5,
) -> list[SearchMoment]:
    """Unified cross-video search combining FTS5 and sqlite-vec.

    Searches across ALL cached videos using both FTS5 and sqlite-vec.

    Args:
        query: Natural language query.
        top_k: Maximum number of results.
        semantic_weight: Weight for semantic scores (0.0 to 1.0).

    Returns:
        List of SearchMoment objects with video context.
    """
    # Get FTS results
    fts_results = _search_fts5_cross_video(query, top_k=top_k * 2)

    # Get vec results
    vec_results = _search_vec_cross_video(query, top_k=top_k * 2)

    # Merge with video-aware deduplication
    return _merge_results_cross_video(fts_results, vec_results, top_k, semantic_weight)


# =============================================================================
# In-Memory Fallback (for backward compatibility)
# =============================================================================


def _search_transcript_text(
    cache_dir: Path,
    query: str,
    top_k: int = 5,
) -> list[SearchMoment]:
    """Search scenes by transcript text matching.

    Uses FTS5 for faster search when available, falls back to
    case-insensitive substring and word matching.

    Args:
        cache_dir: Video cache directory.
        query: Natural language query.
        top_k: Maximum number of results.

    Returns:
        List of SearchMoment objects sorted by relevance.
    """
    # Extract video_id from cache_dir (last component)
    video_id = cache_dir.name

    # Try FTS search first (faster and more accurate)
    fts_results = _search_fts5(video_id, query, top_k)
    if fts_results:
        return fts_results

    # Fallback: in-memory text matching
    return _search_transcript_text_memory(cache_dir, query, top_k)


def _search_transcript_text_memory(
    cache_dir: Path,
    query: str,
    top_k: int = 5,
) -> list[SearchMoment]:
    """Search scenes by transcript text matching (in-memory fallback).

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

            results.append(
                (
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
                )
            )

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
    video_id = cache_dir.name
    return _search_vec(video_id, query, top_k, cache_dir.parent)


async def expand_query(query: str, reasoner: Reasoner) -> list[str]:
    """Generate related search terms using an LLM.

    Asks the reasoner to produce alternative phrasings and related terms
    for the given query, improving recall in text and semantic search.

    Args:
        query: Original user search query.
        reasoner: A Reasoner provider instance.

    Returns:
        List of expanded query strings (excluding the original).
        Returns empty list on failure.
    """
    try:
        result = await reasoner.reason(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate 5 alternative search queries for finding "
                        "moments in a video transcript. Return ONLY the queries, "
                        "one per line, no numbering or bullets. Focus on "
                        "synonyms, related phrases, and different ways to "
                        "express the same concept."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        text = result if isinstance(result, str) else str(result)
        terms = [
            line.strip()
            for line in text.strip().splitlines()
            if line.strip() and line.strip().lower() != query.lower()
        ]
        logger.debug("Query expansion: %r -> %d terms", query, len(terms))
        return terms[:5]
    except Exception as e:
        logger.warning("Query expansion failed (continuing without): %s", e)
        return []


def _search_with_expanded_queries(
    cache_dir: Path,
    original_query: str,
    expanded_queries: list[str],
    top_k: int,
) -> list[SearchMoment]:
    """Run text search with original + expanded queries and merge results.

    Args:
        cache_dir: Video cache directory.
        original_query: The user's original query.
        expanded_queries: Additional queries from LLM expansion.
        top_k: Maximum number of results.

    Returns:
        Merged and deduplicated SearchMoment list.
    """
    # Run original query
    all_results = _search_transcript_text(cache_dir, original_query, top_k)

    # Run expanded queries and collect additional results
    for eq in expanded_queries:
        extra = _search_transcript_text(cache_dir, eq, top_k)
        # Discount expanded query results slightly
        for moment in extra:
            moment.relevance *= 0.8
            moment.match_type = "text+expanded"
        all_results.extend(extra)

    # Deduplicate by scene_id, keeping highest relevance
    by_scene: dict[int, SearchMoment] = {}
    for moment in all_results:
        existing = by_scene.get(moment.scene_id)
        if existing is None or moment.relevance > existing.relevance:
            by_scene[moment.scene_id] = moment

    sorted_moments = sorted(by_scene.values(), key=lambda m: m.relevance, reverse=True)[
        :top_k
    ]

    for i, moment in enumerate(sorted_moments):
        moment.rank = i + 1

    return sorted_moments


def find_moments(
    video_id: str,
    query: str,
    top_k: int = 5,
    cache_dir: Path | None = None,
    strategy: str = "auto",
    reasoner: Reasoner | None = None,
    semantic_weight: float = 0.5,
) -> list[SearchMoment]:
    """Find scenes matching a natural language query.

    Implements tiered search following "Cheap First, Expensive Last":
    1. UNIFIED - Combined FTS5 + sqlite-vec search with score blending
    2. TEXT - Fast transcript text matching (fallback)
    3. QUERY EXPANSION - Optional LLM-powered expansion for better recall

    Args:
        video_id: Video identifier.
        query: Natural language query (e.g., "when do they fix the bug").
        top_k: Maximum number of results to return (default 5).
        cache_dir: Optional cache directory override.
        strategy: Search strategy - "auto" (default), "text", or "semantic".
        reasoner: Optional Reasoner provider for LLM-powered query expansion.
            When provided, the query is expanded into related search terms
            before searching, improving recall. Falls back gracefully if
            the reasoner fails.
        semantic_weight: Weight for semantic scores when combining with text
            scores (0.0 to 1.0). Text weight is ``1 - semantic_weight``.
            Only used when both text and semantic results exist for a scene.
            Default 0.5 (equal weight).

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
        cache_dir = get_cache_dir()
        video_cache_dir = cache_dir / video_id
    else:
        video_cache_dir = cache_dir / video_id

    if not video_cache_dir.exists():
        raise FileNotFoundError(
            f"Video {video_id} not found in cache. Run process_video() first."
        )

    if not has_scenes(video_cache_dir):
        raise ValueError(
            f"Video {video_id} has no scene data. Run scene segmentation first."
        )

    logger.info(f"Searching for '{query}' in video {video_id}")

    # Query expansion (if reasoner is available)
    expanded_queries: list[str] = []
    if reasoner is not None:
        try:
            expanded_queries = asyncio.get_event_loop().run_until_complete(
                expand_query(query, reasoner)
            )
        except RuntimeError:
            # No event loop or already running — try creating one
            try:
                expanded_queries = asyncio.run(expand_query(query, reasoner))
            except Exception as e:
                logger.warning("Query expansion failed: %s", e)

    # Strategy selection
    if strategy == "text":
        if expanded_queries:
            return _search_with_expanded_queries(
                video_cache_dir, query, expanded_queries, top_k
            )
        return _search_transcript_text(video_cache_dir, query, top_k)
    elif strategy == "semantic":
        if not has_vector_index(video_cache_dir):
            raise ValueError(
                f"Video {video_id} has no vector index. "
                "Run build_scene_index() first for semantic search."
            )
        return _search_embedding(video_cache_dir, query, top_k)

    # Auto strategy: use unified search (FTS5 + vec)
    results = unified_search(
        video_id, query, top_k=top_k,
        semantic_weight=semantic_weight,
        cache_dir=cache_dir,
    )

    if results:
        logger.info(f"Unified search found {len(results)} results")
        return results

    # If unified search returned nothing, try expanded queries
    if expanded_queries:
        text_results = _search_with_expanded_queries(
            video_cache_dir, query, expanded_queries, top_k
        )
        if text_results:
            return text_results

    # Last resort: in-memory text search
    return _search_transcript_text_memory(video_cache_dir, query, top_k)


def _merge_results(
    text_results: list[SearchMoment],
    semantic_results: list[SearchMoment],
    top_k: int,
    semantic_weight: float = 0.5,
) -> list[SearchMoment]:
    """Merge text and semantic search results, combining scores for shared scenes.

    When a scene appears in both text and semantic results, the scores are
    blended using the configured weight::

        combined = (1 - semantic_weight) * text_score + semantic_weight * semantic_score

    Scenes that appear in only one result set keep their original score.

    Args:
        text_results: Results from text search.
        semantic_results: Results from semantic search.
        top_k: Maximum number of results.
        semantic_weight: Weight for semantic scores (0.0 to 1.0).
            Text weight is ``1 - semantic_weight``. Default 0.5.

    Returns:
        Merged and deduplicated results sorted by combined relevance.
    """
    text_weight = 1.0 - semantic_weight

    # Index results by scene_id
    text_by_scene: dict[int, SearchMoment] = {}
    for moment in text_results:
        existing = text_by_scene.get(moment.scene_id)
        if existing is None or moment.relevance > existing.relevance:
            text_by_scene[moment.scene_id] = moment

    semantic_by_scene: dict[int, SearchMoment] = {}
    for moment in semantic_results:
        existing = semantic_by_scene.get(moment.scene_id)
        if existing is None or moment.relevance > existing.relevance:
            semantic_by_scene[moment.scene_id] = moment

    # Combine scores
    all_scene_ids = set(text_by_scene) | set(semantic_by_scene)
    by_scene: dict[int, SearchMoment] = {}

    for scene_id in all_scene_ids:
        text_moment = text_by_scene.get(scene_id)
        semantic_moment = semantic_by_scene.get(scene_id)

        if text_moment and semantic_moment:
            # Both present: blend scores
            combined_relevance = (
                text_weight * text_moment.relevance
                + semantic_weight * semantic_moment.relevance
            )
            # Use the text moment as the base (it has a better preview)
            text_moment.relevance = min(combined_relevance, 1.0)
            text_moment.match_type = "text+semantic"
            by_scene[scene_id] = text_moment
        elif text_moment:
            by_scene[scene_id] = text_moment
        else:
            assert semantic_moment is not None
            by_scene[scene_id] = semantic_moment

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


def _merge_results_cross_video(
    text_results: list[SearchMoment],
    semantic_results: list[SearchMoment],
    top_k: int,
    semantic_weight: float = 0.5,
) -> list[SearchMoment]:
    """Merge cross-video results with video-aware deduplication.

    Same as _merge_results but uses (video_id, scene_id) as the key
    for deduplication.

    Args:
        text_results: Results from text search.
        semantic_results: Results from semantic search.
        top_k: Maximum number of results.
        semantic_weight: Weight for semantic scores (0.0 to 1.0).

    Returns:
        Merged and deduplicated results sorted by combined relevance.
    """
    text_weight = 1.0 - semantic_weight

    # Index by (video_id, scene_id)
    def make_key(m: SearchMoment) -> tuple[str | None, int]:
        return (m.video_id, m.scene_id)

    text_by_key: dict[tuple[str | None, int], SearchMoment] = {}
    for moment in text_results:
        key = make_key(moment)
        existing = text_by_key.get(key)
        if existing is None or moment.relevance > existing.relevance:
            text_by_key[key] = moment

    semantic_by_key: dict[tuple[str | None, int], SearchMoment] = {}
    for moment in semantic_results:
        key = make_key(moment)
        existing = semantic_by_key.get(key)
        if existing is None or moment.relevance > existing.relevance:
            semantic_by_key[key] = moment

    # Combine scores
    all_keys = set(text_by_key) | set(semantic_by_key)
    by_key: dict[tuple[str | None, int], SearchMoment] = {}

    for key in all_keys:
        text_moment = text_by_key.get(key)
        semantic_moment = semantic_by_key.get(key)

        if text_moment and semantic_moment:
            combined_relevance = (
                text_weight * text_moment.relevance
                + semantic_weight * semantic_moment.relevance
            )
            text_moment.relevance = min(combined_relevance, 1.0)
            text_moment.match_type = "text+semantic"
            by_key[key] = text_moment
        elif text_moment:
            by_key[key] = text_moment
        else:
            assert semantic_moment is not None
            by_key[key] = semantic_moment

    # Sort by relevance and take top_k
    sorted_moments = sorted(
        by_key.values(),
        key=lambda m: m.relevance,
        reverse=True,
    )[:top_k]

    # Reassign ranks
    for i, moment in enumerate(sorted_moments):
        moment.rank = i + 1

    return sorted_moments
