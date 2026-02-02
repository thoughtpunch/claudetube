"""
Cross-video knowledge graph for playlists.

Builds semantic links between videos in a playlist:
- Shared entities/topics via TF-IDF keyword extraction
- Prerequisite chains for course playlists
- Symlinks to video caches for easy access
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from sklearn.feature_extraction.text import TfidfVectorizer

from claudetube.config.loader import get_cache_dir

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def extract_topic_keywords(texts: list[str], top_n: int = 15) -> list[dict]:
    """Extract important keywords via TF-IDF.

    Args:
        texts: List of text documents (video titles, descriptions)
        top_n: Number of top keywords to return

    Returns:
        List of dicts with 'keyword' and 'score'
    """
    if not texts or all(not t.strip() for t in texts):
        return []

    # Filter out empty texts
    valid_texts = [t for t in texts if t.strip()]
    if not valid_texts:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=100,
            min_df=1,
            ngram_range=(1, 2),  # Include bigrams for phrases
        )
        tfidf_matrix = vectorizer.fit_transform(valid_texts)

        # Sum TF-IDF scores across all documents
        scores = tfidf_matrix.sum(axis=0).A1
        feature_names = vectorizer.get_feature_names_out()

        # Get top keywords with scores
        top_indices = scores.argsort()[-top_n:][::-1]
        keywords = [
            {"keyword": feature_names[i], "score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]

        return keywords
    except ValueError as e:
        # Handle edge cases (e.g., all stop words, empty vocabulary)
        logger.debug(f"TF-IDF extraction failed: {e}")
        return []


def extract_shared_entities(videos: list[dict]) -> list[dict]:
    """Extract entities that appear across multiple videos.

    Uses simple pattern matching for common technical terms,
    names, and concepts mentioned in multiple video titles/descriptions.

    Args:
        videos: List of video dicts with 'title' and optionally 'description'

    Returns:
        List of entity dicts with 'text', 'type', and 'video_count'
    """
    # Collect all text per video
    video_texts = []
    for v in videos:
        text = v.get("title", "") + " " + v.get("description", "")
        video_texts.append(text.lower())

    # Common technical patterns
    patterns = {
        "technology": r"\b(python|javascript|typescript|react|vue|angular|node|docker|kubernetes|aws|api|rest|graphql|sql|nosql|mongodb|redis|linux|git|github)\b",
        "concept": r"\b(function|class|method|variable|loop|array|object|string|integer|boolean|database|server|client|frontend|backend|testing|deployment|security|authentication|authorization)\b",
        "action": r"\b(install|setup|configure|create|build|deploy|test|debug|refactor|optimize)\b",
    }

    entity_counts: dict[str, dict] = {}

    for entity_type, pattern in patterns.items():
        for video_idx, text in enumerate(video_texts):
            matches = re.findall(pattern, text)
            for match in matches:
                key = match.lower()
                if key not in entity_counts:
                    entity_counts[key] = {
                        "text": match,
                        "type": entity_type,
                        "video_indices": set(),
                    }
                entity_counts[key]["video_indices"].add(video_idx)

    # Filter to entities appearing in 2+ videos and sort by count
    shared = [
        {
            "text": e["text"],
            "type": e["type"],
            "video_count": len(e["video_indices"]),
        }
        for e in entity_counts.values()
        if len(e["video_indices"]) >= 2
    ]

    return sorted(shared, key=lambda x: x["video_count"], reverse=True)


def build_prerequisite_chain(videos: list[dict], playlist_type: str) -> list[dict]:
    """Build prerequisite relationships for video sequences.

    For courses/series, each video depends on all previous videos.
    For other types, no prerequisites are assigned.

    Args:
        videos: List of video dicts (must have 'video_id' and 'position')
        playlist_type: Type from classify_playlist_type ('course', 'series', etc.)

    Returns:
        List of video dicts with 'prerequisites' and 'next' fields added
    """
    # Sort by position to ensure correct order
    sorted_videos = sorted(videos, key=lambda v: v.get("position", 0))

    enriched = []
    for i, video in enumerate(sorted_videos):
        video_copy = dict(video)

        if playlist_type in ("course", "series"):
            # Each video requires all previous videos
            video_copy["prerequisites"] = [
                sorted_videos[j]["video_id"] for j in range(i)
            ]
            video_copy["next"] = (
                sorted_videos[i + 1]["video_id"] if i < len(sorted_videos) - 1 else None
            )
            video_copy["previous"] = sorted_videos[i - 1]["video_id"] if i > 0 else None
        else:
            # Collections/conferences don't have implicit prerequisites
            video_copy["prerequisites"] = []
            video_copy["next"] = None
            video_copy["previous"] = None

        enriched.append(video_copy)

    return enriched


def create_video_symlinks(
    videos: list[dict],
    playlist_id: str,
    cache_base: Path | None = None,
) -> dict[str, Path | None]:
    """Create symlinks from playlist directory to cached video directories.

    Args:
        videos: List of video dicts (must have 'video_id')
        playlist_id: Playlist ID for the target directory
        cache_base: Cache base directory (defaults to get_cache_dir())

    Returns:
        Dict mapping video_id to symlink path (None if video not cached)
    """
    cache_base = cache_base or get_cache_dir()

    # Create playlist videos directory
    playlist_videos_dir = cache_base / "playlists" / playlist_id / "videos"
    playlist_videos_dir.mkdir(parents=True, exist_ok=True)

    symlinks = {}

    for video in videos:
        video_id = video.get("video_id")
        if not video_id:
            continue

        video_cache = cache_base / video_id
        symlink_path = playlist_videos_dir / video_id

        if video_cache.exists():
            # Create or update symlink
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()

            try:
                symlink_path.symlink_to(video_cache)
                symlinks[video_id] = symlink_path
                logger.debug(f"Created symlink: {symlink_path} -> {video_cache}")
            except OSError as e:
                logger.warning(f"Failed to create symlink for {video_id}: {e}")
                symlinks[video_id] = None
        else:
            # Video not yet cached
            symlinks[video_id] = None

    return symlinks


def build_knowledge_graph(
    playlist_data: dict,
    cache_base: Path | None = None,
) -> dict:
    """Build a cross-video knowledge graph for a playlist.

    Args:
        playlist_data: Playlist metadata from extract_playlist_metadata()
        cache_base: Cache base directory (defaults to get_cache_dir())

    Returns:
        Knowledge graph dict containing:
        - playlist: Original playlist metadata
        - common_topics: TF-IDF extracted keywords
        - shared_entities: Entities appearing across videos
        - videos: Enriched video list with prerequisites
        - cached_videos: List of video IDs that are cached
    """
    cache_base = cache_base or get_cache_dir()
    videos = playlist_data.get("videos", [])
    playlist_type = playlist_data.get("inferred_type", "collection")

    # Collect text for topic extraction
    texts = []
    for v in videos:
        text_parts = [v.get("title", "")]
        if v.get("description"):
            text_parts.append(v["description"])
        texts.append(" ".join(text_parts))

    # Also include playlist title/description
    playlist_text = (
        playlist_data.get("title", "") + " " + playlist_data.get("description", "")
    )
    if playlist_text.strip():
        texts.append(playlist_text)

    # Extract topics and entities
    common_topics = extract_topic_keywords(texts)
    shared_entities = extract_shared_entities(videos)

    # Build prerequisite chains
    enriched_videos = build_prerequisite_chain(videos, playlist_type)

    # Create symlinks to cached videos
    playlist_id = playlist_data.get("playlist_id", "")
    if playlist_id:
        symlinks = create_video_symlinks(enriched_videos, playlist_id, cache_base)
        cached_video_ids = [vid for vid, path in symlinks.items() if path is not None]
    else:
        cached_video_ids = []

    return {
        "playlist": playlist_data,
        "common_topics": common_topics,
        "shared_entities": shared_entities,
        "videos": enriched_videos,
        "cached_videos": cached_video_ids,
    }


def save_knowledge_graph(
    knowledge_graph: dict,
    cache_base: Path | None = None,
) -> Path:
    """Save knowledge graph to playlist cache directory.

    Args:
        knowledge_graph: Knowledge graph from build_knowledge_graph()
        cache_base: Cache base directory

    Returns:
        Path to saved knowledge_graph.json
    """
    cache_base = cache_base or get_cache_dir()
    playlist_id = knowledge_graph["playlist"]["playlist_id"]

    playlist_dir = cache_base / "playlists" / playlist_id
    playlist_dir.mkdir(parents=True, exist_ok=True)

    graph_file = playlist_dir / "knowledge_graph.json"
    graph_file.write_text(json.dumps(knowledge_graph, indent=2))

    logger.info(f"Saved knowledge graph: {graph_file}")
    return graph_file


def load_knowledge_graph(
    playlist_id: str,
    cache_base: Path | None = None,
) -> dict | None:
    """Load cached knowledge graph for a playlist.

    Args:
        playlist_id: Playlist ID
        cache_base: Cache base directory

    Returns:
        Knowledge graph dict or None if not cached
    """
    cache_base = cache_base or get_cache_dir()
    graph_file = cache_base / "playlists" / playlist_id / "knowledge_graph.json"

    if not graph_file.exists():
        return None

    try:
        return json.loads(graph_file.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load knowledge graph: {e}")
        return None


def get_video_context(
    video_id: str,
    playlist_id: str,
    cache_base: Path | None = None,
) -> dict | None:
    """Get contextual information for a video within a playlist.

    Returns information about prerequisites, related videos, and
    shared topics that provide context for understanding the video.

    Args:
        video_id: Video ID to get context for
        playlist_id: Playlist containing the video
        cache_base: Cache base directory

    Returns:
        Context dict or None if not found
    """
    graph = load_knowledge_graph(playlist_id, cache_base)
    if not graph:
        return None

    # Find the video in the graph
    video = None
    for v in graph.get("videos", []):
        if v.get("video_id") == video_id:
            video = v
            break

    if not video:
        return None

    # Build context
    context = {
        "video": video,
        "playlist_title": graph["playlist"].get("title"),
        "playlist_type": graph["playlist"].get("inferred_type"),
        "position": video.get("position"),
        "total_videos": len(graph.get("videos", [])),
        "prerequisites": video.get("prerequisites", []),
        "next": video.get("next"),
        "previous": video.get("previous"),
        "common_topics": graph.get("common_topics", [])[:10],
        "shared_entities": graph.get("shared_entities", [])[:10],
    }

    # Get titles for prerequisites
    video_titles = {v["video_id"]: v.get("title") for v in graph.get("videos", [])}
    context["prerequisite_titles"] = [
        {"video_id": vid, "title": video_titles.get(vid)}
        for vid in context["prerequisites"]
    ]

    if context["next"]:
        context["next_title"] = video_titles.get(context["next"])
    if context["previous"]:
        context["previous_title"] = video_titles.get(context["previous"])

    return context
