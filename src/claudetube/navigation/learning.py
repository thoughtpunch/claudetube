"""
Learning intelligence for playlist navigation.

Provides smart recommendations, prerequisite awareness, and context bridging
for educational content in playlists.

Architecture: Cheap First, Expensive Last
1. CACHE - Use cached knowledge graph and progress data
2. LOCAL - Analyze prerequisites and topics from metadata
3. NO NEW COMPUTE - Never triggers expensive operations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from claudetube.config.loader import get_cache_dir

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PrerequisiteWarning:
    """Warning about missing prerequisites for a video."""

    video_id: str
    video_title: str
    missing_prerequisites: list[
        dict
    ]  # [{"video_id": str, "title": str, "position": int}]
    total_prerequisites: int
    watched_prerequisites: int

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "missing_prerequisites": self.missing_prerequisites,
            "total_prerequisites": self.total_prerequisites,
            "watched_prerequisites": self.watched_prerequisites,
            "missing_count": len(self.missing_prerequisites),
        }


@dataclass
class Recommendation:
    """A video recommendation with reasoning."""

    video_id: str
    video_title: str
    video_position: int
    reason_type: str  # "sequential", "prerequisite", "goal_driven", "related"
    reason: str
    priority: int  # Lower = higher priority

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "video_position": self.video_position,
            "reason_type": self.reason_type,
            "reason": self.reason,
            "priority": self.priority,
        }


@dataclass
class TopicCoverage:
    """How a topic is covered across a playlist."""

    topic: str
    videos_covering: list[dict]  # [{"video_id", "title", "position", "mentions"}]
    total_mentions: int
    chapters_matching: list[dict]  # [{"video_id", "chapter_title", "start_time"}]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "topic": self.topic,
            "videos_covering": self.videos_covering,
            "video_count": len(self.videos_covering),
            "total_mentions": self.total_mentions,
            "chapters_matching": self.chapters_matching,
            "chapter_count": len(self.chapters_matching),
        }


@dataclass
class VideoContext:
    """Context from previously watched videos relevant to current video."""

    current_video_id: str
    previous_videos_summary: list[dict]  # [{"video_id", "title", "topics"}]
    relevant_prior_content: list[dict]  # [{"video_id", "shared_topics"}]
    prerequisites_covered: list[str]  # video_ids of watched prerequisites

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "current_video_id": self.current_video_id,
            "previous_videos_summary": self.previous_videos_summary,
            "relevant_prior_content": self.relevant_prior_content,
            "prerequisites_covered": self.prerequisites_covered,
        }


def check_prerequisites(
    video_id: str,
    playlist_id: str,
    cache_base: Path | None = None,
) -> PrerequisiteWarning | None:
    """Check if there are unwatched prerequisites for a video.

    Only applies to course/series playlists where videos build on each other.

    Args:
        video_id: Video ID to check prerequisites for.
        playlist_id: Playlist containing the video.
        cache_base: Optional cache base directory.

    Returns:
        PrerequisiteWarning if there are missing prerequisites, None otherwise.
    """
    from claudetube.navigation.context import PlaylistContext
    from claudetube.operations.knowledge_graph import load_knowledge_graph

    cache_base = cache_base or get_cache_dir()

    # Load context
    context = PlaylistContext.load(playlist_id, cache_base)
    if context is None:
        return None

    # Only check prerequisites for courses and series
    if context.playlist_type not in ("course", "series"):
        return None

    # Load knowledge graph for prerequisite data
    knowledge_graph = load_knowledge_graph(playlist_id, cache_base)
    if not knowledge_graph:
        return None

    # Find the video in the knowledge graph
    target_video = None
    for v in knowledge_graph.get("videos", []):
        if v.get("video_id") == video_id:
            target_video = v
            break

    if not target_video:
        return None

    # Get prerequisites
    all_prerequisites = target_video.get("prerequisites", [])
    if not all_prerequisites:
        return None

    # Find unwatched prerequisites
    watched = set(context.progress.watched_videos)
    missing = []

    for prereq_id in all_prerequisites:
        if prereq_id not in watched:
            # Get title from videos list
            prereq_video = context.get_video_by_id(prereq_id)
            if prereq_video:
                missing.append(
                    {
                        "video_id": prereq_id,
                        "title": prereq_video.get("title", ""),
                        "position": prereq_video.get("position", 0) + 1,  # 1-indexed
                    }
                )

    if not missing:
        return None

    return PrerequisiteWarning(
        video_id=video_id,
        video_title=target_video.get("title", ""),
        missing_prerequisites=missing,
        total_prerequisites=len(all_prerequisites),
        watched_prerequisites=len(all_prerequisites) - len(missing),
    )


def get_learning_recommendations(
    playlist_id: str,
    goal: str | None = None,
    cache_base: Path | None = None,
) -> list[Recommendation]:
    """Get recommendations for what to watch next.

    Considers:
    1. Prerequisites (highest priority for courses)
    2. Sequential next (for courses/series)
    3. Goal-driven (if goal provided)
    4. Related by entities

    Args:
        playlist_id: Playlist to get recommendations for.
        goal: Optional learning goal to optimize for.
        cache_base: Optional cache base directory.

    Returns:
        List of Recommendation objects sorted by priority.
    """
    from claudetube.navigation.context import PlaylistContext
    from claudetube.navigation.cross_video import search_playlist_transcripts
    from claudetube.operations.knowledge_graph import load_knowledge_graph

    cache_base = cache_base or get_cache_dir()

    context = PlaylistContext.load(playlist_id, cache_base)
    if context is None:
        return []

    recommendations = []
    watched_ids = set(context.progress.watched_videos)

    # 1. Prerequisite recommendations (highest priority for courses)
    if context.playlist_type in ("course", "series") and context.next_video:
        next_video_id = context.next_video.get("video_id")
        prereq_warning = check_prerequisites(next_video_id, playlist_id, cache_base)

        if prereq_warning and prereq_warning.missing_prerequisites:
            # Recommend the first missing prerequisite
            first_missing = prereq_warning.missing_prerequisites[0]
            recommendations.append(
                Recommendation(
                    video_id=first_missing["video_id"],
                    video_title=first_missing["title"],
                    video_position=first_missing["position"],
                    reason_type="prerequisite",
                    reason=f"Required before '{prereq_warning.video_title}'",
                    priority=0,  # Highest priority
                )
            )

    # 2. Sequential next (for courses/series)
    if context.next_video and context.playlist_type in ("course", "series"):
        next_id = context.next_video.get("video_id")
        if next_id not in watched_ids:
            recommendations.append(
                Recommendation(
                    video_id=next_id,
                    video_title=context.next_video.get("title", ""),
                    video_position=context.next_video.get("position", 0) + 1,
                    reason_type="sequential",
                    reason="Next video in sequence",
                    priority=1,
                )
            )

    # 3. Goal-driven recommendations (if goal provided)
    if goal:
        results = search_playlist_transcripts(
            playlist_id, goal, top_k=3, cache_base=cache_base
        )
        for result in results:
            if result.video_id not in watched_ids:
                recommendations.append(
                    Recommendation(
                        video_id=result.video_id,
                        video_title=result.video_title,
                        video_position=result.video_position + 1,
                        reason_type="goal_driven",
                        reason=f"Matches your goal: '{goal}'",
                        priority=2 if context.playlist_type == "collection" else 3,
                    )
                )
                break  # Only add top result

    # 4. Related by shared entities (for collections/conferences)
    if context.current_video_id:
        knowledge_graph = load_knowledge_graph(playlist_id, cache_base)
        if knowledge_graph:
            # Find videos sharing entities with current video
            current_topics = set()
            for entity in knowledge_graph.get("shared_entities", []):
                if "video_indices" in entity:
                    # Check if current video has this entity
                    # (This is a simplified check - actual implementation may vary)
                    current_topics.add(entity.get("text", ""))

            # Find related unwatched videos
            for video in context.videos:
                vid = video.get("video_id")
                is_candidate = (
                    vid
                    and vid != context.current_video_id
                    and vid not in watched_ids
                    and current_topics  # Has some topics to compare
                )
                if is_candidate:
                    recommendations.append(
                        Recommendation(
                            video_id=vid,
                            video_title=video.get("title", ""),
                            video_position=video.get("position", 0) + 1,
                            reason_type="related",
                            reason="Shares topics with current video",
                            priority=4,
                        )
                    )
                    break  # Only add first related

    # Sort by priority and deduplicate
    seen_ids = set()
    unique_recommendations = []
    for rec in sorted(recommendations, key=lambda r: r.priority):
        if rec.video_id not in seen_ids:
            seen_ids.add(rec.video_id)
            unique_recommendations.append(rec)

    return unique_recommendations


def analyze_topic_coverage(
    playlist_id: str,
    topic: str,
    cache_base: Path | None = None,
) -> TopicCoverage:
    """Analyze how a topic is covered across a playlist.

    Searches transcripts and chapters to understand the topic's presence.

    Args:
        playlist_id: Playlist to analyze.
        topic: Topic to search for.
        cache_base: Optional cache base directory.

    Returns:
        TopicCoverage with analysis results.
    """
    from claudetube.navigation.cross_video import (
        find_chapters_by_topic,
        search_playlist_transcripts,
    )

    cache_base = cache_base or get_cache_dir()

    # Search transcripts
    transcript_results = search_playlist_transcripts(
        playlist_id, topic, top_k=20, cache_base=cache_base
    )

    # Search chapters
    chapter_results = find_chapters_by_topic(
        playlist_id, topic, top_k=10, cache_base=cache_base
    )

    # Group transcript mentions by video
    video_mentions: dict[str, dict] = {}
    for result in transcript_results:
        vid = result.video_id
        if vid not in video_mentions:
            video_mentions[vid] = {
                "video_id": vid,
                "title": result.video_title,
                "position": result.video_position + 1,
                "mentions": 0,
            }
        video_mentions[vid]["mentions"] += 1

    # Convert chapters to dict format
    chapters = [
        {
            "video_id": ch.video_id,
            "chapter_title": ch.chapter_title,
            "start_time": ch.start_time,
            "timestamp_str": ch.timestamp_str,
            "video_title": ch.video_title,
        }
        for ch in chapter_results
    ]

    return TopicCoverage(
        topic=topic,
        videos_covering=list(video_mentions.values()),
        total_mentions=len(transcript_results),
        chapters_matching=chapters,
    )


def get_video_context(
    video_id: str,
    playlist_id: str,
    cache_base: Path | None = None,
) -> VideoContext:
    """Get relevant context from previously watched videos.

    Helps bridge knowledge from prior videos for better understanding.

    Args:
        video_id: Current video ID.
        playlist_id: Playlist containing the video.
        cache_base: Optional cache base directory.

    Returns:
        VideoContext with prior video summaries and relevant content.
    """
    from claudetube.navigation.context import PlaylistContext
    from claudetube.operations.knowledge_graph import load_knowledge_graph

    cache_base = cache_base or get_cache_dir()

    context = PlaylistContext.load(playlist_id, cache_base)
    if context is None:
        return VideoContext(
            current_video_id=video_id,
            previous_videos_summary=[],
            relevant_prior_content=[],
            prerequisites_covered=[],
        )

    knowledge_graph = load_knowledge_graph(playlist_id, cache_base)
    watched_ids = set(context.progress.watched_videos)

    # Build previous videos summary
    previous_summaries = []
    for vid in context.progress.watched_videos:
        if vid == video_id:
            continue
        video_meta = context.get_video_by_id(vid)
        if video_meta:
            # Get topics from knowledge graph
            topics = []
            if knowledge_graph:
                for entity in knowledge_graph.get("shared_entities", [])[:5]:
                    topics.append(entity.get("text", ""))

            previous_summaries.append(
                {
                    "video_id": vid,
                    "title": video_meta.get("title", ""),
                    "position": video_meta.get("position", 0) + 1,
                    "topics": topics[:5],
                }
            )

    # Find relevant prior content (videos that share topics with current)
    relevant_prior = []
    if knowledge_graph:
        # Get current video's info
        current_video = None
        for v in knowledge_graph.get("videos", []):
            if v.get("video_id") == video_id:
                current_video = v
                break

        if current_video:
            # Find watched videos that share entities
            for vid in watched_ids:
                if vid == video_id:
                    continue
                # Simplified: assume shared topics based on knowledge graph
                shared = []
                for entity in knowledge_graph.get("shared_entities", [])[:3]:
                    shared.append(entity.get("text", ""))
                if shared:
                    relevant_prior.append(
                        {
                            "video_id": vid,
                            "title": context.get_video_by_id(vid).get("title", "")
                            if context.get_video_by_id(vid)
                            else "",
                            "shared_topics": shared,
                        }
                    )

    # Get watched prerequisites
    prerequisites_covered = []
    if knowledge_graph:
        for v in knowledge_graph.get("videos", []):
            if v.get("video_id") == video_id:
                prereqs = v.get("prerequisites", [])
                prerequisites_covered = [p for p in prereqs if p in watched_ids]
                break

    return VideoContext(
        current_video_id=video_id,
        previous_videos_summary=previous_summaries,
        relevant_prior_content=relevant_prior,
        prerequisites_covered=prerequisites_covered,
    )


__all__ = [
    "PrerequisiteWarning",
    "Recommendation",
    "TopicCoverage",
    "VideoContext",
    "check_prerequisites",
    "get_learning_recommendations",
    "analyze_topic_coverage",
    "get_video_context",
]
