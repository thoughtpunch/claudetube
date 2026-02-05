"""
Navigation module for playlist-aware video navigation.

Provides progress tracking, navigation tools, cross-video search,
and learning intelligence for moving through playlists.
"""

from claudetube.navigation.context import PlaylistContext
from claudetube.navigation.cross_video import (
    ChapterMatch,
    PlaylistChapterIndex,
    PlaylistSearchResult,
    build_chapter_index,
    find_chapters_by_topic,
    load_chapter_index,
    save_chapter_index,
    search_playlist_transcripts,
)
from claudetube.navigation.learning import (
    PrerequisiteWarning,
    Recommendation,
    TopicCoverage,
    VideoContext,
    analyze_topic_coverage,
    check_prerequisites,
    get_learning_recommendations,
    get_video_context,
)
from claudetube.navigation.progress import PlaylistProgress

__all__ = [
    # Context and progress
    "PlaylistContext",
    "PlaylistProgress",
    # Cross-video search
    "PlaylistSearchResult",
    "ChapterMatch",
    "PlaylistChapterIndex",
    "build_chapter_index",
    "save_chapter_index",
    "load_chapter_index",
    "find_chapters_by_topic",
    "search_playlist_transcripts",
    # Learning intelligence
    "PrerequisiteWarning",
    "Recommendation",
    "TopicCoverage",
    "VideoContext",
    "check_prerequisites",
    "get_learning_recommendations",
    "analyze_topic_coverage",
    "get_video_context",
]
