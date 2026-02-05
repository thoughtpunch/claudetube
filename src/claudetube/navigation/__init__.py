"""
Navigation module for playlist-aware video navigation.

Provides progress tracking, navigation tools, and cross-video search
for moving through playlists.
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
]
