"""Repository classes for claudetube database entities.

Each repository encapsulates CRUD operations and queries for a specific
table or set of related tables. Repositories take a Database instance
and use parameterized queries for all operations.

Usage:
    from claudetube.db import get_database
    from claudetube.db.repos import VideoRepository, AudioTrackRepository, TranscriptionRepository

    db = get_database()
    videos = VideoRepository(db)

    # Insert a video
    uuid = videos.insert(
        video_id="dQw4w9WgXcQ",
        domain="youtube",
        cache_path="~/.claude/video_cache/dQw4w9WgXcQ",
    )

    # Get by natural key
    video = videos.get_by_video_id("dQw4w9WgXcQ")

    # Full-text search
    results = videos.search_fts("rick astley")
"""

from claudetube.db.repos.audio_tracks import AudioTrackRepository
from claudetube.db.repos.transcriptions import TranscriptionRepository
from claudetube.db.repos.videos import VideoRepository

__all__ = [
    "VideoRepository",
    "AudioTrackRepository",
    "TranscriptionRepository",
]
