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

from claudetube.db.repos.audio_descriptions import AudioDescriptionRepository
from claudetube.db.repos.audio_tracks import AudioTrackRepository
from claudetube.db.repos.code_evolution import CodeEvolutionRepository
from claudetube.db.repos.entities import EntityRepository
from claudetube.db.repos.frames import FrameRepository
from claudetube.db.repos.narrative import NarrativeRepository
from claudetube.db.repos.observations import ObservationRepository
from claudetube.db.repos.pipeline import PipelineRepository
from claudetube.db.repos.playlists import PlaylistRepository
from claudetube.db.repos.qa import QARepository
from claudetube.db.repos.scenes import SceneRepository
from claudetube.db.repos.technical_content import TechnicalContentRepository
from claudetube.db.repos.transcriptions import TranscriptionRepository
from claudetube.db.repos.videos import VideoRepository
from claudetube.db.repos.visual_descriptions import VisualDescriptionRepository

__all__ = [
    "AudioDescriptionRepository",
    "AudioTrackRepository",
    "CodeEvolutionRepository",
    "EntityRepository",
    "FrameRepository",
    "NarrativeRepository",
    "ObservationRepository",
    "PipelineRepository",
    "PlaylistRepository",
    "QARepository",
    "SceneRepository",
    "TechnicalContentRepository",
    "TranscriptionRepository",
    "VideoRepository",
    "VisualDescriptionRepository",
]
