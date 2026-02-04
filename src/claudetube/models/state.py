"""
VideoState dataclass for cache state management.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Regex for sanitizing channel/playlist values for filesystem safety
_UNSAFE_CHARS_RE = re.compile(r"[^\w-]")
_MAX_FIELD_LEN = 60


def _sanitize_field(value: str) -> str:
    """Sanitize a string for filesystem safety: replace unsafe chars, truncate."""
    return _UNSAFE_CHARS_RE.sub("_", value)[:_MAX_FIELD_LEN]


@dataclass
class VideoState:
    """Cached video state stored in state.json.

    This is a lean representation of video state for fast per-video reads.
    Queryable metadata (description, categories, tags, view_count, like_count)
    is stored in SQLite as the source of truth and is NOT duplicated here.

    For detailed queryable metadata, use db/queries.py functions.
    """

    video_id: str
    url: str | None = None
    playlist_id: str | None = None

    # Hierarchical path context (populated from URL/metadata)
    domain: str | None = None
    channel_id: str | None = None

    # Source type: "url" for remote videos, "local" for local files
    source_type: str = "url"
    # For local files, the absolute path to the source file
    source_path: str | None = None
    # How the local file was cached: "symlink", "copy", or None
    cache_mode: str | None = None
    # Path to the cached source file (e.g., "source.mp4")
    cached_file: str | None = None

    # Metadata from yt-dlp (fast-access fields only)
    title: str | None = None
    duration: float | None = None
    duration_string: str | None = None
    uploader: str | None = None
    channel: str | None = None
    upload_date: str | None = None
    language: str | None = None
    thumbnail: str | None = None

    # YouTube chapters (if available) - list of {title, start_time, end_time}
    # Kept in JSON for fast scene detection without DB query
    chapters: list[dict] | None = None

    # Processing state
    transcript_complete: bool = False
    transcript_source: str | None = None  # "whisper", "uploaded", "auto-generated"
    whisper_model: str | None = None
    has_thumbnail: bool = False

    # Frame extraction tracking
    frames_count: int | None = None
    frame_interval: int | None = None
    quality_extractions: dict = field(default_factory=dict)

    # Audio Description
    ad_complete: bool = False
    ad_source: str | None = None  # 'source_track' | 'scene_compilation' | 'generated'
    ad_track_available: bool | None = None  # Did source have AD track?

    # Scene processing state
    scenes_processed: bool = False
    scenes_method: str | None = None  # "transcript", "visual", "hybrid"
    scene_count: int | None = None
    visual_transcripts_complete: bool = False
    technical_extraction_complete: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Note: This produces a lean dict for state.json. Queryable metadata
        (description, categories, tags, view_count, like_count) is NOT
        included - that data lives in SQLite as the source of truth.
        """
        return {
            "video_id": self.video_id,
            "url": self.url,
            "playlist_id": self.playlist_id,
            "domain": self.domain,
            "channel_id": self.channel_id,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "cache_mode": self.cache_mode,
            "cached_file": self.cached_file,
            "title": self.title,
            "duration": self.duration,
            "duration_string": self.duration_string,
            "uploader": self.uploader,
            "channel": self.channel,
            "upload_date": self.upload_date,
            "language": self.language,
            "thumbnail": self.thumbnail,
            "chapters": self.chapters,
            "transcript_complete": self.transcript_complete,
            "transcript_source": self.transcript_source,
            "whisper_model": self.whisper_model,
            "has_thumbnail": self.has_thumbnail,
            "frames_count": self.frames_count,
            "frame_interval": self.frame_interval,
            "quality_extractions": self.quality_extractions,
            "ad_complete": self.ad_complete,
            "ad_source": self.ad_source,
            "ad_track_available": self.ad_track_available,
            "scenes_processed": self.scenes_processed,
            "scenes_method": self.scenes_method,
            "scene_count": self.scene_count,
            "visual_transcripts_complete": self.visual_transcripts_complete,
            "technical_extraction_complete": self.technical_extraction_complete,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VideoState:
        """Create from dictionary (JSON deserialization).

        Note: Old state.json files may contain deprecated fields (description,
        categories, tags, view_count, like_count). These are silently ignored -
        the canonical data lives in SQLite.
        """
        return cls(
            video_id=data.get("video_id", ""),
            url=data.get("url"),
            playlist_id=data.get("playlist_id"),
            domain=data.get("domain"),
            channel_id=data.get("channel_id"),
            source_type=data.get("source_type", "url"),
            source_path=data.get("source_path"),
            cache_mode=data.get("cache_mode"),
            cached_file=data.get("cached_file"),
            title=data.get("title"),
            duration=data.get("duration"),
            duration_string=data.get("duration_string"),
            uploader=data.get("uploader"),
            channel=data.get("channel"),
            upload_date=data.get("upload_date"),
            language=data.get("language"),
            thumbnail=data.get("thumbnail"),
            chapters=data.get("chapters"),
            transcript_complete=data.get("transcript_complete", False),
            transcript_source=data.get("transcript_source"),
            whisper_model=data.get("whisper_model"),
            has_thumbnail=data.get("has_thumbnail", False),
            frames_count=data.get("frames_count"),
            frame_interval=data.get("frame_interval"),
            quality_extractions=data.get("quality_extractions", {}),
            ad_complete=data.get("ad_complete", False),
            ad_source=data.get("ad_source"),
            ad_track_available=data.get("ad_track_available"),
            scenes_processed=data.get("scenes_processed", False),
            scenes_method=data.get("scenes_method"),
            scene_count=data.get("scene_count"),
            visual_transcripts_complete=data.get("visual_transcripts_complete", False),
            technical_extraction_complete=data.get(
                "technical_extraction_complete", False
            ),
        )

    @classmethod
    def from_metadata(cls, video_id: str, url: str, meta: dict) -> VideoState:
        """Create from yt-dlp metadata response.

        Note: Queryable fields (description, categories, tags, view_count,
        like_count) are NOT stored in VideoState. They are synced to SQLite
        as the authoritative source. Access sync_video_metadata() in db/sync.py
        to sync these fields to SQLite after creating the VideoState.
        """
        # Extract domain from URL hostname or extractor_key
        domain = cls._extract_domain(url, meta)

        # Extract channel_id from yt-dlp metadata
        channel_id = cls._extract_channel_id(meta)

        # Extract playlist_id from yt-dlp metadata
        playlist_id = cls._extract_playlist_id(meta)

        # Extract chapters from yt-dlp metadata
        # Each chapter: {title, start_time, end_time}
        raw_chapters = meta.get("chapters")
        chapters = None
        if raw_chapters:
            chapters = [
                {
                    "title": ch.get("title"),
                    "start_time": ch.get("start_time"),
                    "end_time": ch.get("end_time"),
                }
                for ch in raw_chapters
                if ch.get("start_time") is not None
            ]

        return cls(
            video_id=video_id,
            url=url,
            source_type="url",
            domain=domain,
            channel_id=channel_id,
            playlist_id=playlist_id,
            title=meta.get("title"),
            duration=meta.get("duration"),
            duration_string=meta.get("duration_string"),
            uploader=meta.get("uploader"),
            channel=meta.get("channel"),
            upload_date=meta.get("upload_date"),
            language=meta.get("language"),
            thumbnail=meta.get("thumbnail"),
            chapters=chapters,
        )

    @staticmethod
    def _extract_domain(url: str, meta: dict) -> str | None:
        """Extract sanitized domain from URL hostname or extractor_key."""
        # Try URL hostname first
        if url:
            try:
                from urllib.parse import urlparse

                from claudetube.models.video_path import sanitize_domain

                parsed = urlparse(url)
                if parsed.netloc:
                    return sanitize_domain(parsed.netloc)
            except (ValueError, ImportError):
                pass

        # Fallback: extractor_key from yt-dlp (e.g. "Youtube", "Vimeo")
        extractor = meta.get("extractor_key") or meta.get("extractor")
        if extractor:
            # Lowercase, strip non-alpha chars
            clean = re.sub(r"[^a-z]", "", extractor.lower())
            if clean:
                return clean

        return None

    @staticmethod
    def _extract_channel_id(meta: dict) -> str | None:
        """Extract channel_id from yt-dlp metadata.

        Priority: channel_id > uploader_id > sanitized channel name.
        """
        for key in ("channel_id", "uploader_id"):
            value = meta.get(key)
            if value:
                return _sanitize_field(str(value))

        # Fallback: channel display name (sanitized)
        channel_name = meta.get("channel")
        if channel_name:
            return _sanitize_field(str(channel_name))

        return None

    @staticmethod
    def _extract_playlist_id(meta: dict) -> str | None:
        """Extract playlist_id from yt-dlp metadata.

        Priority: playlist_id > sanitized playlist_title.
        """
        playlist_id = meta.get("playlist_id")
        if playlist_id:
            return _sanitize_field(str(playlist_id))

        playlist_title = meta.get("playlist_title")
        if playlist_title:
            return _sanitize_field(str(playlist_title))

        return None

    @classmethod
    def from_local_file(
        cls,
        video_id: str,
        source_path: str,
        title: str | None = None,
        duration: float | None = None,
        duration_string: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        codec: str | None = None,
        creation_time: str | None = None,
    ) -> VideoState:
        """Create from a local file with optional ffprobe metadata.

        Note: Video dimensions (width, height, fps, codec) are NOT stored in
        state.json. Sync them to SQLite's videos.description field for queryable
        access using sync_local_file_metadata() in db/sync.py.

        Args:
            video_id: Generated video_id from LocalFile.video_id
            source_path: Absolute path to the source file
            title: Optional title (defaults to filename stem)
            duration: Duration in seconds (from ffprobe)
            duration_string: Human-readable duration (e.g., "1:30")
            width: Video width in pixels (synced to SQLite, not state.json)
            height: Video height in pixels (synced to SQLite, not state.json)
            fps: Frames per second (synced to SQLite, not state.json)
            codec: Video codec name (synced to SQLite, not state.json)
            creation_time: ISO timestamp of creation (if available)
        """
        from pathlib import Path

        path = Path(source_path)

        return cls(
            video_id=video_id,
            url=None,
            source_type="local",
            source_path=source_path,
            title=title or path.stem,
            duration=duration,
            duration_string=duration_string,
            upload_date=creation_time,  # Map creation_time to upload_date
        )
