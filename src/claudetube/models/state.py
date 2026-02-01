"""
VideoState dataclass for cache state management.
"""

from dataclasses import dataclass, field


@dataclass
class VideoState:
    """Cached video state stored in state.json."""

    video_id: str
    url: str | None = None
    playlist_id: str | None = None

    # Source type: "url" for remote videos, "local" for local files
    source_type: str = "url"
    # For local files, the absolute path to the source file
    source_path: str | None = None
    # How the local file was cached: "symlink", "copy", or None
    cache_mode: str | None = None
    # Path to the cached source file (e.g., "source.mp4")
    cached_file: str | None = None

    # Metadata from yt-dlp
    title: str | None = None
    duration: float | None = None
    duration_string: str | None = None
    uploader: str | None = None
    channel: str | None = None
    upload_date: str | None = None
    description: str | None = None
    categories: list[str] | None = None
    tags: list[str] = field(default_factory=list)
    language: str | None = None
    view_count: int | None = None
    like_count: int | None = None
    thumbnail: str | None = None

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
        """Convert to dictionary for JSON serialization."""
        return {
            "video_id": self.video_id,
            "url": self.url,
            "playlist_id": self.playlist_id,
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
            "description": self.description,
            "categories": self.categories,
            "tags": self.tags,
            "language": self.language,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "thumbnail": self.thumbnail,
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
    def from_dict(cls, data: dict) -> "VideoState":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            video_id=data.get("video_id", ""),
            url=data.get("url"),
            playlist_id=data.get("playlist_id"),
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
            description=data.get("description"),
            categories=data.get("categories"),
            tags=data.get("tags", []),
            language=data.get("language"),
            view_count=data.get("view_count"),
            like_count=data.get("like_count"),
            thumbnail=data.get("thumbnail"),
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
            technical_extraction_complete=data.get("technical_extraction_complete", False),
        )

    @classmethod
    def from_metadata(cls, video_id: str, url: str, meta: dict) -> "VideoState":
        """Create from yt-dlp metadata response."""
        return cls(
            video_id=video_id,
            url=url,
            source_type="url",
            title=meta.get("title"),
            duration=meta.get("duration"),
            duration_string=meta.get("duration_string"),
            uploader=meta.get("uploader"),
            channel=meta.get("channel"),
            upload_date=meta.get("upload_date"),
            description=(meta.get("description", "") or "")[:1500],
            categories=meta.get("categories"),
            tags=(meta.get("tags") or [])[:15],
            language=meta.get("language"),
            view_count=meta.get("view_count"),
            like_count=meta.get("like_count"),
            thumbnail=meta.get("thumbnail"),
        )

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
    ) -> "VideoState":
        """Create from a local file with optional ffprobe metadata.

        Args:
            video_id: Generated video_id from LocalFile.video_id
            source_path: Absolute path to the source file
            title: Optional title (defaults to filename stem)
            duration: Duration in seconds (from ffprobe)
            duration_string: Human-readable duration (e.g., "1:30")
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            codec: Video codec name (e.g., "h264")
            creation_time: ISO timestamp of creation (if available)
        """
        from pathlib import Path

        path = Path(source_path)

        state = cls(
            video_id=video_id,
            url=None,
            source_type="local",
            source_path=source_path,
            title=title or path.stem,
            duration=duration,
            duration_string=duration_string,
            upload_date=creation_time,  # Map creation_time to upload_date
        )

        # Store video dimensions and codec in description for now
        # (VideoState doesn't have dedicated fields for these)
        if width and height:
            dimensions = f"{width}x{height}"
            if fps:
                dimensions += f" @ {fps:.1f}fps"
            if codec:
                dimensions += f" ({codec})"
            state.description = dimensions

        return state
