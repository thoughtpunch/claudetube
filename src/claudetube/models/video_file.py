"""
VideoFile - the main result wrapper for processed videos.

Can represent:
- A video downloaded from a URL (in cache)
- A user-provided local video file
- A combination (local file copied to cache for processing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from claudetube.config.loader import get_cache_dir
from claudetube.models.state import VideoState


@dataclass
class VideoFile:
    """
    Result wrapper for a processed video.

    Represents a video that has been processed or is ready for processing.
    Supports both URL-sourced videos (in cache) and local files.
    """

    video_id: str
    cache_dir: Path

    # Original source - either a URL or a local file path
    source_url: str | None = None
    original_path: Path | None = None  # User's local file (if provided)

    # Processing state
    success: bool = True
    error: str | None = None

    # Cached file paths (populated after processing)
    audio_path: Path | None = None
    transcript_srt: Path | None = None
    transcript_txt: Path | None = None
    thumbnail_path: Path | None = None

    # Video state metadata
    state: VideoState | None = None

    # Extracted frames
    frames: list[Path] = field(default_factory=list)

    @property
    def is_local(self) -> bool:
        """True if this video came from a local file."""
        return self.original_path is not None

    @property
    def is_cached(self) -> bool:
        """True if this video has been processed and cached."""
        return self.cache_dir.exists() and (self.cache_dir / "state.json").exists()

    @property
    def has_transcript(self) -> bool:
        """True if a transcript exists."""
        return (
            self.transcript_txt is not None and self.transcript_txt.exists()
        ) or (
            self.transcript_srt is not None and self.transcript_srt.exists()
        )

    @property
    def has_audio(self) -> bool:
        """True if audio file exists in cache."""
        return self.audio_path is not None and self.audio_path.exists()

    @property
    def has_thumbnail(self) -> bool:
        """True if thumbnail exists."""
        return self.thumbnail_path is not None and self.thumbnail_path.exists()

    @property
    def title(self) -> str | None:
        """Get video title from state."""
        return self.state.title if self.state else None

    @property
    def duration(self) -> float | None:
        """Get video duration from state."""
        return self.state.duration if self.state else None

    @property
    def duration_string(self) -> str | None:
        """Get formatted duration string."""
        return self.state.duration_string if self.state else None

    @property
    def metadata(self) -> dict:
        """Get full metadata as dict (for backward compatibility)."""
        return self.state.to_dict() if self.state else {}

    @classmethod
    def from_url(
        cls,
        video_id: str,
        url: str,
        cache_base: Path | None = None,
    ) -> VideoFile:
        """Create a VideoFile for a URL-sourced video."""
        cache_base = cache_base or get_cache_dir()
        cache_dir = cache_base / video_id
        return cls(
            video_id=video_id,
            cache_dir=cache_dir,
            source_url=url,
        )

    @classmethod
    def from_local(
        cls,
        video_id: str,
        local_path: Path,
        cache_base: Path | None = None,
    ) -> VideoFile:
        """Create a VideoFile for a user-provided local file."""
        cache_base = cache_base or get_cache_dir()
        cache_dir = cache_base / video_id
        return cls(
            video_id=video_id,
            cache_dir=cache_dir,
            original_path=local_path,
        )

    @classmethod
    def from_cache(
        cls,
        video_id: str,
        cache_base: Path | None = None,
    ) -> VideoFile | None:
        """Load an existing VideoFile from cache.

        Returns None if not found in cache.
        """
        import json

        cache_base = cache_base or get_cache_dir()
        cache_dir = cache_base / video_id
        state_file = cache_dir / "state.json"

        if not state_file.exists():
            return None

        try:
            state_data = json.loads(state_file.read_text())
            state = VideoState.from_dict(state_data)
        except Exception:
            return None

        vf = cls(
            video_id=video_id,
            cache_dir=cache_dir,
            source_url=state.url,
            state=state,
        )

        # Populate paths if they exist
        audio = cache_dir / "audio.mp3"
        srt = cache_dir / "audio.srt"
        txt = cache_dir / "audio.txt"
        thumb = cache_dir / "thumbnail.jpg"

        if audio.exists():
            vf.audio_path = audio
        if srt.exists():
            vf.transcript_srt = srt
        if txt.exists():
            vf.transcript_txt = txt
        if thumb.exists():
            vf.thumbnail_path = thumb

        return vf

    def populate_paths(self) -> None:
        """Populate file paths from cache directory."""
        if not self.cache_dir.exists():
            return

        audio = self.cache_dir / "audio.mp3"
        srt = self.cache_dir / "audio.srt"
        txt = self.cache_dir / "audio.txt"
        thumb = self.cache_dir / "thumbnail.jpg"

        if audio.exists():
            self.audio_path = audio
        if srt.exists():
            self.transcript_srt = srt
        if txt.exists():
            self.transcript_txt = txt
        if thumb.exists():
            self.thumbnail_path = thumb

    def read_transcript(self, format: str = "txt") -> str | None:
        """Read transcript content.

        Args:
            format: "txt" for plain text, "srt" for subtitles with timestamps
        """
        path = self.transcript_txt if format == "txt" else self.transcript_srt
        if path and path.exists():
            return path.read_text()
        # Fallback to other format
        fallback = self.transcript_srt if format == "txt" else self.transcript_txt
        if fallback and fallback.exists():
            return fallback.read_text()
        return None

    def mark_failed(self, error: str) -> None:
        """Mark this video as failed."""
        self.success = False
        self.error = error

    def __str__(self) -> str:
        status = "cached" if self.is_cached else "pending"
        source = "local" if self.is_local else "url"
        return f"VideoFile({self.video_id}, {source}, {status})"

    def __repr__(self) -> str:
        return (
            f"VideoFile(video_id={self.video_id!r}, "
            f"cache_dir={self.cache_dir!r}, "
            f"source_url={self.source_url!r}, "
            f"original_path={self.original_path!r}, "
            f"success={self.success})"
        )
