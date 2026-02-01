"""
VideoResult dataclass for video processing results.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VideoResult:
    """Result of video processing."""

    success: bool
    video_id: str
    output_dir: Path
    transcript_srt: Path | None = None
    transcript_txt: Path | None = None
    thumbnail: Path | None = None
    frames: list[Path] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    error: str | None = None
