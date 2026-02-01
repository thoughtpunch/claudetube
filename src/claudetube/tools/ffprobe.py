"""
FFprobe tool wrapper for extracting video metadata.
"""

from __future__ import annotations

import contextlib
import json
import logging
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from claudetube.tools.base import ToolResult, VideoTool

if TYPE_CHECKING:
    from pathlib import Path
from claudetube.utils.system import find_tool

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file via ffprobe."""

    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    codec: str | None = None
    creation_time: str | None = None


class FFprobeTool(VideoTool):
    """Wrapper for FFprobe video metadata extraction tool."""

    @property
    def name(self) -> str:
        return "ffprobe"

    def is_available(self) -> bool:
        """Check if ffprobe is installed."""
        try:
            path = self.get_path()
            result = subprocess.run(
                [path, "-version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_path(self) -> str:
        """Get path to ffprobe executable."""
        return find_tool("ffprobe")

    def _run_json(
        self,
        args: list[str],
        timeout: int | None = 30,
    ) -> ToolResult:
        """Run ffprobe with JSON output."""
        cmd = [self.get_path()] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ToolResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.from_error(f"Timeout after {timeout}s")
        except FileNotFoundError:
            return ToolResult.from_error(
                "ffprobe not found. Install ffmpeg: brew install ffmpeg"
            )
        except Exception as e:
            return ToolResult.from_error(str(e))

    def probe(self, file_path: Path | str) -> dict | None:
        """Run ffprobe and return raw JSON output.

        Args:
            file_path: Path to video/audio file

        Returns:
            Parsed JSON dict, or None if failed
        """
        args = [
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(file_path),
        ]

        result = self._run_json(args)
        if not result.success:
            logger.error(f"ffprobe failed: {result.error or result.stderr}")
            return None

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ffprobe JSON: {e}")
            return None

    def get_metadata(self, file_path: Path | str) -> VideoMetadata:
        """Extract video metadata from a file.

        Args:
            file_path: Path to video/audio file

        Returns:
            VideoMetadata with extracted fields (None for unavailable fields)
        """
        probe_data = self.probe(file_path)
        if not probe_data:
            return VideoMetadata()

        metadata = VideoMetadata()

        # Extract duration from format
        format_info = probe_data.get("format", {})
        if "duration" in format_info:
            with contextlib.suppress(ValueError, TypeError):
                metadata.duration = float(format_info["duration"])

        # Extract creation_time from format tags
        tags = format_info.get("tags", {})
        metadata.creation_time = tags.get("creation_time")

        # Find video stream for dimensions, fps, codec
        streams = probe_data.get("streams", [])
        video_stream = next(
            (s for s in streams if s.get("codec_type") == "video"),
            None,
        )

        if video_stream:
            metadata.width = video_stream.get("width")
            metadata.height = video_stream.get("height")
            metadata.codec = video_stream.get("codec_name")

            # Parse frame rate (can be "30/1", "30000/1001", etc.)
            fps_str = video_stream.get("r_frame_rate") or video_stream.get(
                "avg_frame_rate"
            )
            if fps_str:
                metadata.fps = self._parse_frame_rate(fps_str)

        return metadata

    def _parse_frame_rate(self, fps_str: str) -> float | None:
        """Parse frame rate string (e.g., '30/1', '30000/1001') to float."""
        if not fps_str or fps_str == "0/0":
            return None

        try:
            if "/" in fps_str:
                num, den = fps_str.split("/")
                num, den = int(num), int(den)
                if den == 0:
                    return None
                return round(num / den, 3)
            return float(fps_str)
        except (ValueError, ZeroDivisionError):
            return None

    def get_duration(self, file_path: Path | str) -> float | None:
        """Get just the duration of a file (faster than full metadata).

        Args:
            file_path: Path to video/audio file

        Returns:
            Duration in seconds, or None if failed
        """
        args = [
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(file_path),
        ]

        result = self._run_json(args)
        if not result.success:
            return None

        try:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
