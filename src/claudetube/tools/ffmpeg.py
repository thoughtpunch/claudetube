"""
FFmpeg tool wrapper for frame extraction.
"""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

from claudetube.tools.base import ToolResult, VideoTool
from claudetube.utils.system import find_tool

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class FFmpegTool(VideoTool):
    """Wrapper for FFmpeg video processing tool."""

    @property
    def name(self) -> str:
        return "ffmpeg"

    def extract_audio(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int = 16000,
        channels: int = 1,
        bitrate: str = "128k",
    ) -> Path | None:
        """Extract audio from video/audio file to MP3.

        Optimized for speech recognition (whisper) with:
        - 16kHz sample rate (whisper native)
        - Mono channel (speech doesn't need stereo)
        - 128kbps quality (plenty for speech)

        Args:
            input_path: Path to input video/audio file
            output_path: Output MP3 path
            sample_rate: Audio sample rate (default 16000 for whisper)
            channels: Number of audio channels (default 1 for mono)
            bitrate: Audio bitrate (default "128k")

        Returns:
            Path to extracted audio, or None if failed
        """
        args = [
            "-i",
            str(input_path),
            "-vn",  # No video
            "-acodec",
            "libmp3lame",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-ab",
            bitrate,
            "-y",  # Overwrite output
            str(output_path),
        ]

        result = self._run(args)
        if not result.success:
            logger.error(f"Audio extraction failed: {result.stderr}")
            return None

        return output_path if output_path.exists() else None

    def is_available(self) -> bool:
        """Check if ffmpeg is installed."""
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
        """Get path to ffmpeg executable."""
        return find_tool("ffmpeg")

    def _run(
        self,
        args: list[str],
        timeout: int | None = None,
    ) -> ToolResult:
        """Run ffmpeg with given arguments."""
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
        except Exception as e:
            return ToolResult.from_error(str(e))

    def extract_frame(
        self,
        video_path: Path,
        output_path: Path,
        timestamp: float,
        width: int = 480,
        jpeg_quality: int = 5,
    ) -> Path | None:
        """Extract a single frame from video.

        Args:
            video_path: Path to video file
            output_path: Output JPEG path
            timestamp: Time in seconds
            width: Frame width (height auto-calculated)
            jpeg_quality: JPEG quality (2=best, 31=worst)

        Returns:
            Path to extracted frame, or None if failed
        """
        args = [
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-vf",
            f"scale={width}:-1",
            "-q:v",
            str(jpeg_quality),
            "-y",
            str(output_path),
        ]

        self._run(args)
        return output_path if output_path.exists() else None

    def extract_frames_interval(
        self,
        video_path: Path,
        output_dir: Path,
        interval: int = 30,
        width: int = 480,
        jpeg_quality: int = 8,
    ) -> list[Path]:
        """Extract frames at regular intervals.

        Args:
            video_path: Path to video file
            output_dir: Output directory for frames
            interval: Seconds between frames
            width: Frame width
            jpeg_quality: JPEG quality (2=best, 31=worst)

        Returns:
            List of extracted frame paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing frames
        existing = list(output_dir.glob("*.jpg"))
        if existing:
            return sorted(existing)

        args = [
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{interval},scale={width}:-1",
            "-q:v",
            str(jpeg_quality),
            "-vsync",
            "vfr",
            str(output_dir / "frame_%03d.jpg"),
        ]

        result = self._run(args)
        if not result.success:
            return []

        # Rename frames with timestamps
        frames = sorted(output_dir.glob("frame_*.jpg"))
        renamed = []
        for i, f in enumerate(frames):
            ts = i * interval
            ts_str = f"{int(ts // 60):02d}-{int(ts % 60):02d}"
            new_path = output_dir / f"frame_{ts_str}.jpg"
            f.rename(new_path)
            renamed.append(new_path)

        return sorted(output_dir.glob("frame_*.jpg"))

    def extract_frames_range(
        self,
        video_path: Path,
        output_dir: Path,
        start_time: float,
        duration: float,
        interval: float = 1.0,
        width: int = 480,
        jpeg_quality: int = 5,
        seek_offset: float = 0.0,
        prefix: str = "drill",
    ) -> list[Path]:
        """Extract frames for a specific time range.

        Args:
            video_path: Path to video file
            output_dir: Output directory for frames
            start_time: Start time in seconds
            duration: Duration to capture
            interval: Seconds between frames
            width: Frame width
            jpeg_quality: JPEG quality
            seek_offset: Offset for segment-relative timestamps
            prefix: Filename prefix

        Returns:
            List of extracted frame paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        current = start_time
        end_time = start_time + duration

        while current < end_time:
            ts_str = f"{int(current // 60):02d}-{int(current % 60):02d}"
            output = output_dir / f"{prefix}_{ts_str}.jpg"

            frame = self.extract_frame(
                video_path=video_path,
                output_path=output,
                timestamp=current - seek_offset,
                width=width,
                jpeg_quality=jpeg_quality,
            )

            if frame:
                frames.append(frame)
                logger.debug(f"Extracted frame at {ts_str}")

            current += interval

        return frames
