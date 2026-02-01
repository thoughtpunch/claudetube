"""
yt-dlp tool wrapper for video downloading and metadata.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from claudetube.exceptions import DownloadError, MetadataError
from claudetube.tools.base import ToolResult, VideoTool
from claudetube.utils.system import find_tool

logger = logging.getLogger(__name__)


class YtDlpTool(VideoTool):
    """Wrapper for yt-dlp video download tool."""

    @property
    def name(self) -> str:
        return "yt-dlp"

    def is_available(self) -> bool:
        """Check if yt-dlp is installed."""
        try:
            path = self.get_path()
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_path(self) -> str:
        """Get path to yt-dlp executable."""
        return find_tool("yt-dlp")

    def _run(
        self,
        args: list[str],
        timeout: int | None = None,
        retry_mweb: bool = True,
    ) -> ToolResult:
        """Run yt-dlp with given arguments.

        Args:
            args: Command arguments (without the yt-dlp executable)
            timeout: Command timeout in seconds
            retry_mweb: Retry with mweb client on 403 errors
        """
        cmd = [self.get_path()] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # Handle 403 errors by retrying with mweb client
            if result.returncode != 0 and retry_mweb:
                stderr = result.stderr or ""
                if "403" in stderr:
                    logger.info("Retrying with YouTube mweb client (403 workaround)")
                    mweb_args = ["--extractor-args", "youtube:player_client=mweb"]
                    cmd_retry = [self.get_path()] + mweb_args + args
                    result = subprocess.run(
                        cmd_retry,
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

    def get_metadata(self, url: str, timeout: int = 30) -> dict:
        """Fetch video metadata without downloading.

        Returns:
            Dict with video metadata

        Raises:
            MetadataError: If metadata fetch fails
        """
        result = self._run(
            ["--dump-json", "--no-download", url],
            timeout=timeout,
            retry_mweb=False,  # Metadata doesn't hit 403 issues
        )

        if not result.success:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            if "ERROR:" in error_msg:
                error_msg = error_msg.split("ERROR:")[-1].strip()
            raise MetadataError(error_msg[:500])

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise MetadataError(f"Invalid JSON response: {e}") from e

    def download_audio(
        self,
        url: str,
        output_path: Path,
        quality: str = "64K",
    ) -> Path:
        """Download audio from video.

        Args:
            url: Video URL
            output_path: Output file path
            quality: Audio quality (e.g., "64K", "128K")

        Returns:
            Path to downloaded audio file

        Raises:
            DownloadError: If download fails
        """
        args = [
            "-f", "ba",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", quality,
            "--no-playlist",
            "--no-warnings",
            "-o", str(output_path),
            url,
        ]

        result = self._run(args)

        # Fallback: extract from smallest video if no audio-only stream
        if not result.success or not output_path.exists():
            logger.info("No audio-only format, extracting from video...")
            args = [
                "-S", "+size,+br",
                "-x",
                "--audio-format", "mp3",
                "--audio-quality", quality,
                "--no-playlist",
                "--no-warnings",
                "-o", str(output_path),
                url,
            ]
            result = self._run(args)

        if not result.success or not output_path.exists():
            raise DownloadError(f"Audio download failed: {result.stderr[:200]}")

        return output_path

    def download_thumbnail(
        self,
        url: str,
        output_dir: Path,
        timeout: int = 15,
    ) -> Path | None:
        """Download video thumbnail.

        Returns:
            Path to thumbnail file, or None if not available
        """
        args = [
            "--write-thumbnail",
            "--convert-thumbnails", "jpg",
            "--skip-download",
            "--no-playlist",
            "--no-warnings",
            "-o", str(output_dir / "thumbnail"),
            url,
        ]

        try:
            result = self._run(args, timeout=timeout)
            if not result.success:
                return None

            # Find the thumbnail file (yt-dlp may add extension)
            thumb_path = output_dir / "thumbnail.jpg"
            for ext in ["jpg", "webp", "png"]:
                candidate = output_dir / f"thumbnail.{ext}"
                if candidate.exists() and ext != "jpg":
                    candidate.rename(thumb_path)
                    break

            return thumb_path if thumb_path.exists() else None

        except Exception:
            return None

    def fetch_subtitles(
        self,
        url: str,
        output_dir: Path,
        timeout: int = 30,
    ) -> dict | None:
        """Fetch subtitles from video source.

        Returns:
            Dict with 'srt', 'txt', 'source' keys, or None if not available
        """
        import re

        args = [
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs", "en.*,en",
            "--sub-format", "srt/vtt/best",
            "--convert-subs", "srt",
            "--skip-download",
            "--no-playlist",
            "--no-warnings",
            "-o", str(output_dir / "%(id)s.%(ext)s"),
            url,
        ]

        try:
            result = self._run(args, timeout=timeout)
            if not result.success:
                return None
        except Exception:
            return None

        # Look for subtitle files
        sub_files = sorted(output_dir.glob("*.srt"))
        sub_files = [f for f in sub_files if f.name != "audio.srt"]

        if not sub_files:
            return None

        sub_file = sub_files[0]
        raw = sub_file.read_text(errors="replace")

        # Determine source type from filename
        source = "auto-generated" if ".auto." in sub_file.name else "uploaded"

        # Clean HTML tags from auto-generated subs
        raw = re.sub(r"<[^>]+>", "", raw)

        # Parse SRT to extract plain text
        txt_lines = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or re.match(r"^\d+$", line) or "-->" in line:
                continue
            if line not in txt_lines[-1:]:
                txt_lines.append(line)

        if not txt_lines:
            for f in output_dir.glob("*.srt"):
                if f.name != "audio.srt":
                    f.unlink()
            return None

        # Clean up downloaded sub files
        for f in sub_files:
            f.unlink()

        return {
            "srt": raw,
            "txt": "\n".join(txt_lines),
            "source": source,
        }

    def get_formats(self, url: str, timeout: int = 30) -> list[dict]:
        """Fetch available formats for a video.

        Returns:
            List of format dicts from yt-dlp

        Raises:
            MetadataError: If format listing fails
        """
        result = self._run(
            ["-j", "--no-download", url],
            timeout=timeout,
            retry_mweb=False,
        )

        if not result.success:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            if "ERROR:" in error_msg:
                error_msg = error_msg.split("ERROR:")[-1].strip()
            raise MetadataError(error_msg[:500])

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise MetadataError(f"Invalid JSON response: {e}") from e

        return data.get("formats", [])

    def check_audio_description(self, url: str, timeout: int = 30) -> dict | None:
        """Check if the video has an audio description (AD) track.

        Detects AD tracks by looking for keywords in format_note
        and format_id fields.

        Args:
            url: Video URL
            timeout: Timeout for the metadata fetch

        Returns:
            Format dict for the AD track if found, or None
        """
        formats = self.get_formats(url, timeout=timeout)
        ad_indicators = ["description", "descriptive", "ad", "dvs"]

        for fmt in formats:
            note = fmt.get("format_note", "").lower()
            format_id = fmt.get("format_id", "").lower()
            language = fmt.get("language", "").lower() if fmt.get("language") else ""

            # Check format_note for AD indicators
            if any(ind in note for ind in ad_indicators):
                return fmt

            # Check format_id for AD indicators
            if any(ind in format_id for ind in ad_indicators):
                return fmt

            # Check language field (some platforms use language tags like "en-ad")
            if "ad" in language.split("-") or "descriptive" in language:
                return fmt

        return None

    def download_audio_description(
        self,
        url: str,
        output_path: Path,
        format_id: str | None = None,
        quality: str = "64K",
    ) -> Path:
        """Download the audio description track from a video.

        Args:
            url: Video URL
            output_path: Output file path
            format_id: Specific format ID to download (from check_audio_description).
                        If None, will auto-detect the AD track.
            quality: Audio quality (e.g., "64K", "128K")

        Returns:
            Path to downloaded audio file

        Raises:
            DownloadError: If no AD track found or download fails
        """
        if format_id is None:
            ad_format = self.check_audio_description(url)
            if ad_format is None:
                raise DownloadError("No audio description track found")
            format_id = ad_format["format_id"]

        args = [
            "-f", format_id,
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", quality,
            "--no-playlist",
            "--no-warnings",
            "-o", str(output_path),
            url,
        ]

        result = self._run(args)

        if not result.success or not output_path.exists():
            raise DownloadError(
                f"Audio description download failed: {result.stderr[:200] if result.stderr else 'Unknown error'}"
            )

        return output_path

    def download_video_segment(
        self,
        url: str,
        output_path: Path,
        start_time: float,
        end_time: float,
        quality_sort: str = "+res,+size,+br,+fps",
        concurrent_fragments: int = 1,
    ) -> Path | None:
        """Download a video segment for frame extraction.

        Args:
            url: Video URL
            output_path: Output file path
            start_time: Start time in seconds
            end_time: End time in seconds
            quality_sort: yt-dlp format sort string
            concurrent_fragments: Number of concurrent download fragments

        Returns:
            Path to downloaded video, or None if failed
        """
        section_spec = f"*{start_time}-{end_time}"

        args = [
            "-S", quality_sort,
            "-N", str(concurrent_fragments),
            "--download-sections", section_spec,
            "--force-keyframes-at-cuts",
            "--no-playlist",
            "--no-warnings",
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            url,
        ]

        self._run(args)
        return output_path if output_path.exists() else None
