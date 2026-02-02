"""
yt-dlp tool wrapper for video downloading and metadata.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import TYPE_CHECKING

from claudetube.exceptions import DownloadError, MetadataError
from claudetube.tools.base import ToolResult, VideoTool
from claudetube.utils.system import find_tool

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Pattern to detect YouTube URLs
_YOUTUBE_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?(?:youtube\.com|youtu\.be)/", re.IGNORECASE
)


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

    def _subprocess_env(self) -> dict[str, str]:
        """Build env dict that preserves system PATH for subprocess calls.

        When running inside a venv, the subprocess inherits the venv's
        restricted PATH. yt-dlp needs access to system tools like ``deno``
        (for YouTube JS challenge solving) and ``ffmpeg``, so we merge the
        full system PATH back in.
        """
        env = os.environ.copy()
        # Ensure common system tool paths are included (Homebrew, system bin)
        system_paths = ["/usr/local/bin", "/opt/homebrew/bin", "/usr/bin", "/bin"]
        current = env.get("PATH", "")
        for p in system_paths:
            if p not in current:
                current = f"{current}:{p}"
        env["PATH"] = current
        return env

    def _run(
        self,
        args: list[str],
        timeout: int | None = None,
        retry_clients: bool = True,
    ) -> ToolResult:
        """Run yt-dlp with given arguments.

        Args:
            args: Command arguments (without the yt-dlp executable)
            timeout: Command timeout in seconds
            retry_clients: Retry with alternative YouTube clients on 403 /
                format-not-available errors
        """
        cmd = [self.get_path()] + args
        env = self._subprocess_env()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            # Handle 403 / format errors by retrying with alternative clients
            if result.returncode != 0 and retry_clients:
                stderr = result.stderr or ""
                if "403" in stderr or "Requested format is not available" in stderr:
                    logger.info(
                        "Retrying with alternative YouTube clients "
                        "(403 / format-not-available workaround)"
                    )
                    first_stderr = stderr
                    client_args = [
                        "--extractor-args",
                        "youtube:player_client=default,mweb",
                    ]
                    cmd_retry = [self.get_path()] + client_args + args
                    result = subprocess.run(
                        cmd_retry,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        env=env,
                    )
                    # If retry also failed, combine both errors for debugging
                    if result.returncode != 0:
                        retry_stderr = result.stderr or ""
                        combined = (
                            f"[First attempt] {first_stderr.strip()}\n"
                            f"[Retry with default,mweb] {retry_stderr.strip()}"
                        )
                        result = subprocess.CompletedProcess(
                            args=cmd_retry,
                            returncode=result.returncode,
                            stdout=result.stdout,
                            stderr=combined,
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

    @staticmethod
    def _is_youtube_url(url: str) -> bool:
        """Check if a URL points to YouTube."""
        return bool(_YOUTUBE_URL_RE.search(url))

    def _youtube_config_args(self) -> list[str]:
        """Build yt-dlp args from YouTube config (PO token, cookies).

        Reads optional ``youtube`` section from config YAML::

            youtube:
              po_token: "..."
              cookies_file: "/path/to/cookies.txt"

        Returns an empty list when nothing is configured.
        """
        args: list[str] = []
        try:
            from claudetube.config.loader import (
                _find_project_config,
                _get_user_config_path,
                _load_yaml_config,
            )

            yaml_config: dict | None = None
            project_path = _find_project_config()
            if project_path:
                yaml_config = _load_yaml_config(project_path)
            if yaml_config is None:
                user_path = _get_user_config_path()
                yaml_config = _load_yaml_config(user_path)
            if yaml_config is None:
                return args

            yt_cfg = yaml_config.get("youtube", {})
            if not isinstance(yt_cfg, dict):
                return args

            po_token = yt_cfg.get("po_token")
            if po_token:
                args.extend([
                    "--extractor-args",
                    f"youtube:po_token={po_token}",
                ])

            cookies_file = yt_cfg.get("cookies_file")
            if cookies_file:
                from pathlib import Path

                cookies_path = Path(cookies_file).expanduser()
                if cookies_path.exists():
                    args.extend(["--cookies", str(cookies_path)])
                else:
                    logger.warning("Cookies file not found: %s", cookies_path)
        except Exception:
            logger.debug("Failed to load YouTube config", exc_info=True)

        return args

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
            retry_clients=False,  # Metadata doesn't hit 403 issues
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
        # YouTube-specific: add extractor client args and config-driven opts
        yt_args: list[str] = []
        if self._is_youtube_url(url):
            yt_args.extend([
                "--extractor-args",
                "youtube:player_client=default,mweb",
            ])
            yt_args.extend(self._youtube_config_args())

        args = [
            *yt_args,
            "-f",
            "ba",
            "-x",
            "--audio-format",
            "mp3",
            "--audio-quality",
            quality,
            "--no-playlist",
            "--no-warnings",
            "-o",
            str(output_path),
            url,
        ]

        result = self._run(args)
        first_error = result.stderr if not result.success else ""

        # Fallback: extract from smallest video if no audio-only stream
        if not result.success or not output_path.exists():
            logger.info("No audio-only format, extracting from video...")
            args = [
                *yt_args,
                "-S",
                "+size,+br",
                "-x",
                "--audio-format",
                "mp3",
                "--audio-quality",
                quality,
                "--no-playlist",
                "--no-warnings",
                "-o",
                str(output_path),
                url,
            ]
            result = self._run(args)

        if not result.success or not output_path.exists():
            fallback_error = result.stderr[:200] if result.stderr else ""
            # Include both errors when the fallback also fails
            if first_error and fallback_error:
                error_detail = (
                    f"[audio-only] {first_error[:200]} "
                    f"[video fallback] {fallback_error}"
                )
            else:
                error_detail = fallback_error or first_error[:200]
            raise DownloadError(f"Audio download failed: {error_detail}")

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
            "--convert-thumbnails",
            "jpg",
            "--skip-download",
            "--no-playlist",
            "--no-warnings",
            "-o",
            str(output_dir / "thumbnail"),
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
            "--sub-langs",
            "en.*,en",
            "--sub-format",
            "srt/vtt/best",
            "--convert-subs",
            "srt",
            "--skip-download",
            "--no-playlist",
            "--no-warnings",
            "-o",
            str(output_dir / "%(id)s.%(ext)s"),
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
            retry_clients=False,
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
            "-f",
            format_id,
            "-x",
            "--audio-format",
            "mp3",
            "--audio-quality",
            quality,
            "--no-playlist",
            "--no-warnings",
            "-o",
            str(output_path),
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
            "-S",
            quality_sort,
            "-N",
            str(concurrent_fragments),
            "--download-sections",
            section_spec,
            "--force-keyframes-at-cuts",
            "--no-playlist",
            "--no-warnings",
            "--merge-output-format",
            "mp4",
            "-o",
            str(output_path),
            url,
        ]

        self._run(args)
        return output_path if output_path.exists() else None
