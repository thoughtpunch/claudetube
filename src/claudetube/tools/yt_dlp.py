"""
yt-dlp tool wrapper for video downloading and metadata.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from claudetube.exceptions import (
    AgeRestrictedError,
    DownloadError,
    ExtractorError,
    FormatNotAvailableError,
    GeoRestrictedError,
    MetadataError,
    NetworkError,
    RateLimitError,
    VideoUnavailableError,
    YouTubeAuthError,
)
from claudetube.tools.base import ToolResult, VideoTool
from claudetube.utils.system import find_tool

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured Output Types
# ---------------------------------------------------------------------------


@dataclass
class DownloadProgress:
    """Structured progress data from yt-dlp --progress-template.

    Supports both download progress and postprocessor progress.

    Download status values: "downloading", "finished", "error"
    Postprocessor status values: "started", "processing", "finished"
    """

    status: str  # "downloading", "finished", "error", "started", "processing"
    percent: float | None = None  # 0.0-100.0
    speed: str | None = None  # Human-readable speed, e.g., "1.5MiB/s"
    eta: str | None = None  # Human-readable ETA, e.g., "00:30"
    downloaded_bytes: int | None = None
    total_bytes: int | None = None
    fragment_index: int | None = None
    fragment_count: int | None = None
    filename: str | None = None
    # Postprocessor-specific fields
    postprocessor: str | None = None  # Name of postprocessor, e.g., "FFmpegExtractAudio"
    info_dict: dict | None = None  # For accessing other fields like title

    @classmethod
    def from_json_line(cls, line: str) -> DownloadProgress | None:
        """Parse a JSON progress line from yt-dlp --progress-template output.

        Args:
            line: A single line from yt-dlp stdout (may or may not be JSON)

        Returns:
            DownloadProgress if line is valid progress JSON, None otherwise
        """
        line = line.strip()
        if not line.startswith("{"):
            return None

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        # Must have at least a status field to be valid progress
        if "status" not in data:
            return None

        # Parse percent from string like "  5.0%" or "100%"
        percent = None
        if "percent" in data and data["percent"]:
            with contextlib.suppress(ValueError, AttributeError):
                percent = float(str(data["percent"]).strip().rstrip("%"))

        # Parse downloaded/total bytes
        downloaded_bytes = None
        total_bytes = None
        if "downloaded_bytes" in data:
            with contextlib.suppress(ValueError, TypeError):
                downloaded_bytes = int(data["downloaded_bytes"])
        if "total_bytes" in data:
            with contextlib.suppress(ValueError, TypeError):
                total_bytes = int(data["total_bytes"])

        # Parse fragment info
        fragment_index = None
        fragment_count = None
        if "fragment_index" in data:
            with contextlib.suppress(ValueError, TypeError):
                fragment_index = int(data["fragment_index"])
        if "fragment_count" in data:
            with contextlib.suppress(ValueError, TypeError):
                fragment_count = int(data["fragment_count"])

        return cls(
            status=data.get("status", "unknown"),
            percent=percent,
            speed=data.get("speed"),
            eta=data.get("eta"),
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes,
            fragment_index=fragment_index,
            fragment_count=fragment_count,
            filename=data.get("filename"),
            postprocessor=data.get("postprocessor"),
            info_dict=data.get("info_dict"),
        )


class ProgressCallback(Protocol):
    """Protocol for progress callback functions."""

    def __call__(self, progress: DownloadProgress) -> None:
        """Called with progress updates during download."""
        ...


@dataclass
class YtDlpError:
    """Structured error information parsed from yt-dlp stderr.

    Attributes:
        category: Error classification (e.g., "auth", "unavailable", "geo", "network")
        message: The original error message from yt-dlp
        stderr: Full stderr output for debugging
        details: Additional parsed details (error codes, URLs, etc.)
        warnings: List of warning messages extracted from stderr
    """

    category: str
    message: str
    stderr: str
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# Error patterns for classification
# Each tuple: (pattern, category, detail_extractor)
# detail_extractor is optional function to extract extra info from match
_ERROR_PATTERNS: list[tuple[re.Pattern, str, Callable[[re.Match], dict] | None]] = [
    # Authentication / access errors
    (
        re.compile(r"HTTP Error 403", re.IGNORECASE),
        "auth",
        None,
    ),
    (
        re.compile(r"Sign in to confirm your age|age.restricted", re.IGNORECASE),
        "age_restricted",
        None,
    ),
    (
        re.compile(r"private video|video is private", re.IGNORECASE),
        "private",
        None,
    ),
    (
        re.compile(r"members.only|subscriber.only", re.IGNORECASE),
        "members_only",
        None,
    ),
    (
        re.compile(r"PO Token|po_token|Proof of Origin", re.IGNORECASE),
        "po_token",
        None,
    ),
    # Availability errors
    (
        re.compile(
            r"Video unavailable|This video is unavailable|removed by the uploader",
            re.IGNORECASE,
        ),
        "unavailable",
        None,
    ),
    (
        re.compile(r"copyright|blocked.*copyright", re.IGNORECASE),
        "copyright",
        None,
    ),
    (
        re.compile(r"This video has been removed", re.IGNORECASE),
        "removed",
        None,
    ),
    (
        re.compile(r"premiere|Premieres in", re.IGNORECASE),
        "premiere",
        None,
    ),
    (
        re.compile(r"live event|live stream", re.IGNORECASE),
        "live",
        None,
    ),
    # Geo restriction
    (
        re.compile(
            r"not available in your country|geo.?restrict|blocked in your country",
            re.IGNORECASE,
        ),
        "geo_restricted",
        None,
    ),
    # Format errors
    (
        re.compile(r"Requested format is not available", re.IGNORECASE),
        "format_unavailable",
        None,
    ),
    (
        re.compile(r"No video formats found", re.IGNORECASE),
        "no_formats",
        None,
    ),
    # Network errors
    (
        re.compile(r"Connection reset|Connection refused|Connection timed out", re.IGNORECASE),
        "network",
        None,
    ),
    (
        re.compile(r"SSL.*error|certificate verify failed", re.IGNORECASE),
        "ssl",
        None,
    ),
    # Rate limiting (must come before generic HTTP error to catch 429 specifically)
    (
        re.compile(r"429|too many requests|rate.?limit", re.IGNORECASE),
        "rate_limited",
        None,
    ),
    (
        re.compile(r"HTTP Error (\d+)", re.IGNORECASE),
        "http_error",
        lambda m: {"http_code": int(m.group(1))},
    ),
    # Invalid input
    (
        re.compile(r"is not a valid URL|Unsupported URL", re.IGNORECASE),
        "invalid_url",
        None,
    ),
    (
        re.compile(r"No video could be found", re.IGNORECASE),
        "not_found",
        None,
    ),
]


# Patterns for extracting warnings and diagnostic info
_WARNING_PATTERN = re.compile(r"WARNING:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_CLIENT_PATTERN = re.compile(r"\[youtube\].*?Extracting.*?client:\s*(\w+)", re.IGNORECASE)
_SABR_PATTERN = re.compile(r"SABR|Server ABR|serverAbrStreamingUrl", re.IGNORECASE)
_PO_TOKEN_WARNING_PATTERN = re.compile(
    r"(?:PO Token|po_token|Proof of Origin).*?(?:required|needed|missing|invalid)",
    re.IGNORECASE,
)
_NSIG_PATTERN = re.compile(r"nsig.*?(?:function|extract|failed)", re.IGNORECASE)
_PLAYER_PATTERN = re.compile(
    r"player.*?(?:response|error|failed)|Unable to extract.*?player", re.IGNORECASE
)


def _extract_warnings(stderr: str) -> list[str]:
    """Extract warning messages from yt-dlp stderr.

    Args:
        stderr: Full stderr output from yt-dlp

    Returns:
        List of warning message strings
    """
    warnings = []
    for match in _WARNING_PATTERN.finditer(stderr):
        warning_text = match.group(1).strip()
        if warning_text and warning_text not in warnings:
            warnings.append(warning_text)
    return warnings


def _extract_clients_tried(stderr: str) -> list[str]:
    """Extract YouTube client names that were tried from yt-dlp output.

    yt-dlp logs which clients it attempts (default, mweb, android, ios, etc.)
    when extracting video info.

    Args:
        stderr: Full stderr output from yt-dlp

    Returns:
        List of client names tried (e.g., ["default", "mweb", "android_vr"])
    """
    clients = []
    for match in _CLIENT_PATTERN.finditer(stderr):
        client = match.group(1).lower()
        if client not in clients:
            clients.append(client)
    return clients


def _extract_diagnostic_details(stderr: str) -> dict[str, Any]:
    """Extract additional diagnostic details from yt-dlp stderr.

    Looks for:
    - SABR/Server ABR warnings (YouTube's adaptive streaming)
    - PO Token issues
    - nsig extraction failures
    - Player extraction errors

    Args:
        stderr: Full stderr output from yt-dlp

    Returns:
        Dict with diagnostic flags and details
    """
    details: dict[str, Any] = {}

    # Check for SABR (Server ABR) issues
    if _SABR_PATTERN.search(stderr):
        details["sabr_detected"] = True
        details["sabr_note"] = (
            "YouTube is using Server ABR streaming. "
            "This requires proper authentication to access."
        )

    # Check for PO Token warnings
    if _PO_TOKEN_WARNING_PATTERN.search(stderr):
        details["po_token_issue"] = True
        details["po_token_note"] = (
            "A valid PO (Proof of Origin) token is required. "
            "Configure cookies and PO token in config.yaml."
        )

    # Check for nsig extraction failures
    if _NSIG_PATTERN.search(stderr):
        details["nsig_issue"] = True
        details["nsig_note"] = (
            "Failed to extract signature. This often indicates "
            "YouTube has updated their player. Try: pip install -U yt-dlp"
        )

    # Check for player extraction issues
    if _PLAYER_PATTERN.search(stderr):
        details["player_issue"] = True
        details["player_note"] = (
            "Failed to extract player information. "
            "Ensure deno is installed for JS challenge solving."
        )

    return details


def parse_yt_dlp_error(stderr: str) -> YtDlpError:
    """Parse yt-dlp stderr output into structured error information.

    Extracts:
    - Error category (auth, geo_restricted, unavailable, etc.)
    - Main error message
    - Warning messages
    - Diagnostic details (SABR, PO token, nsig issues)
    - HTTP status codes when present

    Args:
        stderr: The stderr output from a failed yt-dlp command

    Returns:
        YtDlpError with category, message, details, and warnings
    """
    # Extract the main ERROR message if present
    error_match = re.search(r"ERROR:\s*(.+?)(?:\n|$)", stderr)
    message = error_match.group(1).strip() if error_match else stderr.strip()

    # Extract warnings
    warnings = _extract_warnings(stderr)

    # Extract diagnostic details
    diagnostic_details = _extract_diagnostic_details(stderr)

    # Try each pattern to classify the error
    for pattern, category, detail_extractor in _ERROR_PATTERNS:
        match = pattern.search(stderr)
        if match:
            details = detail_extractor(match) if detail_extractor else {}
            # Merge diagnostic details
            details.update(diagnostic_details)
            return YtDlpError(
                category=category,
                message=message,
                stderr=stderr,
                details=details,
                warnings=warnings,
            )

    # Default to "unknown" category
    return YtDlpError(
        category="unknown",
        message=message,
        stderr=stderr,
        details=diagnostic_details,
        warnings=warnings,
    )


def yt_dlp_error_to_exception(
    error: YtDlpError,
    *,
    is_youtube: bool = False,
    auth_status: dict[str, Any] | None = None,
    clients_tried: list[str] | None = None,
) -> DownloadError:
    """Convert a YtDlpError to a typed DownloadError exception.

    Args:
        error: The parsed YtDlpError
        is_youtube: Whether this is a YouTube URL (for auth-specific handling)
        auth_status: YouTube auth status dict (from check_youtube_auth_status)
        clients_tried: List of YouTube clients that were attempted

    Returns:
        Appropriate typed exception based on error category
    """
    category = error.category
    message = error.message
    stderr = error.stderr
    details = dict(error.details)  # Copy to avoid mutation

    # Add warnings to details if present
    if error.warnings:
        details["warnings"] = error.warnings

    # Extract clients tried from stderr if not provided
    if clients_tried is None and is_youtube:
        clients_tried = _extract_clients_tried(stderr)

    if clients_tried:
        details["clients_tried"] = clients_tried

    # Map categories to typed exceptions
    if category in ("auth", "po_token"):
        auth_level = auth_status.get("auth_level") if auth_status else None
        return YouTubeAuthError(
            message,
            stderr=stderr,
            details=details,
            auth_level=auth_level,
            clients_tried=clients_tried,
        )

    if category == "age_restricted":
        return AgeRestrictedError(
            message,
            stderr=stderr,
            details=details,
        )

    if category == "geo_restricted":
        return GeoRestrictedError(
            message,
            stderr=stderr,
            details=details,
        )

    if category in ("format_unavailable", "no_formats"):
        return FormatNotAvailableError(
            message,
            stderr=stderr,
            details=details,
        )

    if category == "rate_limited":
        return RateLimitError(
            message,
            stderr=stderr,
            details=details,
        )

    if category in ("network", "ssl"):
        http_code = details.get("http_code")
        return NetworkError(
            message,
            stderr=stderr,
            details=details,
            http_code=http_code,
        )

    if category == "http_error":
        http_code = details.get("http_code")
        # HTTP 403 on YouTube is usually auth-related
        if http_code == 403 and is_youtube:
            auth_level = auth_status.get("auth_level") if auth_status else None
            return YouTubeAuthError(
                message,
                stderr=stderr,
                details=details,
                auth_level=auth_level,
                clients_tried=clients_tried,
            )
        return NetworkError(
            message,
            stderr=stderr,
            details=details,
            http_code=http_code,
        )

    if category in (
        "unavailable",
        "removed",
        "private",
        "members_only",
        "copyright",
        "premiere",
        "live",
    ):
        return VideoUnavailableError(
            message,
            stderr=stderr,
            details=details,
            reason=category,
        )

    if category in ("invalid_url", "not_found"):
        return ExtractorError(
            message,
            stderr=stderr,
            details=details,
        )

    # Default: generic DownloadError with parsed info
    return DownloadError(
        message,
        category=category,
        stderr=stderr,
        details=details,
    )


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

    _deno_checked: bool = False

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

        # One-time check: warn if deno is not discoverable
        if not YtDlpTool._deno_checked:
            YtDlpTool._deno_checked = True
            if not shutil.which("deno", path=current):
                logger.info(
                    "deno not found on PATH. Since yt-dlp 2026.01.29, deno is "
                    "required for full YouTube support (JS challenge solving). "
                    "Install: brew install deno (macOS) or visit https://deno.land"
                )

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
            # NOTE: We try android_vr and web_safari as fallbacks since these
            # often work when default/mweb fail with 403 (they use different
            # auth flows that don't require PO tokens in some cases).
            if result.returncode != 0 and retry_clients:
                stderr = result.stderr or ""
                if "403" in stderr or "Requested format is not available" in stderr:
                    logger.info(
                        "Retrying with alternative YouTube clients "
                        "(403 / format-not-available workaround)"
                    )
                    first_stderr = stderr

                    # Extract clients tried from first attempt
                    first_clients = _extract_clients_tried(first_stderr) or ["default"]

                    # Try android_vr and web_safari - these often work without PO tokens
                    client_args = [
                        "--extractor-args",
                        "youtube:player_client=android_vr,web_safari",
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
                        retry_clients_list = _extract_clients_tried(retry_stderr) or [
                            "android_vr",
                            "web_safari",
                        ]

                        # Build structured error with all clients tried
                        all_clients = list(
                            dict.fromkeys(first_clients + retry_clients_list)
                        )

                        # Append actionable auth diagnostic for 403 errors
                        auth_guidance = ""
                        if "403" in first_stderr or "403" in retry_stderr:
                            try:
                                auth_guidance = self.format_auth_error_message()
                            except Exception:
                                auth_guidance = (
                                    "\nYouTube download failed (403). "
                                    "See: documentation/guides/youtube-auth.md"
                                )

                        # Format combined error with diagnostic info
                        combined_parts = [
                            f"[Clients tried: {', '.join(all_clients)}]",
                            "",
                            f"[First attempt] {first_stderr.strip()}",
                            "",
                            f"[Retry with android_vr,web_safari] {retry_stderr.strip()}",
                        ]

                        # Add warnings summary if present
                        first_warnings = _extract_warnings(first_stderr)
                        retry_warnings = _extract_warnings(retry_stderr)
                        all_warnings = list(
                            dict.fromkeys(first_warnings + retry_warnings)
                        )
                        if all_warnings:
                            combined_parts.append("")
                            combined_parts.append("[Warnings]")
                            for w in all_warnings:
                                combined_parts.append(f"  - {w}")

                        combined_parts.append(auth_guidance)

                        combined = "\n".join(combined_parts)
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

    # JSON progress templates for structured output parsing.
    # Uses yt-dlp's --progress-template to output JSON lines during download.
    # Fields: status, percent, speed, eta, downloaded_bytes, total_bytes,
    #         fragment_index, fragment_count, filename
    _PROGRESS_TEMPLATE_DOWNLOAD = (
        '{"status":"%(progress.status)s",'
        '"percent":"%(progress._percent_str)s",'
        '"speed":"%(progress._speed_str)s",'
        '"eta":"%(progress._eta_str)s",'
        '"downloaded_bytes":%(progress.downloaded_bytes|0)s,'
        '"total_bytes":%(progress.total_bytes|0)s,'
        '"fragment_index":%(progress.fragment_index|0)s,'
        '"fragment_count":%(progress.fragment_count|0)s,'
        '"filename":"%(info.filename)s"}'
    )

    # Postprocessor progress template for operations like MP3 conversion.
    # status: "started", "processing", "finished"
    # postprocessor: name like "FFmpegExtractAudio", "FFmpegVideoConvertor"
    _PROGRESS_TEMPLATE_POSTPROCESS = (
        '{"status":"%(progress.status)s",'
        '"postprocessor":"%(progress.postprocessor)s",'
        '"filename":"%(info.filename)s"}'
    )

    def _run_with_progress(
        self,
        args: list[str],
        on_progress: ProgressCallback | None = None,
        timeout: int | None = None,
    ) -> ToolResult:
        """Run yt-dlp with structured progress output.

        Uses --progress-template to emit JSON progress lines that are parsed
        and passed to the callback. This allows real-time progress tracking
        without parsing human-readable output.

        Supports both download progress and postprocessor progress:
        - Download: status="downloading"|"finished"|"error", percent, speed, eta, etc.
        - Postprocessor: status="started"|"processing"|"finished", postprocessor name

        Args:
            args: Command arguments (without the yt-dlp executable)
            on_progress: Callback invoked with DownloadProgress for each update
            timeout: Command timeout in seconds

        Returns:
            ToolResult with success status and captured output
        """
        # Add progress template args for both download and postprocessor stages
        progress_args = [
            "--progress-template",
            f"download:{self._PROGRESS_TEMPLATE_DOWNLOAD}",
            "--progress-template",
            f"postprocess:{self._PROGRESS_TEMPLATE_POSTPROCESS}",
            "--newline",  # Ensure each progress update is on its own line
        ]
        cmd = [self.get_path()] + progress_args + args
        env = self._subprocess_env()

        try:
            # Use Popen to read output line-by-line for progress parsing
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            stdout_lines: list[str] = []
            stderr_content = ""

            # Read stdout line by line for progress updates
            if process.stdout:
                for line in process.stdout:
                    stdout_lines.append(line)

                    # Try to parse as progress JSON
                    if on_progress:
                        progress = DownloadProgress.from_json_line(line)
                        if progress:
                            # Don't let callback errors stop the download
                            with contextlib.suppress(Exception):
                                on_progress(progress)

            # Wait for completion and get stderr
            _, stderr_content = process.communicate(timeout=timeout)
            returncode = process.returncode

            return ToolResult(
                success=returncode == 0,
                stdout="".join(stdout_lines),
                stderr=stderr_content,
                returncode=returncode,
            )

        except subprocess.TimeoutExpired:
            process.kill()
            return ToolResult.from_error(f"Timeout after {timeout}s")
        except Exception as e:
            return ToolResult.from_error(str(e))

    def parse_error(self, stderr: str) -> YtDlpError:
        """Parse stderr output into structured error information.

        This is a convenience method that wraps the module-level
        parse_yt_dlp_error function.

        Args:
            stderr: The stderr output from a failed yt-dlp command

        Returns:
            YtDlpError with category, message, and details
        """
        return parse_yt_dlp_error(stderr)

    @staticmethod
    def _is_youtube_url(url: str) -> bool:
        """Check if a URL points to YouTube."""
        return bool(_YOUTUBE_URL_RE.search(url))

    # Browsers supported by yt-dlp's --cookies-from-browser flag
    _SUPPORTED_BROWSERS = frozenset(
        {
            "brave",
            "chrome",
            "chromium",
            "edge",
            "firefox",
            "opera",
            "safari",
            "vivaldi",
            "whale",
        }
    )

    def _youtube_config_args(self) -> list[str]:
        """Build yt-dlp args from YouTube config (cookies, PO token, bgutil).

        Reads optional ``youtube`` section from config YAML::

            youtube:
              # Cookie source (pick one; first match wins):
              cookies_from_browser: "firefox"
              cookies_file: "/path/to/cookies.txt"

              # PO token (CLIENT.TYPE+TOKEN or bare token)
              po_token: "mweb.gvs+TOKEN_VALUE_HERE"

              # bgutil-ytdlp-pot-provider
              pot_server_url: "http://127.0.0.1:4416"
              pot_script_path: "/path/to/generate_once.js"

        Cookie source priority: cookies_from_browser > cookies_file.
        Only one cookie source is used (they are mutually exclusive in yt-dlp).

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

            # --- Cookie source (mutually exclusive, first wins) ---
            cookie_source_set = False

            cookies_from_browser = yt_cfg.get("cookies_from_browser")
            if cookies_from_browser:
                browser = str(cookies_from_browser).lower().strip()
                if browser in self._SUPPORTED_BROWSERS:
                    args.extend(["--cookies-from-browser", browser])
                    cookie_source_set = True
                else:
                    logger.warning(
                        "Unsupported browser for cookie extraction: %r. Supported: %s",
                        browser,
                        ", ".join(sorted(self._SUPPORTED_BROWSERS)),
                    )

            if not cookie_source_set:
                cookies_file = yt_cfg.get("cookies_file")
                if cookies_file:
                    from pathlib import Path

                    cookies_path = Path(cookies_file).expanduser()
                    if cookies_path.exists():
                        args.extend(["--cookies", str(cookies_path)])
                        cookie_source_set = True
                    else:
                        logger.warning("Cookies file not found: %s", cookies_path)

            # --- PO token ---
            po_token = yt_cfg.get("po_token")
            if po_token:
                args.extend(
                    [
                        "--extractor-args",
                        f"youtube:po_token={po_token}",
                    ]
                )

            # --- bgutil-ytdlp-pot-provider: HTTP server URL ---
            pot_server_url = yt_cfg.get("pot_server_url")
            if pot_server_url:
                args.extend(
                    [
                        "--extractor-args",
                        f"youtubepot-bgutilhttp:base_url={pot_server_url}",
                    ]
                )

            # --- bgutil-ytdlp-pot-provider: script path (fallback mode) ---
            pot_script_path = yt_cfg.get("pot_script_path")
            if pot_script_path:
                from pathlib import Path

                script_path = Path(pot_script_path).expanduser()
                if script_path.exists():
                    args.extend(
                        [
                            "--extractor-args",
                            f"youtubepot-bgutilscript:script_path={script_path}",
                        ]
                    )
                else:
                    logger.warning("bgutil script not found: %s", script_path)
        except Exception:
            logger.debug("Failed to load YouTube config", exc_info=True)

        return args

    def check_pot_providers(self, timeout: int = 10) -> list[str]:
        """Check which PO Token providers yt-dlp has loaded.

        Runs ``yt-dlp --verbose`` against a dummy YouTube URL (no download)
        and parses the debug output for registered POT providers.

        Returns:
            List of provider strings (e.g. ``["bgutil:http-1.2.2 (external)"]``),
            or an empty list if none are detected.
        """
        result = self._run(
            [
                "--verbose",
                "--skip-download",
                "--no-warnings",
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            ],
            timeout=timeout,
            retry_clients=False,
        )
        providers: list[str] = []
        output = (result.stderr or "") + (result.stdout or "")
        for line in output.splitlines():
            if "PO Token Providers:" in line:
                # Format: [pot] PO Token Providers: bgutil:http-1.2.2 (...), ...
                after = line.split("PO Token Providers:", 1)[1].strip()
                if after and after.lower() != "none":
                    providers = [p.strip() for p in after.split(",") if p.strip()]
                break
        return providers

    def _load_youtube_config(self) -> dict[str, Any]:
        """Load the youtube section from config YAML.

        Returns:
            The ``youtube`` config dict, or empty dict if not found.
        """
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
                return {}

            yt_cfg = yaml_config.get("youtube", {})
            return yt_cfg if isinstance(yt_cfg, dict) else {}
        except Exception:
            return {}

    def check_youtube_auth_status(self) -> dict[str, Any]:
        """Comprehensive YouTube authentication health check.

        Inspects deno availability, PO token providers, cookie config,
        and PO token config to report a structured diagnostic dict.

        Returns:
            Dict with keys:
                deno_available, deno_version, pot_plugin_loaded,
                pot_plugin_version, pot_server_reachable,
                cookies_configured, cookies_source, po_token_configured,
                po_token_type, auth_level (0-4), recommendations.
        """
        env = self._subprocess_env()
        path_str = env.get("PATH", "")

        # --- deno ---
        deno_available = False
        deno_version: str | None = None
        deno_path = shutil.which("deno", path=path_str)
        if deno_path:
            try:
                proc = subprocess.run(
                    [deno_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc.returncode == 0:
                    deno_available = True
                    # First line is typically "deno 2.x.x ..."
                    for line in proc.stdout.splitlines():
                        if line.strip().startswith("deno"):
                            deno_version = (
                                line.strip().split()[1]
                                if len(line.strip().split()) > 1
                                else line.strip()
                            )
                            break
            except Exception:
                pass

        # --- PO token providers (yt-dlp plugins) ---
        pot_plugin_loaded = False
        pot_plugin_version: str | None = None
        try:
            providers = self.check_pot_providers(timeout=10)
            if providers:
                pot_plugin_loaded = True
                # Extract version from e.g. "bgutil:http-1.2.2 (external)"
                for p in providers:
                    match = re.search(r"(\d+\.\d+\.\d+)", p)
                    if match:
                        pot_plugin_version = match.group(1)
                        break
        except Exception:
            pass

        # --- Config inspection ---
        yt_cfg = self._load_youtube_config()

        # Cookie config
        cookies_configured = False
        cookies_source: str | None = None

        cookies_from_browser = yt_cfg.get("cookies_from_browser")
        if cookies_from_browser:
            browser = str(cookies_from_browser).lower().strip()
            if browser in self._SUPPORTED_BROWSERS:
                cookies_configured = True
                cookies_source = f"browser:{browser}"

        if not cookies_configured:
            cookies_file = yt_cfg.get("cookies_file")
            if cookies_file:
                from pathlib import Path as _Path

                cookies_path = _Path(cookies_file).expanduser()
                if cookies_path.exists():
                    cookies_configured = True
                    cookies_source = f"file:{cookies_path}"
                else:
                    cookies_source = f"file:{cookies_path} (NOT FOUND)"

        # PO token config
        po_token_configured = False
        po_token_type: str | None = None
        po_token_value = yt_cfg.get("po_token")
        if po_token_value:
            po_token_configured = True
            # Parse type from "mweb.gvs+TOKEN" or "web.subs+TOKEN"
            if "+" in str(po_token_value):
                po_token_type = str(po_token_value).split("+", 1)[0]

        # POT server reachability
        pot_server_reachable: bool | None = None
        pot_server_url = yt_cfg.get("pot_server_url")
        if pot_server_url:
            try:
                ping_url = f"{pot_server_url.rstrip('/')}/ping"
                req = urllib.request.Request(ping_url, method="GET")
                with urllib.request.urlopen(req, timeout=3) as resp:
                    pot_server_reachable = resp.status == 200
            except Exception:
                pot_server_reachable = False

        # --- Auth level ---
        # Level 0: Nothing configured
        # Level 1: Deno only
        # Level 2: Cookies + deno
        # Level 3: Manual PO token + cookies + deno
        # Level 4: bgutil server + cookies + deno
        auth_level = 0
        if deno_available:
            auth_level = 1
        if cookies_configured and deno_available:
            auth_level = 2
        if po_token_configured and cookies_configured and deno_available:
            auth_level = 3
        if pot_plugin_loaded and cookies_configured and deno_available:
            auth_level = 4
        # Also level 4 if pot_server is reachable
        if pot_server_reachable and cookies_configured and deno_available:
            auth_level = 4

        # --- Recommendations ---
        recommendations: list[str] = []
        if not deno_available:
            recommendations.append(
                "Install deno: brew install deno (macOS) or visit https://deno.land"
            )
        if not cookies_configured:
            recommendations.append(
                "Add browser cookies to config: youtube.cookies_from_browser or youtube.cookies_file"
            )
        if not pot_plugin_loaded and not po_token_configured:
            recommendations.append(
                "Install bgutil PO token provider: pip install bgutil-ytdlp-pot-provider"
            )
        if auth_level < 4 and pot_server_url and pot_server_reachable is False:
            recommendations.append(
                f"POT server not reachable at {pot_server_url} — check if it's running"
            )
        if auth_level >= 2 and not pot_plugin_loaded and not po_token_configured:
            recommendations.append(
                "For full YouTube access, set up automated PO tokens. "
                "See: documentation/guides/youtube-auth.md"
            )

        return {
            "deno_available": deno_available,
            "deno_version": deno_version,
            "pot_plugin_loaded": pot_plugin_loaded,
            "pot_plugin_version": pot_plugin_version,
            "pot_server_reachable": pot_server_reachable,
            "cookies_configured": cookies_configured,
            "cookies_source": cookies_source,
            "po_token_configured": po_token_configured,
            "po_token_type": po_token_type,
            "auth_level": auth_level,
            "recommendations": recommendations,
        }

    def format_auth_error_message(
        self, auth_status: dict[str, Any] | None = None
    ) -> str:
        """Format an actionable error message for YouTube 403 failures.

        Args:
            auth_status: Pre-computed auth status dict, or None to compute it.

        Returns:
            Human-readable error message with diagnostic info and next steps.
        """
        if auth_status is None:
            auth_status = self.check_youtube_auth_status()

        level = auth_status.get("auth_level", 0)
        lines = [
            "",
            "YouTube download failed (403 Forbidden).",
            "",
            "YouTube increasingly requires authentication for video downloads.",
            "For setup instructions, see:",
            "  https://github.com/thoughtpunch/claudetube/blob/main/documentation/guides/youtube-auth.md",
            "",
            f"Your current auth level: {level}/4",
        ]

        # Diagnostic summary
        diag = []
        if auth_status.get("deno_available"):
            diag.append(f"  deno: v{auth_status.get('deno_version', '?')}")
        else:
            diag.append("  deno: NOT INSTALLED")

        if auth_status.get("cookies_configured"):
            diag.append(f"  cookies: {auth_status.get('cookies_source', 'configured')}")
        else:
            diag.append("  cookies: not configured")

        if auth_status.get("pot_plugin_loaded"):
            diag.append(
                f"  PO token plugin: v{auth_status.get('pot_plugin_version', '?')}"
            )
        elif auth_status.get("po_token_configured"):
            diag.append(
                f"  PO token: manual ({auth_status.get('po_token_type', 'unknown type')})"
            )
            diag.append(
                "  Note: Manual PO tokens expire in ~12 hours. Token may have expired."
            )
        else:
            diag.append("  PO token: not configured")

        if auth_status.get("pot_server_reachable") is False:
            diag.append("  POT server: NOT REACHABLE")
        elif auth_status.get("pot_server_reachable") is True:
            diag.append("  POT server: reachable")

        lines.append("")
        lines.extend(diag)

        # Recommendations
        recs = auth_status.get("recommendations", [])
        if recs:
            lines.append("")
            lines.append("Quick fix options (easiest first):")
            for i, rec in enumerate(recs, 1):
                lines.append(f"  {i}. {rec}")

        return "\n".join(lines)

    def get_metadata(self, url: str, timeout: int = 30) -> dict:
        """Fetch video metadata without downloading.

        Returns:
            Dict with video metadata

        Raises:
            MetadataError: If metadata fetch fails
        """
        yt_args: list[str] = []
        if self._is_youtube_url(url):
            yt_args.extend(self._youtube_config_args())

        result = self._run(
            [*yt_args, "--dump-json", "--no-download", url],
            timeout=timeout,
            retry_clients=False,  # Metadata doesn't hit 403 issues
        )

        if not result.success:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            if "ERROR:" in error_msg:
                error_msg = error_msg.split("ERROR:")[-1].strip()
            raise MetadataError(error_msg)

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise MetadataError(f"Invalid JSON response: {e}") from e

    def download_audio(
        self,
        url: str,
        output_path: Path,
        quality: str = "64K",
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        """Download audio from video.

        Args:
            url: Video URL
            output_path: Output file path
            quality: Audio quality (e.g., "64K", "128K")
            on_progress: Optional callback for progress updates.
                If provided, uses structured JSON progress output.

        Returns:
            Path to downloaded audio file

        Raises:
            DownloadError: If download fails
        """
        # YouTube-specific: add config-driven opts (cookies, PO tokens, etc.)
        # NOTE: We do NOT force specific player_client here - let yt-dlp's
        # auto-detection choose the best client (android_vr often works when
        # default/mweb fail with 403).
        yt_args: list[str] = []
        if self._is_youtube_url(url):
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

        # Use progress-enabled runner if callback provided
        if on_progress:
            result = self._run_with_progress(args, on_progress=on_progress)
        else:
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
            # Also use progress-enabled runner for fallback
            if on_progress:
                result = self._run_with_progress(args, on_progress=on_progress)
            else:
                result = self._run(args)

        if not result.success or not output_path.exists():
            fallback_error = result.stderr or ""
            # Include both errors when the fallback also fails
            if first_error and fallback_error:
                combined_stderr = (
                    f"[audio-only] {first_error.strip()}\n"
                    f"[video fallback] {fallback_error.strip()}"
                )
            else:
                combined_stderr = fallback_error or first_error

            # Parse error and raise typed exception
            parsed_error = parse_yt_dlp_error(combined_stderr)
            is_youtube = self._is_youtube_url(url)

            # Get auth status for YouTube errors
            auth_status = None
            clients_tried = ["default", "mweb"]
            if is_youtube and parsed_error.category in ("auth", "http_error", "po_token"):
                with contextlib.suppress(Exception):
                    auth_status = self.check_youtube_auth_status()

            raise yt_dlp_error_to_exception(
                parsed_error,
                is_youtube=is_youtube,
                auth_status=auth_status,
                clients_tried=clients_tried,
            )

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
        yt_args: list[str] = []
        if self._is_youtube_url(url):
            yt_args.extend(self._youtube_config_args())

        args = [
            *yt_args,
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

        yt_args: list[str] = []
        if self._is_youtube_url(url):
            yt_args.extend(self._youtube_config_args())

        args = [
            *yt_args,
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
        yt_args: list[str] = []
        if self._is_youtube_url(url):
            yt_args.extend(self._youtube_config_args())

        result = self._run(
            [*yt_args, "-j", "--no-download", url],
            timeout=timeout,
            retry_clients=False,
        )

        if not result.success:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            if "ERROR:" in error_msg:
                error_msg = error_msg.split("ERROR:")[-1].strip()
            raise MetadataError(error_msg)

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
                raise FormatNotAvailableError(
                    "No audio description track found",
                    details={"requested_format": "audio_description"},
                )
            format_id = ad_format["format_id"]

        yt_args: list[str] = []
        if self._is_youtube_url(url):
            yt_args.extend(self._youtube_config_args())

        args = [
            *yt_args,
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
            stderr = result.stderr or ""
            parsed_error = parse_yt_dlp_error(stderr)
            is_youtube = self._is_youtube_url(url)

            auth_status = None
            if is_youtube and parsed_error.category in ("auth", "http_error", "po_token"):
                with contextlib.suppress(Exception):
                    auth_status = self.check_youtube_auth_status()

            raise yt_dlp_error_to_exception(
                parsed_error,
                is_youtube=is_youtube,
                auth_status=auth_status,
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
        on_progress: ProgressCallback | None = None,
    ) -> Path | None:
        """Download a video segment for frame extraction.

        Args:
            url: Video URL
            output_path: Output file path
            start_time: Start time in seconds
            end_time: End time in seconds
            quality_sort: yt-dlp format sort string
            concurrent_fragments: Number of concurrent download fragments
            on_progress: Optional callback for progress updates.
                If provided, uses structured JSON progress output.

        Returns:
            Path to downloaded video, or None if failed
        """
        section_spec = f"*{start_time}-{end_time}"

        yt_args: list[str] = []
        if self._is_youtube_url(url):
            yt_args.extend(self._youtube_config_args())

        args = [
            *yt_args,
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

        if on_progress:
            self._run_with_progress(args, on_progress=on_progress)
        else:
            self._run(args)
        return output_path if output_path.exists() else None
