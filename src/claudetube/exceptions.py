"""
Custom exceptions for claudetube.

All claudetube exceptions inherit from ClaudetubeError for easy catching.
"""

from __future__ import annotations

from typing import Any


class ClaudetubeError(Exception):
    """Base exception for all claudetube errors."""

    pass


class DownloadError(ClaudetubeError):
    """Error during video/audio download.

    This is the base class for all download-related errors. Specific error
    types inherit from this class for better error handling.

    Attributes:
        message: Human-readable error message
        category: Error classification (e.g., "auth", "geo_restricted")
        stderr: Full stderr output from yt-dlp for debugging
        details: Additional diagnostic information
        suggestion: Recommended remediation steps
    """

    def __init__(
        self,
        message: str,
        *,
        category: str = "unknown",
        stderr: str = "",
        details: dict[str, Any] | None = None,
        suggestion: str = "",
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.stderr = stderr
        self.details = details or {}
        self.suggestion = suggestion

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to a structured dict for MCP error responses."""
        result: dict[str, Any] = {
            "type": self.__class__.__name__,
            "message": self.message,
            "category": self.category,
        }
        if self.details:
            result["details"] = self.details
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result


class YouTubeAuthError(DownloadError):
    """YouTube authentication error (403, PO token issues).

    Raised when YouTube blocks access due to missing or invalid authentication.
    """

    def __init__(
        self,
        message: str,
        *,
        stderr: str = "",
        details: dict[str, Any] | None = None,
        auth_level: int | None = None,
        clients_tried: list[str] | None = None,
    ):
        details = details or {}
        if auth_level is not None:
            details["auth_level"] = auth_level
        if clients_tried:
            details["clients_tried"] = clients_tried

        suggestion = (
            "Set up YouTube authentication. See: documentation/guides/youtube-auth.md"
        )

        super().__init__(
            message,
            category="auth",
            stderr=stderr,
            details=details,
            suggestion=suggestion,
        )
        self.auth_level = auth_level
        self.clients_tried = clients_tried


class GeoRestrictedError(DownloadError):
    """Video is not available in the user's geographic region."""

    def __init__(
        self,
        message: str,
        *,
        stderr: str = "",
        details: dict[str, Any] | None = None,
    ):
        suggestion = "This video is not available in your country. Try using a VPN."
        super().__init__(
            message,
            category="geo_restricted",
            stderr=stderr,
            details=details,
            suggestion=suggestion,
        )


class FormatNotAvailableError(DownloadError):
    """Requested video format is not available."""

    def __init__(
        self,
        message: str,
        *,
        stderr: str = "",
        details: dict[str, Any] | None = None,
        available_formats: list[str] | None = None,
    ):
        details = details or {}
        if available_formats:
            details["available_formats"] = available_formats

        suggestion = "Try a different format or quality setting."
        super().__init__(
            message,
            category="format_unavailable",
            stderr=stderr,
            details=details,
            suggestion=suggestion,
        )
        self.available_formats = available_formats


class RateLimitError(DownloadError):
    """Too many requests, rate limited by the video provider."""

    def __init__(
        self,
        message: str,
        *,
        stderr: str = "",
        details: dict[str, Any] | None = None,
    ):
        suggestion = (
            "Wait a few minutes before retrying. The server is rate limiting requests."
        )
        super().__init__(
            message,
            category="rate_limited",
            stderr=stderr,
            details=details,
            suggestion=suggestion,
        )


class ExtractorError(DownloadError):
    """Extractor-specific failure (site not supported, extraction failed).

    Raised when yt-dlp fails to extract video information from a specific site.
    """

    def __init__(
        self,
        message: str,
        *,
        stderr: str = "",
        details: dict[str, Any] | None = None,
        extractor: str | None = None,
    ):
        details = details or {}
        if extractor:
            details["extractor"] = extractor

        suggestion = (
            "The video may be unavailable, private, or the site may have changed. "
            "Try updating yt-dlp: pip install -U yt-dlp"
        )
        super().__init__(
            message,
            category="extractor",
            stderr=stderr,
            details=details,
            suggestion=suggestion,
        )
        self.extractor = extractor


class NetworkError(DownloadError):
    """Network-related error (connection issues, timeouts, SSL errors)."""

    def __init__(
        self,
        message: str,
        *,
        stderr: str = "",
        details: dict[str, Any] | None = None,
        http_code: int | None = None,
    ):
        details = details or {}
        if http_code:
            details["http_code"] = http_code

        suggestion = "Check your internet connection and try again."
        super().__init__(
            message,
            category="network",
            stderr=stderr,
            details=details,
            suggestion=suggestion,
        )
        self.http_code = http_code


class VideoUnavailableError(DownloadError):
    """Video is unavailable (removed, private, copyright strike, etc.)."""

    def __init__(
        self,
        message: str,
        *,
        stderr: str = "",
        details: dict[str, Any] | None = None,
        reason: str | None = None,
    ):
        details = details or {}
        if reason:
            details["reason"] = reason

        suggestion = "This video is no longer available. It may have been removed or made private."
        super().__init__(
            message,
            category="unavailable",
            stderr=stderr,
            details=details,
            suggestion=suggestion,
        )
        self.reason = reason


class AgeRestrictedError(DownloadError):
    """Video requires age verification."""

    def __init__(
        self,
        message: str,
        *,
        stderr: str = "",
        details: dict[str, Any] | None = None,
    ):
        suggestion = (
            "This video requires age verification. "
            "Configure cookies from a logged-in browser session. "
            "See: documentation/guides/youtube-auth.md"
        )
        super().__init__(
            message,
            category="age_restricted",
            stderr=stderr,
            details=details,
            suggestion=suggestion,
        )


class TranscriptionError(ClaudetubeError):
    """Error during audio transcription."""

    pass


class FrameExtractionError(ClaudetubeError):
    """Error during video frame extraction."""

    pass


class CacheError(ClaudetubeError):
    """Error with cache operations (read/write/corrupt state)."""

    pass


class ToolNotFoundError(ClaudetubeError):
    """Required external tool (yt-dlp, ffmpeg, whisper) not found."""

    def __init__(self, tool_name: str, message: str | None = None):
        self.tool_name = tool_name
        msg = message or f"Required tool '{tool_name}' not found in PATH"
        super().__init__(msg)


class MetadataError(ClaudetubeError):
    """Error fetching video metadata."""

    pass


class SubtitleError(ClaudetubeError):
    """Error fetching or parsing subtitles."""

    pass


class DatabaseError(ClaudetubeError):
    """Error with database operations (connection, migration, query)."""

    pass


class MigrationError(DatabaseError):
    """Error during database schema migration."""

    pass
