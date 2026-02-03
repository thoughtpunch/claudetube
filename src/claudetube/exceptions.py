"""
Custom exceptions for claudetube.

All claudetube exceptions inherit from ClaudetubeError for easy catching.
"""


class ClaudetubeError(Exception):
    """Base exception for all claudetube errors."""

    pass


class DownloadError(ClaudetubeError):
    """Error during video/audio download."""

    pass


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
