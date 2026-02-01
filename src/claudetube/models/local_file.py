"""
LocalFile Pydantic model for local file path parsing.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from pydantic import BaseModel, Field

# Supported video file extensions (lowercase)
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    ".mpeg", ".mpg", ".3gp", ".3g2", ".ogv", ".ts", ".mts", ".m2ts",
    ".vob", ".divx", ".xvid", ".asf", ".rm", ".rmvb",
}

# Supported audio file extensions (for audio-only processing)
SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".m4a", ".wma", ".aiff",
}


class LocalFileError(Exception):
    """Error when processing local file paths."""

    pass


class LocalFile(BaseModel):
    """Parsed and validated local file with metadata."""

    path: Path = Field(..., description="Resolved absolute path to the file")
    original_input: str = Field(..., description="Original input string")
    extension: str = Field(..., description="File extension (lowercase, with dot)")
    is_video: bool = Field(..., description="True if video file, False if audio")

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def parse(cls, input_str: str) -> LocalFile:
        """Parse a local file path and validate it exists.

        Args:
            input_str: File path (absolute, relative, ~, or file:// URI)

        Returns:
            LocalFile with resolved path and metadata

        Raises:
            LocalFileError: If file doesn't exist or is not a supported format
        """
        # Handle file:// URI scheme
        if input_str.startswith("file://"):
            path_str = input_str[7:]
            if path_str.startswith("localhost"):
                path_str = path_str[9:]
            elif path_str.startswith("//"):
                path_str = path_str[1:]
            path_str = unquote(path_str)
        else:
            path_str = input_str

        # Expand ~ and resolve to absolute path
        try:
            path = Path(path_str).expanduser().resolve()
        except Exception as e:
            raise LocalFileError(f"Invalid path '{input_str}': {e}") from e

        if not path.exists():
            raise LocalFileError(f"File not found: {path}")

        if not path.is_file():
            raise LocalFileError(f"Not a file (maybe a directory?): {path}")

        ext = path.suffix.lower()
        is_video = ext in SUPPORTED_VIDEO_EXTENSIONS
        is_audio = ext in SUPPORTED_AUDIO_EXTENSIONS

        if not is_video and not is_audio:
            supported = sorted(SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS)
            raise LocalFileError(
                f"Unsupported file format '{ext}'. "
                f"Supported formats: {', '.join(supported)}"
            )

        return cls(
            path=path,
            original_input=input_str,
            extension=ext,
            is_video=is_video,
        )

    @classmethod
    def try_parse(cls, input_str: str) -> LocalFile | None:
        """Try to parse a local file, returning None on failure."""
        try:
            return cls.parse(input_str)
        except (LocalFileError, Exception):
            return None

    @property
    def filename(self) -> str:
        """Get the filename without path."""
        return self.path.name

    @property
    def stem(self) -> str:
        """Get filename without extension."""
        return self.path.stem

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"LocalFile(path={self.path!r}, is_video={self.is_video})"


def is_local_file(input_str: str) -> bool:
    """Check if input string is a local file path (not a URL).

    Detects:
    - Absolute paths: /path/to/video.mp4
    - Relative paths: ./video.mp4, ../videos/file.mkv
    - Home-relative paths: ~/Videos/file.mp4
    - File URIs: file:///path/to/video.mp4
    """
    if not input_str or not isinstance(input_str, str):
        return False

    input_str = input_str.strip()

    if input_str.startswith(("http://", "https://")):
        return False

    if input_str.startswith("file://"):
        return LocalFile.try_parse(input_str) is not None

    # Absolute path (Unix or Windows)
    if input_str.startswith("/") or (
        len(input_str) > 2 and input_str[1] == ":" and input_str[2] in "/\\"
    ):
        return LocalFile.try_parse(input_str) is not None

    if input_str.startswith("~"):
        return LocalFile.try_parse(input_str) is not None

    if input_str.startswith(("./", "../", ".\\", "..\\")):
        return LocalFile.try_parse(input_str) is not None

    # Check if looks like domain vs file
    if "." in input_str:
        parts = input_str.split("/")[0]
        if "." in parts and not any(
            parts.endswith(ext)
            for ext in (SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS)
        ):
            return False

    return LocalFile.try_parse(input_str) is not None


def is_url(input_str: str) -> bool:
    """Check if input string is a URL (not a local file)."""
    if not input_str or not isinstance(input_str, str):
        return False

    input_str = input_str.strip()
    if not input_str:
        return False

    if input_str.startswith(("http://", "https://")):
        return True

    if input_str.startswith("file://"):
        return False

    return not is_local_file(input_str)
