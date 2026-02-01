"""
Base classes for external tool wrappers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result from running an external tool."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    error: str | None = None

    @classmethod
    def from_error(cls, error: str) -> "ToolResult":
        """Create a failed result from an error message."""
        return cls(success=False, error=error, returncode=-1)

    @classmethod
    def ok(cls, stdout: str = "", stderr: str = "") -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, stdout=stdout, stderr=stderr, returncode=0)


class VideoTool(ABC):
    """Abstract base class for external video processing tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for logging and error messages."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the tool is installed and available."""
        pass

    @abstractmethod
    def get_path(self) -> str:
        """Get the path to the tool executable."""
        pass
