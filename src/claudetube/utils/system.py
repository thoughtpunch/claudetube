"""
System utilities for finding executables.
"""

import shutil
import sys
from pathlib import Path


def find_tool(name: str) -> str:
    """Find executable, checking venv first.

    Args:
        name: Tool name (e.g., "yt-dlp", "ffmpeg")

    Returns:
        Path to executable
    """
    # Check venv bin directory first
    venv = Path(sys.prefix) / "bin" / name
    if venv.exists():
        return str(venv)

    # Fall back to system PATH
    return shutil.which(name) or name
