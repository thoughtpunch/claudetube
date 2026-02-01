"""
JSON storage utilities for cache state.
"""

from __future__ import annotations

import json
from pathlib import Path

from claudetube.exceptions import CacheError
from claudetube.models.state import VideoState


def save_state(state: VideoState, state_file: Path) -> None:
    """Save video state to JSON file.

    Args:
        state: VideoState to save
        state_file: Path to state.json file

    Raises:
        CacheError: If save fails
    """
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state.to_dict(), indent=2))
    except Exception as e:
        raise CacheError(f"Failed to save state: {e}") from e


def load_state(state_file: Path) -> VideoState | None:
    """Load video state from JSON file.

    Args:
        state_file: Path to state.json file

    Returns:
        VideoState or None if file doesn't exist

    Raises:
        CacheError: If file exists but cannot be parsed
    """
    if not state_file.exists():
        return None

    try:
        data = json.loads(state_file.read_text())
        return VideoState.from_dict(data)
    except json.JSONDecodeError as e:
        raise CacheError(f"Invalid state.json: {e}") from e
    except Exception as e:
        raise CacheError(f"Failed to load state: {e}") from e


def load_state_dict(state_file: Path) -> dict | None:
    """Load raw state dict from JSON file.

    Args:
        state_file: Path to state.json file

    Returns:
        Dict or None if file doesn't exist
    """
    if not state_file.exists():
        return None

    try:
        return json.loads(state_file.read_text())
    except Exception:
        return None
