"""
JSON storage utilities for cache state.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
from pathlib import Path

from claudetube.exceptions import CacheError
from claudetube.models.state import VideoState

logger = logging.getLogger(__name__)


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


def cache_local_file(
    source: Path,
    cache_dir: Path,
    copy: bool = False,
) -> tuple[Path, str]:
    """Cache a local file by symlink (default) or copy.

    Args:
        source: Absolute path to the source file
        cache_dir: Cache directory to place the file in
        copy: If True, copy the file; if False (default), create symlink

    Returns:
        Tuple of (dest_path, cache_mode) where cache_mode is "symlink" or "copy"

    Raises:
        CacheError: If caching fails
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"source{source.suffix}"

    if copy:
        try:
            shutil.copy2(source, dest)
            return dest, "copy"
        except Exception as e:
            raise CacheError(f"Failed to copy file: {e}") from e

    # Try symlink (default)
    try:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(source.resolve())
        return dest, "symlink"
    except OSError as e:
        # On Windows, symlinks may fail without admin/dev mode
        if platform.system() == "Windows":
            logger.warning(
                f"Symlink failed on Windows (requires admin/dev mode), falling back to copy: {e}"
            )
            try:
                shutil.copy2(source, dest)
                return dest, "copy"
            except Exception as copy_err:
                raise CacheError(f"Failed to copy file after symlink failed: {copy_err}") from copy_err
        raise CacheError(f"Failed to create symlink: {e}") from e


def check_cached_source(cache_dir: Path, cached_file: str | None) -> tuple[bool, str | None]:
    """Check if a cached source file exists and is valid.

    For symlinks, checks if the target still exists.

    Args:
        cache_dir: Cache directory containing the file
        cached_file: Name of the cached file (e.g., "source.mp4")

    Returns:
        Tuple of (is_valid, warning_message) where warning_message is None if valid
    """
    if not cached_file:
        return False, "No cached file recorded"

    cached_path = cache_dir / cached_file

    if not cached_path.exists() and not cached_path.is_symlink():
        return False, f"Cached file not found: {cached_file}"

    if cached_path.is_symlink():
        try:
            target = cached_path.resolve(strict=True)
            if not target.exists():
                return False, f"Broken symlink: source file moved or deleted ({cached_path.readlink()})"
        except (OSError, FileNotFoundError):
            # resolve(strict=True) raises if target doesn't exist
            try:
                original = os.readlink(cached_path)
                return False, f"Broken symlink: source file moved or deleted ({original})"
            except Exception:
                return False, "Broken symlink: source file moved or deleted"

    return True, None
