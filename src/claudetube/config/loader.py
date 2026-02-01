"""
Unified configuration loader with priority resolution.

Priority order (highest to lowest):
1. Environment variable (CLAUDETUBE_CACHE_DIR)
2. Project config (.claudetube/config.yaml)
3. User config (~/.config/claudetube/config.yaml)
4. Default (~/.claude/video_cache)
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """Source of the configuration value."""

    ENV = "env"
    PROJECT = "project"
    USER = "user"
    DEFAULT = "default"


@dataclass(frozen=True)
class ClaudetubeConfig:
    """Resolved claudetube configuration."""

    cache_dir: Path
    source: ConfigSource

    def __repr__(self) -> str:
        return f"ClaudetubeConfig(cache_dir={self.cache_dir!r}, source={self.source.value!r})"


def _load_yaml_config(config_path: Path) -> dict[str, Any] | None:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config dict, or None if file doesn't exist or fails to parse.
    """
    if not config_path.exists():
        return None

    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
            if config is None:
                return {}
            if not isinstance(config, dict):
                logger.warning(f"Config file {config_path} is not a valid YAML dict")
                return None
            return config
    except ImportError:
        logger.debug("PyYAML not installed, skipping YAML config files")
        return None
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return None


def _get_cache_dir_from_yaml(config: dict[str, Any] | None) -> Path | None:
    """Extract cache_dir from a parsed YAML config."""
    if config is None:
        return None

    cache_dir = config.get("cache_dir")
    if cache_dir is None:
        return None

    path = Path(cache_dir).expanduser()
    return path


def _find_project_config() -> Path | None:
    """Find project-level config by walking up from cwd.

    Returns:
        Path to .claudetube/config.yaml if found, None otherwise.
    """
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        config_path = parent / ".claudetube" / "config.yaml"
        if config_path.exists():
            return config_path
    return None


def _get_user_config_path() -> Path:
    """Get the user-level config path."""
    return Path.home() / ".config" / "claudetube" / "config.yaml"


def _get_default_cache_dir() -> Path:
    """Get the default cache directory."""
    return Path.home() / ".claude" / "video_cache"


def _resolve_config() -> ClaudetubeConfig:
    """Resolve configuration from all sources in priority order.

    Priority:
    1. Environment variable (CLAUDETUBE_CACHE_DIR)
    2. Project config (.claudetube/config.yaml)
    3. User config (~/.config/claudetube/config.yaml)
    4. Default (~/.claude/video_cache)

    Returns:
        Resolved ClaudetubeConfig with cache_dir and source.
    """
    # 1. Check environment variable
    env_cache_dir = os.environ.get("CLAUDETUBE_CACHE_DIR")
    if env_cache_dir:
        cache_dir = Path(env_cache_dir).expanduser()
        logger.debug(f"Using cache_dir from env: {cache_dir}")
        return ClaudetubeConfig(cache_dir=cache_dir, source=ConfigSource.ENV)

    # 2. Check project config
    project_config_path = _find_project_config()
    if project_config_path:
        project_config = _load_yaml_config(project_config_path)
        cache_dir = _get_cache_dir_from_yaml(project_config)
        if cache_dir:
            logger.debug(f"Using cache_dir from project config {project_config_path}: {cache_dir}")
            return ClaudetubeConfig(cache_dir=cache_dir, source=ConfigSource.PROJECT)

    # 3. Check user config
    user_config_path = _get_user_config_path()
    user_config = _load_yaml_config(user_config_path)
    cache_dir = _get_cache_dir_from_yaml(user_config)
    if cache_dir:
        logger.debug(f"Using cache_dir from user config {user_config_path}: {cache_dir}")
        return ClaudetubeConfig(cache_dir=cache_dir, source=ConfigSource.USER)

    # 4. Fall back to default
    cache_dir = _get_default_cache_dir()
    logger.debug(f"Using default cache_dir: {cache_dir}")
    return ClaudetubeConfig(cache_dir=cache_dir, source=ConfigSource.DEFAULT)


@lru_cache(maxsize=1)
def get_config() -> ClaudetubeConfig:
    """Get resolved claudetube configuration.

    Results are cached - configuration is resolved once per process.
    To force re-resolution (e.g., after env change), use clear_config_cache().

    Returns:
        ClaudetubeConfig with cache_dir and source.
    """
    return _resolve_config()


def get_cache_dir() -> Path:
    """Convenience function to get the resolved cache directory.

    Returns:
        Path to the cache directory.
    """
    return get_config().cache_dir


def clear_config_cache() -> None:
    """Clear the cached configuration.

    Call this if environment variables or config files have changed
    and you need to re-resolve the configuration.
    """
    get_config.cache_clear()
