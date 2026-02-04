"""
Unified configuration loader with priority resolution.

Root directory (CLAUDETUBE_ROOT):
- macOS/Linux: ~/.claudetube
- Windows: %APPDATA%\\claudetube
- Override: CLAUDETUBE_ROOT environment variable

Cache directory priority (highest to lowest):
1. Environment variable (CLAUDETUBE_CACHE_DIR) - override
2. Project config (.claudetube/config.yaml)
3. User config ({root_dir}/config.yaml)
4. Default ({root_dir}/cache)

User config location:
- {root_dir}/config.yaml (e.g., ~/.claudetube/config.yaml)
- Project config at .claudetube/config.yaml overrides user config
"""

import logging
import os
import sys
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

    root_dir: Path
    cache_dir: Path
    source: ConfigSource

    def __repr__(self) -> str:
        return (
            f"ClaudetubeConfig(root_dir={self.root_dir!r}, "
            f"cache_dir={self.cache_dir!r}, source={self.source.value!r})"
        )


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


def _get_cache_dir_from_yaml(
    config: dict[str, Any] | None,
    config_path: Path | None = None,
) -> Path | None:
    """Extract cache_dir from a parsed YAML config.

    Args:
        config: Parsed YAML config dict.
        config_path: Path to the config file (for resolving relative paths).

    Returns:
        Resolved cache directory path, or None if not specified.
    """
    if config is None:
        return None

    cache_dir = config.get("cache_dir")
    if cache_dir is None:
        return None

    path = Path(cache_dir).expanduser()

    # Resolve relative paths relative to config file location
    if not path.is_absolute() and config_path is not None:
        path = (config_path.parent / path).resolve()
    else:
        path = path.resolve()

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


def _get_root_dir() -> Path:
    """Get the claudetube root directory.

    Priority:
    1. CLAUDETUBE_ROOT environment variable
    2. Platform-specific default:
       - Windows: %APPDATA%\\claudetube
       - macOS/Linux: ~/.claudetube

    Returns:
        Path to the root directory (may not exist yet).
    """
    # Check environment variable first
    env_root = os.environ.get("CLAUDETUBE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    # Platform-specific defaults
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "claudetube"
        return Path.home() / "AppData" / "Roaming" / "claudetube"
    else:
        # macOS/Linux: ~/.claudetube
        return Path.home() / ".claudetube"


def _get_user_config_path() -> Path:
    """Get the user-level config path.

    Returns:
        Path to user config file at {root_dir}/config.yaml
    """
    return _get_root_dir() / "config.yaml"


def _get_default_cache_dir() -> Path:
    """Get the default cache directory.

    Returns:
        Path to {root_dir}/cache
    """
    return _get_root_dir() / "cache"


def _resolve_config() -> ClaudetubeConfig:
    """Resolve configuration from all sources in priority order.

    Priority for cache_dir:
    1. Environment variable (CLAUDETUBE_CACHE_DIR)
    2. Project config (.claudetube/config.yaml)
    3. User config ({root_dir}/config.yaml)
    4. Default ({root_dir}/cache)

    Returns:
        Resolved ClaudetubeConfig with root_dir, cache_dir and source.
    """
    root_dir = _get_root_dir()

    # 1. Check environment variable for cache_dir override
    env_cache_dir = os.environ.get("CLAUDETUBE_CACHE_DIR")
    if env_cache_dir:
        cache_dir = Path(env_cache_dir).expanduser().resolve()
        logger.info(f"Using cache_dir from CLAUDETUBE_CACHE_DIR: {cache_dir}")
        return ClaudetubeConfig(
            root_dir=root_dir, cache_dir=cache_dir, source=ConfigSource.ENV
        )

    # 2. Check project config
    project_config_path = _find_project_config()
    if project_config_path:
        project_config = _load_yaml_config(project_config_path)
        cache_dir = _get_cache_dir_from_yaml(project_config, project_config_path)
        if cache_dir:
            logger.info(
                f"Using cache_dir from project config {project_config_path}: {cache_dir}"
            )
            return ClaudetubeConfig(
                root_dir=root_dir, cache_dir=cache_dir, source=ConfigSource.PROJECT
            )

    # 3. Check user config
    user_config_path = _get_user_config_path()
    user_config = _load_yaml_config(user_config_path)
    cache_dir = _get_cache_dir_from_yaml(user_config, user_config_path)
    if cache_dir:
        logger.info(f"Using cache_dir from user config {user_config_path}: {cache_dir}")
        return ClaudetubeConfig(
            root_dir=root_dir, cache_dir=cache_dir, source=ConfigSource.USER
        )

    # 4. Fall back to default
    cache_dir = _get_default_cache_dir()
    logger.debug(f"Using default cache_dir: {cache_dir}")
    return ClaudetubeConfig(
        root_dir=root_dir, cache_dir=cache_dir, source=ConfigSource.DEFAULT
    )


@lru_cache(maxsize=1)
def get_config() -> ClaudetubeConfig:
    """Get resolved claudetube configuration.

    Results are cached - configuration is resolved once per process.
    To force re-resolution (e.g., after env change), use clear_config_cache().

    Returns:
        ClaudetubeConfig with cache_dir and source.
    """
    return _resolve_config()


def get_root_dir(ensure_exists: bool = True) -> Path:
    """Get the resolved root directory.

    Args:
        ensure_exists: If True (default), create the directory if it doesn't exist.

    Returns:
        Path to the root directory (e.g., ~/.claudetube).
    """
    root_dir = get_config().root_dir
    if ensure_exists:
        root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir


def get_cache_dir(ensure_exists: bool = True) -> Path:
    """Get the resolved cache directory.

    Args:
        ensure_exists: If True (default), create the directory if it doesn't exist.

    Returns:
        Path to the cache directory.
    """
    cache_dir = get_config().cache_dir
    if ensure_exists:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_db_dir(ensure_exists: bool = True) -> Path:
    """Get the database directory.

    Args:
        ensure_exists: If True (default), create the directory if it doesn't exist.

    Returns:
        Path to the database directory ({root_dir}/db).
    """
    db_dir = get_config().root_dir / "db"
    if ensure_exists:
        db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


def clear_config_cache() -> None:
    """Clear the cached configuration.

    Call this if environment variables or config files have changed
    and you need to re-resolve the configuration.
    """
    get_config.cache_clear()
