"""
Configuration constants for claudetube.

Contains quality tiers, provider definitions, and default settings.
"""

from claudetube.config.defaults import DEFAULT_WHISPER_MODEL
from claudetube.config.loader import (
    ClaudetubeConfig,
    ConfigSource,
    clear_config_cache,
    get_cache_dir,
    get_config,
    get_db_dir,
    get_root_dir,
)
from claudetube.config.output_templates import (
    NO_CHANNEL,
    NO_PLAYLIST,
    TEMPLATES,
    OutputTemplates,
    build_audio_download_args,
    build_cli_args,
    build_outtmpl_dict,
    get_output_path,
)
from claudetube.config.providers import VIDEO_PROVIDERS, get_provider_count
from claudetube.config.quality import (
    QUALITY_LADDER,
    QUALITY_TIERS,
    next_quality,
)

__all__ = [
    "QUALITY_TIERS",
    "QUALITY_LADDER",
    "next_quality",
    "VIDEO_PROVIDERS",
    "get_provider_count",
    "DEFAULT_WHISPER_MODEL",
    # Config loader
    "ClaudetubeConfig",
    "ConfigSource",
    "get_config",
    "get_root_dir",
    "get_cache_dir",
    "get_db_dir",
    "clear_config_cache",
    # Output templates
    "NO_CHANNEL",
    "NO_PLAYLIST",
    "TEMPLATES",
    "OutputTemplates",
    "build_audio_download_args",
    "build_cli_args",
    "build_outtmpl_dict",
    "get_output_path",
]
