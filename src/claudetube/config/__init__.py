"""
Configuration constants for claudetube.

Contains quality tiers, provider definitions, and default settings.
"""

from claudetube.config.defaults import CACHE_DIR, DEFAULT_WHISPER_MODEL
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
    "CACHE_DIR",
    "DEFAULT_WHISPER_MODEL",
]
