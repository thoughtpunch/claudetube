"""
claudetube.providers - AI provider abstraction layer.

This package provides a unified interface for multiple AI providers
(OpenAI, Anthropic, Google, local models) for video analysis tasks.

Public API:
    get_provider(name: str) -> BaseProvider
        Get a configured provider instance by name.

    list_available() -> list[str]
        List all available (configured) provider names.

Example:
    >>> from claudetube.providers import get_provider, list_available
    >>> providers = list_available()
    >>> provider = get_provider("openai")
    >>> result = await provider.analyze_frame(image_data, prompt)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.providers.base import BaseProvider

# Public API - implementations added in future tickets
__all__ = [
    # Provider access
    "get_provider",
    "list_available",
    # Types (from types.py)
    # Base classes (from base.py)
    # Capabilities (from capabilities.py)
]


def get_provider(name: str) -> BaseProvider | Any:
    """
    Get a configured provider instance by name.

    Args:
        name: Provider name (e.g., "openai", "anthropic", "gemini")

    Returns:
        Configured provider instance.

    Raises:
        ProviderNotFoundError: If provider doesn't exist.
        ProviderNotConfiguredError: If provider lacks required config.
    """
    raise NotImplementedError("Provider registry not yet implemented")


def list_available() -> list[str]:
    """
    List all available (configured) provider names.

    Returns:
        List of provider names that are configured and ready to use.
    """
    raise NotImplementedError("Provider registry not yet implemented")
