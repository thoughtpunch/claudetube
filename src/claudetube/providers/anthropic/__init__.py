"""
claudetube.providers.anthropic - Anthropic Claude provider.

Provides vision analysis and reasoning via the Anthropic Messages API.
Supports structured output via tool_choice.

Example:
    >>> from claudetube.providers.registry import get_provider
    >>> provider = get_provider("anthropic")
    >>> result = await provider.analyze_images(
    ...     [Path("frame1.jpg")],
    ...     prompt="What is shown in this frame?",
    ... )
"""

from claudetube.providers.anthropic.client import AnthropicProvider

__all__ = ["AnthropicProvider"]
