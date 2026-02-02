"""
claudetube.providers.litellm - LiteLLM generic reasoning provider.

Provides reasoning via LiteLLM's unified API, which supports 100+ LLM providers
(OpenAI, Anthropic, Cohere, Replicate, Azure, Bedrock, etc.) through a single
OpenAI-compatible interface.

Example:
    >>> from claudetube.providers.registry import get_provider
    >>> provider = get_provider("litellm", model="anthropic/claude-sonnet-4-20250514")
    >>> result = await provider.reason(
    ...     [{"role": "user", "content": "Summarize this transcript..."}]
    ... )
"""

from claudetube.providers.litellm.client import LitellmProvider

__all__ = ["LitellmProvider"]
