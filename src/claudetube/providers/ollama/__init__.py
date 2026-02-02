"""
claudetube.providers.ollama - Ollama local LLM provider.

Provides vision analysis (LLaVA) and reasoning (Llama) via a local Ollama server.
Fully offline, no API keys required.

Example:
    >>> from claudetube.providers.registry import get_provider
    >>> provider = get_provider("ollama")
    >>> result = await provider.analyze_images(
    ...     [Path("frame1.jpg")],
    ...     prompt="What is shown in this frame?",
    ... )
"""

from claudetube.providers.ollama.client import OllamaProvider

__all__ = ["OllamaProvider"]
