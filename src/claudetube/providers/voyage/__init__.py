"""
claudetube.providers.voyage - Voyage AI embedding provider.

Provides multimodal embeddings using Voyage AI's voyage-multimodal-3 model.
Supports text and image inputs for semantic search.

Example:
    >>> from claudetube.providers.registry import get_provider
    >>> provider = get_provider("voyage")
    >>> vector = await provider.embed("What is the main topic?")
    >>> print(f"Embedding dimension: {len(vector)}")
"""

from claudetube.providers.voyage.client import VoyageProvider

__all__ = ["VoyageProvider"]
