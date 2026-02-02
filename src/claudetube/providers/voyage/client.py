"""
claudetube.providers.voyage.client - Voyage AI embedding provider implementation.

Implements the Embedder protocol using Voyage AI's multimodal embedding API.
Supports text-only and text+image (multimodal) embeddings.

Example:
    >>> provider = VoyageProvider()
    >>> vector = await provider.embed("scene description", images=[Path("frame.jpg")])
    >>> print(f"Dimension: {len(vector)}")
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import partial
from typing import TYPE_CHECKING, Any

from claudetube.providers.base import Embedder, Provider
from claudetube.providers.capabilities import PROVIDER_INFO, ProviderInfo

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Default Voyage model for multimodal embeddings
DEFAULT_MODEL = "voyage-multimodal-3"


class VoyageProvider(Provider, Embedder):
    """Voyage AI embedding provider.

    Provides multimodal embeddings combining text and images using
    Voyage AI's voyage-multimodal-3 model. Returns 1024-dimensional
    vectors suitable for semantic search.

    Args:
        model: Voyage model identifier. Defaults to "voyage-multimodal-3".
        api_key: Voyage API key. Defaults to VOYAGE_API_KEY env var.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ):
        self._model = model
        self._api_key = api_key
        self._client: Any = None

    @property
    def info(self) -> ProviderInfo:
        return PROVIDER_INFO["voyage"]

    def _resolve_api_key(self) -> str | None:
        """Resolve API key from init arg or environment."""
        return self._api_key or os.environ.get("VOYAGE_API_KEY")

    def is_available(self) -> bool:
        """Check if voyageai is installed and API key is set."""
        try:
            import voyageai  # noqa: F401
        except ImportError:
            return False
        return self._resolve_api_key() is not None

    def _get_client(self) -> Any:
        """Lazy-load the Voyage AI client."""
        if self._client is None:
            import voyageai

            api_key = self._resolve_api_key()
            if not api_key:
                raise ValueError(
                    "VOYAGE_API_KEY not set. "
                    "Set the environment variable or pass api_key to the provider."
                )
            self._client = voyageai.Client(api_key=api_key)
        return self._client

    def embed_sync(
        self,
        text: str,
        images: list[Path] | None = None,
    ) -> list[float]:
        """Synchronous embedding for use in sync contexts.

        Args:
            text: Text content to embed.
            images: Optional list of image paths (max 3 used).

        Returns:
            Embedding vector as list of floats.
        """
        from PIL import Image as PILImage

        client = self._get_client()

        loaded_images = []
        for path in (images or [])[:3]:
            try:
                loaded_images.append(PILImage.open(path))
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")

        inputs = [[text] + loaded_images] if loaded_images else [[text]]

        result = client.multimodal_embed(
            inputs=inputs,
            model=self._model,
            input_type="document",
        )
        return result.embeddings[0]

    async def embed(
        self,
        text: str,
        images: list[Path] | None = None,
        **kwargs,
    ) -> list[float]:
        """Generate embedding vector for text and optionally images.

        Runs the synchronous Voyage API call in a thread executor
        to avoid blocking the event loop.

        Args:
            text: Text content to embed.
            images: Optional list of image paths for multimodal embedding.
            **kwargs: Additional options (unused, for protocol compatibility).

        Returns:
            List of floats representing the 1024-dimensional embedding vector.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self.embed_sync, text, images),
        )
