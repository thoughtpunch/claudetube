"""
claudetube.providers.local_embedder - Local embedding provider.

Provides text+image embeddings using local models: sentence-transformers
(all-MiniLM-L6-v2) for text and open_clip (ViT-B-32) for images.
No API key required - runs entirely offline.

Output is a 896-dimensional vector: 384d text + 512d image.

Example:
    >>> from claudetube.providers.registry import get_provider
    >>> provider = get_provider("local-embedder")
    >>> vector = await provider.embed("scene description")
    >>> print(f"Dimension: {len(vector)}")  # 896
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from claudetube.providers.base import Embedder, Provider
from claudetube.providers.capabilities import PROVIDER_INFO, ProviderInfo

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Embedding dimensions
TEXT_DIM = 384  # all-MiniLM-L6-v2
IMAGE_DIM = 512  # CLIP ViT-B-32


class LocalEmbedderProvider(Provider, Embedder):
    """Local embedding provider using sentence-transformers and CLIP.

    Produces 896-dimensional embeddings by concatenating:
    - 384d text embedding (all-MiniLM-L6-v2)
    - 512d image embedding (CLIP ViT-B-32, averaged across images)

    When no images are provided or open-clip is not installed,
    the image portion is zero-filled.

    No API key required. Runs entirely offline.
    """

    def __init__(self):
        self._text_model: Any = None
        self._clip_model: Any = None

    @property
    def info(self) -> ProviderInfo:
        return PROVIDER_INFO["local-embedder"]

    def is_available(self) -> bool:
        """Check if sentence-transformers is installed."""
        try:
            import sentence_transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_text_model(self) -> Any:
        """Lazy-load the sentence-transformers model."""
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer

            self._text_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._text_model

    def _get_clip_model(self) -> tuple[Any, Any] | None:
        """Lazy-load the CLIP model and preprocessor.

        Returns:
            Tuple of (model, preprocess) or None if open-clip not available.
        """
        if self._clip_model is None:
            try:
                import open_clip

                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai"
                )
                model.eval()
                self._clip_model = (model, preprocess)
            except ImportError:
                logger.warning("open-clip-torch not installed, image embeddings unavailable")
                return None
        return self._clip_model

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
            Embedding vector as list of floats (896 dimensions).
        """
        # Text embedding
        text_model = self._get_text_model()
        text_emb = text_model.encode(text, convert_to_numpy=True)
        text_emb = text_emb.astype(np.float32)

        # Image embedding
        if images:
            clip = self._get_clip_model()
            if clip is not None:
                import torch
                from PIL import Image

                model, preprocess = clip
                img_embs = []
                for path in images[:3]:
                    try:
                        img = preprocess(Image.open(path)).unsqueeze(0)
                        with torch.no_grad():
                            img_emb = model.encode_image(img).squeeze().numpy()
                        img_embs.append(img_emb)
                    except Exception as e:
                        logger.warning(f"Failed to encode image {path}: {e}")

                if img_embs:
                    avg_img_emb = np.mean(img_embs, axis=0).astype(np.float32)
                else:
                    avg_img_emb = np.zeros(IMAGE_DIM, dtype=np.float32)
            else:
                avg_img_emb = np.zeros(IMAGE_DIM, dtype=np.float32)
        else:
            avg_img_emb = np.zeros(IMAGE_DIM, dtype=np.float32)

        combined = np.concatenate([text_emb, avg_img_emb])
        return combined.tolist()

    async def embed(
        self,
        text: str,
        images: list[Path] | None = None,
        **kwargs,
    ) -> list[float]:
        """Generate embedding vector for text and optionally images.

        Runs the synchronous model inference in a thread executor
        to avoid blocking the event loop.

        Args:
            text: Text content to embed.
            images: Optional list of image paths for multimodal embedding.
            **kwargs: Additional options (unused, for protocol compatibility).

        Returns:
            List of floats representing the 896-dimensional embedding vector.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self.embed_sync, text, images),
        )
