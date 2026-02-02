"""Tests for LocalEmbedderProvider.

Verifies:
1. Provider instantiation and info
2. is_available() behavior
3. embed_sync() with text only
4. embed_sync() with text + images
5. embed_sync() without open-clip (text-only fallback)
6. async embed() method
7. Registry integration
8. Protocol compliance
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from claudetube.providers.base import Embedder
from claudetube.providers.capabilities import Capability
from claudetube.providers.local_embedder import (
    IMAGE_DIM,
    TEXT_DIM,
    LocalEmbedderProvider,
)

# =============================================================================
# Instantiation and Info
# =============================================================================


class TestLocalEmbedderInfo:
    """Tests for provider instantiation and info."""

    def test_instantiation(self):
        provider = LocalEmbedderProvider()
        assert provider._text_model is None
        assert provider._clip_model is None

    def test_info_name(self):
        provider = LocalEmbedderProvider()
        assert provider.info.name == "local-embedder"

    def test_info_capabilities(self):
        provider = LocalEmbedderProvider()
        assert provider.info.can(Capability.EMBED)
        assert not provider.info.can(Capability.TRANSCRIBE)
        assert not provider.info.can(Capability.VISION)

    def test_info_cost_is_free(self):
        provider = LocalEmbedderProvider()
        assert provider.info.cost_per_1m_input_tokens == 0

    def test_protocol_compliance(self):
        provider = LocalEmbedderProvider()
        assert isinstance(provider, Embedder)


# =============================================================================
# Availability
# =============================================================================


class TestLocalEmbedderAvailability:
    """Tests for is_available() behavior."""

    def test_available_with_sentence_transformers(self):
        provider = LocalEmbedderProvider()
        try:
            import sentence_transformers  # noqa: F401

            assert provider.is_available() is True
        except ImportError:
            assert provider.is_available() is False

    def test_unavailable_without_sentence_transformers(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            provider = LocalEmbedderProvider()
            assert provider.is_available() is False


# =============================================================================
# Embedding - Text Only
# =============================================================================


class TestLocalEmbedderTextOnly:
    """Tests for text-only embedding."""

    def test_text_embedding_dimensions(self):
        """Text-only embedding should have TEXT_DIM + IMAGE_DIM dimensions."""
        mock_text_model = MagicMock()
        mock_text_model.encode.return_value = np.random.randn(TEXT_DIM).astype(
            np.float32
        )

        provider = LocalEmbedderProvider()
        provider._text_model = mock_text_model

        result = provider.embed_sync("test text")

        assert len(result) == TEXT_DIM + IMAGE_DIM
        # Image portion should be zeros
        img_part = result[TEXT_DIM:]
        assert all(v == 0.0 for v in img_part)

    def test_text_embedding_values(self):
        """Text embedding should come from the text model."""
        text_vec = np.ones(TEXT_DIM, dtype=np.float32) * 0.5
        mock_text_model = MagicMock()
        mock_text_model.encode.return_value = text_vec

        provider = LocalEmbedderProvider()
        provider._text_model = mock_text_model

        result = provider.embed_sync("test")

        # First TEXT_DIM values should be from text model
        assert np.allclose(result[:TEXT_DIM], 0.5, atol=1e-6)


# =============================================================================
# Embedding - With Images
# =============================================================================


class TestLocalEmbedderWithImages:
    """Tests for text + image embedding."""

    def test_with_images_uses_clip(self, tmp_path):
        """When images are provided and CLIP is available, should use it."""
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake")

        text_vec = np.ones(TEXT_DIM, dtype=np.float32) * 0.5
        mock_text_model = MagicMock()
        mock_text_model.encode.return_value = text_vec

        img_vec = np.ones(IMAGE_DIM, dtype=np.float32) * 0.3
        mock_clip_model = MagicMock()
        mock_clip_model.encode_image.return_value = MagicMock(
            squeeze=MagicMock(
                return_value=MagicMock(numpy=MagicMock(return_value=img_vec))
            )
        )
        mock_preprocess = MagicMock(
            return_value=MagicMock(unsqueeze=MagicMock(return_value=MagicMock()))
        )

        provider = LocalEmbedderProvider()
        provider._text_model = mock_text_model
        provider._clip_model = (mock_clip_model, mock_preprocess)

        mock_pil_image = MagicMock()
        mock_pil_image.open.return_value = MagicMock()
        mock_torch = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "PIL": MagicMock(),
                "PIL.Image": mock_pil_image,
                "torch": mock_torch,
            },
        ):
            result = provider.embed_sync("text", images=[img_path])

        assert len(result) == TEXT_DIM + IMAGE_DIM

    def test_without_clip_falls_back_to_zeros(self, tmp_path):
        """Without open-clip, image portion should be zeros."""
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake")

        text_vec = np.ones(TEXT_DIM, dtype=np.float32) * 0.5
        mock_text_model = MagicMock()
        mock_text_model.encode.return_value = text_vec

        provider = LocalEmbedderProvider()
        provider._text_model = mock_text_model
        # _clip_model stays None, _get_clip_model will try import and fail

        with patch.dict("sys.modules", {"open_clip": None}):
            provider._clip_model = None  # Force re-check
            # _get_clip_model returns None when import fails
            with patch.object(provider, "_get_clip_model", return_value=None):
                result = provider.embed_sync("text", images=[img_path])

        # Image part should be zeros
        img_part = result[TEXT_DIM:]
        assert all(v == 0.0 for v in img_part)

    def test_max_three_images(self, tmp_path):
        """Should only process first 3 images."""
        paths = []
        for i in range(5):
            p = tmp_path / f"img{i}.jpg"
            p.write_bytes(b"fake")
            paths.append(p)

        text_vec = np.ones(TEXT_DIM, dtype=np.float32)
        mock_text_model = MagicMock()
        mock_text_model.encode.return_value = text_vec

        mock_clip = MagicMock()
        mock_clip.encode_image.return_value = MagicMock(
            squeeze=MagicMock(
                return_value=MagicMock(
                    numpy=MagicMock(return_value=np.ones(IMAGE_DIM, dtype=np.float32))
                )
            )
        )
        mock_preprocess = MagicMock(
            return_value=MagicMock(unsqueeze=MagicMock(return_value=MagicMock()))
        )

        provider = LocalEmbedderProvider()
        provider._text_model = mock_text_model
        provider._clip_model = (mock_clip, mock_preprocess)

        mock_pil_image = MagicMock()
        mock_pil_image.open.return_value = MagicMock()
        mock_pil_module = MagicMock()
        mock_pil_module.Image = mock_pil_image

        with patch.dict(
            "sys.modules",
            {
                "PIL": mock_pil_module,
                "PIL.Image": mock_pil_image,
                "torch": MagicMock(),
            },
        ):
            provider.embed_sync("text", images=paths)

        # Should only open 3 images
        assert mock_pil_image.open.call_count == 3


# =============================================================================
# Embedding - Async
# =============================================================================


class TestLocalEmbedderAsync:
    """Tests for async embedding."""

    @pytest.mark.asyncio
    async def test_async_embed_returns_correct_dimensions(self):
        text_vec = np.zeros(TEXT_DIM, dtype=np.float32)
        mock_text_model = MagicMock()
        mock_text_model.encode.return_value = text_vec

        provider = LocalEmbedderProvider()
        provider._text_model = mock_text_model

        result = await provider.embed("test text")

        assert len(result) == TEXT_DIM + IMAGE_DIM


# =============================================================================
# Registry Integration
# =============================================================================


class TestLocalEmbedderRegistry:
    """Tests for registry integration."""

    def test_get_provider_by_name(self):
        from claudetube.providers.registry import clear_cache, get_provider

        clear_cache()
        provider = get_provider("local-embedder")
        assert isinstance(provider, LocalEmbedderProvider)
        clear_cache()

    def test_get_provider_by_alias(self):
        from claudetube.providers.registry import clear_cache, get_provider

        clear_cache()
        provider = get_provider("local")
        assert isinstance(provider, LocalEmbedderProvider)
        clear_cache()
