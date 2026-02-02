"""Tests for VoyageProvider.

Verifies:
1. Provider instantiation and info
2. is_available() behavior
3. embed_sync() with text only
4. embed_sync() with text + images
5. async embed() method
6. Registry integration
7. Protocol compliance
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from claudetube.providers.base import Embedder
from claudetube.providers.capabilities import Capability
from claudetube.providers.voyage.client import VoyageProvider

# =============================================================================
# Instantiation and Info
# =============================================================================


class TestVoyageProviderInfo:
    """Tests for provider instantiation and info."""

    def test_instantiation_defaults(self):
        provider = VoyageProvider()
        assert provider._model == "voyage-multimodal-3"
        assert provider._api_key is None

    def test_instantiation_custom(self):
        provider = VoyageProvider(model="voyage-3", api_key="test-key")
        assert provider._model == "voyage-3"
        assert provider._api_key == "test-key"

    def test_info_name(self):
        provider = VoyageProvider()
        assert provider.info.name == "voyage"

    def test_info_capabilities(self):
        provider = VoyageProvider()
        assert provider.info.can(Capability.EMBED)
        assert not provider.info.can(Capability.TRANSCRIBE)
        assert not provider.info.can(Capability.VISION)

    def test_protocol_compliance(self):
        provider = VoyageProvider()
        assert isinstance(provider, Embedder)


# =============================================================================
# Availability
# =============================================================================


class TestVoyageAvailability:
    """Tests for is_available() behavior."""

    def test_available_with_sdk_and_key(self):
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            provider = VoyageProvider()
            # voyageai may or may not be installed, but if it is:
            try:
                import voyageai  # noqa: F401

                assert provider.is_available() is True
            except ImportError:
                assert provider.is_available() is False

    def test_unavailable_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = VoyageProvider()
            assert provider.is_available() is False

    def test_available_with_explicit_key(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = VoyageProvider(api_key="explicit-key")
            try:
                import voyageai  # noqa: F401

                assert provider.is_available() is True
            except ImportError:
                assert provider.is_available() is False

    def test_unavailable_without_sdk(self):
        with patch.dict("sys.modules", {"voyageai": None}):
            provider = VoyageProvider(api_key="test-key")
            assert provider.is_available() is False


# =============================================================================
# Embedding - Sync
# =============================================================================


class TestVoyageEmbedSync:
    """Tests for synchronous embedding."""

    def test_text_only_embedding(self):
        mock_client = MagicMock()
        mock_client.multimodal_embed.return_value = MagicMock(
            embeddings=[[0.1, 0.2, 0.3]]
        )

        provider = VoyageProvider(api_key="test-key")
        provider._client = mock_client

        with patch.dict("sys.modules", {"PIL": MagicMock(), "PIL.Image": MagicMock()}):
            result = provider.embed_sync("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_client.multimodal_embed.assert_called_once()
        call_kwargs = mock_client.multimodal_embed.call_args
        assert call_kwargs[1]["model"] == "voyage-multimodal-3"
        assert call_kwargs[1]["input_type"] == "document"

    def test_text_with_images(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake image data")

        mock_client = MagicMock()
        mock_client.multimodal_embed.return_value = MagicMock(
            embeddings=[[0.4, 0.5, 0.6]]
        )

        mock_img = MagicMock()
        mock_pil_image = MagicMock()
        mock_pil_image.open.return_value = mock_img

        provider = VoyageProvider(api_key="test-key")
        provider._client = mock_client

        with patch.dict(
            "sys.modules", {"PIL": MagicMock(), "PIL.Image": mock_pil_image}
        ):
            result = provider.embed_sync("test text", images=[img_path])

        assert result == [0.4, 0.5, 0.6]
        call_args = mock_client.multimodal_embed.call_args
        inputs = call_args[1]["inputs"]
        assert len(inputs) == 1
        assert len(inputs[0]) == 2  # text + 1 image

    def test_max_three_images(self, tmp_path):
        paths = []
        for i in range(5):
            p = tmp_path / f"img{i}.jpg"
            p.write_bytes(b"fake")
            paths.append(p)

        mock_client = MagicMock()
        mock_client.multimodal_embed.return_value = MagicMock(embeddings=[[0.1]])

        mock_pil_image = MagicMock()
        mock_pil_module = MagicMock()
        mock_pil_module.Image = mock_pil_image

        provider = VoyageProvider(api_key="test-key")
        provider._client = mock_client

        with patch.dict(
            "sys.modules", {"PIL": mock_pil_module, "PIL.Image": mock_pil_image}
        ):
            provider.embed_sync("text", images=paths)

        # Should only open 3 images
        assert mock_pil_image.open.call_count == 3

    def test_failed_image_load_continues(self, tmp_path):
        img_path = tmp_path / "bad.jpg"
        img_path.write_bytes(b"not an image")

        mock_client = MagicMock()
        mock_client.multimodal_embed.return_value = MagicMock(embeddings=[[0.7, 0.8]])

        mock_pil_image = MagicMock()
        mock_pil_image.open.side_effect = OSError("bad image")
        mock_pil_module = MagicMock()
        mock_pil_module.Image = mock_pil_image

        provider = VoyageProvider(api_key="test-key")
        provider._client = mock_client

        with patch.dict(
            "sys.modules", {"PIL": mock_pil_module, "PIL.Image": mock_pil_image}
        ):
            result = provider.embed_sync("text", images=[img_path])

        # Should still return an embedding (text-only fallback)
        assert result == [0.7, 0.8]
        call_args = mock_client.multimodal_embed.call_args
        inputs = call_args[1]["inputs"]
        assert inputs == [["text"]]  # No images since load failed


# =============================================================================
# Embedding - Async
# =============================================================================


class TestVoyageEmbedAsync:
    """Tests for async embedding."""

    @pytest.mark.asyncio
    async def test_async_embed_delegates_to_sync(self):
        mock_client = MagicMock()
        mock_client.multimodal_embed.return_value = MagicMock(
            embeddings=[[1.0, 2.0, 3.0]]
        )

        provider = VoyageProvider(api_key="test-key")
        provider._client = mock_client

        with patch.dict("sys.modules", {"PIL": MagicMock(), "PIL.Image": MagicMock()}):
            result = await provider.embed("test text")

        assert result == [1.0, 2.0, 3.0]


# =============================================================================
# Registry Integration
# =============================================================================


class TestVoyageRegistry:
    """Tests for registry integration."""

    def test_get_provider_by_name(self):
        from claudetube.providers.registry import clear_cache, get_provider

        clear_cache()
        provider = get_provider("voyage")
        assert isinstance(provider, VoyageProvider)
        clear_cache()

    def test_get_provider_by_alias(self):
        from claudetube.providers.registry import clear_cache, get_provider

        clear_cache()
        provider = get_provider("voyage-ai")
        assert isinstance(provider, VoyageProvider)
        clear_cache()

    def test_get_provider_voyage_3_alias(self):
        from claudetube.providers.registry import clear_cache, get_provider

        clear_cache()
        provider = get_provider("voyage-3")
        assert isinstance(provider, VoyageProvider)
        clear_cache()
