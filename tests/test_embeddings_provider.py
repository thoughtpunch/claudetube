"""Tests for embeddings module provider integration.

Verifies that embed_scene() and related functions correctly use the
provider pattern instead of direct implementations.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from claudetube.analysis.embeddings import (
    SceneEmbedding,
    _build_scene_text,
    _get_embedder,
    embed_scene,
    get_embedding_dim,
    get_embedding_model,
)


# =============================================================================
# get_embedding_model()
# =============================================================================


class TestGetEmbeddingModel:
    """Tests for model selection from environment."""

    def test_default_is_voyage(self):
        with patch.dict("os.environ", {}, clear=True):
            assert get_embedding_model() == "voyage"

    def test_voyage_from_env(self):
        with patch.dict("os.environ", {"CLAUDETUBE_EMBEDDING_MODEL": "voyage"}):
            assert get_embedding_model() == "voyage"

    def test_local_from_env(self):
        with patch.dict("os.environ", {"CLAUDETUBE_EMBEDDING_MODEL": "local"}):
            assert get_embedding_model() == "local"

    def test_unknown_falls_back_to_voyage(self):
        with patch.dict("os.environ", {"CLAUDETUBE_EMBEDDING_MODEL": "unknown"}):
            assert get_embedding_model() == "voyage"


# =============================================================================
# _get_embedder()
# =============================================================================


class TestGetEmbedder:
    """Tests for provider resolution."""

    def test_voyage_resolves_to_voyage_provider(self):
        from claudetube.providers.registry import clear_cache

        clear_cache()
        provider = _get_embedder("voyage")
        from claudetube.providers.voyage.client import VoyageProvider

        assert isinstance(provider, VoyageProvider)
        clear_cache()

    def test_local_resolves_to_local_embedder(self):
        from claudetube.providers.registry import clear_cache

        clear_cache()
        provider = _get_embedder("local")
        from claudetube.providers.local_embedder import LocalEmbedderProvider

        assert isinstance(provider, LocalEmbedderProvider)
        clear_cache()


# =============================================================================
# _build_scene_text()
# =============================================================================


class TestBuildSceneText:
    """Tests for scene text construction."""

    def test_basic_scene_dict(self):
        scene = {"scene_id": 1, "start_time": 0.0, "end_time": 10.0, "transcript_text": "Hello world"}
        result = _build_scene_text(scene)
        assert "Scene 1" in result
        assert "AUDIO: Hello world" in result

    def test_with_visual_data(self):
        scene = {"scene_id": 1, "start_time": 0.0, "end_time": 10.0, "transcript_text": ""}
        visual = {"description": "A person speaking"}
        result = _build_scene_text(scene, visual_data=visual)
        assert "VISUAL: A person speaking" in result

    def test_with_technical_data(self):
        scene = {"scene_id": 1, "start_time": 0.0, "end_time": 10.0, "transcript_text": ""}
        technical = {"frames": [{"regions": [{"text": "def hello()"}]}]}
        result = _build_scene_text(scene, technical_data=technical)
        assert "TEXT ON SCREEN: def hello()" in result


# =============================================================================
# embed_scene()
# =============================================================================


class TestEmbedScene:
    """Tests for embed_scene() with provider pattern."""

    def test_returns_scene_embedding(self):
        mock_provider = MagicMock()
        mock_provider.embed_sync.return_value = [0.1, 0.2, 0.3]

        with patch("claudetube.analysis.embeddings._get_embedder", return_value=mock_provider):
            result = embed_scene(
                scene={"scene_id": 5, "start_time": 0.0, "end_time": 10.0, "transcript_text": "hello"},
                model="voyage",
            )

        assert isinstance(result, SceneEmbedding)
        assert result.scene_id == 5
        assert result.model == "voyage"
        assert len(result.embedding) == 3
        assert np.allclose(result.embedding, [0.1, 0.2, 0.3])

    def test_passes_text_to_provider(self):
        mock_provider = MagicMock()
        mock_provider.embed_sync.return_value = [0.0]

        with patch("claudetube.analysis.embeddings._get_embedder", return_value=mock_provider):
            embed_scene(
                scene={"scene_id": 1, "start_time": 0.0, "end_time": 5.0, "transcript_text": "content"},
                model="voyage",
            )

        call_args = mock_provider.embed_sync.call_args
        text_arg = call_args[0][0]
        assert "content" in text_arg

    def test_passes_image_paths(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"fake")

        mock_provider = MagicMock()
        mock_provider.embed_sync.return_value = [0.0]

        with patch("claudetube.analysis.embeddings._get_embedder", return_value=mock_provider):
            embed_scene(
                scene={"scene_id": 1, "start_time": 0.0, "end_time": 5.0, "transcript_text": ""},
                keyframe_paths=[str(img)],
                model="voyage",
            )

        call_args = mock_provider.embed_sync.call_args
        images_arg = call_args[1].get("images") or call_args[0][1]
        assert len(images_arg) == 1

    def test_uses_env_model_when_none(self):
        mock_provider = MagicMock()
        mock_provider.embed_sync.return_value = [0.0]

        with (
            patch.dict("os.environ", {"CLAUDETUBE_EMBEDDING_MODEL": "local"}),
            patch("claudetube.analysis.embeddings._get_embedder", return_value=mock_provider) as mock_get,
        ):
            embed_scene(
                scene={"scene_id": 1, "start_time": 0.0, "end_time": 5.0, "transcript_text": ""},
            )

        mock_get.assert_called_once_with("local")

    def test_scene_boundary_object(self):
        """Works with SceneBoundary-like objects."""
        mock_scene = MagicMock()
        mock_scene.scene_id = 3
        mock_scene.start_time = 10.0
        mock_scene.end_time = 20.0
        mock_scene.transcript_text = "test content"

        mock_provider = MagicMock()
        mock_provider.embed_sync.return_value = [0.1, 0.2]

        with patch("claudetube.analysis.embeddings._get_embedder", return_value=mock_provider):
            result = embed_scene(scene=mock_scene, model="voyage")

        assert result.scene_id == 3

    def test_to_dict(self):
        mock_provider = MagicMock()
        mock_provider.embed_sync.return_value = [0.1, 0.2, 0.3]

        with patch("claudetube.analysis.embeddings._get_embedder", return_value=mock_provider):
            result = embed_scene(
                scene={"scene_id": 1, "start_time": 0.0, "end_time": 5.0, "transcript_text": ""},
                model="voyage",
            )

        d = result.to_dict()
        assert d["scene_id"] == 1
        assert d["model"] == "voyage"
        assert d["embedding_dim"] == 3


# =============================================================================
# get_embedding_dim()
# =============================================================================


class TestGetEmbeddingDim:
    """Tests for embedding dimension lookup."""

    def test_voyage_dim(self):
        assert get_embedding_dim("voyage") == 1024

    def test_local_dim(self):
        assert get_embedding_dim("local") == 896

    def test_default_dim(self):
        with patch.dict("os.environ", {"CLAUDETUBE_EMBEDDING_MODEL": "voyage"}):
            assert get_embedding_dim() == 1024
