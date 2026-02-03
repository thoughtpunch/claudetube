"""
Tests for multimodal scene embeddings module.
"""

import importlib.util
import json

import numpy as np
import pytest


def _has_sentence_transformers():
    """Check if sentence-transformers is installed."""
    return importlib.util.find_spec("sentence_transformers") is not None


class TestGetEmbeddingModel:
    """Tests for get_embedding_model function."""

    def test_default_model(self, monkeypatch):
        """Should return 'voyage' by default."""
        from claudetube.analysis.embeddings import get_embedding_model

        monkeypatch.delenv("CLAUDETUBE_EMBEDDING_MODEL", raising=False)
        assert get_embedding_model() == "voyage"

    def test_voyage_model(self, monkeypatch):
        """Should return 'voyage' when configured."""
        from claudetube.analysis.embeddings import get_embedding_model

        monkeypatch.setenv("CLAUDETUBE_EMBEDDING_MODEL", "voyage")
        assert get_embedding_model() == "voyage"

    def test_local_model(self, monkeypatch):
        """Should return 'local' when configured."""
        from claudetube.analysis.embeddings import get_embedding_model

        monkeypatch.setenv("CLAUDETUBE_EMBEDDING_MODEL", "local")
        assert get_embedding_model() == "local"

    def test_case_insensitive(self, monkeypatch):
        """Should handle case variations."""
        from claudetube.analysis.embeddings import get_embedding_model

        monkeypatch.setenv("CLAUDETUBE_EMBEDDING_MODEL", "LOCAL")
        assert get_embedding_model() == "local"

        monkeypatch.setenv("CLAUDETUBE_EMBEDDING_MODEL", "Voyage")
        assert get_embedding_model() == "voyage"

    def test_invalid_model_fallback(self, monkeypatch):
        """Should fall back to voyage for invalid models."""
        from claudetube.analysis.embeddings import get_embedding_model

        monkeypatch.setenv("CLAUDETUBE_EMBEDDING_MODEL", "invalid")
        assert get_embedding_model() == "voyage"


class TestBuildSceneText:
    """Tests for _build_scene_text function."""

    def test_basic_scene_dict(self):
        """Should build text from scene dict."""
        from claudetube.analysis.embeddings import _build_scene_text

        scene = {
            "scene_id": 1,
            "start_time": 10.0,
            "end_time": 30.0,
            "transcript_text": "Hello, welcome to the tutorial.",
        }

        text = _build_scene_text(scene)

        assert "Scene 1" in text
        assert "10.0s - 30.0s" in text
        assert "Hello, welcome to the tutorial" in text

    def test_with_visual_data(self):
        """Should include visual description."""
        from claudetube.analysis.embeddings import _build_scene_text

        scene = {
            "scene_id": 0,
            "start_time": 0.0,
            "end_time": 10.0,
            "transcript_text": "Introduction",
        }
        visual_data = {"description": "Person standing at a whiteboard"}

        text = _build_scene_text(scene, visual_data=visual_data)

        assert "VISUAL: Person standing at a whiteboard" in text

    def test_with_technical_data(self):
        """Should include OCR text."""
        from claudetube.analysis.embeddings import _build_scene_text

        scene = {
            "scene_id": 2,
            "start_time": 60.0,
            "end_time": 90.0,
            "transcript_text": "Let me show you the code",
        }
        technical_data = {
            "frames": [
                {
                    "regions": [
                        {"text": "def hello():"},
                        {"text": "    return 'world'"},
                    ]
                }
            ]
        }

        text = _build_scene_text(scene, technical_data=technical_data)

        assert "TEXT ON SCREEN:" in text
        assert "def hello():" in text

    def test_scene_boundary_object(self):
        """Should work with SceneBoundary objects."""
        from claudetube.analysis.embeddings import _build_scene_text
        from claudetube.cache.scenes import SceneBoundary

        scene = SceneBoundary(
            scene_id=5,
            start_time=100.0,
            end_time=120.0,
            transcript_text="This is the main content.",
        )

        text = _build_scene_text(scene)

        assert "Scene 5" in text
        assert "This is the main content" in text


class TestSceneEmbedding:
    """Tests for SceneEmbedding dataclass."""

    def test_to_dict(self):
        """Should convert to dict without embedding array."""
        from claudetube.analysis.embeddings import SceneEmbedding

        emb = SceneEmbedding(
            scene_id=3,
            embedding=np.zeros(512, dtype=np.float32),
            model="local",
        )

        data = emb.to_dict()

        assert data["scene_id"] == 3
        assert data["model"] == "local"
        assert data["embedding_dim"] == 512
        assert "embedding" not in data


class TestGetEmbeddingDim:
    """Tests for get_embedding_dim function."""

    def test_voyage_dim(self):
        """Should return 1024 for voyage."""
        from claudetube.analysis.embeddings import get_embedding_dim

        assert get_embedding_dim("voyage") == 1024

    def test_local_dim(self):
        """Should return 896 for local (384 + 512)."""
        from claudetube.analysis.embeddings import get_embedding_dim

        assert get_embedding_dim("local") == 896


class TestSaveLoadEmbeddings:
    """Tests for save_embeddings and load_embeddings functions."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Should save and load embeddings correctly."""
        from claudetube.analysis.embeddings import (
            SceneEmbedding,
            load_embeddings,
            save_embeddings,
        )

        embeddings = [
            SceneEmbedding(
                scene_id=0,
                embedding=np.random.randn(512).astype(np.float32),
                model="local",
            ),
            SceneEmbedding(
                scene_id=1,
                embedding=np.random.randn(512).astype(np.float32),
                model="local",
            ),
        ]

        save_embeddings(tmp_path, embeddings)

        # Check files created
        assert (tmp_path / "embeddings" / "scene_embeddings.npy").exists()
        assert (tmp_path / "embeddings" / "scene_ids.json").exists()
        assert (tmp_path / "embeddings" / "metadata.json").exists()

        # Load and verify
        loaded = load_embeddings(tmp_path)
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].scene_id == 0
        assert loaded[1].scene_id == 1
        assert loaded[0].model == "local"
        assert np.allclose(loaded[0].embedding, embeddings[0].embedding)
        assert np.allclose(loaded[1].embedding, embeddings[1].embedding)

    def test_load_nonexistent(self, tmp_path):
        """Should return None for nonexistent embeddings."""
        from claudetube.analysis.embeddings import load_embeddings

        result = load_embeddings(tmp_path)
        assert result is None

    def test_save_empty_list(self, tmp_path):
        """Should handle empty embedding list."""
        from claudetube.analysis.embeddings import save_embeddings

        save_embeddings(tmp_path, [])

        # Should not create files for empty list
        assert not (tmp_path / "embeddings" / "scene_embeddings.npy").exists()

    def test_metadata_content(self, tmp_path):
        """Should save correct metadata."""
        from claudetube.analysis.embeddings import SceneEmbedding, save_embeddings

        embeddings = [
            SceneEmbedding(
                scene_id=0,
                embedding=np.zeros(1024, dtype=np.float32),
                model="voyage",
            ),
        ]

        save_embeddings(tmp_path, embeddings)

        metadata = json.loads((tmp_path / "embeddings" / "metadata.json").read_text())
        assert metadata["model"] == "voyage"
        assert metadata["embedding_dim"] == 1024
        assert metadata["num_scenes"] == 1
        assert metadata["scene_ids"] == [0]


class TestHasEmbeddings:
    """Tests for has_embeddings function."""

    def test_no_embeddings(self, tmp_path):
        """Should return False when no embeddings exist."""
        from claudetube.analysis.embeddings import has_embeddings

        assert has_embeddings(tmp_path) is False

    def test_with_embeddings(self, tmp_path):
        """Should return True when embeddings exist."""
        from claudetube.analysis.embeddings import (
            SceneEmbedding,
            has_embeddings,
            save_embeddings,
        )

        embeddings = [
            SceneEmbedding(
                scene_id=0,
                embedding=np.zeros(512, dtype=np.float32),
                model="local",
            ),
        ]
        save_embeddings(tmp_path, embeddings)

        assert has_embeddings(tmp_path) is True


class TestEmbedSceneLocal:
    """Tests for local embedding (when dependencies available)."""

    @pytest.mark.skipif(
        not _has_sentence_transformers(),
        reason="sentence-transformers not installed",
    )
    def test_embed_text_only(self, monkeypatch):
        """Should create embedding with text only."""
        from claudetube.analysis.embeddings import embed_scene

        monkeypatch.setenv("CLAUDETUBE_EMBEDDING_MODEL", "local")

        scene = {
            "scene_id": 0,
            "start_time": 0.0,
            "end_time": 10.0,
            "transcript_text": "Hello world, this is a test.",
        }

        result = embed_scene(scene, model="local")

        assert result.scene_id == 0
        assert result.model == "local"
        # Should have text (384) + image (512) dimensions
        assert len(result.embedding) == 896


class TestEmbedSceneVoyage:
    """Tests for Voyage AI embedding."""

    def test_voyage_requires_api_key(self, monkeypatch):
        """Should raise error without API key."""
        from claudetube.analysis.embeddings import embed_scene

        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.setenv("CLAUDETUBE_EMBEDDING_MODEL", "voyage")

        scene = {
            "scene_id": 0,
            "start_time": 0.0,
            "end_time": 10.0,
            "transcript_text": "Test",
        }

        # Should raise error about missing API key or missing voyageai package
        with pytest.raises((RuntimeError, ImportError, ValueError)):
            embed_scene(scene, model="voyage")
