"""
Multimodal scene embeddings for semantic search.

Creates unified embeddings per scene combining visual + audio + text.
Supports Voyage AI multimodal-3 (best quality) and local fallback
(CLIP + sentence-transformers).

Architecture: Cheap First, Expensive Last
1. CACHE - Check for cached embeddings first
2. TEXT-ONLY - Transcript embeddings are cheaper than multimodal
3. MULTIMODAL - Add visual embeddings only when needed

Config via CLAUDETUBE_EMBEDDING_MODEL env var:
- "voyage" (default): Use Voyage AI multimodal-3
- "local": Use CLIP + sentence-transformers
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from claudetube.cache.scenes import SceneBoundary

logger = logging.getLogger(__name__)

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "voyage"

# Voyage AI embedding dimension
VOYAGE_EMBEDDING_DIM = 1024

# Local embedding dimensions
LOCAL_TEXT_DIM = 384  # all-MiniLM-L6-v2
LOCAL_IMAGE_DIM = 512  # CLIP ViT-B-32
LOCAL_COMBINED_DIM = LOCAL_TEXT_DIM + LOCAL_IMAGE_DIM  # 896


@dataclass
class SceneEmbedding:
    """Embedding for a single scene."""

    scene_id: int
    embedding: np.ndarray
    model: str  # "voyage" or "local"

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (without embedding array)."""
        return {
            "scene_id": self.scene_id,
            "model": self.model,
            "embedding_dim": len(self.embedding),
        }


def get_embedding_model() -> str:
    """Get the configured embedding model from environment.

    Returns:
        "voyage" or "local" based on CLAUDETUBE_EMBEDDING_MODEL env var.
    """
    model = os.environ.get("CLAUDETUBE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).lower()
    if model not in ("voyage", "local"):
        logger.warning(f"Unknown embedding model '{model}', falling back to 'voyage'")
        return DEFAULT_EMBEDDING_MODEL
    return model


def _build_scene_text(
    scene: dict | SceneBoundary,
    visual_data: dict | None = None,
    technical_data: dict | None = None,
) -> str:
    """Build combined text content for a scene.

    Args:
        scene: Scene data (dict or SceneBoundary).
        visual_data: Optional visual.json data for the scene.
        technical_data: Optional technical.json data for the scene.

    Returns:
        Combined text for embedding.
    """
    parts = []

    # Scene timing
    if hasattr(scene, "scene_id"):
        scene_id = scene.scene_id
        start = scene.start_time
        end = scene.end_time
        transcript_text = scene.transcript_text
    else:
        scene_id = scene.get("scene_id", 0)
        start = scene.get("start_time", 0)
        end = scene.get("end_time", 0)
        transcript_text = scene.get("transcript_text", "")

    parts.append(f"Scene {scene_id} ({start:.1f}s - {end:.1f}s)")

    # Transcript/audio content
    if transcript_text:
        parts.append(f"AUDIO: {transcript_text}")

    # Visual description
    if visual_data:
        description = visual_data.get("description", "")
        if description:
            parts.append(f"VISUAL: {description}")

    # OCR text (from technical.json)
    if technical_data:
        ocr_texts = []
        for frame in technical_data.get("frames", []):
            for region in frame.get("regions", []):
                text = region.get("text", "").strip()
                if text:
                    ocr_texts.append(text)
        if ocr_texts:
            parts.append(f"TEXT ON SCREEN: {' '.join(ocr_texts)}")

    return "\n\n".join(parts)


def _embed_scene_voyage(
    scene_text: str,
    keyframe_paths: list[str | Path],
) -> np.ndarray:
    """Create multimodal embedding using Voyage AI.

    Args:
        scene_text: Combined text content for the scene.
        keyframe_paths: Paths to keyframe images (max 3 used).

    Returns:
        Embedding vector as numpy array.

    Raises:
        ImportError: If voyageai not installed.
        RuntimeError: If VOYAGE_API_KEY not set.
    """
    try:
        import voyageai
    except ImportError as e:
        raise ImportError(
            "Voyage AI client required for embeddings. "
            "Install with: pip install voyageai"
        ) from e

    if not os.environ.get("VOYAGE_API_KEY"):
        raise RuntimeError(
            "VOYAGE_API_KEY environment variable not set. "
            "Get an API key from https://dash.voyageai.com/"
        )

    # Lazy import PIL only when needed
    from PIL import Image

    voyage = voyageai.Client()

    # Load keyframe images (max 3)
    images = []
    for path in keyframe_paths[:3]:
        try:
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            logger.warning(f"Failed to load keyframe {path}: {e}")

    # Build multimodal input
    # Voyage multimodal_embed accepts list of inputs where each input
    # is a list of [text_or_image, ...]
    inputs = [[scene_text] + images] if images else [[scene_text]]

    result = voyage.multimodal_embed(
        inputs=inputs,
        model="voyage-multimodal-3",
        input_type="document",
    )

    return np.array(result.embeddings[0], dtype=np.float32)


def _embed_scene_local(
    scene_text: str,
    keyframe_paths: list[str | Path],
) -> np.ndarray:
    """Create embedding using local models (CLIP + sentence-transformers).

    Args:
        scene_text: Combined text content for the scene.
        keyframe_paths: Paths to keyframe images (max 3 used).

    Returns:
        Concatenated embedding vector (text: 384d + image: 512d = 896d).

    Raises:
        ImportError: If required packages not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers required for local embeddings. "
            "Install with: pip install sentence-transformers"
        ) from e

    # Text embedding
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    text_emb = text_model.encode(scene_text, convert_to_numpy=True)
    text_emb = text_emb.astype(np.float32)

    # Image embedding (if keyframes available)
    if keyframe_paths:
        try:
            import open_clip
            import torch
            from PIL import Image

            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            clip_model.eval()

            img_embs = []
            for path in keyframe_paths[:3]:
                try:
                    img = preprocess(Image.open(path)).unsqueeze(0)
                    with torch.no_grad():
                        img_emb = clip_model.encode_image(img).squeeze().numpy()
                    img_embs.append(img_emb)
                except Exception as e:
                    logger.warning(f"Failed to encode keyframe {path}: {e}")

            if img_embs:
                avg_img_emb = np.mean(img_embs, axis=0).astype(np.float32)
            else:
                avg_img_emb = np.zeros(LOCAL_IMAGE_DIM, dtype=np.float32)

        except ImportError:
            logger.warning("open-clip-torch not installed, using text-only embedding")
            avg_img_emb = np.zeros(LOCAL_IMAGE_DIM, dtype=np.float32)
    else:
        avg_img_emb = np.zeros(LOCAL_IMAGE_DIM, dtype=np.float32)

    # Concatenate text and image embeddings
    return np.concatenate([text_emb, avg_img_emb])


def embed_scene(
    scene: dict | SceneBoundary,
    keyframe_paths: list[str | Path] | None = None,
    visual_data: dict | None = None,
    technical_data: dict | None = None,
    model: str | None = None,
) -> SceneEmbedding:
    """Create embedding for a single scene.

    Args:
        scene: Scene data (dict or SceneBoundary).
        keyframe_paths: Paths to keyframe images for this scene.
        visual_data: Optional visual.json data.
        technical_data: Optional technical.json data (OCR results).
        model: Override embedding model ("voyage" or "local").
            If None, uses CLAUDETUBE_EMBEDDING_MODEL env var.

    Returns:
        SceneEmbedding with scene_id, embedding vector, and model name.
    """
    if model is None:
        model = get_embedding_model()

    # Build combined text
    scene_text = _build_scene_text(scene, visual_data, technical_data)

    # Get scene_id
    if hasattr(scene, "scene_id"):
        scene_id = scene.scene_id
    else:
        scene_id = scene.get("scene_id", 0)

    # Generate embedding
    kf_paths = keyframe_paths or []
    if model == "voyage":
        embedding = _embed_scene_voyage(scene_text, kf_paths)
    else:
        embedding = _embed_scene_local(scene_text, kf_paths)

    return SceneEmbedding(
        scene_id=scene_id,
        embedding=embedding,
        model=model,
    )


def embed_scenes(
    scenes: list[dict | SceneBoundary],
    cache_dir: Path,
    model: str | None = None,
    skip_cached: bool = True,
) -> list[SceneEmbedding]:
    """Embed multiple scenes, using cache when available.

    Args:
        scenes: List of scenes to embed.
        cache_dir: Video cache directory.
        model: Override embedding model.
        skip_cached: If True (default), skip scenes with cached embeddings.

    Returns:
        List of SceneEmbedding objects.
    """
    if model is None:
        model = get_embedding_model()

    # Check for cached embeddings
    cached = load_embeddings(cache_dir)
    cached_ids = set()
    if cached and skip_cached:
        cached_ids = {e.scene_id for e in cached}
        logger.info(f"Found {len(cached_ids)} cached scene embeddings")

    results = list(cached) if cached else []

    # Import scene utilities
    from claudetube.cache.scenes import (
        get_technical_json_path,
        get_visual_json_path,
        list_scene_keyframes,
    )

    for scene in scenes:
        if hasattr(scene, "scene_id"):
            scene_id = scene.scene_id
        else:
            scene_id = scene.get("scene_id", 0)

        if scene_id in cached_ids:
            logger.debug(f"Skipping cached scene {scene_id}")
            continue

        # Load visual and technical data if available
        visual_data = None
        visual_path = get_visual_json_path(cache_dir, scene_id)
        if visual_path.exists():
            try:
                visual_data = json.loads(visual_path.read_text())
            except Exception as e:
                logger.warning(f"Failed to load visual.json for scene {scene_id}: {e}")

        technical_data = None
        technical_path = get_technical_json_path(cache_dir, scene_id)
        if technical_path.exists():
            try:
                technical_data = json.loads(technical_path.read_text())
            except Exception as e:
                logger.warning(f"Failed to load technical.json for scene {scene_id}: {e}")

        # Get keyframes
        keyframe_paths = [str(p) for p in list_scene_keyframes(cache_dir, scene_id)]

        # Embed
        try:
            emb = embed_scene(
                scene=scene,
                keyframe_paths=keyframe_paths,
                visual_data=visual_data,
                technical_data=technical_data,
                model=model,
            )
            results.append(emb)
            logger.info(f"Embedded scene {scene_id}")
        except Exception as e:
            logger.error(f"Failed to embed scene {scene_id}: {e}")

    return results


def save_embeddings(
    cache_dir: Path,
    embeddings: list[SceneEmbedding],
) -> None:
    """Save embeddings to cache directory.

    Saves:
    - embeddings/scene_embeddings.npy: numpy array of all embeddings
    - embeddings/scene_ids.json: list of scene IDs (for mapping)
    - embeddings/metadata.json: model info and dimensions

    Args:
        cache_dir: Video cache directory.
        embeddings: List of SceneEmbedding objects.
    """
    if not embeddings:
        logger.warning("No embeddings to save")
        return

    emb_dir = cache_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Stack embeddings into array
    emb_array = np.stack([e.embedding for e in embeddings])

    # Save numpy array
    np.save(emb_dir / "scene_embeddings.npy", emb_array)

    # Save scene IDs for mapping
    scene_ids = [e.scene_id for e in embeddings]
    (emb_dir / "scene_ids.json").write_text(json.dumps(scene_ids))

    # Save metadata
    first_emb = embeddings[0]
    metadata = {
        "model": first_emb.model,
        "embedding_dim": len(first_emb.embedding),
        "num_scenes": len(embeddings),
        "scene_ids": scene_ids,
    }
    (emb_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info(f"Saved {len(embeddings)} embeddings to {emb_dir}")


def load_embeddings(cache_dir: Path) -> list[SceneEmbedding] | None:
    """Load cached embeddings from cache directory.

    Args:
        cache_dir: Video cache directory.

    Returns:
        List of SceneEmbedding objects, or None if not found.
    """
    emb_dir = cache_dir / "embeddings"
    emb_path = emb_dir / "scene_embeddings.npy"
    ids_path = emb_dir / "scene_ids.json"
    meta_path = emb_dir / "metadata.json"

    if not emb_path.exists():
        return None

    try:
        emb_array = np.load(emb_path)
        scene_ids = json.loads(ids_path.read_text())
        metadata = json.loads(meta_path.read_text())

        model = metadata.get("model", "unknown")

        embeddings = []
        for i, scene_id in enumerate(scene_ids):
            embeddings.append(
                SceneEmbedding(
                    scene_id=scene_id,
                    embedding=emb_array[i],
                    model=model,
                )
            )

        logger.info(f"Loaded {len(embeddings)} embeddings from {emb_dir}")
        return embeddings

    except Exception as e:
        logger.warning(f"Failed to load embeddings from {emb_dir}: {e}")
        return None


def has_embeddings(cache_dir: Path) -> bool:
    """Check if embeddings exist for this video.

    Args:
        cache_dir: Video cache directory.

    Returns:
        True if scene_embeddings.npy exists.
    """
    return (cache_dir / "embeddings" / "scene_embeddings.npy").exists()


def get_embedding_dim(model: str | None = None) -> int:
    """Get embedding dimension for a model.

    Args:
        model: Model name ("voyage" or "local").
            If None, uses configured model.

    Returns:
        Embedding dimension.
    """
    if model is None:
        model = get_embedding_model()

    if model == "voyage":
        return VOYAGE_EMBEDDING_DIM
    else:
        return LOCAL_COMBINED_DIM
