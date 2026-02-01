"""
Vector index for semantic scene search using ChromaDB.

Provides fast similarity search over scene embeddings with metadata.

Architecture: Cheap First, Expensive Last
1. CACHE - Check for existing index
2. BUILD - Create index from cached embeddings
3. QUERY - Sub-second similarity search

Config:
- Index stored in cache_dir/embeddings/chroma/
- Persistent by default (survives restarts)
- Metadata includes timestamps, transcript previews
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from claudetube.analysis.embeddings import SceneEmbedding
    from claudetube.cache.scenes import SceneBoundary

logger = logging.getLogger(__name__)

# Collection name for scenes
SCENES_COLLECTION = "scenes"

# Maximum transcript preview length in metadata
MAX_TRANSCRIPT_PREVIEW = 500


@dataclass
class SearchResult:
    """Single search result from vector index."""

    scene_id: int
    distance: float  # Lower is better for L2, higher for cosine
    start_time: float
    end_time: float
    transcript_preview: str
    visual_description: str


def _get_chroma_path(cache_dir: Path) -> Path:
    """Get path to ChromaDB storage directory.

    Args:
        cache_dir: Video cache directory.

    Returns:
        Path to embeddings/chroma/ directory.
    """
    return cache_dir / "embeddings" / "chroma"


def has_vector_index(cache_dir: Path) -> bool:
    """Check if vector index exists for this video.

    Args:
        cache_dir: Video cache directory.

    Returns:
        True if ChromaDB index exists.
    """
    chroma_path = _get_chroma_path(cache_dir)
    return chroma_path.exists() and (chroma_path / "chroma.sqlite3").exists()


def build_scene_index(
    cache_dir: Path,
    scenes: list[dict | SceneBoundary],
    embeddings: list[SceneEmbedding],
    video_id: str | None = None,
) -> int:
    """Create searchable vector index of video scenes.

    Stores scene embeddings in ChromaDB with metadata for fast retrieval.

    Args:
        cache_dir: Video cache directory.
        scenes: List of scene data (dicts or SceneBoundary objects).
        embeddings: List of SceneEmbedding objects from embed_scenes().
        video_id: Optional video ID for metadata.

    Returns:
        Number of scenes indexed.

    Raises:
        ImportError: If chromadb not installed.
        ValueError: If scenes and embeddings don't match.
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as e:
        raise ImportError(
            "chromadb required for vector search. "
            "Install with: pip install 'claudetube[search]'"
        ) from e

    if not embeddings:
        logger.warning("No embeddings provided, skipping index build")
        return 0

    # Build scene_id -> embedding mapping
    emb_by_id = {e.scene_id: e for e in embeddings}

    # Load visual descriptions if available
    visual_by_id: dict[int, str] = {}
    for scene in scenes:
        if hasattr(scene, "scene_id"):
            scene_id = scene.scene_id
        else:
            scene_id = scene.get("scene_id", 0)

        visual_path = cache_dir / "scenes" / f"scene_{scene_id:03d}" / "visual.json"
        if visual_path.exists():
            try:
                visual_data = json.loads(visual_path.read_text())
                visual_by_id[scene_id] = visual_data.get("description", "")[:MAX_TRANSCRIPT_PREVIEW]
            except Exception:
                pass

    # Create persistent ChromaDB client
    chroma_path = _get_chroma_path(cache_dir)
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete existing collection if present (for rebuilds)
    with contextlib.suppress(Exception):
        client.delete_collection(SCENES_COLLECTION)

    # Create new collection
    collection = client.create_collection(
        name=SCENES_COLLECTION,
        metadata={"video_id": video_id or "unknown"},
    )

    # Prepare data for batch add
    ids = []
    embs = []
    metadatas = []
    documents = []

    for scene in scenes:
        if hasattr(scene, "scene_id"):
            scene_id = scene.scene_id
            start = scene.start_time
            end = scene.end_time
            transcript = scene.transcript_text or ""
        else:
            scene_id = scene.get("scene_id", 0)
            start = scene.get("start_time", 0)
            end = scene.get("end_time", 0)
            transcript = scene.get("transcript_text", "")

        # Skip if no embedding for this scene
        if scene_id not in emb_by_id:
            logger.warning(f"No embedding for scene {scene_id}, skipping")
            continue

        emb = emb_by_id[scene_id]

        ids.append(f"scene_{scene_id}")
        embs.append(emb.embedding.tolist())
        metadatas.append({
            "scene_id": scene_id,
            "start_time": start,
            "end_time": end,
            "transcript_preview": transcript[:MAX_TRANSCRIPT_PREVIEW],
            "visual_description": visual_by_id.get(scene_id, ""),
        })
        documents.append(transcript)

    if not ids:
        logger.warning("No scenes with embeddings to index")
        return 0

    # Batch add to collection
    collection.add(
        ids=ids,
        embeddings=embs,
        metadatas=metadatas,
        documents=documents,
    )

    logger.info(f"Built vector index with {len(ids)} scenes at {chroma_path}")
    return len(ids)


def load_scene_index(cache_dir: Path):
    """Load existing vector index for a video.

    Args:
        cache_dir: Video cache directory.

    Returns:
        ChromaDB collection or None if not found.

    Raises:
        ImportError: If chromadb not installed.
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as e:
        raise ImportError(
            "chromadb required for vector search. "
            "Install with: pip install 'claudetube[search]'"
        ) from e

    chroma_path = _get_chroma_path(cache_dir)
    if not has_vector_index(cache_dir):
        return None

    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        return client.get_collection(SCENES_COLLECTION)
    except Exception as e:
        logger.warning(f"Failed to load vector index: {e}")
        return None


def search_scenes(
    cache_dir: Path,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> list[SearchResult]:
    """Search for scenes similar to query embedding.

    Args:
        cache_dir: Video cache directory.
        query_embedding: Query embedding vector (same dim as scene embeddings).
        top_k: Number of results to return.

    Returns:
        List of SearchResult objects, sorted by similarity.

    Raises:
        ImportError: If chromadb not installed.
        ValueError: If no index exists.
    """
    collection = load_scene_index(cache_dir)
    if collection is None:
        raise ValueError(
            f"No vector index found at {cache_dir}. "
            "Run build_scene_index() first."
        )

    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )

    # Parse results
    search_results = []
    for meta, dist in zip(
        results["metadatas"][0],
        results["distances"][0],
        strict=True,
    ):
        search_results.append(SearchResult(
            scene_id=meta["scene_id"],
            distance=dist,
            start_time=meta["start_time"],
            end_time=meta["end_time"],
            transcript_preview=meta.get("transcript_preview", ""),
            visual_description=meta.get("visual_description", ""),
        ))

    return search_results


def search_scenes_by_text(
    cache_dir: Path,
    query_text: str,
    top_k: int = 5,
    model: str | None = None,
) -> list[SearchResult]:
    """Search for scenes using text query.

    Embeds the query text and searches the vector index.

    Args:
        cache_dir: Video cache directory.
        query_text: Natural language query.
        top_k: Number of results to return.
        model: Embedding model to use (must match index).

    Returns:
        List of SearchResult objects, sorted by similarity.

    Raises:
        ImportError: If required packages not installed.
        ValueError: If no index exists.
    """
    from claudetube.analysis.embeddings import get_embedding_model

    if model is None:
        model = get_embedding_model()

    # Embed the query
    if model == "voyage":
        query_embedding = _embed_text_voyage(query_text)
    else:
        query_embedding = _embed_text_local(query_text)

    return search_scenes(cache_dir, query_embedding, top_k)


def _embed_text_voyage(text: str) -> np.ndarray:
    """Embed text using Voyage AI."""
    import os

    try:
        import voyageai
    except ImportError as e:
        raise ImportError(
            "voyageai required for Voyage embeddings. "
            "Install with: pip install voyageai"
        ) from e

    if not os.environ.get("VOYAGE_API_KEY"):
        raise RuntimeError("VOYAGE_API_KEY environment variable not set")

    voyage = voyageai.Client()
    result = voyage.embed(
        texts=[text],
        model="voyage-3",
        input_type="query",
    )
    return np.array(result.embeddings[0], dtype=np.float32)


def _embed_text_local(text: str) -> np.ndarray:
    """Embed text using local model (sentence-transformers)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers required for local embeddings. "
            "Install with: pip install sentence-transformers"
        ) from e

    from claudetube.analysis.embeddings import LOCAL_IMAGE_DIM

    model = SentenceTransformer("all-MiniLM-L6-v2")
    text_emb = model.encode(text, convert_to_numpy=True).astype(np.float32)

    # Pad with zeros for image dimension to match scene embeddings
    img_zeros = np.zeros(LOCAL_IMAGE_DIM, dtype=np.float32)
    return np.concatenate([text_emb, img_zeros])


def delete_scene_index(cache_dir: Path) -> bool:
    """Delete vector index for a video.

    Args:
        cache_dir: Video cache directory.

    Returns:
        True if deleted, False if not found.
    """
    import shutil

    chroma_path = _get_chroma_path(cache_dir)
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        logger.info(f"Deleted vector index at {chroma_path}")
        return True
    return False


def get_index_stats(cache_dir: Path) -> dict | None:
    """Get statistics about the vector index.

    Args:
        cache_dir: Video cache directory.

    Returns:
        Dict with index stats or None if not found.
    """
    collection = load_scene_index(cache_dir)
    if collection is None:
        return None

    count = collection.count()
    metadata = collection.metadata

    return {
        "num_scenes": count,
        "video_id": metadata.get("video_id", "unknown"),
        "collection_name": SCENES_COLLECTION,
        "path": str(_get_chroma_path(cache_dir)),
    }
