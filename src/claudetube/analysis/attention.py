"""
Attention priority modeling for human-like video comprehension.

Scores scenes by multiple factors to model where a human expert would focus:
- Relevance to user goal
- Information density
- Novelty relative to previously seen content
- Visual salience (code, diagrams, etc.)
- Audio emphasis (speaker cues)
- Structural importance (intro, conclusion, demos)

Video-type-specific weights adjust factor importance based on content type.

Architecture: Cheap First, Expensive Last
1. TEXT - Use transcript features first (cheap)
2. CACHED - Use pre-computed analysis if available
3. EMBEDDINGS - Use vector similarity only when needed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from claudetube.analysis.embeddings import SceneEmbedding
    from claudetube.cache.scenes import SceneBoundary

logger = logging.getLogger(__name__)


# Default weights for unknown video types
DEFAULT_WEIGHTS = {
    "relevance": 0.20,
    "density": 0.15,
    "visual": 0.15,
    "novelty": 0.15,
    "audio": 0.15,
    "structure": 0.20,
}

# Video-type-specific weight configurations
VIDEO_TYPE_WEIGHTS = {
    "coding_tutorial": {
        "relevance": 0.30,
        "density": 0.25,
        "visual": 0.25,
        "novelty": 0.10,
        "audio": 0.05,
        "structure": 0.05,
    },
    "lecture": {
        "relevance": 0.25,
        "audio": 0.25,
        "structure": 0.20,
        "novelty": 0.15,
        "density": 0.10,
        "visual": 0.05,
    },
    "demo": {
        "relevance": 0.30,
        "visual": 0.30,
        "novelty": 0.20,
        "density": 0.10,
        "audio": 0.05,
        "structure": 0.05,
    },
    "interview": {
        "relevance": 0.30,
        "audio": 0.30,
        "structure": 0.15,
        "novelty": 0.15,
        "density": 0.05,
        "visual": 0.05,
    },
    "presentation": {
        "relevance": 0.25,
        "visual": 0.25,
        "structure": 0.20,
        "density": 0.15,
        "audio": 0.10,
        "novelty": 0.05,
    },
    "screencast": {
        "relevance": 0.25,
        "visual": 0.30,
        "density": 0.20,
        "novelty": 0.15,
        "audio": 0.05,
        "structure": 0.05,
    },
}


@dataclass
class AttentionFactors:
    """Individual attention factors for a scene.

    Each factor is a float from 0.0 to 1.0 indicating the scene's
    score for that dimension.

    Attributes:
        relevance_to_goal: How relevant the scene is to the user's question.
        information_density: How much information is packed into the scene.
        novelty: How different this scene is from previously examined content.
        visual_salience: Whether the scene has visually important content
            (code, diagrams, text).
        audio_emphasis: Whether the speaker emphasizes this content.
        structural_importance: Position/role in video structure
            (intro, conclusion, key demo).
    """

    relevance_to_goal: float
    information_density: float
    novelty: float
    visual_salience: float
    audio_emphasis: float
    structural_importance: float

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "relevance_to_goal": self.relevance_to_goal,
            "information_density": self.information_density,
            "novelty": self.novelty,
            "visual_salience": self.visual_salience,
            "audio_emphasis": self.audio_emphasis,
            "structural_importance": self.structural_importance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AttentionFactors:
        """Create from dictionary."""
        return cls(
            relevance_to_goal=data.get("relevance_to_goal", 0.0),
            information_density=data.get("information_density", 0.0),
            novelty=data.get("novelty", 0.0),
            visual_salience=data.get("visual_salience", 0.0),
            audio_emphasis=data.get("audio_emphasis", 0.0),
            structural_importance=data.get("structural_importance", 0.0),
        )


def get_weights_for_video_type(video_type: str) -> dict[str, float]:
    """Get attention weights for a video type.

    Args:
        video_type: Video category (e.g., 'coding_tutorial', 'lecture').

    Returns:
        Dict mapping factor names to weights (sum to 1.0).
    """
    return VIDEO_TYPE_WEIGHTS.get(video_type, DEFAULT_WEIGHTS)


def calculate_relevance(
    scene: dict | SceneBoundary,
    user_goal: str,
    scene_embedding: SceneEmbedding | None = None,
    goal_embedding: np.ndarray | None = None,
) -> float:
    """Calculate relevance of scene to user's goal.

    Uses embedding similarity if available, otherwise keyword matching.

    Args:
        scene: Scene data (dict or SceneBoundary).
        user_goal: User's question or goal.
        scene_embedding: Pre-computed scene embedding (optional).
        goal_embedding: Pre-computed goal embedding (optional).

    Returns:
        Relevance score from 0.0 to 1.0.
    """
    # Try semantic similarity if embeddings available
    if scene_embedding is not None and goal_embedding is not None:
        try:
            scene_emb = scene_embedding.embedding
            # Cosine similarity
            dot = np.dot(scene_emb, goal_embedding)
            norm = np.linalg.norm(scene_emb) * np.linalg.norm(goal_embedding)
            if norm > 0:
                similarity = dot / norm
                # Convert from [-1, 1] to [0, 1] range
                return float((similarity + 1) / 2)
        except Exception as e:
            logger.debug(f"Embedding similarity failed: {e}")

    # Fallback: keyword matching
    if hasattr(scene, "transcript_text"):
        transcript = scene.transcript_text or ""
    else:
        transcript = scene.get("transcript_text", "")

    if not transcript or not user_goal:
        return 0.0

    transcript_lower = transcript.lower()
    goal_words = set(user_goal.lower().split())

    if not goal_words:
        return 0.0

    # Calculate word overlap
    transcript_words = set(transcript_lower.split())
    matches = sum(1 for w in goal_words if w in transcript_words)
    word_score = matches / len(goal_words)

    # Bonus for exact phrase match
    phrase_bonus = 0.3 if user_goal.lower() in transcript_lower else 0.0

    return min(1.0, word_score + phrase_bonus)


def estimate_information_density(scene: dict | SceneBoundary) -> float:
    """Estimate how much information is packed into a scene.

    Higher scores indicate more content (words, on-screen text, code).

    Args:
        scene: Scene data with transcript_text and optional technical data.

    Returns:
        Information density score from 0.0 to 1.0.
    """
    # Get transcript text
    if hasattr(scene, "transcript_text"):
        transcript = scene.transcript_text or ""
    else:
        transcript = scene.get("transcript_text", "")

    # Get scene duration
    if hasattr(scene, "start_time"):
        duration = (scene.end_time or 0) - (scene.start_time or 0)
    else:
        duration = scene.get("end_time", 0) - scene.get("start_time", 0)

    # Transcript density (words per second)
    word_count = len(transcript.split())
    if duration > 0:
        words_per_second = word_count / duration
        # Normal speech is ~2-3 words/second, dense is 4+
        word_density = min(1.0, words_per_second / 4.0)
    else:
        word_density = min(1.0, word_count / 100)

    # Technical content density (from technical data)
    technical = scene.get("technical", {}) if isinstance(scene, dict) else {}
    ocr_items = len(technical.get("ocr_text", []))
    code_blocks = len(technical.get("code_blocks", []))

    # Visual text density (OCR text + code is high-value content)
    visual_density = min(1.0, (ocr_items + code_blocks * 3) / 10)

    # Combine factors
    return (word_density * 0.6) + (visual_density * 0.4)


def calculate_novelty(
    scene: dict | SceneBoundary,
    previous_scenes: list[dict | SceneBoundary],
    scene_embedding: SceneEmbedding | None = None,
    previous_embeddings: list[SceneEmbedding] | None = None,
) -> float:
    """Calculate how different this scene is from previously examined content.

    High novelty indicates new information the watcher hasn't seen yet.

    Args:
        scene: Current scene to evaluate.
        previous_scenes: Previously examined scenes.
        scene_embedding: Pre-computed embedding for this scene.
        previous_embeddings: Pre-computed embeddings for previous scenes.

    Returns:
        Novelty score from 0.0 to 1.0 (1.0 = completely novel).
    """
    if not previous_scenes:
        return 0.5  # Neutral for first scene

    # Try embedding-based novelty
    if scene_embedding is not None and previous_embeddings:
        try:
            scene_emb = scene_embedding.embedding
            prev_embs = [e.embedding for e in previous_embeddings if e.embedding is not None]

            if prev_embs:
                # Calculate similarity to each previous scene
                similarities = []
                for prev_emb in prev_embs:
                    dot = np.dot(scene_emb, prev_emb)
                    norm = np.linalg.norm(scene_emb) * np.linalg.norm(prev_emb)
                    if norm > 0:
                        similarities.append(dot / norm)

                if similarities:
                    # Novelty = inverse of max similarity to any previous scene
                    max_similarity = max(similarities)
                    # Convert from [-1, 1] similarity to [0, 1] novelty
                    return float(1.0 - (max_similarity + 1) / 2)
        except Exception as e:
            logger.debug(f"Embedding novelty calculation failed: {e}")

    # Fallback: keyword-based novelty
    if hasattr(scene, "transcript_text"):
        transcript = scene.transcript_text or ""
    else:
        transcript = scene.get("transcript_text", "")

    if not transcript:
        return 0.5

    current_words = set(transcript.lower().split())

    # Collect all words from previous scenes
    previous_words: set[str] = set()
    for prev in previous_scenes:
        if hasattr(prev, "transcript_text"):
            prev_text = prev.transcript_text or ""
        else:
            prev_text = prev.get("transcript_text", "")
        previous_words.update(prev_text.lower().split())

    if not current_words:
        return 0.5

    # Novelty = proportion of words not seen before
    new_words = current_words - previous_words
    return len(new_words) / len(current_words)


def detect_visual_salience(scene: dict | SceneBoundary) -> float:
    """Detect if scene has visually important content.

    Higher scores for code, diagrams, slides, terminal output.

    Args:
        scene: Scene data with optional technical analysis.

    Returns:
        Visual salience score from 0.0 to 1.0.
    """
    # Get technical data
    technical = scene.get("technical", {}) if isinstance(scene, dict) else {}

    content_type = technical.get("content_type", "unknown")

    # Content types mapped to salience scores
    salience_by_type = {
        "code": 0.9,
        "diagram": 0.85,
        "slides": 0.7,
        "terminal": 0.75,
        "talking_head": 0.2,
        "unknown": 0.3,
    }

    base_salience = salience_by_type.get(content_type, 0.3)

    # Bonus for OCR text presence
    ocr_text = technical.get("ocr_text", [])
    if ocr_text:
        base_salience = min(1.0, base_salience + 0.1)

    # Bonus for code blocks
    code_blocks = technical.get("code_blocks", [])
    if code_blocks:
        base_salience = min(1.0, base_salience + 0.15)

    return base_salience


def detect_audio_emphasis(scene: dict | SceneBoundary) -> float:
    """Detect if speaker emphasizes this content.

    Looks for verbal cues indicating importance.

    Args:
        scene: Scene data with transcript.

    Returns:
        Audio emphasis score from 0.0 to 1.0.
    """
    # Get transcript
    if hasattr(scene, "transcript_text"):
        transcript = scene.transcript_text or ""
    else:
        transcript = scene.get("transcript_text", "")

    if not transcript:
        return 0.0

    transcript_lower = transcript.lower()

    # Emphasis phrases indicating important content
    emphasis_phrases = [
        # Direct importance markers
        "important",
        "key point",
        "remember",
        "crucial",
        "essential",
        "critical",
        "note that",
        "pay attention",
        "the main",
        "keep in mind",
        "don't forget",
        # Structure markers
        "in summary",
        "to summarize",
        "the takeaway",
        "bottom line",
        "the key is",
        "most importantly",
        # Demonstration markers
        "let me show",
        "watch this",
        "here's how",
        "here is how",
        "now look at",
        "notice that",
        "you'll see",
        # Warning/error markers
        "common mistake",
        "gotcha",
        "be careful",
        "watch out",
        "don't do",
        "avoid",
    ]

    matches = sum(1 for phrase in emphasis_phrases if phrase in transcript_lower)

    # Normalize: 3+ matches = high emphasis
    return min(1.0, matches * 0.25)


def get_structural_weight(
    scene: dict | SceneBoundary,
    video_type: str,
    total_scenes: int,
    video_duration: float,
) -> float:
    """Get structural importance weight based on scene position.

    Intro, conclusion, and key demo positions get higher weights.

    Args:
        scene: Scene data.
        video_type: Video category.
        total_scenes: Total number of scenes in video.
        video_duration: Total video duration in seconds.

    Returns:
        Structural importance score from 0.0 to 1.0.
    """
    # Get scene timing
    if hasattr(scene, "scene_id"):
        scene_id = scene.scene_id
        start_time = scene.start_time or 0
    else:
        scene_id = scene.get("scene_id", 0)
        start_time = scene.get("start_time", 0)

    # Calculate relative position
    position_ratio = start_time / video_duration if video_duration > 0 else 0

    # Base weight
    weight = 0.3

    # Intro bonus (first 10% of video)
    if position_ratio < 0.1:
        weight += 0.3

    # Conclusion bonus (last 10% of video)
    if position_ratio > 0.9:
        weight += 0.25

    # Demo section bonus for tutorials (20-80% is typically where demos happen)
    if video_type in ("coding_tutorial", "demo", "screencast") and 0.2 < position_ratio < 0.8:
            weight += 0.1

    # Chapter start bonus (scene_id <= 2 is often a chapter start)
    if scene_id <= 2:
        weight += 0.15

    return min(1.0, weight)


def calculate_attention_factors(
    scene: dict | SceneBoundary,
    user_goal: str,
    previous_scenes: list[dict | SceneBoundary] | None = None,
    video_type: str = "unknown",
    total_scenes: int = 1,
    video_duration: float = 0.0,
    scene_embedding: SceneEmbedding | None = None,
    goal_embedding: np.ndarray | None = None,
    previous_embeddings: list[SceneEmbedding] | None = None,
) -> AttentionFactors:
    """Calculate all attention factors for a scene.

    Args:
        scene: Scene data to evaluate.
        user_goal: User's question or goal.
        previous_scenes: Previously examined scenes.
        video_type: Video category for structural weighting.
        total_scenes: Total number of scenes.
        video_duration: Total video duration.
        scene_embedding: Pre-computed scene embedding.
        goal_embedding: Pre-computed goal embedding.
        previous_embeddings: Pre-computed embeddings for previous scenes.

    Returns:
        AttentionFactors with all factor scores.
    """
    previous = previous_scenes or []

    return AttentionFactors(
        relevance_to_goal=calculate_relevance(
            scene, user_goal, scene_embedding, goal_embedding
        ),
        information_density=estimate_information_density(scene),
        novelty=calculate_novelty(
            scene, previous, scene_embedding, previous_embeddings
        ),
        visual_salience=detect_visual_salience(scene),
        audio_emphasis=detect_audio_emphasis(scene),
        structural_importance=get_structural_weight(
            scene, video_type, total_scenes, video_duration
        ),
    )


def calculate_attention_priority(
    scene: dict | SceneBoundary,
    user_goal: str,
    video_type: str,
    previous_scenes: list[dict | SceneBoundary] | None = None,
    total_scenes: int = 1,
    video_duration: float = 0.0,
    scene_embedding: SceneEmbedding | None = None,
    goal_embedding: np.ndarray | None = None,
    previous_embeddings: list[SceneEmbedding] | None = None,
    custom_weights: dict[str, float] | None = None,
) -> float:
    """Calculate how much attention a scene deserves.

    Combines multiple factors weighted by video type to produce a single
    priority score. Higher scores indicate scenes that should be examined
    more carefully.

    Args:
        scene: Scene data to evaluate.
        user_goal: User's question or goal.
        video_type: Video category (e.g., 'coding_tutorial', 'lecture').
        previous_scenes: Previously examined scenes (for novelty calculation).
        total_scenes: Total number of scenes in the video.
        video_duration: Total video duration in seconds.
        scene_embedding: Pre-computed scene embedding (optional).
        goal_embedding: Pre-computed goal embedding (optional).
        previous_embeddings: Embeddings for previous scenes (optional).
        custom_weights: Override video-type weights if provided.

    Returns:
        Priority score from 0.0 to 1.0.

    Example:
        >>> scene = {"scene_id": 0, "transcript_text": "Let me show you the code"}
        >>> priority = calculate_attention_priority(
        ...     scene=scene,
        ...     user_goal="show me the code",
        ...     video_type="coding_tutorial",
        ... )
        >>> 0.0 <= priority <= 1.0
        True
    """
    # Calculate all factors
    factors = calculate_attention_factors(
        scene=scene,
        user_goal=user_goal,
        previous_scenes=previous_scenes,
        video_type=video_type,
        total_scenes=total_scenes,
        video_duration=video_duration,
        scene_embedding=scene_embedding,
        goal_embedding=goal_embedding,
        previous_embeddings=previous_embeddings,
    )

    # Get weights for video type
    weights = custom_weights or get_weights_for_video_type(video_type)

    # Calculate weighted sum
    priority = (
        factors.relevance_to_goal * weights["relevance"]
        + factors.information_density * weights["density"]
        + factors.novelty * weights["novelty"]
        + factors.visual_salience * weights["visual"]
        + factors.audio_emphasis * weights["audio"]
        + factors.structural_importance * weights["structure"]
    )

    # Clamp to [0, 1]
    return max(0.0, min(1.0, priority))


def rank_scenes_by_attention(
    scenes: list[dict | SceneBoundary],
    user_goal: str,
    video_type: str = "unknown",
    video_duration: float = 0.0,
    embeddings: list[SceneEmbedding] | None = None,
    goal_embedding: np.ndarray | None = None,
    examined_scene_ids: set[int] | None = None,
) -> list[dict]:
    """Rank all scenes by attention priority.

    Convenience function to rank multiple scenes at once, tracking
    which have been examined for novelty calculation.

    Args:
        scenes: List of scenes to rank.
        user_goal: User's question or goal.
        video_type: Video category.
        video_duration: Total video duration.
        embeddings: Pre-computed embeddings for scenes.
        goal_embedding: Pre-computed goal embedding.
        examined_scene_ids: Scene IDs already examined (excluded from results).

    Returns:
        List of dicts with scene_id, priority, factors, sorted by priority desc.
    """
    examined = examined_scene_ids or set()
    total_scenes = len(scenes)

    # Build embedding lookup
    embedding_by_id: dict[int, SceneEmbedding] = {}
    if embeddings:
        for emb in embeddings:
            embedding_by_id[emb.scene_id] = emb

    results = []
    previous_scenes: list[dict | SceneBoundary] = []
    previous_embeddings: list[SceneEmbedding] = []

    for scene in scenes:
        # Get scene_id
        if hasattr(scene, "scene_id"):
            scene_id = scene.scene_id
        else:
            scene_id = scene.get("scene_id", 0)

        # Skip already examined scenes
        if scene_id in examined:
            previous_scenes.append(scene)
            if scene_id in embedding_by_id:
                previous_embeddings.append(embedding_by_id[scene_id])
            continue

        # Get embedding for this scene
        scene_embedding = embedding_by_id.get(scene_id)

        # Calculate factors
        factors = calculate_attention_factors(
            scene=scene,
            user_goal=user_goal,
            previous_scenes=previous_scenes,
            video_type=video_type,
            total_scenes=total_scenes,
            video_duration=video_duration,
            scene_embedding=scene_embedding,
            goal_embedding=goal_embedding,
            previous_embeddings=previous_embeddings,
        )

        # Get weights and calculate priority
        weights = get_weights_for_video_type(video_type)
        priority = (
            factors.relevance_to_goal * weights["relevance"]
            + factors.information_density * weights["density"]
            + factors.novelty * weights["novelty"]
            + factors.visual_salience * weights["visual"]
            + factors.audio_emphasis * weights["audio"]
            + factors.structural_importance * weights["structure"]
        )
        priority = max(0.0, min(1.0, priority))

        results.append({
            "scene_id": scene_id,
            "priority": priority,
            "factors": factors.to_dict(),
        })

    # Sort by priority descending
    results.sort(key=lambda x: x["priority"], reverse=True)
    return results
