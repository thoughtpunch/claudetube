"""
Vocabulary shift detection in transcripts using TF-IDF.

Detects topic transitions by identifying windows where vocabulary suddenly
changes, using cosine similarity between TF-IDF vectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from claudetube.analysis.linguistic import Boundary

if TYPE_CHECKING:
    import numpy as np
    from scipy.sparse import spmatrix

# Default window size in seconds
DEFAULT_WINDOW_SECONDS = 30

# Similarity threshold for detecting shifts (below this = topic change)
SIMILARITY_THRESHOLD = 0.3

# Fixed confidence for vocabulary shift boundaries
VOCABULARY_SHIFT_CONFIDENCE = 0.6


def _group_into_windows(
    segments: list[dict],
    window_seconds: int,
) -> list[dict]:
    """Group transcript segments into time-based windows.

    Args:
        segments: List of segment dicts with 'start' and 'text' keys.
        window_seconds: Size of each window in seconds.

    Returns:
        List of window dicts with 'start' and 'text' keys.
    """
    if not segments:
        return []

    windows: list[dict] = []
    current_window: dict = {"start": 0.0, "text": ""}

    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_text = seg.get("text", "")

        # Start a new window if we've exceeded the window duration
        if seg_start >= current_window["start"] + window_seconds:
            if current_window["text"].strip():
                windows.append(current_window)
            current_window = {"start": seg_start, "text": ""}

        current_window["text"] += " " + seg_text

    # Don't forget the last window
    if current_window["text"].strip():
        windows.append(current_window)

    return windows


def _get_top_terms(
    tfidf_row: spmatrix,
    feature_names: np.ndarray,
    n: int,
) -> list[str]:
    """Get top N terms from a TF-IDF row by weight.

    Args:
        tfidf_row: Single row from TF-IDF matrix.
        feature_names: Array of feature names from vectorizer.
        n: Number of top terms to return.

    Returns:
        List of top N terms sorted by TF-IDF weight descending.
    """
    arr = tfidf_row.toarray()[0]
    # Get indices sorted by value descending
    indices = arr.argsort()[-n:][::-1]
    # Filter out zero-weight terms
    return [feature_names[i] for i in indices if arr[i] > 0]


def detect_vocabulary_shifts(
    transcript_segments: list[dict],
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[Boundary]:
    """Detect vocabulary shifts in transcript segments using TF-IDF.

    Groups transcript into time windows, computes TF-IDF vectors for each,
    and identifies boundaries where cosine similarity drops below threshold.

    Args:
        transcript_segments: List of segment dicts with 'start' (float)
            and 'text' (str) keys.
        window_seconds: Size of each time window in seconds. Default 30.
        similarity_threshold: Similarity below this indicates a topic shift.
            Default 0.3.

    Returns:
        List of Boundary tuples with timestamp, type='vocabulary_shift',
        trigger_text showing top keywords, and confidence=0.6.

    Example:
        >>> segments = [
        ...     {"start": 0.0, "text": "Python is a programming language"},
        ...     {"start": 30.0, "text": "Machine learning uses neural networks"},
        ...     {"start": 60.0, "text": "Neural networks are trained with data"},
        ... ]
        >>> boundaries = detect_vocabulary_shifts(segments, window_seconds=30)
        >>> # May detect shift between "Python programming" and "ML neural networks"
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for vocabulary shift detection. "
            "Install with: pip install scikit-learn"
        ) from e

    # Group segments into time windows
    windows = _group_into_windows(transcript_segments, window_seconds)

    if len(windows) < 2:
        return []

    # Extract text from windows
    texts = [w["text"] for w in windows]

    # Build TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=100,
        ngram_range=(1, 2),  # Unigrams and bigrams
    )

    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        # Empty vocabulary (all stop words or empty text)
        return []

    feature_names = vectorizer.get_feature_names_out()
    boundaries: list[Boundary] = []

    # Compare adjacent windows
    for i in range(1, len(windows)):
        sim = cosine_similarity(tfidf[i - 1], tfidf[i])[0][0]

        if sim < similarity_threshold:
            # Get top keywords for context
            keywords_before = _get_top_terms(tfidf[i - 1], feature_names, 5)
            keywords_after = _get_top_terms(tfidf[i], feature_names, 5)

            # Format trigger text with keywords
            trigger_text = (
                f"vocab shift (sim={sim:.2f}): "
                f"{', '.join(keywords_before[:3])} → {', '.join(keywords_after[:3])}"
            )

            boundaries.append(
                Boundary(
                    timestamp=windows[i]["start"],
                    type="vocabulary_shift",
                    trigger_text=trigger_text[:50],  # Truncate to 50 chars
                    confidence=VOCABULARY_SHIFT_CONFIDENCE,
                )
            )

    return boundaries
