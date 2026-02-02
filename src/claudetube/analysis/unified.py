"""
Unified cheap boundary detection.

Combines all cheap boundary detection methods:
1. YouTube chapters (highest confidence, 0.95)
2. Description-parsed timestamps (high confidence, 0.9)
3. Linguistic transition cues (medium confidence, 0.7)
4. Significant pauses (variable confidence, 0.5-0.8)
5. Vocabulary shifts (medium confidence, 0.6)

Merges nearby boundaries and boosts confidence when multiple signals agree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from claudetube.analysis.linguistic import Boundary, detect_linguistic_boundaries
from claudetube.analysis.pause import detect_pause_boundaries
from claudetube.analysis.vocabulary import detect_vocabulary_shifts
from claudetube.operations.chapters import extract_youtube_chapters

if TYPE_CHECKING:
    from pathlib import Path

# Merge threshold in seconds - boundaries closer than this are merged
MERGE_THRESHOLD_SECONDS = 5.0

# Confidence boost when multiple signals agree on same boundary
CONFIDENCE_BOOST = 0.1

# Maximum confidence after boosting
MAX_CONFIDENCE = 0.95


def _chapter_to_boundary(chapter) -> Boundary:
    """Convert a Chapter object to a Boundary.

    Args:
        chapter: Chapter object with title, start, source, confidence

    Returns:
        Boundary with chapter information
    """
    trigger_text = f"[{chapter.source}] {chapter.title}"[:50]
    return Boundary(
        timestamp=chapter.start,
        type="chapter",
        trigger_text=trigger_text,
        confidence=chapter.confidence,
    )


def detect_boundaries_cheap(
    video_info: dict | None = None,
    transcript_segments: list[dict] | None = None,
    srt_path: str | Path | None = None,
    merge_threshold: float = MERGE_THRESHOLD_SECONDS,
) -> list[Boundary]:
    """Detect topic boundaries using all cheap text-based methods.

    Calls multiple detection methods in order of confidence:
    1. YouTube chapters (from video metadata)
    2. Linguistic transition cues (from transcript text)
    3. Significant pauses (from SRT timing)
    4. Vocabulary shifts (from transcript text, uses TF-IDF)

    Merges nearby boundaries (<5s by default) keeping highest confidence
    and boosting confidence when multiple signals agree.

    Args:
        video_info: yt-dlp metadata dict. Used for chapter extraction.
        transcript_segments: List of segment dicts with 'start' and 'text'.
            Used for linguistic and vocabulary detection.
        srt_path: Path to SRT file. Used for pause detection.
        merge_threshold: Seconds threshold for merging nearby boundaries.
            Default 5.0.

    Returns:
        List of Boundary tuples sorted by timestamp. Each boundary has:
        - timestamp: float seconds from start
        - type: str indicating source ('chapter', 'linguistic_cue', 'pause', 'vocabulary_shift')
        - trigger_text: str with context (truncated to 50 chars)
        - confidence: float 0.0-1.0

    Performance:
        Targets <2s for a 30-minute video. All processing is text-based
        (no video decoding). TF-IDF vocabulary analysis is the bottleneck
        (~500ms).

    Example:
        >>> video_info = {"chapters": [{"title": "Intro", "start_time": 0.0, "end_time": 60.0}]}
        >>> segments = [
        ...     {"start": 0.0, "text": "Welcome to the tutorial"},
        ...     {"start": 60.0, "text": "Now let's talk about setup"},
        ... ]
        >>> boundaries = detect_boundaries_cheap(
        ...     video_info=video_info,
        ...     transcript_segments=segments,
        ... )
        >>> len(boundaries)  # Chapter + linguistic cue, possibly merged
        2
    """
    all_boundaries: list[Boundary] = []

    # 1. YouTube chapters (highest confidence: 0.95 native, 0.9 description)
    if video_info:
        chapters = extract_youtube_chapters(video_info)
        all_boundaries.extend(_chapter_to_boundary(ch) for ch in chapters)

    # 2. Linguistic transitions (confidence: 0.7)
    if transcript_segments:
        linguistic = detect_linguistic_boundaries(transcript_segments)
        all_boundaries.extend(linguistic)

    # 3. Pauses (confidence: 0.5-0.8)
    if srt_path:
        pauses = detect_pause_boundaries(srt_path=srt_path)
        all_boundaries.extend(pauses)

    # 4. Vocabulary shifts (confidence: 0.6)
    if transcript_segments:
        vocab = detect_vocabulary_shifts(transcript_segments)
        all_boundaries.extend(vocab)

    # Merge nearby boundaries
    return merge_nearby_boundaries(all_boundaries, threshold=merge_threshold)


def merge_nearby_boundaries(
    boundaries: list[Boundary],
    threshold: float = MERGE_THRESHOLD_SECONDS,
) -> list[Boundary]:
    """Merge boundaries within threshold seconds of each other.

    When boundaries are close together:
    - Keeps the timestamp of the highest-confidence boundary
    - Boosts confidence by 0.1 for each additional signal (capped at 0.95)
    - Combines types as comma-separated list
    - Combines trigger texts

    Args:
        boundaries: List of Boundary tuples to merge.
        threshold: Maximum seconds between boundaries to merge. Default 5.0.

    Returns:
        List of merged Boundary tuples, sorted by timestamp.

    Example:
        >>> from claudetube.analysis.linguistic import Boundary
        >>> boundaries = [
        ...     Boundary(10.0, "chapter", "Intro", 0.95),
        ...     Boundary(12.0, "linguistic_cue", "now let's", 0.7),
        ...     Boundary(50.0, "pause", "3.0s pause", 0.59),
        ... ]
        >>> merged = merge_nearby_boundaries(boundaries, threshold=5.0)
        >>> len(merged)  # 10s and 12s merge, 50s stays separate
        2
        >>> merged[0].confidence  # 0.95 + 0.1 boost = 1.0 capped to 0.95
        0.95
    """
    if not boundaries:
        return []

    # Sort by timestamp
    sorted_b = sorted(boundaries, key=lambda x: x.timestamp)

    # Group nearby boundaries
    groups: list[list[Boundary]] = []
    current_group: list[Boundary] = [sorted_b[0]]

    for b in sorted_b[1:]:
        # Check if this boundary is within threshold of the last one in current group
        if b.timestamp - current_group[-1].timestamp < threshold:
            current_group.append(b)
        else:
            groups.append(current_group)
            current_group = [b]

    # Don't forget the last group
    groups.append(current_group)

    # Merge each group into a single boundary
    merged: list[Boundary] = []
    for group in groups:
        merged.append(_merge_group(group))

    return merged


def _merge_group(group: list[Boundary]) -> Boundary:
    """Merge a group of nearby boundaries into one.

    Args:
        group: List of Boundary tuples to merge (must not be empty).

    Returns:
        Single merged Boundary.
    """
    if len(group) == 1:
        return group[0]

    # Sort by confidence descending to pick best as base
    by_confidence = sorted(group, key=lambda x: x.confidence, reverse=True)
    best = by_confidence[0]

    # Collect all unique types
    types = []
    seen_types = set()
    for b in group:
        if b.type not in seen_types:
            types.append(b.type)
            seen_types.add(b.type)

    # Combine types (first is the primary/highest confidence)
    combined_type = "+".join(types)

    # Boost confidence: +0.1 for each additional signal
    num_signals = len(group)
    boosted_confidence = min(
        best.confidence + (num_signals - 1) * CONFIDENCE_BOOST, MAX_CONFIDENCE
    )

    # Use best trigger text (highest confidence source)
    trigger_text = best.trigger_text

    return Boundary(
        timestamp=best.timestamp,
        type=combined_type,
        trigger_text=trigger_text,
        confidence=round(boosted_confidence, 2),
    )
