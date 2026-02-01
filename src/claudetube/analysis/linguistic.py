"""
Linguistic transition cue detection in transcripts.

Detects topic transitions from speech patterns like 'next let's talk about...',
'step 1', 'moving on', etc. using pre-compiled regex patterns.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Transition pattern groups
TRANSITION_PATTERNS = [
    # Explicit transitions
    r"\b(next|now)\s+(let'?s|we('ll)?|i('ll)?)\b",
    r"\b(moving on|let's move|let's talk about)\b",
    r"\bnow\s+(that|we|i)\b",
    r"\b(first|second|third|finally|lastly)\b",
    r"\bso\s+(now|let's|we)\b",
    r"\b(okay|alright|all right)\s*,?\s*(so|now|let's)\b",
    # Section markers
    r"\b(step\s+\d+|part\s+\d+)\b",
    r"\bin\s+this\s+(section|part|video)\b",
    r"\b(to\s+summarize|in\s+summary|to\s+recap)\b",
    # Topic shifts
    r"\b(another\s+(thing|way|approach|important))\b",
    r"\b(the\s+(next|last|final)\s+(thing|step|part))\b",
]

# Compile once for performance
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in TRANSITION_PATTERNS]


class Boundary(NamedTuple):
    """Detected linguistic boundary in transcript."""

    timestamp: float
    type: str
    trigger_text: str
    confidence: float


def detect_linguistic_boundaries(
    transcript_segments: list[dict],
) -> list[Boundary]:
    """Detect linguistic transition cues in transcript segments.

    Scans transcript segments for speech patterns that indicate topic
    transitions, section markers, and topic shifts.

    Args:
        transcript_segments: List of segment dicts with 'start' (float)
            and 'text' (str) keys. Typically from SRT parsing or
            Whisper output.

    Returns:
        List of Boundary tuples with timestamp, type, trigger_text,
        and confidence (fixed at 0.7 for linguistic cues).

    Example:
        >>> segments = [
        ...     {"start": 0.0, "text": "Welcome to the tutorial"},
        ...     {"start": 30.0, "text": "Now let's talk about setup"},
        ...     {"start": 60.0, "text": "Step 1 is installing dependencies"},
        ... ]
        >>> boundaries = detect_linguistic_boundaries(segments)
        >>> len(boundaries)
        2
        >>> boundaries[0].timestamp
        30.0
    """
    boundaries: list[Boundary] = []

    for seg in transcript_segments:
        text = seg.get("text", "")
        start = seg.get("start", 0.0)

        for pattern in COMPILED_PATTERNS:
            if pattern.search(text):
                boundaries.append(
                    Boundary(
                        timestamp=start,
                        type="linguistic_cue",
                        trigger_text=text[:50],
                        confidence=0.7,
                    )
                )
                break  # One match per segment

    return boundaries
