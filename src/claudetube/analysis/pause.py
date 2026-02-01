"""
Pause-based boundary detection in transcripts.

Detects significant pauses (>2 seconds) between transcript segments
as potential topic boundaries. Longer pauses indicate higher confidence.
"""

from __future__ import annotations

import re
from pathlib import Path

from claudetube.analysis.linguistic import Boundary

# SRT timestamp pattern: HH:MM:SS,mmm
SRT_TIME_PATTERN = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")

# Minimum gap in seconds to consider as a boundary
MIN_PAUSE_SECONDS = 2.0

# Confidence calculation parameters
BASE_CONFIDENCE = 0.5
CONFIDENCE_PER_SECOND = 0.03
MAX_CONFIDENCE = 0.8


def parse_srt_timestamp(timestamp: str) -> float:
    """Parse SRT timestamp to seconds.

    Args:
        timestamp: SRT format timestamp (HH:MM:SS,mmm)

    Returns:
        Time in seconds as float.

    Raises:
        ValueError: If timestamp format is invalid.
    """
    match = SRT_TIME_PATTERN.match(timestamp)
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {timestamp}")
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


def parse_srt_file(srt_path: str | Path) -> list[dict]:
    """Parse SRT file into segments with start/end times.

    Args:
        srt_path: Path to SRT file.

    Returns:
        List of dicts with 'start', 'end', and 'text' keys.
    """
    path = Path(srt_path)
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8")
    segments = []

    # Split into subtitle blocks (separated by blank lines)
    blocks = re.split(r"\n\n+", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        # Find the timestamp line (contains " --> ")
        timestamp_line = None
        text_start_idx = 0
        for i, line in enumerate(lines):
            if " --> " in line:
                timestamp_line = line
                text_start_idx = i + 1
                break

        if not timestamp_line:
            continue

        # Parse timestamps: "00:00:01,000 --> 00:00:04,500"
        parts = timestamp_line.split(" --> ")
        if len(parts) != 2:
            continue

        try:
            start = parse_srt_timestamp(parts[0].strip())
            end = parse_srt_timestamp(parts[1].strip())
        except ValueError:
            continue

        # Join remaining lines as text
        text = " ".join(lines[text_start_idx:]).strip()

        segments.append({"start": start, "end": end, "text": text})

    return segments


def detect_pause_boundaries(
    srt_path: str | Path | None = None,
    segments: list[dict] | None = None,
) -> list[Boundary]:
    """Detect significant pauses as potential topic boundaries.

    Identifies gaps >2 seconds between transcript segments. Longer pauses
    result in higher confidence scores.

    Args:
        srt_path: Path to SRT file. Either this or segments must be provided.
        segments: Pre-parsed segments with 'start' and 'end' keys.
            If provided, srt_path is ignored.

    Returns:
        List of Boundary tuples with timestamp, type='pause',
        trigger_text showing gap duration, and calculated confidence.

    Example:
        >>> # From SRT file
        >>> boundaries = detect_pause_boundaries(srt_path="video.srt")
        >>> for b in boundaries:
        ...     print(f"{b.timestamp}s: {b.trigger_text} (conf={b.confidence})")

        >>> # From pre-parsed segments
        >>> segs = [
        ...     {"start": 0.0, "end": 5.0, "text": "Hello"},
        ...     {"start": 10.0, "end": 15.0, "text": "World"},
        ... ]
        >>> boundaries = detect_pause_boundaries(segments=segs)
        >>> len(boundaries)  # 5-second gap detected
        1
    """
    if segments is None:
        if srt_path is None:
            return []
        segments = parse_srt_file(srt_path)

    if len(segments) < 2:
        return []

    boundaries: list[Boundary] = []

    for i in range(1, len(segments)):
        prev_seg = segments[i - 1]
        curr_seg = segments[i]

        # Need both end of previous and start of current
        prev_end = prev_seg.get("end")
        curr_start = curr_seg.get("start")

        if prev_end is None or curr_start is None:
            continue

        gap = curr_start - prev_end

        if gap > MIN_PAUSE_SECONDS:
            # Calculate confidence: 0.5 base + 0.03 per second, max 0.8
            confidence = min(BASE_CONFIDENCE + (gap * CONFIDENCE_PER_SECOND), MAX_CONFIDENCE)

            boundaries.append(
                Boundary(
                    timestamp=curr_start,
                    type="pause",
                    trigger_text=f"{gap:.1f}s pause",
                    confidence=round(confidence, 2),
                )
            )

    return boundaries
