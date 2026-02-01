"""
Analysis module for claudetube.

Provides boundary detection and transcript analysis utilities.
"""

from claudetube.analysis.linguistic import (
    Boundary,
    detect_linguistic_boundaries,
)
from claudetube.analysis.pause import (
    detect_pause_boundaries,
    parse_srt_file,
    parse_srt_timestamp,
)
from claudetube.analysis.vocabulary import (
    detect_vocabulary_shifts,
)

__all__ = [
    "Boundary",
    "detect_linguistic_boundaries",
    "detect_pause_boundaries",
    "detect_vocabulary_shifts",
    "parse_srt_file",
    "parse_srt_timestamp",
]
