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
from claudetube.analysis.unified import (
    detect_boundaries_cheap,
    merge_nearby_boundaries,
)
from claudetube.analysis.vocabulary import (
    detect_vocabulary_shifts,
)

__all__ = [
    "Boundary",
    "detect_boundaries_cheap",
    "detect_linguistic_boundaries",
    "detect_pause_boundaries",
    "detect_vocabulary_shifts",
    "merge_nearby_boundaries",
    "parse_srt_file",
    "parse_srt_timestamp",
]
