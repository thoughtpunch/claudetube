"""
Analysis module for claudetube.

Provides boundary detection and transcript analysis utilities.
"""

from claudetube.analysis.alignment import (
    align_transcript_to_scenes,
    align_transcript_to_scenes_simple,
)
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
from claudetube.analysis.visual import (
    detect_visual_boundaries,
    detect_visual_boundaries_fast,
    should_use_visual_detection,
)
from claudetube.analysis.vocabulary import (
    detect_vocabulary_shifts,
)

__all__ = [
    "Boundary",
    "align_transcript_to_scenes",
    "align_transcript_to_scenes_simple",
    "detect_boundaries_cheap",
    "detect_linguistic_boundaries",
    "detect_pause_boundaries",
    "detect_visual_boundaries",
    "detect_visual_boundaries_fast",
    "detect_vocabulary_shifts",
    "merge_nearby_boundaries",
    "parse_srt_file",
    "parse_srt_timestamp",
    "should_use_visual_detection",
]
