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
from claudetube.analysis.ocr import (
    FrameOCRResult,
    TextRegion,
    extract_text_from_frame,
    extract_text_from_scene,
    load_ocr_results,
    save_ocr_results,
)
from claudetube.analysis.code import (
    CodeBlock,
    FrameCodeResult,
    analyze_frame_for_code,
    detect_language,
    extract_code_blocks,
    is_likely_code,
    load_code_results,
    save_code_results,
)

__all__ = [
    "Boundary",
    "CodeBlock",
    "FrameCodeResult",
    "FrameOCRResult",
    "TextRegion",
    "align_transcript_to_scenes",
    "align_transcript_to_scenes_simple",
    "analyze_frame_for_code",
    "detect_boundaries_cheap",
    "detect_language",
    "detect_linguistic_boundaries",
    "detect_pause_boundaries",
    "detect_visual_boundaries",
    "detect_visual_boundaries_fast",
    "detect_vocabulary_shifts",
    "extract_code_blocks",
    "extract_text_from_frame",
    "extract_text_from_scene",
    "is_likely_code",
    "load_code_results",
    "load_ocr_results",
    "merge_nearby_boundaries",
    "parse_srt_file",
    "parse_srt_timestamp",
    "save_code_results",
    "save_ocr_results",
    "should_use_visual_detection",
]
