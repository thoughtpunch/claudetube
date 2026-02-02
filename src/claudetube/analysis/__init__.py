"""
Analysis module for claudetube.

Provides boundary detection and transcript analysis utilities.
"""

from claudetube.analysis.alignment import (
    align_transcript_to_scenes,
    align_transcript_to_scenes_simple,
)
from claudetube.analysis.attention import (
    AttentionFactors,
    calculate_attention_factors,
    calculate_attention_priority,
    calculate_relevance,
    detect_audio_emphasis,
    detect_visual_salience,
    estimate_information_density,
    get_structural_weight,
    get_weights_for_video_type,
    rank_scenes_by_attention,
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
from claudetube.analysis.comprehension import (
    VerificationResult,
    answer_from_understanding,
    generate_self_test_questions,
    verify_answer,
    verify_comprehension,
)
from claudetube.analysis.embeddings import (
    SceneEmbedding,
    embed_scene,
    embed_scenes,
    get_embedding_dim,
    get_embedding_model,
    has_embeddings,
    load_embeddings,
    save_embeddings,
)
from claudetube.analysis.linguistic import (
    Boundary,
    detect_linguistic_boundaries,
)
from claudetube.analysis.ocr import (
    FrameOCRResult,
    TextRegion,
    extract_text_from_frame,
    extract_text_from_scene,
    load_ocr_results,
    save_ocr_results,
)
from claudetube.analysis.pause import (
    detect_pause_boundaries,
    parse_srt_file,
    parse_srt_timestamp,
)
from claudetube.analysis.search import (
    SearchMoment,
    expand_query,
    find_moments,
    format_timestamp,
)
from claudetube.analysis.unified import (
    detect_boundaries_cheap,
    merge_nearby_boundaries,
)
from claudetube.analysis.vector_index import (
    SearchResult,
    build_scene_index,
    delete_scene_index,
    get_index_stats,
    has_vector_index,
    load_scene_index,
    search_scenes,
    search_scenes_by_text,
)
from claudetube.analysis.visual import (
    detect_visual_boundaries,
    detect_visual_boundaries_fast,
    should_use_visual_detection,
)
from claudetube.analysis.vocabulary import (
    detect_vocabulary_shifts,
)
from claudetube.analysis.watcher import (
    ActiveVideoWatcher,
    Hypothesis,
    WatcherAction,
)

__all__ = [
    # Attention priority modeling
    "AttentionFactors",
    "calculate_attention_factors",
    "calculate_attention_priority",
    "calculate_relevance",
    "detect_audio_emphasis",
    "detect_visual_salience",
    "estimate_information_density",
    "get_structural_weight",
    "get_weights_for_video_type",
    "rank_scenes_by_attention",
    # Comprehension verification
    "VerificationResult",
    "answer_from_understanding",
    "generate_self_test_questions",
    "verify_answer",
    "verify_comprehension",
    # Other exports
    "Boundary",
    "CodeBlock",
    "FrameCodeResult",
    "FrameOCRResult",
    "SceneEmbedding",
    "SearchMoment",
    "SearchResult",
    "TextRegion",
    "align_transcript_to_scenes",
    "align_transcript_to_scenes_simple",
    "analyze_frame_for_code",
    "build_scene_index",
    "delete_scene_index",
    "detect_boundaries_cheap",
    "detect_language",
    "detect_linguistic_boundaries",
    "detect_pause_boundaries",
    "detect_visual_boundaries",
    "detect_visual_boundaries_fast",
    "detect_vocabulary_shifts",
    "embed_scene",
    "embed_scenes",
    "expand_query",
    "extract_code_blocks",
    "extract_text_from_frame",
    "extract_text_from_scene",
    "find_moments",
    "format_timestamp",
    "get_embedding_dim",
    "get_embedding_model",
    "get_index_stats",
    "has_embeddings",
    "has_vector_index",
    "is_likely_code",
    "load_code_results",
    "load_embeddings",
    "load_ocr_results",
    "load_scene_index",
    "merge_nearby_boundaries",
    "parse_srt_file",
    "parse_srt_timestamp",
    "save_code_results",
    "save_embeddings",
    "save_ocr_results",
    "search_scenes",
    "search_scenes_by_text",
    "should_use_visual_detection",
    # Active video watcher
    "ActiveVideoWatcher",
    "Hypothesis",
    "WatcherAction",
]
