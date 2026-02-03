"""
High-level video processing operations.
"""

from claudetube.operations.analysis_depth import (
    AnalysisDepth,
    AnalysisResult,
    Entities,
    TechnicalContent,
    analyze_video,
    get_analysis_status,
)
from claudetube.operations.audio_description import (
    AudioDescriptionGenerator,
    compile_scene_descriptions,
    get_scene_descriptions,
)
from claudetube.operations.change_detection import (
    detect_scene_changes,
    get_major_transitions,
    get_scene_changes,
)
from claudetube.operations.chapters import (
    extract_youtube_chapters,
    parse_timestamp,
)
from claudetube.operations.code_evolution import (
    CodeEvolutionData,
    CodeSnapshot,
    CodeUnit,
    get_code_evolution,
    query_code_evolution,
    track_code_evolution,
)
from claudetube.operations.download import (
    DownloadProgress,
    ProgressCallback,
    download_audio,
    download_thumbnail,
    download_video_segment,
    extract_audio_local,
    fetch_metadata,
    fetch_subtitles,
)
from claudetube.operations.entity_extraction import (
    EntityExtractionOperation,
    EntityExtractionSceneResult,
    extract_entities_for_video,
    get_extracted_entities,
)
from claudetube.operations.extract_frames import (
    extract_frames,
    extract_frames_local,
    extract_hq_frames,
    extract_hq_frames_local,
)
from claudetube.operations.factory import (
    OperationFactory,
    clear_factory_cache,
    get_factory,
)
from claudetube.operations.knowledge_graph import (
    build_knowledge_graph,
    build_prerequisite_chain,
    create_video_symlinks,
    extract_shared_entities,
    extract_topic_keywords,
    get_video_context,
    load_knowledge_graph,
    save_knowledge_graph,
)
from claudetube.operations.narrative_structure import (
    NarrativeStructure,
    Section,
    detect_narrative_structure,
    get_narrative_structure,
)
from claudetube.operations.person_tracking import (
    PersonTrackingOperation,
    get_people_tracking,
    track_people,
)
from claudetube.operations.playlist import (
    classify_playlist_type,
    extract_playlist_metadata,
    list_cached_playlists,
    load_playlist_metadata,
    save_playlist_metadata,
)
from claudetube.operations.processor import process_local_video, process_video
from claudetube.operations.segmentation import (
    boundaries_to_segments,
    segment_video_smart,
)
from claudetube.operations.subtitles import (
    fetch_local_subtitles,
    find_embedded_subtitles,
    find_sidecar_subtitles,
)
from claudetube.operations.transcribe import (
    TranscribeOperation,
    transcribe_audio,
    transcribe_video,
)
from claudetube.operations.visual_transcript import (
    VisualTranscriptOperation,
    generate_visual_transcript,
    get_visual_transcript,
)
from claudetube.operations.watch import (
    examine_scene_deep,
    examine_scene_quick,
    watch_video,
)

__all__ = [
    "process_video",
    "process_local_video",
    # Progress tracking
    "DownloadProgress",
    "ProgressCallback",
    # Download operations
    "fetch_metadata",
    "download_audio",
    "download_video_segment",
    "download_thumbnail",
    "extract_audio_local",
    "fetch_subtitles",
    "TranscribeOperation",
    "transcribe_audio",
    "transcribe_video",
    "extract_frames",
    "extract_frames_local",
    "extract_hq_frames",
    "extract_hq_frames_local",
    "extract_youtube_chapters",
    "parse_timestamp",
    "fetch_local_subtitles",
    "find_embedded_subtitles",
    "find_sidecar_subtitles",
    "boundaries_to_segments",
    "segment_video_smart",
    "VisualTranscriptOperation",
    "generate_visual_transcript",
    "get_visual_transcript",
    "PersonTrackingOperation",
    "get_people_tracking",
    "track_people",
    "detect_scene_changes",
    "get_scene_changes",
    "get_major_transitions",
    "AnalysisDepth",
    "AnalysisResult",
    "TechnicalContent",
    "Entities",
    "analyze_video",
    "get_analysis_status",
    "compile_scene_descriptions",
    "get_scene_descriptions",
    "AudioDescriptionGenerator",
    "EntityExtractionOperation",
    "EntityExtractionSceneResult",
    "extract_entities_for_video",
    "get_extracted_entities",
    "OperationFactory",
    "get_factory",
    "clear_factory_cache",
    "watch_video",
    "examine_scene_quick",
    "examine_scene_deep",
    # code_evolution
    "CodeSnapshot",
    "CodeUnit",
    "CodeEvolutionData",
    "track_code_evolution",
    "get_code_evolution",
    "query_code_evolution",
    # playlist
    "extract_playlist_metadata",
    "save_playlist_metadata",
    "load_playlist_metadata",
    "list_cached_playlists",
    "classify_playlist_type",
    # knowledge_graph (operations-level)
    "build_knowledge_graph",
    "save_knowledge_graph",
    "load_knowledge_graph",
    "get_video_context",
    "extract_topic_keywords",
    "extract_shared_entities",
    "build_prerequisite_chain",
    "create_video_symlinks",
    # narrative_structure
    "Section",
    "NarrativeStructure",
    "detect_narrative_structure",
    "get_narrative_structure",
]
