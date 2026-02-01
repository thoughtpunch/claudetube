"""
Cache management for claudetube.
"""

from claudetube.cache.entities import (
    ConceptMention,
    ObjectAppearance,
    TrackedConcept,
    TrackedObject,
    get_concepts_json_path,
    get_entities_dir,
    get_objects_json_path,
    load_concepts,
    load_objects,
    save_concepts,
    save_objects,
    track_concepts_from_scenes,
    track_entities,
    track_objects_from_scenes,
)
from claudetube.cache.manager import CacheManager
from claudetube.cache.memory import (
    Observation,
    QAPair,
    VideoMemory,
    get_memory_dir,
    has_memory,
)
from claudetube.cache.scenes import (
    SceneBoundary,
    ScenesData,
    SceneStatus,
    get_all_scene_statuses,
    get_keyframes_dir,
    get_scene_dir,
    get_scene_status,
    get_scenes_dir,
    get_scenes_json_path,
    get_technical_json_path,
    get_visual_json_path,
    has_scenes,
    list_scene_keyframes,
    load_scenes_data,
    save_scenes_data,
)
from claudetube.cache.storage import (
    cache_local_file,
    check_cached_source,
    load_state,
    save_state,
)

__all__ = [
    "CacheManager",
    "cache_local_file",
    "check_cached_source",
    "load_state",
    "save_state",
    # Scene cache utilities
    "SceneBoundary",
    "ScenesData",
    "SceneStatus",
    "get_keyframes_dir",
    "get_scene_dir",
    "get_scene_status",
    "get_scenes_dir",
    "get_scenes_json_path",
    "get_technical_json_path",
    "get_visual_json_path",
    "has_scenes",
    "list_scene_keyframes",
    "load_scenes_data",
    "save_scenes_data",
    "get_all_scene_statuses",
    # Memory utilities
    "Observation",
    "QAPair",
    "VideoMemory",
    "get_memory_dir",
    "has_memory",
    # Entity tracking utilities
    "ConceptMention",
    "ObjectAppearance",
    "TrackedConcept",
    "TrackedObject",
    "get_concepts_json_path",
    "get_entities_dir",
    "get_objects_json_path",
    "load_concepts",
    "load_objects",
    "save_concepts",
    "save_objects",
    "track_concepts_from_scenes",
    "track_entities",
    "track_objects_from_scenes",
]
