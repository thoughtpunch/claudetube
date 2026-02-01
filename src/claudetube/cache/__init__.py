"""
Cache management for claudetube.
"""

from claudetube.cache.manager import CacheManager
from claudetube.cache.storage import load_state, save_state
from claudetube.cache.scenes import (
    SceneBoundary,
    ScenesData,
    SceneStatus,
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
    get_all_scene_statuses,
)

__all__ = [
    "CacheManager",
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
]
