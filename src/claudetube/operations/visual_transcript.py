"""
Visual transcript generation for scenes.

Generates natural language descriptions of what's happening on screen for each scene.
Follows the "Cheap First, Expensive Last" principle:
1. CACHE - Return instantly if visual.json already exists
2. SKIP - Skip scenes where transcript provides sufficient context
3. COMPUTE - Only generate for scenes that need visual context
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from claudetube.cache.manager import CacheManager

if TYPE_CHECKING:
    from pathlib import Path

    from claudetube.providers.base import VisionAnalyzer

from claudetube.cache.scenes import (
    SceneBoundary,
    get_scene_dir,
    get_visual_json_path,
    list_scene_keyframes,
    load_scenes_data,
)
from claudetube.config.loader import get_cache_dir
from claudetube.tools.ffmpeg import FFmpegTool
from claudetube.utils.logging import log_timed

logger = logging.getLogger(__name__)

# Prompt template for visual description generation.
# Context placeholder {context} is appended when transcript text is available.
VISUAL_PROMPT = """\
Describe what is visually happening in these frames from a video.
Focus on: visual actions, people present, objects visible, any text on screen, \
and the setting/environment.
Be specific and factual. Avoid speculation about things not visible.{context}"""


@dataclass
class VisualDescription:
    """Visual description for a scene."""

    scene_id: int
    description: str
    people: list[str]
    objects: list[str]
    text_on_screen: list[str]
    actions: list[str]
    setting: str | None = None
    keyframe_count: int = 0
    model_used: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scene_id": self.scene_id,
            "description": self.description,
            "people": self.people,
            "objects": self.objects,
            "text_on_screen": self.text_on_screen,
            "actions": self.actions,
            "setting": self.setting,
            "keyframe_count": self.keyframe_count,
            "model_used": self.model_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VisualDescription:
        """Create from dictionary."""
        return cls(
            scene_id=data.get("scene_id", 0),
            description=data.get("description", ""),
            people=data.get("people", []),
            objects=data.get("objects", []),
            text_on_screen=data.get("text_on_screen", []),
            actions=data.get("actions", []),
            setting=data.get("setting"),
            keyframe_count=data.get("keyframe_count", 0),
            model_used=data.get("model_used"),
        )


class VisualTranscriptOperation:
    """Generate visual descriptions for video scenes using a VisionAnalyzer.

    Accepts any provider implementing the VisionAnalyzer protocol via constructor
    injection. The execute() method analyzes keyframe images for a single scene
    and returns a VisualDescription.

    Uses the Pydantic VisualDescription schema from providers.types for structured
    output, producing consistent results across different vision providers.

    Args:
        vision_analyzer: Provider implementing the VisionAnalyzer protocol.

    Example:
        >>> from claudetube.providers import get_provider
        >>> provider = get_provider("anthropic")
        >>> op = VisualTranscriptOperation(provider)
        >>> desc = await op.execute(scene_id=0, keyframes=[Path("frame.jpg")], scene=scene)
    """

    def __init__(self, vision_analyzer: VisionAnalyzer):
        self.vision = vision_analyzer

    def _build_prompt(self, scene: SceneBoundary) -> str:
        """Build the visual analysis prompt with optional transcript context.

        Args:
            scene: Scene boundary with transcript data.

        Returns:
            Formatted prompt string.
        """
        context = ""
        if scene.transcript_text:
            context = f"\n\nTranscript context: {scene.transcript_text[:500]}"
        return VISUAL_PROMPT.format(context=context)

    async def execute(
        self,
        scene_id: int,
        keyframes: list[Path],
        scene: SceneBoundary,
    ) -> VisualDescription:
        """Analyze keyframes and produce a structured visual description.

        Args:
            scene_id: Scene identifier.
            keyframes: List of keyframe image paths to analyze.
            scene: Scene boundary for transcript context.

        Returns:
            VisualDescription populated from the vision provider's response.
        """
        from claudetube.providers.types import get_visual_description_model

        visual_schema = get_visual_description_model()

        prompt = self._build_prompt(scene)
        result = await self.vision.analyze_images(
            keyframes,
            prompt,
            schema=visual_schema,
        )

        # Provider returns dict when schema is provided, str otherwise
        data = result if isinstance(result, dict) else json.loads(result)

        # Determine model name from provider info
        model_name = None
        if hasattr(self.vision, "info"):
            model_name = self.vision.info.name

        return VisualDescription(
            scene_id=scene_id,
            description=data.get("description", ""),
            people=data.get("people", []),
            objects=data.get("objects", []),
            text_on_screen=data.get("text_on_screen", []),
            actions=data.get("actions", []),
            setting=data.get("setting"),
            keyframe_count=len(keyframes),
            model_used=model_name,
        )


def _select_keyframes_for_scene(
    scene: SceneBoundary,
    video_id: str,
    cache_dir: Path,
    n: int = 3,
) -> list[Path]:
    """Select representative keyframes from a scene.

    Extracts frames at evenly-distributed timestamps within the scene.
    Uses existing keyframes if available, otherwise extracts new ones.

    Args:
        scene: Scene boundary with start/end times
        video_id: Video ID
        cache_dir: Video cache directory
        n: Maximum number of keyframes to select (default: 3)

    Returns:
        List of keyframe paths
    """
    # Check for existing keyframes
    existing = list_scene_keyframes(cache_dir, scene.scene_id)
    if existing:
        # Use up to n existing keyframes, evenly distributed
        if len(existing) <= n:
            return existing
        step = len(existing) // n
        return [existing[i * step] for i in range(n)]

    # Extract new keyframes
    duration = scene.duration()
    if duration < 3:
        # Short scene: just middle frame
        timestamps = [scene.start_time + duration / 2]
    else:
        # Distribute evenly across scene
        timestamps = [scene.start_time + i * duration / (n - 1) for i in range(n)]

    # Extract frames to scene keyframes directory
    keyframes_dir = get_scene_dir(cache_dir, scene.scene_id) / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = FFmpegTool()

    # Get video source path from state.json
    state_file = cache_dir / "state.json"
    if not state_file.exists():
        logger.warning(f"No state.json found for {video_id}")
        return []

    state = json.loads(state_file.read_text())
    video_path = None

    # Try cached file first (for local videos)
    cached_file = state.get("cached_file")
    if cached_file:
        video_path = cache_dir / cached_file
        if not video_path.exists():
            video_path = None

    # For URL videos, we need to download a segment
    if not video_path and state.get("url"):
        from claudetube.operations.download import download_video_segment

        # Download segment covering the scene
        seg_path = cache_dir / f"scene_{scene.scene_id:03d}_segment.mp4"
        if not seg_path.exists():
            download_video_segment(
                url=state["url"],
                output_path=seg_path,
                start_time=max(0, scene.start_time - 1),
                end_time=scene.end_time + 1,
                quality_sort="res:720",  # Medium quality for keyframes
                concurrent_fragments=2,
            )
        if seg_path.exists():
            video_path = seg_path

    if not video_path or not video_path.exists():
        logger.warning(f"No video source available for scene {scene.scene_id}")
        return []

    # Extract frames
    frames = []
    for ts in timestamps:
        # Adjust timestamp for segment videos
        if state.get("url") and not state.get("cached_file"):
            seek_ts = ts - max(0, scene.start_time - 1)
        else:
            seek_ts = ts

        output_path = keyframes_dir / f"kf_{int(ts // 60):02d}_{int(ts % 60):02d}.jpg"
        frame = ffmpeg.extract_frame(
            video_path=video_path,
            output_path=output_path,
            timestamp=seek_ts,
            width=854,  # Medium quality for visual analysis
            jpeg_quality=4,
        )
        if frame:
            frames.append(frame)

    # Clean up segment file if we created one
    if state.get("url") and not state.get("cached_file"):
        seg_path = cache_dir / f"scene_{scene.scene_id:03d}_segment.mp4"
        if seg_path.exists():
            seg_path.unlink()

    # Dual-write: sync keyframes to SQLite (fire-and-forget)
    try:
        from claudetube.db.sync import get_video_uuid, record_pipeline_step, sync_frame

        video_uuid = get_video_uuid(video_id)
        if video_uuid and frames:
            for frame_path in frames:
                # Extract timestamp from filename (format: kf_MM_SS.jpg)
                try:
                    name = frame_path.stem
                    parts = name.split("_")
                    if len(parts) >= 3:
                        minutes = int(parts[1])
                        seconds = int(parts[2])
                        timestamp = minutes * 60 + seconds
                    else:
                        timestamp = scene.start_time
                except (ValueError, IndexError):
                    timestamp = scene.start_time

                # Get relative path from cache_dir
                relative_path = str(frame_path.relative_to(cache_dir))

                sync_frame(
                    video_uuid=video_uuid,
                    timestamp=float(timestamp),
                    extraction_type="keyframe",
                    file_path=relative_path,
                    scene_id=scene.scene_id,
                    quality_tier="medium",
                    width=854,
                )

            # Record pipeline step for keyframe extraction
            record_pipeline_step(
                video_id,
                step_type="keyframe_extract",
                status="completed",
                scene_id=scene.scene_id,
            )
    except Exception:
        # Fire-and-forget: don't disrupt keyframe extraction
        pass

    return frames


def _should_skip_scene(scene: SceneBoundary) -> bool:
    """Determine if a scene can be skipped for visual transcript.

    Scenes with good transcript coverage may not need visual descriptions.
    This helps avoid unnecessary API calls for talking-head videos.

    Args:
        scene: Scene boundary with transcript data

    Returns:
        True if scene can be skipped, False if visual description needed
    """
    # Skip if scene has substantial transcript (likely talking head)
    transcript_text = scene.transcript_text or scene.transcript_segment or ""

    # If scene has good transcript density, visual may be unnecessary
    duration = scene.duration()
    if duration > 0:
        words = len(transcript_text.split())
        words_per_second = words / duration
        # Typical speech is 2-3 words/second; if we have good coverage, skip
        if words_per_second > 2.0 and duration < 60:
            return True

    return False


def _get_default_vision_analyzer() -> VisionAnalyzer:
    """Get the default vision analyzer via the ProviderRouter.

    Uses the ProviderRouter to select the best available vision provider,
    respecting user configuration preferences and fallback chains. This
    supports all configured providers (Ollama, OpenAI, Anthropic,
    claude-code, etc.) without hardcoding provider names.

    Returns:
        A VisionAnalyzer provider instance.

    Raises:
        RuntimeError: If no vision provider is available.
    """
    from claudetube.providers.base import VisionAnalyzer as VisionAnalyzerProtocol
    from claudetube.providers.capabilities import Capability
    from claudetube.providers.router import NoProviderError, ProviderRouter

    try:
        router = ProviderRouter()
        provider = router.get_for_capability(Capability.VISION)
        if isinstance(provider, VisionAnalyzerProtocol):
            return provider
    except NoProviderError:
        pass
    except Exception as e:
        logger.debug(f"ProviderRouter failed for VISION: {e}")

    raise RuntimeError(
        "No vision provider available. Configure a vision provider "
        "(e.g., anthropic, openai, ollama) or run within Claude Code."
    )


def generate_visual_transcript(
    video_id: str,
    scene_id: int | None = None,
    force: bool = False,
    output_base: Path | None = None,
    vision_analyzer: VisionAnalyzer | None = None,
) -> dict:
    """Generate visual transcript for a video's scenes.

    Follows "Cheap First, Expensive Last" principle:
    1. Return cached visual.json instantly if available
    2. Skip scenes with good transcript coverage
    3. Only call vision API when visual context adds value

    Args:
        video_id: Video ID
        scene_id: Optional specific scene ID (None = all scenes)
        force: Re-generate even if cached
        output_base: Cache directory
        vision_analyzer: Optional VisionAnalyzer provider. If None, auto-selects
            from available providers (anthropic -> claude-code).

    Returns:
        Dict with results and any errors
    """
    t0 = time.time()
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    if not cache_dir.exists():
        return {
            "error": "Video not cached. Run process_video first.",
            "video_id": video_id,
        }

    # Load scenes data
    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return {"error": "No scenes found. Run get_scenes first.", "video_id": video_id}

    # Determine which scenes to process
    if scene_id is not None:
        scenes = [s for s in scenes_data.scenes if s.scene_id == scene_id]
        if not scenes:
            return {"error": f"Scene {scene_id} not found", "video_id": video_id}
    else:
        scenes = scenes_data.scenes

    results = []
    skipped = []
    errors = []

    # Lazily resolve vision analyzer only when we actually need it
    operation = None

    for scene in scenes:
        # 1. CACHE - Return instantly if already exists
        visual_path = get_visual_json_path(cache_dir, scene.scene_id)
        if not force and visual_path.exists():
            try:
                data = json.loads(visual_path.read_text())
                results.append(VisualDescription.from_dict(data))
                log_timed(f"Scene {scene.scene_id}: loaded from cache", t0)
                continue
            except (json.JSONDecodeError, KeyError):
                pass  # Re-generate if cached data is invalid

        # 2. SKIP - Check if scene needs visual description
        if not force and _should_skip_scene(scene):
            skipped.append(scene.scene_id)
            log_timed(f"Scene {scene.scene_id}: skipped (good transcript coverage)", t0)
            continue

        # 3. COMPUTE - Generate visual description
        log_timed(f"Scene {scene.scene_id}: extracting keyframes...", t0)
        keyframes = _select_keyframes_for_scene(scene, video_id, cache_dir)

        if not keyframes:
            errors.append(
                {"scene_id": scene.scene_id, "error": "No keyframes available"}
            )
            continue

        # Lazily create operation on first actual use
        if operation is None:
            try:
                analyzer = vision_analyzer or _get_default_vision_analyzer()
                operation = VisualTranscriptOperation(analyzer)
            except RuntimeError as e:
                errors.append({"scene_id": scene.scene_id, "error": str(e)})
                continue

        log_timed(f"Scene {scene.scene_id}: generating visual description...", t0)

        try:
            visual_desc = asyncio.run(
                operation.execute(scene.scene_id, keyframes, scene)
            )
        except Exception as e:
            logger.error(f"Scene {scene.scene_id}: vision analysis failed: {e}")
            errors.append({"scene_id": scene.scene_id, "error": str(e)})
            continue

        # Save to cache
        visual_path.parent.mkdir(parents=True, exist_ok=True)
        visual_path.write_text(json.dumps(visual_desc.to_dict(), indent=2))
        results.append(visual_desc)

        # Dual-write: sync visual description to SQLite (fire-and-forget)
        try:
            from claudetube.db.sync import (
                get_video_uuid,
                record_pipeline_step,
                sync_visual_description,
            )

            video_uuid = get_video_uuid(video_id)
            if video_uuid:
                # Get relative path for storage
                relative_path = str(visual_path.relative_to(cache_dir))

                # Determine provider name
                provider_name = None
                if hasattr(operation, "vision") and hasattr(operation.vision, "info"):
                    provider_name = operation.vision.info.name

                sync_visual_description(
                    video_uuid=video_uuid,
                    scene_id=scene.scene_id,
                    description=visual_desc.description,
                    provider=provider_name,
                    file_path=relative_path,
                )

                # Record pipeline step for visual analysis
                record_pipeline_step(
                    video_id,
                    step_type="visual_analyze",
                    status="completed",
                    provider=provider_name,
                    scene_id=scene.scene_id,
                )
        except Exception:
            # Fire-and-forget: don't disrupt visual transcript generation
            pass

        log_timed(f"Scene {scene.scene_id}: visual transcript generated", t0)

    # Update state.json if all scenes processed
    if scene_id is None:  # Only update when processing all scenes
        state_file = cache_dir / "state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            all_complete = len(results) + len(skipped) == len(scenes_data.scenes)
            state["visual_transcripts_complete"] = all_complete
            state_file.write_text(json.dumps(state, indent=2))

    log_timed(
        f"Visual transcript complete: {len(results)} generated, {len(skipped)} skipped, {len(errors)} errors",
        t0,
    )

    return {
        "video_id": video_id,
        "generated": len(results),
        "skipped": len(skipped),
        "skipped_scene_ids": skipped,
        "errors": errors,
        "results": [r.to_dict() for r in results],
    }


def get_visual_transcript(
    video_id: str,
    scene_id: int | None = None,
    output_base: Path | None = None,
) -> dict:
    """Get cached visual transcript for a video's scenes.

    Does NOT generate new transcripts - use generate_visual_transcript for that.

    Args:
        video_id: Video ID
        scene_id: Optional specific scene ID (None = all scenes)
        output_base: Cache directory

    Returns:
        Dict with cached visual descriptions
    """
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    if not cache_dir.exists():
        return {"error": "Video not cached", "video_id": video_id}

    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return {"error": "No scenes found", "video_id": video_id}

    if scene_id is not None:
        scenes = [s for s in scenes_data.scenes if s.scene_id == scene_id]
    else:
        scenes = scenes_data.scenes

    results = []
    missing = []

    for scene in scenes:
        visual_path = get_visual_json_path(cache_dir, scene.scene_id)
        if visual_path.exists():
            try:
                data = json.loads(visual_path.read_text())
                results.append(data)
            except json.JSONDecodeError:
                missing.append(scene.scene_id)
        else:
            missing.append(scene.scene_id)

    return {
        "video_id": video_id,
        "count": len(results),
        "missing": missing,
        "results": results,
    }
