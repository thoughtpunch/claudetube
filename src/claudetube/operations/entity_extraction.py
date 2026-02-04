"""
Entity extraction from video scenes.

Extracts visual entities (objects, people, text) and semantic concepts (topics,
themes) from video content. Follows the entities-first architecture:
1. Extract entities using best available AI provider
2. Generate visual.json as a DERIVED artifact from entities

Follows the "Cheap First, Expensive Last" principle:
1. CACHE - Return instantly if entities.json already exists
2. SKIP - Skip scenes where extraction adds minimal value
3. COMPUTE - Only call AI providers when needed
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from claudetube.cache.manager import CacheManager

if TYPE_CHECKING:
    from pathlib import Path

    from claudetube.providers.base import Reasoner, VideoAnalyzer, VisionAnalyzer

from claudetube.cache.scenes import (
    SceneBoundary,
    get_entities_json_path,
    get_visual_json_path,
    list_scene_keyframes,
    load_scenes_data,
)
from claudetube.config.loader import get_cache_dir
from claudetube.utils.logging import log_timed

logger = logging.getLogger(__name__)

# Prompt for visual entity extraction from keyframes
VISUAL_ENTITY_PROMPT = """\
Analyze these video frames and extract all entities you can identify.

For each entity, provide:
- name: A clear, specific name
- category: One of "object", "person", "text", "code", "ui_element"
- first_seen_sec: Approximate timestamp (use {start_time:.1f} as base)
- confidence: How confident you are (0.0-1.0)
- attributes: Any notable attributes (color, size, position, etc.)

Be thorough but factual. Only report entities you can clearly see.{context}"""

# Prompt for semantic concept extraction from transcript
SEMANTIC_CONCEPT_PROMPT = """\
Analyze this video transcript and extract the key concepts discussed.

For each concept, provide:
- term: The concept name or phrase
- definition: A brief definition in context of this video
- importance: "primary" (central topic), "secondary" (supporting), or "mentioned" (brief reference)
- first_mention_sec: Approximate timestamp of first mention (use {start_time:.1f} as base)
- related_terms: Other terms related to this concept

Transcript:
{transcript}"""

# Prompt for video-level entity extraction (VideoAnalyzer / Gemini)
VIDEO_ENTITY_PROMPT = """\
Analyze this video segment and extract all entities you can identify.

For each entity, provide:
- name: A clear, specific name
- category: One of "object", "person", "text", "code", "ui_element"
- first_seen_sec: Approximate timestamp (scene runs from {start_time:.1f}s to {end_time:.1f}s)
- confidence: How confident you are (0.0-1.0)
- attributes: Any notable attributes (color, size, position, etc.)

Be thorough but factual. Only report entities you can clearly see.{context}"""


@dataclass
class EntityExtractionSceneResult:
    """Entity extraction result for a single scene."""

    scene_id: int
    objects: list[dict] = field(default_factory=list)
    people: list[dict] = field(default_factory=list)
    text_on_screen: list[dict] = field(default_factory=list)
    concepts: list[dict] = field(default_factory=list)
    code_snippets: list[dict] = field(default_factory=list)
    model_used: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scene_id": self.scene_id,
            "objects": self.objects,
            "people": self.people,
            "text_on_screen": self.text_on_screen,
            "concepts": self.concepts,
            "code_snippets": self.code_snippets,
            "model_used": self.model_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EntityExtractionSceneResult:
        """Create from dictionary."""
        return cls(
            scene_id=data.get("scene_id", 0),
            objects=data.get("objects", []),
            people=data.get("people", []),
            text_on_screen=data.get("text_on_screen", []),
            concepts=data.get("concepts", []),
            code_snippets=data.get("code_snippets", []),
            model_used=data.get("model_used"),
        )

    def to_visual_json(self) -> dict:
        """Derive a visual.json-compatible dict from extracted entities.

        This implements the entities-first architecture: entities are PRIMARY,
        and visual.json is DERIVED from them.

        Returns:
            Dict compatible with VisualDescription format.
        """
        # Build description from entities
        parts = []
        if self.people:
            names = [p.get("name", "") for p in self.people]
            parts.append(f"People: {', '.join(names)}")
        if self.objects:
            names = [o.get("name", "") for o in self.objects]
            parts.append(f"Objects: {', '.join(names)}")
        if self.text_on_screen:
            texts = [t.get("name", "") for t in self.text_on_screen]
            parts.append(f"Text on screen: {', '.join(texts)}")

        return {
            "scene_id": self.scene_id,
            "description": "; ".join(parts) if parts else "",
            "people": [p.get("name", "") for p in self.people],
            "objects": [o.get("name", "") for o in self.objects],
            "text_on_screen": [t.get("name", "") for t in self.text_on_screen],
            "actions": [],
            "setting": None,
            "keyframe_count": 0,
            "model_used": self.model_used,
        }


class EntityExtractionOperation:
    """Extract entities from video scenes using AI providers.

    Accepts VideoAnalyzer for native video analysis (most efficient),
    VisionAnalyzer for frame-by-frame visual entities, and Reasoner for
    semantic concepts. Uses the Pydantic EntityExtractionResult schema
    for structured output.

    Provider priority:
    1. VideoAnalyzer (Gemini) - Analyze video segment directly
    2. VisionAnalyzer - Frame-by-frame analysis
    3. Reasoner only - Transcript-based concept extraction

    Args:
        video_analyzer: Provider implementing VideoAnalyzer for native video
            analysis. If available, used instead of VisionAnalyzer.
        vision_analyzer: Provider implementing VisionAnalyzer for frame analysis.
            Falls back to this when VideoAnalyzer is unavailable.
        reasoner: Provider implementing Reasoner for transcript analysis.
            If None, only visual extraction is performed.

    Example:
        >>> from claudetube.providers import get_provider
        >>> provider = get_provider("google")
        >>> op = EntityExtractionOperation(video_analyzer=provider, vision_analyzer=provider)
        >>> result = await op.execute(scene_id=0, keyframes=[], scene=scene)
    """

    def __init__(
        self,
        video_analyzer: VideoAnalyzer | None = None,
        vision_analyzer: VisionAnalyzer | None = None,
        reasoner: Reasoner | None = None,
    ):
        self.video_analyzer = video_analyzer
        self.vision = vision_analyzer
        self.reasoner = reasoner

    def _build_visual_prompt(self, scene: SceneBoundary) -> str:
        """Build the visual entity extraction prompt.

        Args:
            scene: Scene boundary with transcript data.

        Returns:
            Formatted prompt string.
        """
        context = ""
        if scene.transcript_text:
            context = f"\n\nTranscript context: {scene.transcript_text[:500]}"
        return VISUAL_ENTITY_PROMPT.format(
            start_time=scene.start_time,
            context=context,
        )

    def _build_concept_prompt(self, scene: SceneBoundary) -> str:
        """Build the semantic concept extraction prompt.

        Args:
            scene: Scene boundary with transcript data.

        Returns:
            Formatted prompt string.
        """
        transcript = scene.transcript_text or ""
        return SEMANTIC_CONCEPT_PROMPT.format(
            start_time=scene.start_time,
            transcript=transcript[:2000],
        )

    def _build_video_prompt(self, scene: SceneBoundary) -> str:
        """Build the video-level entity extraction prompt.

        Args:
            scene: Scene boundary with transcript data.

        Returns:
            Formatted prompt string.
        """
        context = ""
        if scene.transcript_text:
            context = f"\n\nTranscript context: {scene.transcript_text[:500]}"
        return VIDEO_ENTITY_PROMPT.format(
            start_time=scene.start_time,
            end_time=scene.end_time,
            context=context,
        )

    async def _extract_video_entities(
        self,
        video_path: Path,
        scene: SceneBoundary,
    ) -> dict:
        """Extract entities from a scene using native video analysis.

        Uses VideoAnalyzer to process the video segment directly, which is
        more efficient than frame-by-frame analysis.

        Args:
            video_path: Path to video file.
            scene: Scene boundary for time range and context.

        Returns:
            Dict with objects, people, text_on_screen, code_snippets keys.
        """
        if not self.video_analyzer:
            return {}

        from claudetube.providers.types import get_entity_extraction_result_model

        schema = get_entity_extraction_result_model()
        prompt = self._build_video_prompt(scene)

        result = await self.video_analyzer.analyze_video(
            video_path,
            prompt,
            schema=schema,
            start_time=scene.start_time,
            end_time=scene.end_time,
        )

        data = result if isinstance(result, dict) else json.loads(result)
        return data

    async def _extract_visual_entities(
        self,
        keyframes: list[Path],
        scene: SceneBoundary,
    ) -> dict:
        """Extract visual entities from keyframes using VisionAnalyzer.

        Args:
            keyframes: Keyframe image paths to analyze.
            scene: Scene boundary for context.

        Returns:
            Dict with objects, people, text_on_screen, code_snippets keys.
            May contain _delegate_to_host marker if using claude-code provider.
        """
        if not self.vision or not keyframes:
            return {}

        from claudetube.providers.types import get_entity_extraction_result_model

        schema = get_entity_extraction_result_model()
        prompt = self._build_visual_prompt(scene)

        result = await self.vision.analyze_images(
            keyframes,
            prompt,
            schema=schema,
        )

        data = result if isinstance(result, dict) else json.loads(result)

        # Check for claude-code delegation marker - pass it through
        if isinstance(data, dict) and data.get("_delegate_to_host"):
            return data

        return data

    async def _extract_semantic_concepts(
        self,
        scene: SceneBoundary,
    ) -> list[dict]:
        """Extract semantic concepts from transcript using Reasoner.

        Args:
            scene: Scene boundary with transcript data.

        Returns:
            List of concept dicts.
        """
        if not self.reasoner or not scene.transcript_text:
            return []

        from pydantic import BaseModel, Field

        from claudetube.providers.types import get_semantic_concept_model

        semantic_concept_model = get_semantic_concept_model()

        class ConceptListResult(BaseModel):
            concepts: list[semantic_concept_model] = Field(  # type: ignore[valid-type]
                default_factory=list,
                description="Key concepts extracted from transcript",
            )

        prompt = self._build_concept_prompt(scene)
        messages = [
            {"role": "user", "content": prompt},
        ]

        result = await self.reasoner.reason(
            messages,
            schema=ConceptListResult,
        )

        data = result if isinstance(result, dict) else json.loads(result)
        return data.get("concepts", [])

    async def execute(
        self,
        scene_id: int,
        keyframes: list[Path],
        scene: SceneBoundary,
        video_path: Path | None = None,
    ) -> EntityExtractionSceneResult | dict:
        """Extract entities from a scene using available providers.

        Tries VideoAnalyzer first (native video, most efficient), falls back
        to VisionAnalyzer (frame-by-frame), and runs semantic concept extraction
        concurrently when Reasoner is available.

        Args:
            scene_id: Scene identifier.
            keyframes: List of keyframe image paths to analyze.
            scene: Scene boundary for transcript context.
            video_path: Path to video file (required for VideoAnalyzer).

        Returns:
            EntityExtractionSceneResult with all extracted entities.
            If using claude-code provider, may return a dict with
            _delegate_to_host marker for the MCP tool to handle.
        """
        # Run visual and semantic extraction concurrently
        tasks = []
        has_video = bool(self.video_analyzer and video_path and video_path.exists())
        has_visual = bool(self.vision and keyframes)
        has_reasoner = bool(self.reasoner and scene.transcript_text)

        # VideoAnalyzer takes priority over VisionAnalyzer
        use_video = has_video
        use_visual = has_visual and not has_video

        if use_video:
            tasks.append(self._extract_video_entities(video_path, scene))
        elif use_visual:
            tasks.append(self._extract_visual_entities(keyframes, scene))
        if has_reasoner:
            tasks.append(self._extract_semantic_concepts(scene))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unpack results
        visual_data: dict = {}
        concepts: list[dict] = []
        video_failed = False

        idx = 0
        if use_video:
            if isinstance(results[idx], dict):
                visual_data = results[idx]
                # Check for claude-code delegation marker
                if visual_data.get("_delegate_to_host"):
                    visual_data["scene_id"] = scene_id
                    return visual_data
            elif isinstance(results[idx], Exception):
                logger.warning(
                    f"Scene {scene_id}: video entity extraction failed, "
                    f"falling back to vision: {results[idx]}"
                )
                video_failed = True
            idx += 1
        elif use_visual:
            if isinstance(results[idx], dict):
                visual_data = results[idx]
                # Check for claude-code delegation marker
                if visual_data.get("_delegate_to_host"):
                    visual_data["scene_id"] = scene_id
                    return visual_data
            elif isinstance(results[idx], Exception):
                logger.warning(
                    f"Scene {scene_id}: visual extraction failed: {results[idx]}"
                )
            idx += 1
        if has_reasoner:
            if isinstance(results[idx], list):
                concepts = results[idx]
            elif isinstance(results[idx], dict) and results[idx].get("_delegate_to_host"):
                # Reasoner delegation - return it
                results[idx]["scene_id"] = scene_id
                return results[idx]
            elif isinstance(results[idx], Exception):
                logger.warning(
                    f"Scene {scene_id}: concept extraction failed: {results[idx]}"
                )

        # If VideoAnalyzer failed, fall back to VisionAnalyzer
        if video_failed and has_visual:
            try:
                visual_data = await self._extract_visual_entities(keyframes, scene)
                # Check for delegation after fallback
                if isinstance(visual_data, dict) and visual_data.get("_delegate_to_host"):
                    visual_data["scene_id"] = scene_id
                    return visual_data
            except Exception as e:
                logger.warning(f"Scene {scene_id}: vision fallback also failed: {e}")

        # Determine model name
        model_name = None
        if (
            use_video
            and not video_failed
            and self.video_analyzer
            and hasattr(self.video_analyzer, "info")
        ):
            model_name = self.video_analyzer.info.name
        elif self.vision and hasattr(self.vision, "info"):
            model_name = self.vision.info.name
        elif self.reasoner and hasattr(self.reasoner, "info"):
            model_name = self.reasoner.info.name

        # Normalize entity dicts - ensure they're plain dicts
        def _normalize_entities(entities: list) -> list[dict]:
            normalized = []
            for e in entities:
                if hasattr(e, "model_dump"):
                    normalized.append(e.model_dump())
                elif isinstance(e, dict):
                    normalized.append(e)
            return normalized

        return EntityExtractionSceneResult(
            scene_id=scene_id,
            objects=_normalize_entities(visual_data.get("objects", [])),
            people=_normalize_entities(visual_data.get("people", [])),
            text_on_screen=_normalize_entities(visual_data.get("text_on_screen", [])),
            concepts=_normalize_entities(concepts),
            code_snippets=_normalize_entities(visual_data.get("code_snippets", [])),
            model_used=model_name,
        )


def _get_video_path(cache_dir: Path) -> Path | None:
    """Get the video file path from state.json.

    .. deprecated::
        Use ``CacheManager.get_video_path()`` instead. This wrapper exists
        only for backward compatibility and will be removed in a future release.
    """
    state_file = cache_dir / "state.json"
    if not state_file.exists():
        return None

    try:
        state = json.loads(state_file.read_text())
    except json.JSONDecodeError:
        return None

    cached_file = state.get("cached_file")
    if cached_file:
        video_path = cache_dir / cached_file
        if video_path.exists():
            return video_path

    return None


def _should_skip_entity_extraction(scene: SceneBoundary) -> bool:
    """Determine if a scene can be skipped for entity extraction.

    Very short scenes with no transcript and no keyframes provide
    minimal value for entity extraction.

    Args:
        scene: Scene boundary with transcript data.

    Returns:
        True if scene can be skipped.
    """
    duration = scene.duration()
    has_transcript = bool(scene.transcript_text or scene.transcript_segment)

    # Skip very short scenes with no transcript
    return duration < 2.0 and not has_transcript


def _get_default_providers() -> tuple[
    VideoAnalyzer | None, VisionAnalyzer | None, Reasoner | None
]:
    """Get default video, vision, and reasoner providers for entity extraction.

    Entity extraction requires structured output (JSON schemas), so providers
    that don't support structured output (like claude-code) are filtered out.

    Uses the ProviderRouter to respect user configuration and fallback chains.

    Returns:
        Tuple of (video_analyzer, vision_analyzer, reasoner) - any may be None.
    """
    from claudetube.providers.router import NoProviderError, ProviderRouter

    video: VideoAnalyzer | None = None
    vision: VisionAnalyzer | None = None
    reasoner: Reasoner | None = None

    try:
        router = ProviderRouter()

        # VideoAnalyzer is optional (only Gemini supports it)
        video = router.get_video_analyzer()

        # VisionAnalyzer must support structured output
        try:
            vision = router.get_vision_analyzer_for_structured_output()
        except NoProviderError:
            logger.warning(
                "No vision provider with structured output available for entity extraction"
            )

        # Reasoner must support structured output
        try:
            reasoner = router.get_reasoner_for_structured_output()
        except NoProviderError:
            logger.warning(
                "No reasoning provider with structured output available for entity extraction"
            )

    except Exception as e:
        logger.warning(f"Failed to initialize providers: {e}")

    return video, vision, reasoner


def extract_entities_for_video(
    video_id: str,
    scene_id: int | None = None,
    force: bool = False,
    generate_visual: bool = True,
    output_base: Path | None = None,
    video_analyzer: VideoAnalyzer | None = None,
    vision_analyzer: VisionAnalyzer | None = None,
    reasoner: Reasoner | None = None,
) -> dict:
    """Extract entities for a video's scenes.

    Follows "Cheap First, Expensive Last" principle:
    1. Return cached entities.json instantly if available
    2. Skip scenes with minimal content
    3. Only call AI providers when needed

    Entities-first architecture: entities are PRIMARY, visual.json is
    DERIVED from entities when generate_visual=True.

    Args:
        video_id: Video ID.
        scene_id: Optional specific scene ID (None = all scenes).
        force: Re-extract even if cached.
        generate_visual: Generate visual.json from entities (default True).
        output_base: Cache directory.
        video_analyzer: Optional VideoAnalyzer provider. If None, auto-selects.
        vision_analyzer: Optional VisionAnalyzer provider. If None, auto-selects.
        reasoner: Optional Reasoner provider. If None, auto-selects.

    Returns:
        Dict with results and any errors.
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

    # Lazily resolve providers only when we actually need them
    operation = None
    video_path = None

    for scene in scenes:
        # 1. CACHE - Return instantly if already exists
        entities_path = get_entities_json_path(cache_dir, scene.scene_id)
        if not force and entities_path.exists():
            try:
                data = json.loads(entities_path.read_text())
                result = EntityExtractionSceneResult.from_dict(data)
                results.append(result)
                log_timed(f"Scene {scene.scene_id}: entities loaded from cache", t0)
                continue
            except (json.JSONDecodeError, KeyError):
                pass  # Re-extract if cached data is invalid

        # 2. SKIP - Check if scene needs entity extraction
        if not force and _should_skip_entity_extraction(scene):
            skipped.append(scene.scene_id)
            log_timed(f"Scene {scene.scene_id}: skipped (minimal content)", t0)
            continue

        # 3. COMPUTE - Extract entities
        keyframes = list_scene_keyframes(cache_dir, scene.scene_id)

        # Lazily create operation on first actual use
        if operation is None:
            try:
                if (
                    video_analyzer is None
                    and vision_analyzer is None
                    and reasoner is None
                ):
                    video_analyzer, vision_analyzer, reasoner = _get_default_providers()
                if (
                    video_analyzer is None
                    and vision_analyzer is None
                    and reasoner is None
                ):
                    errors.append(
                        {
                            "scene_id": scene.scene_id,
                            "error": "No AI provider available for entity extraction",
                        }
                    )
                    continue
                operation = EntityExtractionOperation(
                    video_analyzer=video_analyzer,
                    vision_analyzer=vision_analyzer,
                    reasoner=reasoner,
                )
                # Resolve video path for VideoAnalyzer
                if video_analyzer is not None:
                    video_path = cache.get_video_path(video_id)
            except RuntimeError as e:
                errors.append({"scene_id": scene.scene_id, "error": str(e)})
                continue

        log_timed(f"Scene {scene.scene_id}: extracting entities...", t0)

        try:
            entity_result = asyncio.run(
                operation.execute(
                    scene.scene_id, keyframes, scene, video_path=video_path
                )
            )
        except Exception as e:
            logger.error(f"Scene {scene.scene_id}: entity extraction failed: {e}")
            errors.append({"scene_id": scene.scene_id, "error": str(e)})
            continue

        # Check for claude-code delegation response
        if isinstance(entity_result, dict) and entity_result.get("_delegate_to_host"):
            # Return delegation info for MCP tool to handle
            return {
                "video_id": video_id,
                "_delegate_to_host": True,
                "delegation": entity_result,
                "message": (
                    "Entity extraction requires analyzing images. "
                    "The images and prompt are provided for analysis."
                ),
            }

        # Save entities.json to cache
        entities_path.parent.mkdir(parents=True, exist_ok=True)
        entities_path.write_text(json.dumps(entity_result.to_dict(), indent=2))

        # Generate visual.json from entities (entities-first)
        if generate_visual:
            visual_path = get_visual_json_path(cache_dir, scene.scene_id)
            if force or not visual_path.exists():
                visual_data = entity_result.to_visual_json()
                visual_data["keyframe_count"] = len(keyframes)
                visual_path.parent.mkdir(parents=True, exist_ok=True)
                visual_path.write_text(json.dumps(visual_data, indent=2))

        results.append(entity_result)
        log_timed(f"Scene {scene.scene_id}: entities extracted", t0)

    # Update state.json if processing all scenes
    if scene_id is None:
        state_file = cache_dir / "state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            all_complete = len(results) + len(skipped) == len(scenes_data.scenes)
            state["entity_extraction_complete"] = all_complete
            state_file.write_text(json.dumps(state, indent=2))

    log_timed(
        f"Entity extraction complete: {len(results)} extracted, "
        f"{len(skipped)} skipped, {len(errors)} errors",
        t0,
    )

    return {
        "video_id": video_id,
        "extracted": len(results),
        "skipped": len(skipped),
        "skipped_scene_ids": skipped,
        "errors": errors,
        "results": [r.to_dict() for r in results],
    }


def get_extracted_entities(
    video_id: str,
    scene_id: int | None = None,
    output_base: Path | None = None,
) -> dict:
    """Get cached entity extraction results for a video's scenes.

    Does NOT trigger new extraction - use extract_entities_for_video for that.

    Args:
        video_id: Video ID.
        scene_id: Optional specific scene ID (None = all scenes).
        output_base: Cache directory.

    Returns:
        Dict with cached entity extraction results.
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
        entities_path = get_entities_json_path(cache_dir, scene.scene_id)
        if entities_path.exists():
            try:
                data = json.loads(entities_path.read_text())
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
