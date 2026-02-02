"""
Multi-pass video analysis at configurable depths.

Implements analysis tiers following "Cheap First, Expensive Last":
- QUICK:      Scenes + transcript only (~2s)
- STANDARD:   + Visual transcripts (~30s)
- DEEP:       + OCR, code extraction, entities (~2min)
- EXHAUSTIVE: + Frame-by-frame for focus sections (~5min+)

Each level builds on the previous, caching results incrementally.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from claudetube.cache.manager import CacheManager
from claudetube.cache.scenes import (
    SceneBoundary,
    get_scene_dir,
    get_technical_json_path,
    get_visual_json_path,
    has_scenes,
    load_scenes_data,
)
from claudetube.config.loader import get_cache_dir
from claudetube.utils.logging import log_timed

logger = logging.getLogger(__name__)


class AnalysisDepth(Enum):
    """Analysis depth levels.

    Each level includes all processing from previous levels.

    Attributes:
        QUICK: Scenes + transcript only. Fast, no API calls.
        STANDARD: + visual transcripts for each scene. Uses vision API.
        DEEP: + OCR, code extraction, entity detection. Heavy compute.
        EXHAUSTIVE: + frame-by-frame analysis for focus sections.
    """

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    EXHAUSTIVE = "exhaustive"


@dataclass
class TechnicalContent:
    """Technical content extracted from a scene."""

    scene_id: int
    ocr_text: list[str] = field(default_factory=list)
    code_blocks: list[dict] = field(default_factory=list)
    content_types: list[str] = field(default_factory=list)  # 'code', 'slides', etc.

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TechnicalContent:
        """Create from dictionary."""
        return cls(
            scene_id=data.get("scene_id", 0),
            ocr_text=data.get("ocr_text", []),
            code_blocks=data.get("code_blocks", []),
            content_types=data.get("content_types", []),
        )


@dataclass
class Entities:
    """Entities extracted from a scene."""

    scene_id: int
    people: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Entities:
        """Create from dictionary."""
        return cls(
            scene_id=data.get("scene_id", 0),
            people=data.get("people", []),
            topics=data.get("topics", []),
            technologies=data.get("technologies", []),
            keywords=data.get("keywords", []),
        )


@dataclass
class AnalysisResult:
    """Result of multi-pass video analysis."""

    video_id: str
    depth: AnalysisDepth
    scenes: list[dict]
    method: str
    processing_time: float
    focus_sections: list[int] | None = None
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_id": self.video_id,
            "depth": self.depth.value,
            "scene_count": len(self.scenes),
            "scenes": self.scenes,
            "method": self.method,
            "processing_time": self.processing_time,
            "focus_sections": self.focus_sections,
            "errors": self.errors,
        }


def _get_entities_path(cache_dir: Path, scene_id: int) -> Path:
    """Get path to entities.json for a scene."""
    return get_scene_dir(cache_dir, scene_id) / "entities.json"


def _load_cached_technical(cache_dir: Path, scene_id: int) -> TechnicalContent | None:
    """Load cached technical content for a scene."""
    tech_path = get_technical_json_path(cache_dir, scene_id)
    if not tech_path.exists():
        return None
    try:
        data = json.loads(tech_path.read_text())
        return TechnicalContent.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        return None


def _save_technical_content(
    cache_dir: Path, scene_id: int, content: TechnicalContent
) -> None:
    """Save technical content for a scene."""
    tech_path = get_technical_json_path(cache_dir, scene_id)
    tech_path.parent.mkdir(parents=True, exist_ok=True)
    tech_path.write_text(json.dumps(content.to_dict(), indent=2))


def _load_cached_entities(cache_dir: Path, scene_id: int) -> Entities | None:
    """Load cached entities for a scene."""
    ent_path = _get_entities_path(cache_dir, scene_id)
    if not ent_path.exists():
        return None
    try:
        data = json.loads(ent_path.read_text())
        return Entities.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        return None


def _save_entities(cache_dir: Path, scene_id: int, entities: Entities) -> None:
    """Save entities for a scene."""
    ent_path = _get_entities_path(cache_dir, scene_id)
    ent_path.parent.mkdir(parents=True, exist_ok=True)
    ent_path.write_text(json.dumps(entities.to_dict(), indent=2))


def extract_technical_content(
    video_id: str,
    scene: SceneBoundary,
    cache_dir: Path,
    force: bool = False,
) -> TechnicalContent | None:
    """Extract technical content from a scene using OCR and code detection.

    Args:
        video_id: Video identifier.
        scene: Scene to analyze.
        cache_dir: Video cache directory.
        force: Re-extract even if cached.

    Returns:
        TechnicalContent or None if extraction failed.
    """
    # Check cache first
    if not force:
        cached = _load_cached_technical(cache_dir, scene.scene_id)
        if cached is not None:
            return cached

    # Get keyframes for the scene
    from claudetube.cache.scenes import list_scene_keyframes

    keyframes = list_scene_keyframes(cache_dir, scene.scene_id)

    if not keyframes:
        logger.debug(f"No keyframes available for scene {scene.scene_id}")
        return None

    # Run OCR on keyframes
    try:
        from claudetube.analysis.code import analyze_frame_for_code
        from claudetube.analysis.ocr import FrameOCRResult, extract_text_from_scene

        scene_dir = get_scene_dir(cache_dir, scene.scene_id) / "keyframes"
        ocr_results: list[FrameOCRResult] = extract_text_from_scene(
            scene_dir,
            keyframe_paths=keyframes,
            skip_low_likelihood=True,
        )

        # Collect OCR text
        ocr_text = []
        for result in ocr_results:
            for region in result.regions:
                if region.text.strip():
                    ocr_text.append(region.text.strip())

        # Detect code blocks
        code_blocks = []
        content_types = set()
        for ocr_result in ocr_results:
            code_result = analyze_frame_for_code(ocr_result)
            for block in code_result.code_blocks:
                code_blocks.append(block.to_dict())
            content_types.add(ocr_result.content_type)

        content = TechnicalContent(
            scene_id=scene.scene_id,
            ocr_text=ocr_text,
            code_blocks=code_blocks,
            content_types=list(content_types),
        )

        # Cache the result
        _save_technical_content(cache_dir, scene.scene_id, content)

        return content

    except ImportError as e:
        logger.warning(f"OCR dependencies not available: {e}")
        return None
    except Exception as e:
        logger.warning(f"Technical extraction failed for scene {scene.scene_id}: {e}")
        return None


def extract_entities(
    video_id: str,
    scene: SceneBoundary,
    cache_dir: Path,
    force: bool = False,
) -> Entities | None:
    """Extract entities (people, topics, technologies) from a scene.

    Uses transcript text and visual descriptions to identify entities.

    Args:
        video_id: Video identifier.
        scene: Scene to analyze.
        cache_dir: Video cache directory.
        force: Re-extract even if cached.

    Returns:
        Entities or None if extraction failed.
    """
    # Check cache first
    if not force:
        cached = _load_cached_entities(cache_dir, scene.scene_id)
        if cached is not None:
            return cached

    # Gather text sources
    text_sources = []

    # From transcript
    if scene.transcript_text:
        text_sources.append(scene.transcript_text)

    # From visual description
    visual_path = get_visual_json_path(cache_dir, scene.scene_id)
    if visual_path.exists():
        try:
            visual_data = json.loads(visual_path.read_text())
            if visual_data.get("description"):
                text_sources.append(visual_data["description"])
            if visual_data.get("people"):
                text_sources.extend(visual_data["people"])
        except json.JSONDecodeError:
            pass

    if not text_sources:
        return None

    combined_text = " ".join(text_sources)

    # Extract entities using simple pattern matching
    # (Could be enhanced with NER models in the future)
    entities = Entities(scene_id=scene.scene_id)

    # Technology keywords to look for
    tech_keywords = [
        "python",
        "javascript",
        "typescript",
        "react",
        "vue",
        "angular",
        "node",
        "django",
        "flask",
        "fastapi",
        "rust",
        "go",
        "java",
        "kotlin",
        "swift",
        "docker",
        "kubernetes",
        "aws",
        "gcp",
        "azure",
        "postgresql",
        "mysql",
        "mongodb",
        "redis",
        "git",
        "github",
        "vscode",
        "vim",
        "api",
        "rest",
        "graphql",
        "websocket",
        "http",
        "json",
        "yaml",
        "html",
        "css",
        "sql",
        "nosql",
        "oauth",
        "jwt",
        "webpack",
        "npm",
        "yarn",
        "pip",
        "cargo",
    ]

    text_lower = combined_text.lower()
    for tech in tech_keywords:
        if tech in text_lower:
            entities.technologies.append(tech)

    # Extract potential people mentions (simple pattern: capitalized words)
    import re

    # Match patterns like "John", "John Doe", "Dr. Smith"
    people_pattern = r"\b(?:Dr\.|Mr\.|Ms\.|Mrs\.)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
    people_matches = re.findall(people_pattern, combined_text)
    # Filter out common non-names
    non_names = {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "Here",
        "There",
        "Now",
        "Then",
        "Chapter",
        "Section",
        "Part",
        "Step",
    }
    people = [p for p in people_matches if p not in non_names and len(p) > 2]
    entities.people = list(set(people))[:10]  # Limit to 10

    # Extract keywords (nouns that appear multiple times)
    words = re.findall(r"\b[a-z]{4,}\b", text_lower)
    word_counts: dict[str, int] = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    # Filter to words appearing 2+ times, excluding common words
    common_words = {
        "that",
        "this",
        "with",
        "from",
        "have",
        "been",
        "were",
        "will",
        "would",
        "could",
        "should",
        "there",
        "their",
        "about",
        "which",
        "when",
        "what",
        "where",
        "your",
        "just",
        "like",
        "make",
        "going",
        "know",
        "want",
        "here",
        "some",
        "also",
        "more",
        "then",
        "them",
        "only",
        "come",
        "very",
        "into",
        "over",
        "such",
        "other",
        "than",
        "these",
        "those",
    }
    keywords = [
        word
        for word, count in word_counts.items()
        if count >= 2 and word not in common_words and word not in tech_keywords
    ]
    entities.keywords = sorted(keywords, key=lambda w: word_counts[w], reverse=True)[
        :15
    ]

    # Cache the result
    _save_entities(cache_dir, scene.scene_id, entities)

    return entities


def analyze_all_frames(
    video_id: str,
    scene: SceneBoundary,
    cache_dir: Path,
) -> list[dict]:
    """Frame-by-frame analysis for a scene (EXHAUSTIVE mode).

    Extracts frames at higher density and runs full analysis on each.

    Args:
        video_id: Video identifier.
        scene: Scene to analyze.
        cache_dir: Video cache directory.

    Returns:
        List of frame analysis results.
    """
    from claudetube.operations.extract_frames import extract_frames

    results = []

    # Extract frames at 1-second intervals throughout the scene
    duration = scene.duration()
    if duration <= 0:
        return results

    interval = min(1.0, duration / 10)  # At most 10 frames, at least 1/sec

    try:
        frames = extract_frames(
            video_id,
            start_time=scene.start_time,
            duration=duration,
            interval=interval,
            output_base=cache_dir.parent,  # Go up to cache base
            quality="medium",
        )

        for frame_path in frames:
            frame_result = {"path": str(frame_path), "timestamp": 0.0}

            # Extract timestamp from filename if possible
            try:
                # Filename format: frame_MMM_SS.jpg
                name = Path(frame_path).stem
                parts = name.split("_")
                if len(parts) >= 3:
                    minutes = int(parts[1])
                    seconds = int(parts[2])
                    frame_result["timestamp"] = minutes * 60 + seconds
            except (ValueError, IndexError):
                pass

            # Run OCR if dependencies available
            try:
                from claudetube.analysis.ocr import extract_text_from_frame

                ocr_result = extract_text_from_frame(frame_path)
                frame_result["ocr_text"] = [r.text for r in ocr_result.regions]
                frame_result["content_type"] = ocr_result.content_type
            except ImportError:
                pass

            results.append(frame_result)

    except Exception as e:
        logger.warning(f"Frame extraction failed for scene {scene.scene_id}: {e}")

    return results


def _extract_entities_with_operation(
    entity_op,
    video_id: str,
    scene: SceneBoundary,
    cache_dir: Path,
    video_path: Path | None,
    force: bool,
) -> Entities | None:
    """Extract entities using the OperationFactory operation, with regex fallback.

    Tries the AI-powered EntityExtractionOperation first. If no operation is
    available (no providers configured), falls back to the regex-based
    ``extract_entities()`` function.

    The AI result (``EntityExtractionSceneResult``) is converted to the
    ``Entities`` dataclass to maintain backward compatibility in
    ``scene_dict["entities"]``.

    Args:
        entity_op: An ``EntityExtractionOperation`` instance, or None.
        video_id: Video identifier.
        scene: Scene to extract entities from.
        cache_dir: Video cache directory.
        video_path: Path to video file (for VideoAnalyzer).
        force: Re-extract even if cached.

    Returns:
        Entities or None if extraction failed.
    """
    if entity_op is None:
        return extract_entities(video_id, scene, cache_dir, force=force)

    # Check cache first (same file the operation writes to)
    if not force:
        cached = _load_cached_entities(cache_dir, scene.scene_id)
        if cached is not None:
            return cached

    import asyncio

    from claudetube.cache.scenes import get_entities_json_path, list_scene_keyframes

    keyframes = list_scene_keyframes(cache_dir, scene.scene_id)

    try:
        result = asyncio.run(
            entity_op.execute(scene.scene_id, keyframes, scene, video_path=video_path)
        )
    except Exception as e:
        logger.warning(
            "AI entity extraction failed for scene %d, falling back to regex: %s",
            scene.scene_id,
            e,
        )
        return extract_entities(video_id, scene, cache_dir, force=force)

    # Save the full EntityExtractionSceneResult to entities.json
    entities_path = get_entities_json_path(cache_dir, scene.scene_id)
    entities_path.parent.mkdir(parents=True, exist_ok=True)
    entities_path.write_text(json.dumps(result.to_dict(), indent=2))

    # Convert to Entities dataclass for backward compatibility
    return Entities(
        scene_id=scene.scene_id,
        people=[p.get("name", "") for p in result.people],
        topics=[c.get("term", "") for c in result.concepts],
        technologies=[],  # Not directly available from AI extraction
        keywords=[o.get("name", "") for o in result.objects],
    )


def analyze_video(
    video_id: str,
    depth: AnalysisDepth = AnalysisDepth.STANDARD,
    focus_sections: list[int] | None = None,
    force: bool = False,
    output_base: Path | None = None,
) -> AnalysisResult:
    """Analyze video at specified depth.

    Implements progressive analysis levels:
    - QUICK: Returns cached scenes + transcript (instant)
    - STANDARD: Adds visual transcripts for each scene
    - DEEP: Adds OCR, code extraction, entity detection
    - EXHAUSTIVE: Adds frame-by-frame analysis for focus sections

    Each level builds on previous, caching results incrementally.

    Args:
        video_id: Video identifier.
        depth: Analysis depth level.
        focus_sections: Optional list of scene IDs for EXHAUSTIVE analysis.
            If None and depth=EXHAUSTIVE, analyzes all scenes.
        force: Re-run analysis even if cached.
        output_base: Cache base directory.

    Returns:
        AnalysisResult with enriched scene data.
    """
    t0 = time.time()
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    errors: list[dict] = []

    # Verify video is cached
    if not cache_dir.exists():
        return AnalysisResult(
            video_id=video_id,
            depth=depth,
            scenes=[],
            method="error",
            processing_time=time.time() - t0,
            errors=[{"error": "Video not cached. Run process_video first."}],
        )

    # Step 1: Load scenes (required for all depths)
    if not has_scenes(cache_dir):
        # Run segmentation first
        from claudetube.operations.segmentation import segment_video_smart

        state_file = cache_dir / "state.json"
        if not state_file.exists():
            return AnalysisResult(
                video_id=video_id,
                depth=depth,
                scenes=[],
                method="error",
                processing_time=time.time() - t0,
                errors=[{"error": "No state.json found"}],
            )

        state = json.loads(state_file.read_text())
        video_info = {
            "duration": state.get("duration"),
            "description": state.get("description", ""),
        }

        transcript_segments = None
        srt_path = cache_dir / "audio.srt"
        if srt_path.exists():
            from claudetube.analysis.pause import parse_srt_file

            transcript_segments = parse_srt_file(srt_path)

        segment_video_smart(
            video_id=video_id,
            video_path=None,
            transcript_segments=transcript_segments,
            video_info=video_info,
            cache_dir=cache_dir,
            srt_path=srt_path if srt_path.exists() else None,
        )

    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return AnalysisResult(
            video_id=video_id,
            depth=depth,
            scenes=[],
            method="error",
            processing_time=time.time() - t0,
            errors=[{"error": "Failed to load scenes"}],
        )

    log_timed(f"Loaded {len(scenes_data.scenes)} scenes", t0)

    # Convert scenes to enrichable dicts
    scenes = [s.to_dict() for s in scenes_data.scenes]

    # QUICK depth: just scenes + transcript
    if depth == AnalysisDepth.QUICK:
        return AnalysisResult(
            video_id=video_id,
            depth=depth,
            scenes=scenes,
            method=scenes_data.method,
            processing_time=time.time() - t0,
        )

    # STANDARD depth: add visual transcripts
    if depth.value in ("standard", "deep", "exhaustive"):
        log_timed("Enriching with visual transcripts...", t0)
        for scene_dict in scenes:
            scene_id = scene_dict["scene_id"]

            # Skip if not in focus sections (for targeted analysis)
            if focus_sections and scene_id not in focus_sections:
                continue

            # Check for cached visual
            visual_path = get_visual_json_path(cache_dir, scene_id)
            if visual_path.exists():
                try:
                    visual_data = json.loads(visual_path.read_text())
                    scene_dict["visual"] = visual_data
                except json.JSONDecodeError:
                    pass
            elif not force:
                # Visual not cached and not forcing - skip
                continue
            else:
                # Generate visual transcript
                try:
                    from claudetube.operations.visual_transcript import (
                        generate_visual_transcript,
                    )

                    result = generate_visual_transcript(
                        video_id,
                        scene_id=scene_id,
                        force=force,
                        output_base=output_base,
                    )
                    if result.get("results"):
                        scene_dict["visual"] = result["results"][0]
                except Exception as e:
                    errors.append(
                        {"scene_id": scene_id, "stage": "visual", "error": str(e)}
                    )

    if depth == AnalysisDepth.STANDARD:
        return AnalysisResult(
            video_id=video_id,
            depth=depth,
            scenes=scenes,
            method=scenes_data.method,
            processing_time=time.time() - t0,
            focus_sections=focus_sections,
            errors=errors,
        )

    # DEEP depth: add technical content + entities
    if depth.value in ("deep", "exhaustive"):
        log_timed("Extracting technical content...", t0)

        # Try to get an EntityExtractionOperation from the factory
        entity_op = None
        video_path = None
        try:
            from claudetube.operations.factory import get_factory

            factory = get_factory()
            entity_op = factory.get_entity_extraction_operation()
            # Check if the operation has at least one provider
            if not (entity_op.video_analyzer or entity_op.vision or entity_op.reasoner):
                entity_op = None
            elif entity_op.video_analyzer:
                from claudetube.operations.entity_extraction import _get_video_path

                video_path = _get_video_path(cache_dir)
        except Exception as e:
            logger.debug("OperationFactory unavailable, falling back to regex: %s", e)

        for i, scene_dict in enumerate(scenes):
            scene_id = scene_dict["scene_id"]
            scene = scenes_data.scenes[i]

            # Skip if not in focus sections
            if focus_sections and scene_id not in focus_sections:
                continue

            # Technical content
            try:
                tech = extract_technical_content(
                    video_id, scene, cache_dir, force=force
                )
                if tech:
                    scene_dict["technical"] = tech.to_dict()
            except Exception as e:
                errors.append(
                    {"scene_id": scene_id, "stage": "technical", "error": str(e)}
                )

            # Entities via OperationFactory (preferred) or regex fallback
            try:
                ent = _extract_entities_with_operation(
                    entity_op, video_id, scene, cache_dir, video_path, force
                )
                if ent:
                    scene_dict["entities"] = ent.to_dict()
            except Exception as e:
                errors.append(
                    {"scene_id": scene_id, "stage": "entities", "error": str(e)}
                )

    if depth == AnalysisDepth.DEEP:
        return AnalysisResult(
            video_id=video_id,
            depth=depth,
            scenes=scenes,
            method=scenes_data.method,
            processing_time=time.time() - t0,
            focus_sections=focus_sections,
            errors=errors,
        )

    # EXHAUSTIVE depth: frame-by-frame for focus sections
    if depth == AnalysisDepth.EXHAUSTIVE:
        log_timed("Running frame-by-frame analysis...", t0)

        # Default to all scenes if no focus specified
        target_ids = (
            focus_sections if focus_sections else [s["scene_id"] for s in scenes]
        )

        for i, scene_dict in enumerate(scenes):
            scene_id = scene_dict["scene_id"]
            if scene_id not in target_ids:
                continue

            scene = scenes_data.scenes[i]
            try:
                frame_analysis = analyze_all_frames(video_id, scene, cache_dir)
                if frame_analysis:
                    scene_dict["frame_analysis"] = frame_analysis
            except Exception as e:
                errors.append(
                    {"scene_id": scene_id, "stage": "frames", "error": str(e)}
                )

    return AnalysisResult(
        video_id=video_id,
        depth=depth,
        scenes=scenes,
        method=scenes_data.method,
        processing_time=time.time() - t0,
        focus_sections=focus_sections,
        errors=errors,
    )


def get_analysis_status(video_id: str, output_base: Path | None = None) -> dict:
    """Get current analysis status for a video.

    Returns what analysis has been completed for each scene.

    Args:
        video_id: Video identifier.
        output_base: Cache base directory.

    Returns:
        Dict with status information.
    """
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    if not cache_dir.exists():
        return {"error": "Video not cached", "video_id": video_id}

    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return {"error": "No scenes found", "video_id": video_id, "has_scenes": False}

    scene_statuses = []
    for scene in scenes_data.scenes:
        scene_id = scene.scene_id
        status = {
            "scene_id": scene_id,
            "has_transcript": bool(scene.transcript_text),
            "has_visual": get_visual_json_path(cache_dir, scene_id).exists(),
            "has_technical": get_technical_json_path(cache_dir, scene_id).exists(),
            "has_entities": _get_entities_path(cache_dir, scene_id).exists(),
        }
        scene_statuses.append(status)

    # Determine max completed depth
    all_have_transcript = all(s["has_transcript"] for s in scene_statuses)
    all_have_visual = all(s["has_visual"] for s in scene_statuses)
    all_have_technical = all(s["has_technical"] for s in scene_statuses)
    all_have_entities = all(s["has_entities"] for s in scene_statuses)

    if all_have_technical and all_have_entities:
        max_depth = "deep"
    elif all_have_visual:
        max_depth = "standard"
    elif all_have_transcript or scene_statuses:
        max_depth = "quick"
    else:
        max_depth = "none"

    return {
        "video_id": video_id,
        "scene_count": len(scenes_data.scenes),
        "max_completed_depth": max_depth,
        "scenes": scene_statuses,
        "summary": {
            "with_transcript": sum(1 for s in scene_statuses if s["has_transcript"]),
            "with_visual": sum(1 for s in scene_statuses if s["has_visual"]),
            "with_technical": sum(1 for s in scene_statuses if s["has_technical"]),
            "with_entities": sum(1 for s in scene_statuses if s["has_entities"]),
        },
    }
