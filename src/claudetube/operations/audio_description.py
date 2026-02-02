"""
Audio description generation and compilation from scene analysis data.

Supports three strategies, following "Cheap First, Expensive Last":
1. CACHE     - Return instantly if AD files already exist
2. COMPILE   - Format existing visual.json + transcript into WebVTT
3. GENERATE  - Use provider system (VIDEO → VISION fallback) when no visual data exists

The AudioDescriptionGenerator class integrates with the provider abstraction
to generate descriptions on-demand using the best available AI provider.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from claudetube.cache.manager import CacheManager
from claudetube.cache.scenes import (
    SceneBoundary,
    ScenesData,
    get_visual_json_path,
    list_scene_keyframes,
    load_scenes_data,
)
from claudetube.config.loader import get_cache_dir
from claudetube.operations.visual_transcript import VisualDescription
from claudetube.utils.logging import log_timed

if TYPE_CHECKING:
    from pathlib import Path

    from claudetube.providers.base import Provider

logger = logging.getLogger(__name__)


def _format_vtt_timestamp(seconds: float) -> str:
    """Format seconds as WebVTT timestamp (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted VTT timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _format_description(
    visual: VisualDescription | None,
    transcript_text: str,
    title: str | None,
) -> str:
    """Format a scene description from visual and transcript data.

    Combines visual description, text on screen, actions, and setting
    into a concise audio description line.

    Args:
        visual: VisualDescription from visual.json (may be None)
        transcript_text: Scene transcript text
        title: Optional scene title (from chapters)

    Returns:
        Formatted description string
    """
    parts = []

    # Scene title/chapter if available
    if title:
        parts.append(title + ".")

    if visual:
        # Primary visual description
        if visual.description:
            parts.append(visual.description)

        # Setting context (only if not already implied by description)
        if visual.setting and visual.setting.lower() not in (visual.description or "").lower():
            parts.append(f"Setting: {visual.setting}.")

        # Text on screen (code, diagrams, etc.) - important for accessibility
        if visual.text_on_screen:
            text_items = ", ".join(visual.text_on_screen[:5])  # Limit to 5 items
            parts.append(f"On screen: {text_items}.")

        # Actions if they add context beyond the description
        if visual.actions and not visual.description:
            action_text = ", ".join(visual.actions[:3])
            parts.append(action_text.capitalize() + ".")
    else:
        # No visual data - use transcript summary as fallback
        if transcript_text:
            # Take first sentence or first 100 chars as a basic description
            first_sentence = transcript_text.split(".")[0].strip()
            if first_sentence and len(first_sentence) > 10:
                parts.append(f"Speaker: {first_sentence[:150]}.")

    return " ".join(parts).strip()


def _compile_vtt(
    scenes_data: ScenesData,
    cache_dir: Path,
) -> tuple[list[str], list[str]]:
    """Compile scene data into VTT lines and plain text lines.

    Args:
        scenes_data: Loaded scenes data with boundaries
        cache_dir: Video cache directory

    Returns:
        Tuple of (vtt_lines, txt_lines)
    """
    vtt_lines = [
        "WEBVTT",
        "Kind: descriptions",
        "Language: en",
        "",
    ]
    txt_lines = []
    cue_index = 0

    for scene in scenes_data.scenes:
        # Load visual.json if it exists
        visual = None
        visual_path = get_visual_json_path(cache_dir, scene.scene_id)
        if visual_path.exists():
            try:
                data = json.loads(visual_path.read_text())
                visual = VisualDescription.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Invalid visual.json for scene {scene.scene_id}")

        # Format description from available data
        description = _format_description(
            visual=visual,
            transcript_text=scene.transcript_text,
            title=scene.title,
        )

        if not description:
            continue

        cue_index += 1

        # VTT cue
        start_ts = _format_vtt_timestamp(scene.start_time)
        end_ts = _format_vtt_timestamp(scene.end_time)
        vtt_lines.append(str(cue_index))
        vtt_lines.append(f"{start_ts} --> {end_ts}")
        vtt_lines.append(description)
        vtt_lines.append("")

        # Plain text line with timestamp
        minutes = int(scene.start_time // 60)
        seconds = int(scene.start_time % 60)
        txt_lines.append(f"[{minutes:02d}:{seconds:02d}] {description}")

    return vtt_lines, txt_lines


def compile_scene_descriptions(
    video_id: str,
    force: bool = False,
    output_base: Path | None = None,
) -> dict:
    """Compile scene analysis into WebVTT audio descriptions.

    Reads existing visual.json files and scene transcript data, then
    compiles them into .ad.vtt and .ad.txt formats. This is a compilation
    step, not AI generation.

    Args:
        video_id: Video ID
        force: Re-compile even if cached AD files exist
        output_base: Cache base directory override

    Returns:
        Dict with compilation results
    """
    t0 = time.time()
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    if not cache_dir.exists():
        return {"error": "Video not cached. Run process_video first.", "video_id": video_id}

    # 1. CACHE - Return instantly if AD already exists
    if not force and cache.has_ad(video_id):
        vtt_path, txt_path = cache.get_ad_paths(video_id)
        log_timed("Audio descriptions loaded from cache", t0)
        return {
            "video_id": video_id,
            "status": "cached",
            "vtt_path": str(vtt_path),
            "txt_path": str(txt_path),
        }

    # Load scenes data
    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return {"error": "No scenes found. Run scene analysis first.", "video_id": video_id}

    if not scenes_data.scenes:
        return {"error": "No scene boundaries found.", "video_id": video_id}

    # 2. COMPILE - Format existing data into VTT
    log_timed(f"Compiling audio descriptions for {len(scenes_data.scenes)} scenes...", t0)
    vtt_lines, txt_lines = _compile_vtt(scenes_data, cache_dir)

    if not txt_lines:
        return {
            "error": "No descriptions could be generated from available scene data.",
            "video_id": video_id,
        }

    # Write output files
    vtt_path, txt_path = cache.get_ad_paths(video_id)

    vtt_path.write_text("\n".join(vtt_lines))
    txt_path.write_text("\n".join(txt_lines))

    # Update state
    state = cache.get_state(video_id)
    if state:
        state.ad_complete = True
        state.ad_source = "scene_compilation"
        cache.save_state(video_id, state)

    log_timed(f"Audio descriptions compiled: {len(txt_lines)} cues", t0)

    return {
        "video_id": video_id,
        "status": "compiled",
        "cue_count": len(txt_lines),
        "vtt_path": str(vtt_path),
        "txt_path": str(txt_path),
        "source": "scene_compilation",
    }


def get_scene_descriptions(
    video_id: str,
    output_base: Path | None = None,
) -> dict:
    """Get cached audio descriptions for a video.

    Does NOT compile new descriptions - use compile_scene_descriptions for that.

    Args:
        video_id: Video ID
        output_base: Cache base directory override

    Returns:
        Dict with cached AD content or error
    """
    cache = CacheManager(output_base or get_cache_dir())

    if not cache.has_ad(video_id):
        return {"error": "No audio descriptions found. Run compile_scene_descriptions first.", "video_id": video_id}

    vtt_path, txt_path = cache.get_ad_paths(video_id)

    result: dict = {"video_id": video_id}

    if vtt_path.exists():
        result["vtt"] = vtt_path.read_text()
        result["vtt_path"] = str(vtt_path)

    if txt_path.exists():
        result["txt"] = txt_path.read_text()
        result["txt_path"] = str(txt_path)

    return result


# =============================================================================
# Provider-Integrated Audio Description Generator
# =============================================================================

_AD_SCENE_PROMPT = """\
Describe what is visually happening in this scene for a vision-impaired viewer.
Focus on: visual actions, people present, objects visible, any text on screen,
and the setting/environment. Be concise (1-3 sentences). Avoid describing audio
or dialogue — those are already in the transcript.

{context}

Return a JSON object with these keys:
- description: A 1-3 sentence audio description of what's visually happening
- people: Array of brief person descriptions (e.g., ["man in blue shirt"])
- objects: Array of notable objects (e.g., ["laptop", "whiteboard"])
- text_on_screen: Array of visible text (e.g., ["def main():", "Chapter 3"])
- actions: Array of visual actions (e.g., ["typing on keyboard"])
- setting: Brief setting description (e.g., "office", "outdoors")

Output only valid JSON, no markdown code blocks."""

_AD_VIDEO_PROMPT = """\
Generate audio descriptions for a vision-impaired viewer of this video segment
({start_fmt} to {end_fmt}). Describe what is visually happening — people,
actions, objects, text on screen, and setting. Do not describe audio/dialogue.
Be concise (1-3 sentences per scene).

Return a JSON object with these keys:
- description: A 1-3 sentence audio description of what's visually happening
- people: Array of brief person descriptions
- objects: Array of notable objects
- text_on_screen: Array of visible text
- actions: Array of visual actions
- setting: Brief setting description

Output only valid JSON, no markdown code blocks."""


def _find_provider_for_capability(capability_name: str) -> Provider | None:
    """Find the first available provider with a given capability.

    Uses the provider registry to discover available providers and checks
    capabilities via the ProviderInfo metadata.

    Args:
        capability_name: Capability enum name (e.g., "VIDEO", "VISION", "TRANSCRIBE")

    Returns:
        A Provider instance or None if no available provider has the capability.
    """
    from claudetube.providers.capabilities import Capability
    from claudetube.providers.registry import get_provider, list_available

    target = Capability[capability_name]
    for name in list_available():
        try:
            provider = get_provider(name)
            if provider.info.can(target):
                return provider
        except (ImportError, ValueError):
            continue
    return None


class AudioDescriptionGenerator:
    """Generate audio descriptions using the provider abstraction.

    Supports three generation strategies, selected based on available
    provider capabilities:

    1. **Native video** (Capability.VIDEO) — Send video directly to a
       provider that understands video natively (e.g., Gemini). Most
       efficient for long videos.
    2. **Frame-by-frame vision** (Capability.VISION) — Extract keyframes
       per scene and describe them via a VisionAnalyzer. Works with any
       vision-capable provider (OpenAI, Anthropic, Claude Code, etc.).
    3. **Compile-only** — Fall back to compiling existing visual.json data
       without any AI generation. Always available.

    The generator also supports transcribing downloaded AD audio tracks
    via any provider implementing the Transcriber protocol.

    Args:
        video_provider: Explicit VideoAnalyzer provider (optional).
        vision_provider: Explicit VisionAnalyzer provider (optional).
        transcription_provider: Explicit Transcriber provider (optional).

    If providers are not explicitly passed, the generator auto-discovers
    them from the provider registry at generation time.
    """

    def __init__(
        self,
        video_provider: Provider | None = None,
        vision_provider: Provider | None = None,
        transcription_provider: Provider | None = None,
    ):
        self._video_provider = video_provider
        self._vision_provider = vision_provider
        self._transcription_provider = transcription_provider

    def _get_video_provider(self) -> Provider | None:
        """Get a VideoAnalyzer provider (explicit or auto-discovered)."""
        if self._video_provider is not None:
            return self._video_provider
        return _find_provider_for_capability("VIDEO")

    def _get_vision_provider(self) -> Provider | None:
        """Get a VisionAnalyzer provider (explicit or auto-discovered)."""
        if self._vision_provider is not None:
            return self._vision_provider
        return _find_provider_for_capability("VISION")

    def _get_transcription_provider(self) -> Provider | None:
        """Get a Transcriber provider (explicit or auto-discovered)."""
        if self._transcription_provider is not None:
            return self._transcription_provider
        return _find_provider_for_capability("TRANSCRIBE")

    async def generate(
        self,
        video_id: str,
        force: bool = False,
        output_base: Path | None = None,
    ) -> dict:
        """Generate audio descriptions for a video.

        Follows "Cheap First, Expensive Last":
        1. CACHE — Return instantly if AD files already exist
        2. NATIVE VIDEO — Use VideoAnalyzer if available (Gemini)
        3. FRAME-BY-FRAME — Use VisionAnalyzer with keyframes
        4. COMPILE-ONLY — Format existing visual.json data

        Args:
            video_id: Video ID
            force: Re-generate even if cached
            output_base: Cache base directory override

        Returns:
            Dict with generation results including paths and source method.
        """
        t0 = time.time()
        cache = CacheManager(output_base or get_cache_dir())
        cache_dir = cache.get_cache_dir(video_id)

        if not cache_dir.exists():
            return {"error": "Video not cached. Run process_video first.", "video_id": video_id}

        # 1. CACHE
        if not force and cache.has_ad(video_id):
            vtt_path, txt_path = cache.get_ad_paths(video_id)
            log_timed("Audio descriptions loaded from cache", t0)
            return {
                "video_id": video_id,
                "status": "cached",
                "vtt_path": str(vtt_path),
                "txt_path": str(txt_path),
            }

        # Load scenes data
        scenes_data = load_scenes_data(cache_dir)
        if not scenes_data or not scenes_data.scenes:
            return {"error": "No scenes found. Run scene analysis first.", "video_id": video_id}

        # Determine generation strategy
        video_prov = self._get_video_provider()
        vision_prov = self._get_vision_provider()

        if video_prov is not None:
            log_timed(f"Using native video provider: {video_prov.info.name}", t0)
            result = await self._generate_native(
                video_id, scenes_data, cache, cache_dir, video_prov, t0,
            )
        elif vision_prov is not None:
            log_timed(f"Using vision provider: {vision_prov.info.name}", t0)
            result = await self._generate_from_frames(
                video_id, scenes_data, cache, cache_dir, vision_prov, t0,
            )
        else:
            log_timed("No AI providers available; falling back to compile-only", t0)
            return compile_scene_descriptions(video_id, force=True, output_base=output_base)

        return result

    async def _generate_native(
        self,
        video_id: str,
        scenes_data: ScenesData,
        cache: CacheManager,
        cache_dir: Path,
        provider: Provider,
        t0: float,
    ) -> dict:
        """Generate AD using native video analysis (e.g., Gemini).

        Sends each scene's time range to the VideoAnalyzer provider for
        direct video understanding without frame extraction.
        """

        from claudetube.providers.base import VideoAnalyzer

        if not isinstance(provider, VideoAnalyzer):
            logger.warning(f"Provider {provider.info.name} does not implement VideoAnalyzer")
            return {"error": f"Provider {provider.info.name} lacks VIDEO capability", "video_id": video_id}

        # Locate video source
        video_path = self._resolve_video_path(cache_dir, video_id)
        if video_path is None:
            logger.warning("No video source available for native analysis; falling back to frames")
            vision_prov = self._get_vision_provider()
            if vision_prov is not None:
                return await self._generate_from_frames(
                    video_id, scenes_data, cache, cache_dir, vision_prov, t0,
                )
            return compile_scene_descriptions(video_id, force=True)

        descriptions: list[VisualDescription] = []
        errors: list[dict] = []

        for scene in scenes_data.scenes:
            context = ""
            if scene.transcript_text:
                context = f"Transcript context: {scene.transcript_text[:500]}"

            start_fmt = _format_vtt_timestamp(scene.start_time)[:8]
            end_fmt = _format_vtt_timestamp(scene.end_time)[:8]

            prompt = _AD_VIDEO_PROMPT.format(
                start_fmt=start_fmt,
                end_fmt=end_fmt,
            )
            if context:
                prompt += f"\n\n{context}"

            try:
                result = await provider.analyze_video(
                    video=video_path,
                    prompt=prompt,
                    start_time=scene.start_time,
                    end_time=scene.end_time,
                )
                desc = self._parse_description_response(result, scene)
                if desc:
                    desc.model_used = provider.info.name
                    descriptions.append(desc)
                    # Cache the visual.json
                    visual_path = get_visual_json_path(cache_dir, scene.scene_id)
                    visual_path.parent.mkdir(parents=True, exist_ok=True)
                    visual_path.write_text(json.dumps(desc.to_dict(), indent=2))
                else:
                    errors.append({"scene_id": scene.scene_id, "error": "Failed to parse response"})
            except Exception as e:
                logger.warning(f"Native video analysis failed for scene {scene.scene_id}: {e}")
                errors.append({"scene_id": scene.scene_id, "error": str(e)})

        return self._finalize(
            video_id, scenes_data, cache, cache_dir, descriptions, errors,
            source="native_video", provider_name=provider.info.name, t0=t0,
        )

    async def _generate_from_frames(
        self,
        video_id: str,
        scenes_data: ScenesData,
        cache: CacheManager,
        cache_dir: Path,
        provider: Provider,
        t0: float,
    ) -> dict:
        """Generate AD using frame-by-frame vision analysis.

        Extracts keyframes for each scene and sends them to a
        VisionAnalyzer provider for description.
        """
        from claudetube.providers.base import VisionAnalyzer

        if not isinstance(provider, VisionAnalyzer):
            logger.warning(f"Provider {provider.info.name} does not implement VisionAnalyzer")
            return compile_scene_descriptions(video_id, force=True)

        descriptions: list[VisualDescription] = []
        errors: list[dict] = []

        for scene in scenes_data.scenes:
            # Check for cached visual.json first
            visual_path = get_visual_json_path(cache_dir, scene.scene_id)
            if visual_path.exists():
                try:
                    data = json.loads(visual_path.read_text())
                    descriptions.append(VisualDescription.from_dict(data))
                    continue
                except (json.JSONDecodeError, KeyError):
                    pass

            # Get keyframes for this scene
            keyframes = self._get_scene_keyframes(scene, video_id, cache_dir)
            if not keyframes:
                errors.append({"scene_id": scene.scene_id, "error": "No keyframes available"})
                continue

            context = ""
            if scene.transcript_text:
                context = f"Transcript context: {scene.transcript_text[:500]}"

            prompt = _AD_SCENE_PROMPT.format(context=context)

            try:
                result = await provider.analyze_images(
                    images=keyframes,
                    prompt=prompt,
                )
                desc = self._parse_description_response(result, scene)
                if desc:
                    desc.model_used = provider.info.name
                    desc.keyframe_count = len(keyframes)
                    descriptions.append(desc)
                    # Cache the visual.json
                    visual_path.parent.mkdir(parents=True, exist_ok=True)
                    visual_path.write_text(json.dumps(desc.to_dict(), indent=2))
                else:
                    errors.append({"scene_id": scene.scene_id, "error": "Failed to parse response"})
            except Exception as e:
                logger.warning(f"Vision analysis failed for scene {scene.scene_id}: {e}")
                errors.append({"scene_id": scene.scene_id, "error": str(e)})

        return self._finalize(
            video_id, scenes_data, cache, cache_dir, descriptions, errors,
            source="frame_vision", provider_name=provider.info.name, t0=t0,
        )

    async def transcribe_ad_track(
        self,
        video_id: str,
        ad_audio_path: Path,
        output_base: Path | None = None,
    ) -> dict:
        """Transcribe a downloaded audio description track.

        Uses the Transcriber protocol to convert an AD audio track into
        text, then writes VTT and TXT output files.

        Args:
            video_id: Video ID
            ad_audio_path: Path to the downloaded AD audio file
            output_base: Cache base directory override

        Returns:
            Dict with transcription results.
        """
        from pathlib import Path as _Path

        from claudetube.providers.base import Transcriber

        t0 = time.time()
        cache = CacheManager(output_base or get_cache_dir())

        provider = self._get_transcription_provider()
        if provider is None:
            return {"error": "No transcription provider available", "video_id": video_id}

        if not isinstance(provider, Transcriber):
            return {"error": f"Provider {provider.info.name} lacks TRANSCRIBE capability", "video_id": video_id}

        if not _Path(ad_audio_path).exists():
            return {"error": f"AD audio file not found: {ad_audio_path}", "video_id": video_id}

        log_timed(f"Transcribing AD track with {provider.info.name}...", t0)

        try:
            result = await provider.transcribe(_Path(ad_audio_path))
        except Exception as e:
            logger.error(f"AD track transcription failed: {e}")
            return {"error": f"Transcription failed: {e}", "video_id": video_id}

        # Write VTT output from transcription segments
        vtt_lines = [
            "WEBVTT",
            "Kind: descriptions",
            "Language: en",
            "",
        ]
        txt_lines = []

        for i, seg in enumerate(result.segments, 1):
            start_ts = _format_vtt_timestamp(seg.start)
            end_ts = _format_vtt_timestamp(seg.end)
            vtt_lines.append(str(i))
            vtt_lines.append(f"{start_ts} --> {end_ts}")
            vtt_lines.append(seg.text.strip())
            vtt_lines.append("")

            minutes = int(seg.start // 60)
            seconds = int(seg.start % 60)
            txt_lines.append(f"[{minutes:02d}:{seconds:02d}] {seg.text.strip()}")

        vtt_path, txt_path = cache.get_ad_paths(video_id)
        vtt_path.write_text("\n".join(vtt_lines))
        txt_path.write_text("\n".join(txt_lines))

        # Update state
        state = cache.get_state(video_id)
        if state:
            state.ad_complete = True
            state.ad_source = "source_track"
            cache.save_state(video_id, state)

        log_timed(f"AD track transcribed: {len(result.segments)} segments", t0)

        return {
            "video_id": video_id,
            "status": "transcribed",
            "segment_count": len(result.segments),
            "vtt_path": str(vtt_path),
            "txt_path": str(txt_path),
            "source": "source_track",
            "provider": provider.info.name,
        }

    # ── Private helpers ────────────────────────────────────────────────

    @staticmethod
    def _resolve_video_path(cache_dir: Path, video_id: str) -> Path | None:
        """Locate the video source file in the cache directory."""

        state_file = cache_dir / "state.json"
        if not state_file.exists():
            return None

        state = json.loads(state_file.read_text())

        # Try cached local file first
        cached_file = state.get("cached_file")
        if cached_file:
            path = cache_dir / cached_file
            if path.exists():
                return path

        return None

    @staticmethod
    def _get_scene_keyframes(
        scene: SceneBoundary,
        video_id: str,
        cache_dir: Path,
        n: int = 3,
    ) -> list[Path]:
        """Get existing keyframes for a scene, or extract them.

        Prefers existing keyframes from the scene cache. If none exist,
        attempts to extract them via FFmpeg.

        Args:
            scene: Scene boundary
            video_id: Video ID
            cache_dir: Video cache directory
            n: Maximum keyframes to return

        Returns:
            List of keyframe paths (may be empty).
        """
        existing = list_scene_keyframes(cache_dir, scene.scene_id)
        if existing:
            if len(existing) <= n:
                return existing
            step = len(existing) // n
            return [existing[i * step] for i in range(n)]

        # Try extracting keyframes on demand
        try:
            from claudetube.operations.visual_transcript import (
                _select_keyframes_for_scene,
            )

            return _select_keyframes_for_scene(scene, video_id, cache_dir, n=n)
        except Exception as e:
            logger.warning(f"Failed to extract keyframes for scene {scene.scene_id}: {e}")
            return []

    @staticmethod
    def _parse_description_response(
        response: str | dict,
        scene: SceneBoundary,
    ) -> VisualDescription | None:
        """Parse a provider response into a VisualDescription.

        Handles both string (JSON text) and dict responses.

        Args:
            response: Provider response (str or dict)
            scene: Scene boundary for metadata

        Returns:
            VisualDescription or None on parse failure.
        """
        try:
            if isinstance(response, str):
                text = response.strip()
                # Handle markdown code blocks
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                data = json.loads(text)
            else:
                data = response

            return VisualDescription(
                scene_id=scene.scene_id,
                description=data.get("description", ""),
                people=data.get("people", []),
                objects=data.get("objects", []),
                text_on_screen=data.get("text_on_screen", []),
                actions=data.get("actions", []),
                setting=data.get("setting"),
            )
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.warning(f"Failed to parse description response for scene {scene.scene_id}: {e}")
            return None

    @staticmethod
    def _finalize(
        video_id: str,
        scenes_data: ScenesData,
        cache: CacheManager,
        cache_dir: Path,
        descriptions: list[VisualDescription],
        errors: list[dict],
        source: str,
        provider_name: str,
        t0: float,
    ) -> dict:
        """Compile generated descriptions into VTT/TXT and update state.

        Called after all scenes have been processed (or attempted) to
        produce the final output files.
        """
        # Compile the VTT from all descriptions (generated + already-cached)
        vtt_lines, txt_lines = _compile_vtt(scenes_data, cache_dir)

        if not txt_lines:
            return {
                "error": "No descriptions could be generated.",
                "video_id": video_id,
                "errors": errors,
            }

        # Write output files
        vtt_path, txt_path = cache.get_ad_paths(video_id)
        vtt_path.write_text("\n".join(vtt_lines))
        txt_path.write_text("\n".join(txt_lines))

        # Update state
        state = cache.get_state(video_id)
        if state:
            state.ad_complete = True
            state.ad_source = "generated"
            cache.save_state(video_id, state)

        log_timed(
            f"Audio descriptions generated: {len(descriptions)} scenes, "
            f"{len(errors)} errors, source={source}",
            t0,
        )

        return {
            "video_id": video_id,
            "status": "generated",
            "cue_count": len(txt_lines),
            "generated_count": len(descriptions),
            "vtt_path": str(vtt_path),
            "txt_path": str(txt_path),
            "source": source,
            "provider": provider_name,
            "errors": errors,
        }
