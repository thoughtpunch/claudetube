"""
claudetube MCP server — expose video processing tools via Model Context Protocol.

Run as: claudetube-mcp (stdio transport)
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Import search functionality
from claudetube.analysis.search import find_moments, format_timestamp
from claudetube.cache.enrichment import (
    get_enrichment_stats,
    get_scene_context,
    record_frame_examination,
    record_qa_interaction,
    search_cached_qa,
)
from claudetube.cache.knowledge_graph import get_knowledge_graph, index_video_to_graph
from claudetube.cache.manager import CacheManager
from claudetube.cache.scenes import has_scenes, load_scenes_data
from claudetube.config import get_cache_dir
from claudetube.models.local_file import is_local_file
from claudetube.operations.extract_frames import (
    extract_frames as get_frames_at,
)
from claudetube.operations.extract_frames import (
    extract_hq_frames as get_hq_frames_at,
)
from claudetube.operations.factory import get_factory
from claudetube.operations.processor import process_local_video, process_video
from claudetube.operations.transcribe import transcribe_video as _transcribe_video
from claudetube.operations.watch import watch_video
from claudetube.parsing.utils import extract_video_id
from claudetube.providers.capabilities import PROVIDER_INFO, Capability
from claudetube.providers.registry import list_all, list_available

# All logging goes to stderr so stdout stays clean for JSON-RPC
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

TRANSCRIPT_INLINE_CAP = 50_000

mcp = FastMCP("claudetube")


@mcp.tool()
async def process_video_tool(
    url: str,
    whisper_model: str = "tiny",
    copy: bool = False,
) -> str:
    """Process a video from URL or local file path.

    Returns JSON with metadata, transcript text (capped at 50k chars),
    and file paths for the full transcript and thumbnail.

    Args:
        url: Video URL, video ID, or local file path.
             Supports YouTube, Vimeo, and 1500+ sites via yt-dlp.
             Local paths can be absolute (/path/to/video.mp4),
             relative (./video.mp4), home-relative (~/Videos/file.mp4),
             or file URIs (file:///path/to/video.mp4).
        whisper_model: Whisper model size (tiny/base/small/medium/large).
        copy: For local files only - if True, copy the file to cache instead of symlink.
    """
    # Detect if input is a local file or URL
    if is_local_file(url):
        result = await asyncio.to_thread(
            process_local_video,
            url,
            output_base=get_cache_dir(),
            whisper_model=whisper_model,
            copy=copy,
        )
    else:
        result = await asyncio.to_thread(
            process_video,
            url,
            output_base=get_cache_dir(),
            whisper_model=whisper_model,
        )

    if not result.success:
        return json.dumps({"error": result.error})

    transcript_text = ""
    if result.transcript_txt and result.transcript_txt.exists():
        full = result.transcript_txt.read_text()
        transcript_text = full[:TRANSCRIPT_INLINE_CAP]
        if len(full) > TRANSCRIPT_INLINE_CAP:
            transcript_text += (
                f"\n\n[Transcript truncated at {TRANSCRIPT_INLINE_CAP} chars. "
                f"Use get_transcript for the full text.]"
            )

    return json.dumps(
        {
            "video_id": result.video_id,
            "metadata": result.metadata,
            "transcript": transcript_text,
            "transcript_srt_path": (
                str(result.transcript_srt) if result.transcript_srt else None
            ),
            "transcript_txt_path": (
                str(result.transcript_txt) if result.transcript_txt else None
            ),
            "thumbnail_path": str(result.thumbnail) if result.thumbnail else None,
            "output_dir": str(result.output_dir),
        },
        indent=2,
    )


@mcp.tool()
async def get_frames(
    video_id_or_url: str,
    start_time: float,
    duration: float = 5.0,
    interval: float = 1.0,
    quality: str = "lowest",
) -> str:
    """Extract frames from a cached video at a specific time range.

    The video must have been processed first with process_video.
    Frame examinations are automatically recorded for progressive learning.

    Args:
        video_id_or_url: YouTube video ID or URL.
        start_time: Start time in seconds.
        duration: Duration to capture in seconds.
        interval: Seconds between frames.
        quality: Quality tier (lowest/low/medium/high/highest).
    """
    video_id = extract_video_id(video_id_or_url)
    cache_dir = get_cache_dir() / video_id

    frames = await asyncio.to_thread(
        get_frames_at,
        video_id_or_url,
        start_time=start_time,
        duration=duration,
        interval=interval,
        output_base=get_cache_dir(),
        quality=quality,
    )

    # Record frame examination for progressive learning
    enrichment = None
    if frames and cache_dir.exists():
        enrichment = await asyncio.to_thread(
            record_frame_examination,
            video_id,
            cache_dir,
            start_time,
            duration,
            quality,
        )

    result = {
        "frame_count": len(frames),
        "frame_paths": [str(f) for f in frames],
    }

    if enrichment:
        result["learning"] = {
            "scene_examined": enrichment["scene_id"],
            "relevance_boost": enrichment["new_boost"],
        }

    return json.dumps(result, indent=2)


@mcp.tool()
async def get_hq_frames(
    video_id_or_url: str,
    start_time: float,
    duration: float = 5.0,
    interval: float = 1.0,
    width: int = 1280,
) -> str:
    """Extract HIGH QUALITY frames for reading text, code, or small UI elements.

    Downloads best available quality (larger file, slower). The video must
    have been processed first with process_video.
    Frame examinations are automatically recorded for progressive learning.

    Args:
        video_id_or_url: YouTube video ID or URL.
        start_time: Start time in seconds.
        duration: Duration to capture in seconds.
        interval: Seconds between frames.
        width: Frame width in pixels.
    """
    video_id = extract_video_id(video_id_or_url)
    cache_dir = get_cache_dir() / video_id

    frames = await asyncio.to_thread(
        get_hq_frames_at,
        video_id_or_url,
        start_time=start_time,
        duration=duration,
        interval=interval,
        output_base=get_cache_dir(),
        width=width,
    )

    # Record frame examination for progressive learning
    enrichment = None
    if frames and cache_dir.exists():
        enrichment = await asyncio.to_thread(
            record_frame_examination,
            video_id,
            cache_dir,
            start_time,
            duration,
            "hq",
        )

    result = {
        "frame_count": len(frames),
        "frame_paths": [str(f) for f in frames],
    }

    if enrichment:
        result["learning"] = {
            "scene_examined": enrichment["scene_id"],
            "relevance_boost": enrichment["new_boost"],
        }

    return json.dumps(result, indent=2)


@mcp.tool()
async def transcribe_video(
    video_id_or_url: str,
    whisper_model: str = "small",
    force: bool = False,
    provider: str | None = None,
) -> str:
    """Transcribe a video's audio using Whisper or another transcription provider.

    Returns cached transcript immediately if available, otherwise runs
    transcription. Use force=True to re-transcribe with a different model
    or provider.

    Args:
        video_id_or_url: Video ID or URL.
        whisper_model: Whisper model size (tiny/base/small/medium/large).
        force: Re-transcribe even if a cached transcript exists.
        provider: Override transcription provider (e.g., "whisper-local", "openai",
            "deepgram"). If None, uses configured preference.
    """
    transcriber = None
    if provider:
        from claudetube.providers import get_provider

        transcriber = get_provider(provider)
    else:
        try:
            factory = get_factory()
            transcriber = factory.get_transcriber(model_size=whisper_model)
        except (RuntimeError, ImportError):
            pass  # Fall back to default in _transcribe_video

    result = await _transcribe_video(
        video_id_or_url,
        whisper_model=whisper_model,
        force=force,
        output_base=get_cache_dir(),
        transcriber=transcriber,
    )

    if not result["success"]:
        return json.dumps({"error": result["message"]})

    # Read transcript text for inline return
    transcript_text = ""
    txt_path = result.get("transcript_txt")
    if txt_path:
        path = Path(txt_path)
        if path.exists():
            full = path.read_text()
            transcript_text = full[:TRANSCRIPT_INLINE_CAP]
            if len(full) > TRANSCRIPT_INLINE_CAP:
                transcript_text += (
                    f"\n\n[Transcript truncated at {TRANSCRIPT_INLINE_CAP} chars. "
                    f"Use get_transcript for the full text.]"
                )

    return json.dumps(
        {
            "video_id": result["video_id"],
            "source": result["source"],
            "whisper_model": result["whisper_model"],
            "message": result["message"],
            "transcript": transcript_text,
            "transcript_srt_path": result["transcript_srt"],
            "transcript_txt_path": result["transcript_txt"],
        },
        indent=2,
    )


@mcp.tool()
async def list_cached_videos() -> str:
    """List all videos that have been processed and cached.

    Returns JSON with video ID, title, duration, and transcript source
    for each cached video.
    """
    cache_dir = get_cache_dir()
    videos = []
    if cache_dir.exists():
        for state_file in sorted(cache_dir.glob("*/state.json")):
            try:
                state = json.loads(state_file.read_text())
                videos.append(
                    {
                        "video_id": state.get("video_id", state_file.parent.name),
                        "title": state.get("title"),
                        "duration_string": state.get("duration_string"),
                        "transcript_complete": state.get("transcript_complete", False),
                        "transcript_source": state.get("transcript_source"),
                        "cache_dir": str(state_file.parent),
                    }
                )
            except (json.JSONDecodeError, OSError):
                continue

    return json.dumps({"count": len(videos), "videos": videos}, indent=2)


@mcp.tool()
async def get_playlist(
    playlist_url: str,
) -> str:
    """Extract metadata from a playlist URL.

    Fetches playlist info including title, description, channel, and video list
    without downloading any content. Useful for understanding course structure
    or video series before processing individual videos.

    The playlist is cached for future reference.

    Args:
        playlist_url: URL to a playlist on YouTube or other supported sites.
    """
    from claudetube.operations.playlist import (
        extract_playlist_metadata,
        save_playlist_metadata,
    )

    try:
        data = extract_playlist_metadata(playlist_url)
        save_playlist_metadata(data)
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def list_playlists() -> str:
    """List all cached playlists.

    Returns playlist ID, title, video count, and inferred type (course/series/
    conference/collection) for each cached playlist.
    """
    from claudetube.operations.playlist import list_cached_playlists

    playlists = list_cached_playlists()
    return json.dumps({"count": len(playlists), "playlists": playlists}, indent=2)


@mcp.tool()
async def get_transcript(
    video_id: str,
    format: str = "txt",
) -> str:
    """Get the full transcript for a cached video.

    Use this to retrieve the complete transcript without the 50k char cap
    applied by process_video.

    Args:
        video_id: YouTube video ID.
        format: Transcript format — "txt" for plain text or "srt" for subtitles.
    """
    video_id = extract_video_id(video_id)
    video_dir = get_cache_dir() / video_id

    if not video_dir.exists():
        return json.dumps({"error": f"No cached video found for '{video_id}'"})

    if format == "srt":
        transcript_path = video_dir / "audio.srt"
    else:
        transcript_path = video_dir / "audio.txt"

    if not transcript_path.exists():
        # Try the other format as fallback
        fallback = video_dir / ("audio.srt" if format == "txt" else "audio.txt")
        if fallback.exists():
            transcript_path = fallback
        else:
            return json.dumps({"error": "No transcript file found"})

    text = transcript_path.read_text()
    return json.dumps(
        {
            "video_id": video_id,
            "format": transcript_path.suffix.lstrip("."),
            "length": len(text),
            "transcript": text,
        },
        indent=2,
    )


def _get_scenes_sync(video_id: str, force: bool = False, enrich: bool = False) -> dict:
    """Get scene structure for a cached video (sync version).

    Returns cached scenes if available, otherwise runs smart segmentation.
    Implements "Cheap First, Expensive Last" - returns cached data instantly.

    Args:
        video_id: Video ID
        force: Re-run segmentation even if cached
        enrich: Run visual enrichment (generate visual.json) for each scene

    Returns:
        Dict with scene data or error
    """
    cache_dir = get_cache_dir() / video_id

    if not cache_dir.exists():
        return {
            "error": "Video not cached. Run process_video first.",
            "video_id": video_id,
        }

    # Check state.json exists
    state_file = cache_dir / "state.json"
    if not state_file.exists():
        return {
            "error": "Video not processed. Run process_video first.",
            "video_id": video_id,
        }

    # Fast path: return cached scenes
    if not force and has_scenes(cache_dir):
        scenes_data = load_scenes_data(cache_dir)
        if scenes_data:
            # Trigger visual enrichment if requested
            if enrich:
                from claudetube.operations.visual_transcript import (
                    generate_visual_transcript,
                )

                generate_visual_transcript(
                    video_id=video_id, output_base=cache_dir.parent
                )

            result = scenes_data.to_dict()
            # Enrich with visual descriptions if available
            for scene in result.get("scenes", []):
                scene_id = scene.get("scene_id", 0)
                visual_file = (
                    cache_dir / "scenes" / f"scene_{scene_id:03d}" / "visual.json"
                )
                if visual_file.exists():
                    visual_text = visual_file.read_text()
                    if visual_text.strip():
                        visual_data = json.loads(visual_text)
                        if visual_data:
                            scene["visual"] = visual_data

            # Enrich with people tracking data if available
            people_file = cache_dir / "entities" / "people.json"
            if people_file.exists():
                try:
                    people_data = json.loads(people_file.read_text())
                    result["people_tracking"] = people_data
                except json.JSONDecodeError:
                    pass

            return result

    # No cached scenes - run segmentation
    try:
        state = json.loads(state_file.read_text())
    except json.JSONDecodeError:
        return {"error": "Invalid state.json", "video_id": video_id}

    # Build video_info dict for chapter extraction
    # (use stored state data - includes description for chapter parsing)
    video_info = {
        "duration": state.get("duration"),
        "description": state.get("description", ""),
    }

    # Load transcript segments from SRT
    transcript_segments = None
    srt_path = cache_dir / "audio.srt"
    if srt_path.exists():
        from claudetube.analysis.pause import parse_srt_file

        transcript_segments = parse_srt_file(srt_path)

    # Run smart segmentation
    from claudetube.operations.segmentation import segment_video_smart

    # Find video path for visual fallback (if needed)
    video_path = None
    cached_file = state.get("cached_file")
    if cached_file:
        video_path = cache_dir / cached_file
        if not video_path.exists():
            video_path = None

    scenes_data = segment_video_smart(
        video_id=video_id,
        video_path=video_path,
        transcript_segments=transcript_segments,
        video_info=video_info,
        cache_dir=cache_dir,
        srt_path=srt_path if srt_path.exists() else None,
        force=force,
    )

    # Trigger visual enrichment if requested
    if enrich:
        from claudetube.operations.visual_transcript import generate_visual_transcript

        enrich_result = generate_visual_transcript(
            video_id=video_id, output_base=cache_dir.parent
        )
        # Merge visual data into scene dicts
        result = scenes_data.to_dict()
        visual_by_scene = {}
        for vr in enrich_result.get("results", []):
            visual_by_scene[vr.get("scene_id")] = vr
        for scene in result.get("scenes", []):
            scene_id = scene.get("scene_id", 0)
            if scene_id in visual_by_scene:
                scene["visual"] = visual_by_scene[scene_id]
        return result

    return scenes_data.to_dict()


@mcp.tool()
async def get_scenes(
    video_id: str,
    force: bool = False,
    enrich: bool = False,
) -> str:
    """Get scene structure of a processed video.

    Returns scene list with timestamps, transcript summaries, and visual
    descriptions for understanding video structure. Uses cached scenes if
    available, otherwise runs smart segmentation.

    This is useful for:
    - Understanding video organization before asking questions
    - Finding specific sections by topic
    - Getting an overview of video content

    Args:
        video_id: Video ID of a previously processed video.
        force: Re-run segmentation even if cached (default: False).
        enrich: Generate visual descriptions for each scene using a VisionAnalyzer
            (default: False). This is an expensive operation that calls a vision API.
            Visual descriptions are cached, so subsequent calls are free.
    """
    video_id = extract_video_id(video_id)

    result = await asyncio.to_thread(_get_scenes_sync, video_id, force, enrich)

    return json.dumps(result, indent=2)


@mcp.tool()
async def generate_visual_transcripts(
    video_id: str,
    scene_id: int | None = None,
    force: bool = False,
    provider: str | None = None,
) -> str:
    """Generate visual descriptions for video scenes.

    Uses vision AI to describe what's happening visually in each scene.
    Results are cached in scene_{NNN}/visual.json.

    Follows "Cheap First, Expensive Last" principle:
    - Returns cached descriptions instantly if available
    - Skips scenes with good transcript coverage (talking heads)
    - Only calls vision API when visual context adds value

    Args:
        video_id: Video ID of a previously processed video.
        scene_id: Optional specific scene ID (None = all scenes).
        force: Re-generate even if cached (default: False).
        provider: Override vision provider (e.g., "anthropic", "openai", "google",
            "claude-code"). If None, uses configured preference.
    """
    from claudetube.operations.visual_transcript import generate_visual_transcript

    vision_analyzer = None
    if provider:
        from claudetube.providers import get_provider

        vision_analyzer = get_provider(provider)
    else:
        try:
            factory = get_factory()
            vision_analyzer = factory.get_vision_analyzer()
        except (RuntimeError, ImportError):
            pass  # Fall back to default in generate_visual_transcript

    video_id = extract_video_id(video_id)
    result = await asyncio.to_thread(
        generate_visual_transcript,
        video_id,
        scene_id=scene_id,
        force=force,
        output_base=get_cache_dir(),
        vision_analyzer=vision_analyzer,
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def extract_entities_tool(
    video_id: str,
    scene_id: int | None = None,
    force: bool = False,
    generate_visual: bool = True,
    provider: str | None = None,
) -> str:
    """Extract entities (objects, people, text, concepts) from video scenes.

    Uses AI-powered extraction with structured output schemas for consistent
    results. Extracts visual entities from keyframes (VisionAnalyzer) and
    semantic concepts from transcripts (Reasoner) concurrently.

    Follows "Cheap First, Expensive Last" principle:
    - Returns cached entities.json instantly if available
    - Skips scenes with minimal content
    - Only calls AI providers when needed

    Entities-first architecture: entities are PRIMARY, visual.json is DERIVED
    from entities when generate_visual=True (backwards compatible).

    Args:
        video_id: Video ID of a previously processed video.
        scene_id: Optional specific scene ID (None = all scenes).
        force: Re-extract even if cached (default: False).
        generate_visual: Generate visual.json from entities (default: True).
        provider: Override AI provider (e.g., "anthropic", "openai", "google",
            "claude-code"). Provider must support VisionAnalyzer and/or Reasoner.
            If None, uses configured preference.
    """
    from claudetube.operations.entity_extraction import extract_entities_for_video

    vision_analyzer = None
    reasoner = None
    if provider:
        from claudetube.providers import get_provider as _get_provider
        from claudetube.providers.base import Reasoner as ReasonerProto
        from claudetube.providers.base import VisionAnalyzer as VisionProto

        p = _get_provider(provider)
        if isinstance(p, VisionProto):
            vision_analyzer = p
        if isinstance(p, ReasonerProto):
            reasoner = p
    else:
        try:
            factory = get_factory()
            vision_analyzer = factory.get_vision_analyzer()
            reasoner = factory.get_reasoner()
        except (RuntimeError, ImportError):
            pass  # Fall back to default in extract_entities_for_video

    video_id = extract_video_id(video_id)
    result = await asyncio.to_thread(
        extract_entities_for_video,
        video_id,
        scene_id=scene_id,
        force=force,
        generate_visual=generate_visual,
        output_base=get_cache_dir(),
        vision_analyzer=vision_analyzer,
        reasoner=reasoner,
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def track_people_tool(
    video_id: str,
    force: bool = False,
    use_face_recognition: bool = False,
    provider: str | None = None,
) -> str:
    """Track people across scenes in a video.

    Identifies distinct people and tracks their appearances with timestamps.
    Uses visual transcript data by default (cheap), with optional face_recognition
    for more accurate tracking (expensive, requires pip install face_recognition).

    Results are cached in entities/people.json.

    Args:
        video_id: Video ID of a previously processed video.
        force: Re-generate even if cached (default: False).
        use_face_recognition: Use face_recognition library for accurate tracking.
            Requires: pip install face_recognition (default: False).
        provider: Override vision/video provider (e.g., "google", "anthropic",
            "openai"). If None, uses configured preference.
    """
    from claudetube.operations.person_tracking import track_people

    video_analyzer = None
    vision_analyzer = None
    if provider:
        from claudetube.providers import get_provider
        from claudetube.providers.base import VideoAnalyzer, VisionAnalyzer

        p = get_provider(provider)
        if isinstance(p, VideoAnalyzer):
            video_analyzer = p
        if isinstance(p, VisionAnalyzer):
            vision_analyzer = p
    else:
        try:
            factory = get_factory()
            video_analyzer = factory.get_video_analyzer()
            vision_analyzer = factory.get_vision_analyzer()
        except (RuntimeError, ImportError):
            pass

    video_id = extract_video_id(video_id)
    result = await asyncio.to_thread(
        track_people,
        video_id,
        force=force,
        use_face_recognition=use_face_recognition,
        output_base=get_cache_dir(),
        video_analyzer=video_analyzer,
        vision_analyzer=vision_analyzer,
    )

    return json.dumps(result, indent=2)


def _format_moments_for_claude(moments: list[dict]) -> str:
    """Format moments as readable text for Claude.

    Args:
        moments: List of moment dicts from SearchMoment.to_dict().

    Returns:
        Human-readable formatted string.
    """
    if not moments:
        return "No relevant moments found."

    lines = [
        f"Found {len(moments)} relevant moment{'s' if len(moments) != 1 else ''}:\n"
    ]

    for m in moments:
        end_str = format_timestamp(m["end_time"])
        lines.append(
            f"{m['rank']}. [{m['timestamp_str']}-{end_str}] "
            f"(relevance: {m['relevance']:.0%})\n"
            f"   {m['preview']}\n"
        )

    return "\n".join(lines)


@mcp.tool()
async def find_moments_tool(
    video_id: str,
    query: str,
    top_k: int = 5,
    semantic_weight: float = 0.5,
) -> str:
    """Find moments in a video matching a natural language query.

    Uses tiered search strategy (Cheap First, Expensive Last):
    1. TEXT - Fast transcript text matching (~100ms)
    2. SEMANTIC - Vector embedding similarity (if text search fails)

    When both text and semantic results exist for a scene, scores are blended
    using semantic_weight. Falls back to text-only if embeddings are unavailable.

    The video must have been processed with process_video and have scene data.

    Example: find_moments_tool('abc123', 'when they fix the auth bug')

    Args:
        video_id: Video ID of a previously processed video.
        query: Natural language query (e.g., "when they discuss authentication").
        top_k: Maximum number of results to return (default: 5).
        semantic_weight: Weight for semantic similarity scores (0.0 to 1.0).
            Text weight is 1 - semantic_weight. Default: 0.5 (equal blend).
    """
    video_id = extract_video_id(video_id)

    try:
        moments = await asyncio.to_thread(
            find_moments,
            video_id,
            query,
            top_k=top_k,
            semantic_weight=semantic_weight,
        )
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})

    # Convert to dicts for JSON serialization
    moment_dicts = [m.to_dict() for m in moments]

    # Format for Claude
    output = {
        "video_id": video_id,
        "query": query,
        "results": moment_dicts,
        "formatted": _format_moments_for_claude(moment_dicts),
    }

    return json.dumps(output, indent=2)


@mcp.tool()
async def analyze_deep_tool(
    video_id: str,
    force: bool = False,
) -> str:
    """Deep analysis of video with OCR and entity extraction.

    Performs comprehensive analysis including:
    - Scene segmentation
    - Visual transcripts for each scene
    - OCR text extraction
    - Code block detection
    - Entity extraction (people, technologies, keywords)

    This is more expensive than standard analysis (~2 min for 30-min video).
    Results are cached for subsequent queries.

    Args:
        video_id: Video ID of a previously processed video.
        force: Re-run analysis even if cached (default: False).
    """
    from claudetube.operations.analysis_depth import AnalysisDepth, analyze_video

    video_id = extract_video_id(video_id)

    result = await asyncio.to_thread(
        analyze_video,
        video_id,
        depth=AnalysisDepth.DEEP,
        force=force,
        output_base=get_cache_dir(),
    )

    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
async def analyze_focus_tool(
    video_id: str,
    start_time: float,
    end_time: float,
    force: bool = False,
) -> str:
    """Exhaustive analysis of a specific video section.

    Performs frame-by-frame analysis on scenes within the specified time range.
    Use this for detailed investigation of specific moments (e.g., code demos,
    bug introductions, key explanations).

    This is the most expensive analysis mode. Use sparingly.

    Args:
        video_id: Video ID of a previously processed video.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        force: Re-run analysis even if cached (default: False).
    """
    from claudetube.cache.scenes import load_scenes_data
    from claudetube.operations.analysis_depth import AnalysisDepth, analyze_video

    video_id = extract_video_id(video_id)
    cache_dir = get_cache_dir() / video_id

    # Find scenes in time range
    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return json.dumps(
            {"error": "No scenes found. Run get_scenes first.", "video_id": video_id}
        )

    focus_ids = [
        s.scene_id
        for s in scenes_data.scenes
        if s.start_time >= start_time and s.end_time <= end_time
    ]

    # Also include scenes that overlap with the range
    if not focus_ids:
        focus_ids = [
            s.scene_id
            for s in scenes_data.scenes
            if not (s.end_time < start_time or s.start_time > end_time)
        ]

    if not focus_ids:
        return json.dumps(
            {
                "error": f"No scenes found between {start_time}s and {end_time}s",
                "video_id": video_id,
            }
        )

    result = await asyncio.to_thread(
        analyze_video,
        video_id,
        depth=AnalysisDepth.EXHAUSTIVE,
        focus_sections=focus_ids,
        force=force,
        output_base=get_cache_dir(),
    )

    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
async def get_analysis_status_tool(
    video_id: str,
) -> str:
    """Get current analysis status for a video.

    Shows what analysis has been completed for each scene:
    - Transcript coverage
    - Visual descriptions
    - Technical content (OCR, code)
    - Entity extraction

    Use this to understand what analysis is cached before running
    more expensive operations.

    Args:
        video_id: Video ID of a previously processed video.
    """
    from claudetube.operations.analysis_depth import get_analysis_status

    video_id = extract_video_id(video_id)

    result = await asyncio.to_thread(
        get_analysis_status,
        video_id,
        output_base=get_cache_dir(),
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def record_qa_tool(
    video_id: str,
    question: str,
    answer: str,
) -> str:
    """Record a question-answer interaction for progressive learning.

    Call this after answering a question about a video to cache the Q&A
    for future reference. Subsequent similar questions can be answered faster
    by checking the cached Q&A history.

    The system automatically identifies which scenes are relevant to the Q&A
    and boosts their relevance scores for future searches.

    Args:
        video_id: Video ID of the video the question was about.
        question: The question that was asked.
        answer: The answer that was given.
    """
    video_id = extract_video_id(video_id)
    cache_dir = get_cache_dir() / video_id

    if not cache_dir.exists():
        return json.dumps({"error": f"Video '{video_id}' not found in cache"})

    result = await asyncio.to_thread(
        record_qa_interaction,
        video_id,
        cache_dir,
        question,
        answer,
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def search_qa_history_tool(
    video_id: str,
    query: str,
) -> str:
    """Search for previously answered questions about a video.

    Use this before answering a new question to check if a similar question
    has been answered before. This enables "second query faster than first"
    by returning cached answers.

    Args:
        video_id: Video ID to search Q&A history for.
        query: The question to search for (keyword matching).
    """
    video_id = extract_video_id(video_id)
    cache_dir = get_cache_dir() / video_id

    if not cache_dir.exists():
        return json.dumps({"error": f"Video '{video_id}' not found in cache"})

    results = await asyncio.to_thread(
        search_cached_qa,
        video_id,
        cache_dir,
        query,
    )

    return json.dumps(
        {
            "video_id": video_id,
            "query": query,
            "match_count": len(results),
            "matches": results,
        },
        indent=2,
    )


@mcp.tool()
async def get_scene_context_tool(
    video_id: str,
    scene_id: int,
) -> str:
    """Get all learned context for a specific scene.

    Returns observations made about the scene, related Q&A pairs,
    and the scene's relevance boost. Use this when revisiting a scene
    to leverage prior analysis.

    Args:
        video_id: Video ID.
        scene_id: Scene index (0-based).
    """
    video_id = extract_video_id(video_id)
    cache_dir = get_cache_dir() / video_id

    if not cache_dir.exists():
        return json.dumps({"error": f"Video '{video_id}' not found in cache"})

    context = await asyncio.to_thread(
        get_scene_context,
        video_id,
        cache_dir,
        scene_id,
    )

    return json.dumps(
        {
            "video_id": video_id,
            "scene_id": scene_id,
            **context,
        },
        indent=2,
    )


@mcp.tool()
async def get_enrichment_stats_tool(
    video_id: str,
) -> str:
    """Get statistics about cache enrichment for a video.

    Shows how much progressive learning has occurred:
    - Number of observations recorded
    - Number of Q&A pairs cached
    - Number of scenes with relevance boosts

    Args:
        video_id: Video ID.
    """
    video_id = extract_video_id(video_id)
    cache_dir = get_cache_dir() / video_id

    if not cache_dir.exists():
        return json.dumps({"error": f"Video '{video_id}' not found in cache"})

    stats = await asyncio.to_thread(
        get_enrichment_stats,
        cache_dir,
    )

    return json.dumps(
        {
            "video_id": video_id,
            **stats,
        },
        indent=2,
    )


@mcp.tool()
async def find_related_videos_tool(
    query: str,
) -> str:
    """Find videos related to a topic across all cached videos.

    Searches the cross-video knowledge graph for videos sharing
    entities or concepts matching the query. Use this to discover
    connections between videos or find all videos about a topic.

    The query performs case-insensitive substring matching against:
    - Entities: People, technologies, objects mentioned in videos
    - Concepts: Key topics and terms extracted from transcripts

    Args:
        query: Search query (e.g., "python", "machine learning", "authentication")
    """
    graph = get_knowledge_graph()

    if graph.video_count == 0:
        return json.dumps(
            {
                "query": query,
                "error": "No videos indexed yet. Use index_video_to_graph_tool first.",
                "matches": [],
            }
        )

    matches = graph.find_related_videos(query)

    return json.dumps(
        {
            "query": query,
            "match_count": len(matches),
            "matches": [m.to_dict() for m in matches],
            "graph_stats": graph.get_stats(),
        },
        indent=2,
    )


@mcp.tool()
async def index_video_to_graph_tool(
    video_id: str,
    force: bool = False,
) -> str:
    """Index a video's entities and concepts into the knowledge graph.

    Adds the video to the cross-video knowledge graph, enabling
    cross-video search and relationship discovery. The video must
    have been processed and have entity tracking data.

    Indexing is automatic if the video has entities/concepts.json.
    Use force=True to re-index with updated entity data.

    Args:
        video_id: Video ID of a previously processed video.
        force: Re-index even if already present (default: False).
    """
    video_id = extract_video_id(video_id)
    cache_dir = get_cache_dir() / video_id

    if not cache_dir.exists():
        return json.dumps({"error": f"No cached video found for '{video_id}'"})

    result = await asyncio.to_thread(
        index_video_to_graph,
        video_id,
        cache_dir,
        force=force,
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def get_video_connections_tool(
    video_id: str,
) -> str:
    """Get videos connected to a specific video.

    Finds other videos that share entities or concepts with the
    specified video. Useful for discovering related content or
    building viewing paths through a video collection.

    Args:
        video_id: Video ID to find connections for.
    """
    video_id = extract_video_id(video_id)
    graph = get_knowledge_graph()

    video = graph.get_video(video_id)
    if not video:
        return json.dumps(
            {
                "error": f"Video '{video_id}' not in knowledge graph. Index it first.",
                "video_id": video_id,
            }
        )

    connected_ids = graph.get_video_connections(video_id)

    # Get full info for connected videos
    connections = []
    for vid in connected_ids:
        v = graph.get_video(vid)
        if v:
            connections.append(v.to_dict())

    return json.dumps(
        {
            "video_id": video_id,
            "video_title": video.title,
            "connection_count": len(connections),
            "connections": connections,
        },
        indent=2,
    )


@mcp.tool()
async def get_descriptions(
    video_id_or_url: str,
    format: str = "vtt",
    regenerate: bool = False,
) -> str:
    """Get visual descriptions for accessibility (audio description).

    Returns cached descriptions instantly if available. Otherwise generates
    them following "Cheap First, Expensive Last":
    1. Return cached .ad.vtt/.ad.txt if they exist
    2. Check for source AD track via yt-dlp and transcribe it
    3. Compile from existing scene visual.json data
    4. Generate via AI provider (VIDEO -> VISION fallback)

    Args:
        video_id_or_url: Video ID or URL.
        format: Output format — "vtt" for WebVTT or "txt" for plain text.
        regenerate: Re-generate even if cached (default: False).
    """
    from claudetube.operations.audio_description import (
        AudioDescriptionGenerator,
        compile_scene_descriptions,
        get_scene_descriptions,
    )

    video_id = extract_video_id(video_id_or_url)
    cache_dir = get_cache_dir() / video_id

    if not cache_dir.exists():
        return json.dumps(
            {
                "error": "Video not cached. Run process_video first.",
                "video_id": video_id,
            }
        )

    cache = CacheManager(get_cache_dir())

    # 1. CACHE — Return instantly if AD files exist
    if not regenerate and cache.has_ad(video_id):
        result = get_scene_descriptions(video_id, output_base=get_cache_dir())
        if format == "txt" and "txt" in result:
            result["content"] = result.pop("txt", "")
            result.pop("vtt", None)
        elif "vtt" in result:
            result["content"] = result.pop("vtt", "")
            result.pop("txt", None)
        result["source"] = "cache"
        return json.dumps(result, indent=2)

    # 2. YT-DLP — Check for source AD track
    state = cache.get_state(video_id)
    if state and state.url and state.ad_track_available is None:
        try:
            from claudetube.tools.yt_dlp import YtDlpTool

            yt = YtDlpTool()
            ad_format = await asyncio.to_thread(yt.check_audio_description, state.url)
            state.ad_track_available = ad_format is not None
            cache.save_state(video_id, state)

            if ad_format is not None:
                ad_audio_path = cache_dir / "audio_description.mp3"
                if not ad_audio_path.exists():
                    await asyncio.to_thread(
                        yt.download_audio_description,
                        state.url,
                        ad_audio_path,
                        format_id=ad_format.get("format_id"),
                    )

                if ad_audio_path.exists():
                    try:
                        generator = get_factory().get_audio_description_generator()
                    except Exception:
                        generator = AudioDescriptionGenerator()
                    result = await generator.transcribe_ad_track(
                        video_id,
                        ad_audio_path,
                        output_base=get_cache_dir(),
                    )
                    if "error" not in result:
                        ad_result = get_scene_descriptions(
                            video_id, output_base=get_cache_dir()
                        )
                        if format == "txt" and "txt" in ad_result:
                            ad_result["content"] = ad_result.pop("txt", "")
                            ad_result.pop("vtt", None)
                        elif "vtt" in ad_result:
                            ad_result["content"] = ad_result.pop("vtt", "")
                            ad_result.pop("txt", None)
                        ad_result["source"] = "source_track"
                        return json.dumps(ad_result, indent=2)
        except Exception as e:
            logger.warning(f"AD track detection failed: {e}")

    # 3. SCENE COMPILATION — Compile from existing visual.json
    if has_scenes(cache_dir):
        result = await asyncio.to_thread(
            compile_scene_descriptions,
            video_id,
            force=regenerate,
            output_base=get_cache_dir(),
        )
        if "error" not in result:
            ad_result = get_scene_descriptions(video_id, output_base=get_cache_dir())
            if format == "txt" and "txt" in ad_result:
                ad_result["content"] = ad_result.pop("txt", "")
                ad_result.pop("vtt", None)
            elif "vtt" in ad_result:
                ad_result["content"] = ad_result.pop("vtt", "")
                ad_result.pop("txt", None)
            ad_result["source"] = result.get("source", "scene_compilation")
            return json.dumps(ad_result, indent=2)

    # 4. PROVIDER GENERATION — Use AI providers (expensive)
    try:
        try:
            generator = get_factory().get_audio_description_generator()
        except Exception:
            generator = AudioDescriptionGenerator()
        result = await generator.generate(
            video_id,
            force=regenerate,
            output_base=get_cache_dir(),
        )
        if "error" not in result:
            ad_result = get_scene_descriptions(video_id, output_base=get_cache_dir())
            if format == "txt" and "txt" in ad_result:
                ad_result["content"] = ad_result.pop("txt", "")
                ad_result.pop("vtt", None)
            elif "vtt" in ad_result:
                ad_result["content"] = ad_result.pop("vtt", "")
                ad_result.pop("txt", None)
            ad_result["source"] = result.get("source", "generated")
            return json.dumps(ad_result, indent=2)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps(
            {
                "error": f"Audio description generation failed: {e}",
                "video_id": video_id,
                "suggestion": "Run get_scenes and generate_visual_transcripts first, then try again.",
            }
        )


@mcp.tool()
async def describe_moment(
    video_id_or_url: str,
    timestamp: float,
    context: str | None = None,
) -> str:
    """Describe visual content at a specific moment for accessibility.

    Extracts HQ frames around the timestamp and generates an audio
    description using the best available AI vision provider. This is
    an expensive on-demand operation.

    Args:
        video_id_or_url: Video ID or URL.
        timestamp: Time in seconds to describe.
        context: Optional context about what the viewer is interested in.
    """
    video_id = extract_video_id(video_id_or_url)
    cache_dir = get_cache_dir() / video_id

    if not cache_dir.exists():
        return json.dumps(
            {
                "error": "Video not cached. Run process_video first.",
                "video_id": video_id,
            }
        )

    # Extract HQ frames around the timestamp
    frames = await asyncio.to_thread(
        get_hq_frames_at,
        video_id_or_url,
        start_time=max(0, timestamp - 1),
        duration=3,
        interval=1,
        output_base=get_cache_dir(),
        width=1280,
    )

    if not frames:
        return json.dumps(
            {
                "error": "Could not extract frames at the given timestamp.",
                "video_id": video_id,
                "timestamp": timestamp,
            }
        )

    # Try to get a vision provider for description
    try:
        factory = get_factory()
        vision = factory.get_vision_analyzer()
    except Exception:
        vision = None

    if vision is None:
        return json.dumps(
            {
                "video_id": video_id,
                "timestamp": timestamp,
                "frame_count": len(frames),
                "frame_paths": [str(f) for f in frames],
                "description": None,
                "note": "No vision provider available. Frames extracted for manual inspection.",
            },
            indent=2,
        )

    # Build prompt
    prompt_parts = [
        "Describe what is visually happening at this moment for a vision-impaired viewer.",
        "Focus on: people present, their actions, objects visible, any text on screen,",
        "and the setting. Be concise (2-4 sentences). Do not describe audio or dialogue.",
    ]
    if context:
        prompt_parts.append(f"\nContext: {context}")

    prompt = " ".join(prompt_parts)

    try:
        result = await vision.analyze_images(
            images=[Path(f) for f in frames],
            prompt=prompt,
        )
        description = result if isinstance(result, str) else json.dumps(result)
    except Exception as e:
        description = None
        logger.warning(f"Vision analysis failed for moment at {timestamp}s: {e}")

    return json.dumps(
        {
            "video_id": video_id,
            "timestamp": timestamp,
            "frame_count": len(frames),
            "frame_paths": [str(f) for f in frames],
            "description": description,
            "provider": vision.info.name if vision else None,
        },
        indent=2,
    )


@mcp.tool()
async def get_accessible_transcript(
    video_id_or_url: str,
    format: str = "txt",
) -> str:
    """Get a merged transcript with audio descriptions interspersed.

    Combines the spoken transcript with visual descriptions to create
    a fully accessible text version. Audio descriptions are marked with
    [AD] tags to distinguish them from dialogue.

    The video must have both a transcript and audio descriptions cached.

    Args:
        video_id_or_url: Video ID or URL.
        format: Output format — "txt" for plain text or "srt" for subtitles.
    """
    from claudetube.operations.audio_description import get_scene_descriptions

    video_id = extract_video_id(video_id_or_url)
    cache_dir = get_cache_dir() / video_id

    if not cache_dir.exists():
        return json.dumps(
            {
                "error": "Video not cached. Run process_video first.",
                "video_id": video_id,
            }
        )

    # Load transcript
    if format == "srt":
        transcript_path = cache_dir / "audio.srt"
    else:
        transcript_path = cache_dir / "audio.txt"

    if not transcript_path.exists():
        fallback = cache_dir / ("audio.srt" if format == "txt" else "audio.txt")
        if fallback.exists():
            transcript_path = fallback
        else:
            return json.dumps(
                {
                    "error": "No transcript found. Run process_video or transcribe_video first.",
                    "video_id": video_id,
                }
            )

    transcript_text = transcript_path.read_text()

    # Load audio descriptions
    cache = CacheManager(get_cache_dir())
    if not cache.has_ad(video_id):
        return json.dumps(
            {
                "error": "No audio descriptions found. Run get_descriptions first.",
                "video_id": video_id,
                "transcript_available": True,
            }
        )

    ad_data = get_scene_descriptions(video_id, output_base=get_cache_dir())
    ad_txt = ad_data.get("txt", "")

    if not ad_txt:
        return json.dumps(
            {
                "error": "Audio description file is empty.",
                "video_id": video_id,
            }
        )

    # Parse AD lines into (timestamp_seconds, description) tuples
    ad_entries = []
    for line in ad_txt.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: [MM:SS] description text
        if line.startswith("[") and "]" in line:
            bracket_end = line.index("]")
            ts_str = line[1:bracket_end]
            desc = line[bracket_end + 1 :].strip()
            parts = ts_str.split(":")
            try:
                if len(parts) == 2:
                    seconds = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                else:
                    continue
                ad_entries.append((seconds, desc))
            except ValueError:
                continue

    # Build merged output with [AD] markers
    merged_lines = [
        "=== ACCESSIBLE TRANSCRIPT ===",
        f"Video ID: {video_id}",
        "",
        "--- Spoken Transcript ---",
        transcript_text,
        "",
        "--- Visual Descriptions (Audio Description) ---",
    ]
    for ts, desc in ad_entries:
        minutes = ts // 60
        secs = ts % 60
        merged_lines.append(f"[AD {minutes:02d}:{secs:02d}] {desc}")

    merged = "\n".join(merged_lines)

    return json.dumps(
        {
            "video_id": video_id,
            "format": "accessible_transcript",
            "transcript_length": len(transcript_text),
            "ad_entry_count": len(ad_entries),
            "content": merged,
        },
        indent=2,
    )


@mcp.tool()
async def has_audio_description(
    video_id_or_url: str,
) -> str:
    """Check if a video has audio description content available.

    Checks three sources:
    1. Cached AD files (.ad.vtt / .ad.txt)
    2. Source AD track from the video platform (via yt-dlp)
    3. Generated AD from scene analysis

    Args:
        video_id_or_url: Video ID or URL.
    """
    video_id = extract_video_id(video_id_or_url)
    cache_dir = get_cache_dir() / video_id
    cache = CacheManager(get_cache_dir())

    result: dict = {
        "video_id": video_id,
        "has_cached_ad": False,
        "has_source_ad_track": None,
        "ad_source": None,
        "ad_complete": False,
    }

    if not cache_dir.exists():
        result["error"] = "Video not cached. Run process_video first."
        return json.dumps(result, indent=2)

    # 1. Check cached AD files
    result["has_cached_ad"] = cache.has_ad(video_id)

    # 2. Check state for AD info
    state = cache.get_state(video_id)
    if state:
        result["ad_complete"] = state.ad_complete
        result["ad_source"] = state.ad_source
        result["has_source_ad_track"] = state.ad_track_available

    # 3. If source AD track status unknown, check via yt-dlp
    if state and state.url and state.ad_track_available is None:
        try:
            from claudetube.tools.yt_dlp import YtDlpTool

            yt = YtDlpTool()
            ad_format = await asyncio.to_thread(yt.check_audio_description, state.url)
            has_track = ad_format is not None
            result["has_source_ad_track"] = has_track

            state.ad_track_available = has_track
            cache.save_state(video_id, state)

            if ad_format:
                result["source_ad_format"] = {
                    "format_id": ad_format.get("format_id"),
                    "format_note": ad_format.get("format_note"),
                    "language": ad_format.get("language"),
                }
        except Exception as e:
            logger.warning(f"AD track check failed: {e}")
            result["has_source_ad_track"] = None
            result["source_check_error"] = str(e)

    return json.dumps(result, indent=2)


@mcp.tool()
async def get_knowledge_graph_stats_tool() -> str:
    """Get statistics about the cross-video knowledge graph.

    Returns counts of indexed videos, entities, and concepts,
    along with the graph storage location.
    """
    graph = get_knowledge_graph()
    stats = graph.get_stats()

    # Add list of indexed video IDs
    videos = graph.get_all_videos()
    stats["videos"] = [{"video_id": v.video_id, "title": v.title} for v in videos]

    return json.dumps(stats, indent=2)


@mcp.tool()
async def watch_video_tool(
    video_id: str,
    question: str,
    max_iterations: int = 15,
) -> str:
    """Actively watch and reason about a video to answer a question.

    Uses an intelligent watching strategy that:
    1. Checks cached Q&A for previously answered questions
    2. Identifies most relevant scenes via attention modeling
    3. Examines them progressively (quick first, deep if needed)
    4. Builds hypotheses and gathers evidence
    5. Verifies comprehension before answering

    This is the most thorough analysis mode - use when you need
    detailed, evidence-backed answers about video content.

    Args:
        video_id: Video ID of a previously processed video.
        question: Natural language question about the video.
        max_iterations: Maximum scene examinations (default: 15).
    """
    video_id = extract_video_id(video_id)

    result = await asyncio.to_thread(
        watch_video,
        video_id,
        question,
        max_iterations=max_iterations,
        output_base=get_cache_dir(),
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def list_providers_tool() -> str:
    """List available AI providers and their capabilities.

    Shows which providers are installed and configured, organized by
    capability (transcription, vision, video, reasoning, embedding).
    Also shows the currently configured preference for each capability.

    Use this to understand what providers are available before overriding
    the default with the provider parameter on other tools.
    """
    available = list_available()
    all_providers = list_all()

    # Build capability -> provider mapping
    capabilities: dict[str, dict] = {}
    for cap in Capability:
        cap_providers = []
        for name in all_providers:
            info = PROVIDER_INFO.get(name)
            if info and info.can(cap):
                cap_providers.append(
                    {
                        "name": name,
                        "available": name in available,
                    }
                )
        capabilities[cap.name.lower()] = {
            "providers": cap_providers,
        }

    # Add configured preferences
    try:
        factory = get_factory()
        config = factory.config
        capabilities["transcribe"]["preferred"] = config.transcription_provider
        capabilities["transcribe"]["fallbacks"] = config.transcription_fallbacks
        capabilities["vision"]["preferred"] = config.vision_provider
        capabilities["vision"]["fallbacks"] = config.vision_fallbacks
        capabilities["video"]["preferred"] = config.video_provider
        capabilities["reason"]["preferred"] = config.reasoning_provider
        capabilities["reason"]["fallbacks"] = config.reasoning_fallbacks
        capabilities["embed"]["preferred"] = config.embedding_provider
    except (RuntimeError, ImportError):
        pass

    return json.dumps(
        {
            "available_providers": available,
            "all_providers": all_providers,
            "capabilities": capabilities,
        },
        indent=2,
    )


@mcp.tool()
async def detect_narrative_structure_tool(
    video_id: str,
    force: bool = False,
) -> str:
    """Detect narrative structure of a video.

    Identifies sections (introduction, main content, conclusion, transitions)
    using scene clustering and classifies the video type (coding_tutorial,
    lecture, demo, etc.). Results are cached in structure/narrative.json.

    Args:
        video_id: Video ID of a previously processed video.
        force: Re-detect even if cached (default: False).
    """
    from claudetube.operations.narrative_structure import detect_narrative_structure

    video_id = extract_video_id(video_id)

    result = await asyncio.to_thread(
        detect_narrative_structure,
        video_id,
        force=force,
        output_base=get_cache_dir(),
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def get_narrative_structure_tool(
    video_id: str,
) -> str:
    """Get cached narrative structure for a video.

    Returns the previously detected sections and video type classification.
    Run detect_narrative_structure_tool first if no structure is cached.

    Args:
        video_id: Video ID of a previously processed video.
    """
    from claudetube.operations.narrative_structure import get_narrative_structure

    video_id = extract_video_id(video_id)

    result = await asyncio.to_thread(
        get_narrative_structure,
        video_id,
        output_base=get_cache_dir(),
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def detect_changes_tool(
    video_id: str,
    force: bool = False,
) -> str:
    """Detect changes between consecutive scenes in a video.

    Analyzes visual changes, topic shifts, and content type transitions
    between adjacent scenes. Results are cached in structure/changes.json.

    Args:
        video_id: Video ID of a previously processed video.
        force: Re-detect even if cached (default: False).
    """
    from claudetube.operations.change_detection import detect_scene_changes

    video_id = extract_video_id(video_id)

    result = await asyncio.to_thread(
        detect_scene_changes,
        video_id,
        force=force,
        output_base=get_cache_dir(),
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def get_changes_tool(
    video_id: str,
) -> str:
    """Get cached scene change data for a video.

    Returns previously detected changes between consecutive scenes.
    Run detect_changes_tool first if no changes are cached.

    Args:
        video_id: Video ID of a previously processed video.
    """
    from claudetube.operations.change_detection import get_scene_changes

    video_id = extract_video_id(video_id)

    result = await asyncio.to_thread(
        get_scene_changes,
        video_id,
        output_base=get_cache_dir(),
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def get_major_transitions_tool(
    video_id: str,
) -> str:
    """Get only the major transitions between scenes in a video.

    Filters scene changes to return only significant transitions (topic shifts,
    content type changes). Useful for understanding video structure at a glance.

    Args:
        video_id: Video ID of a previously processed video.
    """
    from claudetube.operations.change_detection import get_major_transitions

    video_id = extract_video_id(video_id)

    result = await asyncio.to_thread(
        get_major_transitions,
        video_id,
        output_base=get_cache_dir(),
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def track_code_evolution_tool(
    video_id: str,
    force: bool = False,
) -> str:
    """Track how code evolves across scenes in a video.

    Analyzes code snapshots from entity extraction data to track how
    code changes over time — additions, modifications, refactoring.
    Results are cached in entities/code_evolution.json.

    Best used on coding tutorials, live coding sessions, and code review videos.

    Args:
        video_id: Video ID of a previously processed video.
        force: Re-track even if cached (default: False).
    """
    from claudetube.operations.code_evolution import track_code_evolution

    video_id = extract_video_id(video_id)

    result = await asyncio.to_thread(
        track_code_evolution,
        video_id,
        force=force,
        output_base=get_cache_dir(),
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def get_code_evolution_tool(
    video_id: str,
) -> str:
    """Get cached code evolution data for a video.

    Returns how code changed across scenes. Run track_code_evolution_tool
    first if no evolution data is cached.

    Args:
        video_id: Video ID of a previously processed video.
    """
    from claudetube.operations.code_evolution import get_code_evolution

    video_id = extract_video_id(video_id)

    result = await asyncio.to_thread(
        get_code_evolution,
        video_id,
        output_base=get_cache_dir(),
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def query_code_evolution_tool(
    video_id: str,
    query: str,
) -> str:
    """Query code evolution for a specific file or code unit.

    Searches the code evolution data for a specific filename, function,
    or code pattern. Returns the evolution history for matching code units.

    Args:
        video_id: Video ID of a previously processed video.
        query: Filename, function name, or code pattern to search for.
    """
    from claudetube.operations.code_evolution import query_code_evolution

    video_id = extract_video_id(video_id)

    result = await asyncio.to_thread(
        query_code_evolution,
        video_id,
        query,
        output_base=get_cache_dir(),
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def build_knowledge_graph_tool(
    playlist_id: str,
) -> str:
    """Build a cross-video knowledge graph for a playlist.

    Analyzes relationships between videos in a playlist: shared topics,
    entities, and prerequisite chains. The playlist must have been fetched
    with get_playlist first.

    Args:
        playlist_id: Playlist ID (from get_playlist results).
    """
    from claudetube.operations.knowledge_graph import (
        build_knowledge_graph,
        save_knowledge_graph,
    )
    from claudetube.operations.playlist import load_playlist_metadata

    playlist_data = load_playlist_metadata(playlist_id)
    if not playlist_data:
        return json.dumps(
            {
                "error": f"Playlist '{playlist_id}' not found in cache. Run get_playlist first."
            }
        )

    graph = await asyncio.to_thread(build_knowledge_graph, playlist_data)
    await asyncio.to_thread(save_knowledge_graph, graph)

    return json.dumps(graph, indent=2, default=str)


@mcp.tool()
async def get_playlist_video_context_tool(
    video_id: str,
    playlist_id: str,
) -> str:
    """Get contextual information for a video within a playlist.

    Returns the video's position in the playlist, related topics, prerequisite
    videos, and shared entities with other videos. Requires a knowledge graph
    to have been built for the playlist.

    Args:
        video_id: Video ID to get context for.
        playlist_id: Playlist ID containing the video.
    """
    from claudetube.operations.knowledge_graph import get_video_context

    result = await asyncio.to_thread(
        get_video_context,
        video_id,
        playlist_id,
    )

    if result is None:
        return json.dumps(
            {
                "error": f"No context found. Build the knowledge graph for playlist '{playlist_id}' first.",
                "video_id": video_id,
                "playlist_id": playlist_id,
            }
        )

    return json.dumps(result, indent=2, default=str)


def main():
    """Entry point for the claudetube-mcp command."""
    mcp.run()


if __name__ == "__main__":
    main()
