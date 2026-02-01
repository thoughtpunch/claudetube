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
from claudetube.cache.knowledge_graph import get_knowledge_graph, index_video_to_graph
from claudetube.cache.scenes import has_scenes, load_scenes_data
from claudetube.config import get_cache_dir
from claudetube.models.local_file import is_local_file
from claudetube.operations.extract_frames import (
    extract_frames as get_frames_at,
)
from claudetube.operations.extract_frames import (
    extract_hq_frames as get_hq_frames_at,
)
from claudetube.operations.processor import process_local_video, process_video
from claudetube.operations.transcribe import transcribe_video as _transcribe_video
from claudetube.parsing.utils import extract_video_id

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

    Args:
        video_id_or_url: YouTube video ID or URL.
        start_time: Start time in seconds.
        duration: Duration to capture in seconds.
        interval: Seconds between frames.
        quality: Quality tier (lowest/low/medium/high/highest).
    """
    frames = await asyncio.to_thread(
        get_frames_at,
        video_id_or_url,
        start_time=start_time,
        duration=duration,
        interval=interval,
        output_base=get_cache_dir(),
        quality=quality,
    )

    return json.dumps(
        {
            "frame_count": len(frames),
            "frame_paths": [str(f) for f in frames],
        },
        indent=2,
    )


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

    Args:
        video_id_or_url: YouTube video ID or URL.
        start_time: Start time in seconds.
        duration: Duration to capture in seconds.
        interval: Seconds between frames.
        width: Frame width in pixels.
    """
    frames = await asyncio.to_thread(
        get_hq_frames_at,
        video_id_or_url,
        start_time=start_time,
        duration=duration,
        interval=interval,
        output_base=get_cache_dir(),
        width=width,
    )

    return json.dumps(
        {
            "frame_count": len(frames),
            "frame_paths": [str(f) for f in frames],
        },
        indent=2,
    )


@mcp.tool()
async def transcribe_video(
    video_id_or_url: str,
    whisper_model: str = "small",
    force: bool = False,
) -> str:
    """Transcribe a video's audio using Whisper.

    Returns cached transcript immediately if available, otherwise runs
    Whisper transcription. Use force=True to re-transcribe with a
    different model.

    Args:
        video_id_or_url: Video ID or URL.
        whisper_model: Whisper model size (tiny/base/small/medium/large).
        force: Re-transcribe even if a cached transcript exists.
    """
    result = await asyncio.to_thread(
        _transcribe_video,
        video_id_or_url,
        whisper_model=whisper_model,
        force=force,
        output_base=get_cache_dir(),
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


def _get_scenes_sync(video_id: str, force: bool = False) -> dict:
    """Get scene structure for a cached video (sync version).

    Returns cached scenes if available, otherwise runs smart segmentation.
    Implements "Cheap First, Expensive Last" - returns cached data instantly.

    Args:
        video_id: Video ID
        force: Re-run segmentation even if cached

    Returns:
        Dict with scene data or error
    """
    cache_dir = get_cache_dir() / video_id

    if not cache_dir.exists():
        return {"error": "Video not cached. Run process_video first.", "video_id": video_id}

    # Check state.json exists
    state_file = cache_dir / "state.json"
    if not state_file.exists():
        return {"error": "Video not processed. Run process_video first.", "video_id": video_id}

    # Fast path: return cached scenes
    if not force and has_scenes(cache_dir):
        scenes_data = load_scenes_data(cache_dir)
        if scenes_data:
            result = scenes_data.to_dict()
            # Enrich with visual descriptions if available
            for scene in result.get("scenes", []):
                scene_id = scene.get("scene_id", 0)
                visual_file = cache_dir / "scenes" / f"scene_{scene_id:03d}" / "visual.json"
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

    return scenes_data.to_dict()


@mcp.tool()
async def get_scenes(
    video_id: str,
    force: bool = False,
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
    """
    video_id = extract_video_id(video_id)

    result = await asyncio.to_thread(_get_scenes_sync, video_id, force)

    return json.dumps(result, indent=2)


@mcp.tool()
async def generate_visual_transcripts(
    video_id: str,
    scene_id: int | None = None,
    force: bool = False,
) -> str:
    """Generate visual descriptions for video scenes.

    Uses vision AI (Claude Haiku by default) to describe what's happening
    visually in each scene. Results are cached in scene_{NNN}/visual.json.

    Follows "Cheap First, Expensive Last" principle:
    - Returns cached descriptions instantly if available
    - Skips scenes with good transcript coverage (talking heads)
    - Only calls vision API when visual context adds value

    Requires ANTHROPIC_API_KEY environment variable to be set.

    Args:
        video_id: Video ID of a previously processed video.
        scene_id: Optional specific scene ID (None = all scenes).
        force: Re-generate even if cached (default: False).
    """
    from claudetube.operations.visual_transcript import generate_visual_transcript

    video_id = extract_video_id(video_id)
    result = await asyncio.to_thread(
        generate_visual_transcript,
        video_id,
        scene_id=scene_id,
        force=force,
        output_base=get_cache_dir(),
    )

    return json.dumps(result, indent=2)


@mcp.tool()
async def track_people_tool(
    video_id: str,
    force: bool = False,
    use_face_recognition: bool = False,
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
    """
    from claudetube.operations.person_tracking import track_people

    video_id = extract_video_id(video_id)
    result = await asyncio.to_thread(
        track_people,
        video_id,
        force=force,
        use_face_recognition=use_face_recognition,
        output_base=get_cache_dir(),
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

    lines = [f"Found {len(moments)} relevant moment{'s' if len(moments) != 1 else ''}:\n"]

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
) -> str:
    """Find moments in a video matching a natural language query.

    Uses tiered search strategy (Cheap First, Expensive Last):
    1. TEXT - Fast transcript text matching (~100ms)
    2. SEMANTIC - Vector embedding similarity (if text search fails)

    The video must have been processed with process_video and have scene data.

    Example: find_moments_tool('abc123', 'when they fix the auth bug')

    Args:
        video_id: Video ID of a previously processed video.
        query: Natural language query (e.g., "when they discuss authentication").
        top_k: Maximum number of results to return (default: 5).
    """
    video_id = extract_video_id(video_id)

    try:
        moments = await asyncio.to_thread(
            find_moments,
            video_id,
            query,
            top_k=top_k,
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
        return json.dumps({"error": "No scenes found. Run get_scenes first.", "video_id": video_id})

    focus_ids = [
        s.scene_id for s in scenes_data.scenes
        if s.start_time >= start_time and s.end_time <= end_time
    ]

    # Also include scenes that overlap with the range
    if not focus_ids:
        focus_ids = [
            s.scene_id for s in scenes_data.scenes
            if not (s.end_time < start_time or s.start_time > end_time)
        ]

    if not focus_ids:
        return json.dumps({
            "error": f"No scenes found between {start_time}s and {end_time}s",
            "video_id": video_id,
        })

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
        return json.dumps({
            "query": query,
            "error": "No videos indexed yet. Use index_video_to_graph_tool first.",
            "matches": [],
        })

    matches = graph.find_related_videos(query)

    return json.dumps({
        "query": query,
        "match_count": len(matches),
        "matches": [m.to_dict() for m in matches],
        "graph_stats": graph.get_stats(),
    }, indent=2)


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
        return json.dumps({
            "error": f"Video '{video_id}' not in knowledge graph. Index it first.",
            "video_id": video_id,
        })

    connected_ids = graph.get_video_connections(video_id)

    # Get full info for connected videos
    connections = []
    for vid in connected_ids:
        v = graph.get_video(vid)
        if v:
            connections.append(v.to_dict())

    return json.dumps({
        "video_id": video_id,
        "video_title": video.title,
        "connection_count": len(connections),
        "connections": connections,
    }, indent=2)


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
    stats["videos"] = [
        {"video_id": v.video_id, "title": v.title}
        for v in videos
    ]

    return json.dumps(stats, indent=2)


def main():
    """Entry point for the claudetube-mcp command."""
    mcp.run()


if __name__ == "__main__":
    main()
