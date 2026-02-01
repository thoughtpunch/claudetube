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
        load_playlist_metadata,
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


def main():
    """Entry point for the claudetube-mcp command."""
    mcp.run()


if __name__ == "__main__":
    main()
