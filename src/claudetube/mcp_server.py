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

from claudetube.operations.extract_frames import (
    extract_frames as get_frames_at,
)
from claudetube.operations.extract_frames import (
    extract_hq_frames as get_hq_frames_at,
)
from claudetube.operations.processor import process_video
from claudetube.operations.transcribe import transcribe_video as _transcribe_video
from claudetube.parsing.utils import extract_video_id

# All logging goes to stderr so stdout stays clean for JSON-RPC
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".claude" / "video_cache"
TRANSCRIPT_INLINE_CAP = 50_000

mcp = FastMCP("claudetube")


@mcp.tool()
async def process_video_tool(
    url: str,
    whisper_model: str = "tiny",
) -> str:
    """Process a YouTube video: download, transcribe, and cache.

    Returns JSON with metadata, transcript text (capped at 50k chars),
    and file paths for the full transcript and thumbnail.

    Args:
        url: YouTube video URL or video ID.
        whisper_model: Whisper model size (tiny/base/small/medium/large).
    """
    result = await asyncio.to_thread(
        process_video,
        url,
        output_base=CACHE_DIR,
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
        output_base=CACHE_DIR,
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
        output_base=CACHE_DIR,
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
        output_base=CACHE_DIR,
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
    videos = []
    if CACHE_DIR.exists():
        for state_file in sorted(CACHE_DIR.glob("*/state.json")):
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
    video_dir = CACHE_DIR / video_id

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


def main():
    """Entry point for the claudetube-mcp command."""
    mcp.run()


if __name__ == "__main__":
    main()
