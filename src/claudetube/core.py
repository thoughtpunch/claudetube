"""
Core video processing for claudetube.

Downloads YouTube videos, transcribes with faster-whisper, and extracts
frames on-demand for visual analysis.
"""

import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

QUALITY_TIERS = {
    "lowest": {
        "sort": "+res,+size,+br,+fps",
        "width": 480,
        "jpeg_q": 5,
    },
    "low": {
        "sort": "res:360,+size,+br",
        "width": 640,
        "jpeg_q": 4,
    },
    "medium": {
        "sort": "res:480",
        "width": 854,
        "jpeg_q": 3,
    },
    "high": {
        "sort": "res:720",
        "width": 1280,
        "jpeg_q": 2,
    },
    "highest": {
        "sort": "res:1080",
        "width": 1280,
        "jpeg_q": 2,
    },
}
QUALITY_LADDER = ["lowest", "low", "medium", "high", "highest"]


def next_quality(current: str) -> str | None:
    """Return the next quality tier, or None if already at highest."""
    idx = QUALITY_LADDER.index(current)
    return QUALITY_LADDER[idx + 1] if idx + 1 < len(QUALITY_LADDER) else None


def _log(msg: str, start_time: float | None = None):
    """Print timestamped log message."""
    elapsed = f"[{time.time() - start_time:.1f}s]" if start_time else "[START]"
    print(f"{elapsed} {msg}", flush=True)


def _find_tool(name: str) -> str:
    """Find executable, checking venv first."""
    venv = Path(sys.prefix) / "bin" / name
    if venv.exists():
        return str(venv)
    return shutil.which(name) or name


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url.replace("/", "_").replace(":", "_")[:20]


def extract_playlist_id(url: str) -> str | None:
    """Extract YouTube playlist ID from URL (list= parameter)."""
    match = re.search(r"[?&]list=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def extract_url_context(url: str) -> dict:
    """Extract video ID, playlist ID, and other context from URL."""
    return {
        "video_id": extract_video_id(url),
        "playlist_id": extract_playlist_id(url),
        "original_url": url,
        "clean_url": re.sub(r"[&?]list=[^&]+", "", url),  # URL without playlist
    }


@dataclass
class VideoResult:
    """Result of video processing."""

    success: bool
    video_id: str
    output_dir: Path
    transcript_srt: Path | None = None
    transcript_txt: Path | None = None
    frames: list[Path] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    error: str | None = None


def process_video(
    url: str,
    output_base: Path | None = None,
    whisper_model: str = "tiny",
    extract_frames: bool = False,
    frame_interval: int = 30,
) -> VideoResult:
    """
    Process a YouTube video - transcript first, frames optional.

    Args:
        url: YouTube URL
        output_base: Base directory for cache
        whisper_model: tiny|base|small|medium|large
        extract_frames: Whether to extract frames (default: False for speed)
        frame_interval: Seconds between frames if extracting

    Returns:
        VideoResult with transcript and optional frames
    """
    t0 = time.time()
    _log("Starting video processing", t0)

    # Default cache location: ~/.claude/video_cache/
    if output_base is None:
        output_base = Path.home() / ".claude" / "video_cache"

    # Extract context from URL (video_id, playlist_id, etc.)
    url_context = extract_url_context(url)
    video_id = url_context["video_id"]
    playlist_id = url_context["playlist_id"]

    output_dir = Path(output_base) / video_id
    state_file = output_dir / "state.json"

    # Check cache
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state.get("transcript_complete"):
            _log(f"Cache hit for {video_id}", t0)
            return VideoResult(
                success=True,
                video_id=video_id,
                output_dir=output_dir,
                transcript_srt=(
                    output_dir / "audio.srt"
                    if (output_dir / "audio.srt").exists()
                    else None
                ),
                transcript_txt=(
                    output_dir / "audio.txt"
                    if (output_dir / "audio.txt").exists()
                    else None
                ),
                frames=(
                    sorted(output_dir.glob("frames/*.jpg")) if extract_frames else []
                ),
                metadata=state,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "audio.mp3"

    # STEP 1: Fetch metadata (fast, ~2s)
    _log("Fetching video metadata...", t0)
    meta = _get_metadata(url)
    if not meta:
        return VideoResult(
            False, video_id, output_dir, error="Failed to fetch metadata"
        )

    state = {
        "video_id": video_id,
        "url": url,
        "playlist_id": playlist_id,  # For future playlist-aware features / YouTube API
        "title": meta.get("title"),
        "duration": meta.get("duration"),
        "duration_string": meta.get("duration_string"),
        "uploader": meta.get("uploader"),
        "channel": meta.get("channel"),
        "upload_date": meta.get("upload_date"),
        "description": meta.get("description", "")[:1500],
        "categories": meta.get("categories"),
        "tags": meta.get("tags", [])[:15],
        "language": meta.get("language"),
        "view_count": meta.get("view_count"),
        "like_count": meta.get("like_count"),
        "thumbnail": meta.get("thumbnail"),
        "transcript_complete": False,
    }
    state_file.write_text(json.dumps(state, indent=2))
    _log(f"Metadata: '{state['title']}' ({state['duration_string']})", t0)

    srt_path = output_dir / "audio.srt"
    txt_path = output_dir / "audio.txt"

    # STEP 2: Try subtitles first (fast, ~2-3s)
    sub_result = _fetch_subtitles(url, output_dir, t0)
    if sub_result:
        srt_path.write_text(sub_result["srt"])
        txt_path.write_text(sub_result["txt"])
        state["transcript_complete"] = True
        state["transcript_source"] = sub_result["source"]
        state_file.write_text(json.dumps(state, indent=2))
        _log(
            f"DONE via subtitles ({sub_result['source']}) in {time.time() - t0:.1f}s",
            t0,
        )
        return VideoResult(
            success=True,
            video_id=video_id,
            output_dir=output_dir,
            transcript_srt=srt_path,
            transcript_txt=txt_path,
            metadata=state,
        )

    # STEP 3: No subtitles available — download audio for whisper
    _log("No subtitles available, falling back to whisper...", t0)
    if not audio_path.exists():
        _log("Downloading audio...", t0)
        cmd = [
            _find_tool("yt-dlp"),
            "-f",
            "ba",
            "-x",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "64K",
            "--no-playlist",
            "--no-warnings",
            "-o",
            str(audio_path),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not audio_path.exists():
            _log(f"Audio download failed: {result.stderr[:100]}", t0)
            return VideoResult(
                False,
                video_id,
                output_dir,
                error="Audio download failed",
                metadata=state,
            )
        size_mb = audio_path.stat().st_size / 1024 / 1024
        _log(f"Audio downloaded: {size_mb:.1f}MB", t0)

    # STEP 4: Transcribe with whisper (~30-60s for tiny model)
    if not srt_path.exists():
        _log(f"Transcribing with faster-whisper ({whisper_model})...", t0)
        transcript_text = _transcribe_faster_whisper(audio_path, whisper_model, t0)

        if transcript_text:
            txt_path.write_text(transcript_text["txt"])
            srt_path.write_text(transcript_text["srt"])
            _log("Transcription complete", t0)
        else:
            _log("Transcription failed, trying fallback...", t0)
            cmd = [
                "whisper",
                str(audio_path),
                "--model",
                whisper_model,
                "--language",
                "en",
                "--output_format",
                "all",
                "--output_dir",
                str(output_dir),
            ]
            subprocess.run(cmd, capture_output=True)
            _log("Fallback transcription complete", t0)

    # STEP 5: Optional frames (downloads video only if needed)
    frames = []
    if extract_frames:
        video_path = output_dir / "video.mp4"
        if not video_path.exists():
            _log("Downloading video for frames...", t0)
            cmd = [
                _find_tool("yt-dlp"),
                "-f",
                "160+139/160+140/worst[height<=360]/worst",
                "-S",
                "+size,+br",
                "--no-playlist",
                "--no-warnings",
                "--merge-output-format",
                "mp4",
                "-o",
                str(video_path),
                url,
            ]
            subprocess.run(cmd, capture_output=True, text=True)
        if video_path.exists():
            frames = _extract_frames(
                video_path, output_dir / "frames", frame_interval, t0
            )
            video_path.unlink()
            _log("Cleaned up video file", t0)

    # Update state
    state["transcript_complete"] = True
    state["transcript_source"] = "whisper"
    state["whisper_model"] = whisper_model
    if frames:
        state["frames_count"] = len(frames)
        state["frame_interval"] = frame_interval
    state_file.write_text(json.dumps(state, indent=2))

    _log(f"DONE in {time.time() - t0:.1f}s", t0)

    return VideoResult(
        success=True,
        video_id=video_id,
        output_dir=output_dir,
        transcript_srt=srt_path if srt_path.exists() else None,
        transcript_txt=txt_path if txt_path.exists() else None,
        frames=frames,
        metadata=state,
    )


def get_frames_at(
    video_id_or_url: str,
    start_time: float,
    duration: float = 5.0,
    interval: float = 1.0,
    output_base: Path = Path("./video_cache"),
    width: int | None = None,
    quality: str = "lowest",
) -> list[Path]:
    """
    Extract frames for a specific time range (drill-in feature).

    Use this when Claude needs visual context for a specific part of the video.

    Args:
        video_id_or_url: Video ID or URL
        start_time: Start time in seconds
        duration: Duration to capture (default: 5s)
        interval: Seconds between frames (default: 1s)
        output_base: Cache directory
        width: Frame width (overrides tier default if set)
        quality: Quality tier (lowest/low/medium/high/highest)

    Returns:
        List of frame paths
    """
    t0 = time.time()

    # Validate quality
    if quality not in QUALITY_TIERS:
        raise ValueError(
            f"Invalid quality '{quality}'. Must be one of: {', '.join(QUALITY_LADDER)}"
        )
    tier = QUALITY_TIERS[quality]
    effective_width = width or tier["width"]

    # Get video ID
    video_id = extract_video_id(video_id_or_url)
    output_dir = Path(output_base) / video_id
    drill_dir = output_dir / f"drill_{quality}"
    drill_dir.mkdir(parents=True, exist_ok=True)

    # Download only the needed section (+ 2s buffer for keyframes)
    section_start = max(0, start_time - 2)
    section_end = start_time + duration + 2
    section_spec = f"*{section_start}-{section_end}"

    seg_name = f"segment_{quality}_{int(start_time)}_{int(start_time + duration)}.mp4"
    video_path = output_dir / seg_name
    state_file = output_dir / "state.json"

    if not video_path.exists() and state_file.exists():
        state = json.loads(state_file.read_text())
        url = state.get("url")
        if url:
            _log(
                f"Downloading {quality} segment"
                f" ({section_start}s-{section_end}s)...",
                t0,
            )
            cmd = [
                _find_tool("yt-dlp"),
                "-S",
                tier["sort"],
                "--download-sections",
                section_spec,
                "--force-keyframes-at-cuts",
                "--no-playlist",
                "--no-warnings",
                "--merge-output-format",
                "mp4",
                "-o",
                str(video_path),
                url,
            ]
            subprocess.run(cmd, capture_output=True)

    if not video_path.exists():
        _log("No video available for drill-in", t0)
        return []

    # Extract frames for the time range
    # Since we downloaded a section starting at section_start,
    # ffmpeg timestamps are relative to the section start
    frames = []
    current = start_time
    end_time = start_time + duration
    seek_offset = section_start  # adjust for section-relative timestamps

    _log(f"Extracting {quality} frames from {start_time}s to {end_time}s...", t0)

    while current < end_time:
        ts_str = f"{int(current // 60):02d}-{int(current % 60):02d}"
        output = drill_dir / f"drill_{ts_str}.jpg"

        cmd = [
            "ffmpeg",
            "-ss",
            str(current - seek_offset),
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-vf",
            f"scale={effective_width}:-1",
            "-q:v",
            str(tier["jpeg_q"]),
            "-y",
            str(output),
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and output.exists():
            frames.append(output)
            _log(f"  Frame at {ts_str}", t0)

        current += interval

    # Always clean up segment files (they're cheap to re-download)
    if video_path.exists():
        video_path.unlink()
        _log(f"Cleaned up {quality} segment file", t0)

    # Track extraction in state.json
    if state_file.exists():
        state = json.loads(state_file.read_text())
        extractions = state.get("quality_extractions", {})
        extractions[quality] = {
            "start_time": start_time,
            "duration": duration,
            "frames": len(frames),
            "width": effective_width,
        }
        state["quality_extractions"] = extractions
        state_file.write_text(json.dumps(state, indent=2))

    _log(f"Drill-in complete ({quality}): {len(frames)} frames", t0)
    return frames


def get_hq_frames_at(
    video_id_or_url: str,
    start_time: float,
    duration: float = 5.0,
    interval: float = 1.0,
    output_base: Path | None = None,
    width: int = 1280,
) -> list[Path]:
    """
    Extract HIGH QUALITY frames for a specific time range.

    Use this when the low-quality drill-in frames aren't clear enough
    (e.g., reading text, code, small UI elements).

    Downloads best available quality video (larger file, slower).

    Args:
        video_id_or_url: Video ID or URL
        start_time: Start time in seconds
        duration: Duration to capture (default: 5s)
        interval: Seconds between frames (default: 1s)
        output_base: Cache directory (default: ~/.claude/video_cache)
        width: Frame width (default: 1280 for HD)

    Returns:
        List of frame paths
    """
    t0 = time.time()

    if output_base is None:
        output_base = Path.home() / ".claude" / "video_cache"

    video_id = extract_video_id(video_id_or_url)
    output_dir = Path(output_base) / video_id
    hq_dir = output_dir / "hq"
    hq_dir.mkdir(parents=True, exist_ok=True)

    state_file = output_dir / "state.json"

    # Get URL from state
    if not state_file.exists():
        _log("No state.json found - run process_video first", t0)
        return []

    state = json.loads(state_file.read_text())
    url = state.get("url")
    if not url:
        _log("No URL in state.json", t0)
        return []

    # Download only the needed section (+ 2s buffer for keyframes)
    section_start = max(0, start_time - 2)
    section_end = start_time + duration + 2
    section_spec = f"*{section_start}-{section_end}"

    seg_name = f"segment_hq_{int(start_time)}_{int(start_time + duration)}.mp4"
    hq_video_path = output_dir / seg_name

    if not hq_video_path.exists():
        _log(
            f"Downloading HQ segment ({section_start}s-{section_end}s)...",
            t0,
        )
        cmd = [
            _find_tool("yt-dlp"),
            "-S",
            "res:1080",
            "--download-sections",
            section_spec,
            "--force-keyframes-at-cuts",
            "--no-playlist",
            "--no-warnings",
            "--merge-output-format",
            "mp4",
            "-o",
            str(hq_video_path),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not hq_video_path.exists():
            _log(f"HQ segment download failed: {result.stderr[:200]}", t0)
            return []
        size_mb = hq_video_path.stat().st_size / 1024 / 1024
        _log(f"Downloaded HQ segment: {size_mb:.1f}MB", t0)

    # Extract HQ frames (timestamps relative to section start)
    frames = []
    current = start_time
    end_time = start_time + duration
    seek_offset = section_start

    _log(f"Extracting HQ frames from {start_time}s to {end_time}s...", t0)

    while current < end_time:
        ts_str = f"{int(current // 60):02d}-{int(current % 60):02d}"
        output = hq_dir / f"hq_{ts_str}.jpg"

        cmd = [
            "ffmpeg",
            "-ss",
            str(current - seek_offset),
            "-i",
            str(hq_video_path),
            "-vframes",
            "1",
            "-vf",
            f"scale={width}:-1",
            "-q:v",
            "2",  # Higher quality JPEG
            "-y",
            str(output),
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and output.exists():
            frames.append(output)
            _log(f"  HQ frame at {ts_str}", t0)

        current += interval

    # Clean up segment file
    if hq_video_path.exists():
        hq_video_path.unlink()

    _log(f"HQ drill-in complete: {len(frames)} frames", t0)
    return frames


def _fetch_subtitles(url: str, output_dir: Path, t0: float) -> dict | None:
    """Try to fetch subtitles from the video source (faster than whisper).

    Attempts uploaded subs first, then auto-generated subs.
    Returns {"srt": ..., "txt": ..., "source": ...} or None.
    """
    _log("Checking for subtitles...", t0)
    cmd = [
        _find_tool("yt-dlp"),
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        "en.*,en",
        "--sub-format",
        "srt/vtt/best",
        "--convert-subs",
        "srt",
        "--skip-download",
        "--no-playlist",
        "--no-warnings",
        "-o",
        str(output_dir / "%(id)s.%(ext)s"),
        url,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception as e:
        _log(f"Subtitle fetch failed: {e}", t0)
        return None

    # Look for subtitle files (yt-dlp names them {id}.{lang}.srt)
    sub_files = sorted(output_dir.glob("*.srt"))
    # Exclude our own audio.srt if it exists from a previous run
    sub_files = [f for f in sub_files if f.name != "audio.srt"]
    if not sub_files:
        _log("No subtitles found", t0)
        return None

    sub_file = sub_files[0]
    _log(f"Found subtitle: {sub_file.name}", t0)
    raw = sub_file.read_text(errors="replace")

    # Determine source type from filename
    source = "auto-generated" if ".auto." in sub_file.name else "uploaded"

    # Clean up: strip HTML tags from auto-generated subs
    raw = re.sub(r"<[^>]+>", "", raw)

    # Parse SRT to extract plain text
    txt_lines = []
    for line in raw.splitlines():
        line = line.strip()
        # Skip sequence numbers, timestamps, empty lines
        if not line or re.match(r"^\d+$", line) or "-->" in line:
            continue
        if line not in txt_lines[-1:]:  # deduplicate consecutive lines
            txt_lines.append(line)

    if not txt_lines:
        _log("Subtitle file was empty after parsing", t0)
        # Clean up downloaded sub files
        for f in output_dir.glob("*.srt"):
            if f.name != "audio.srt":
                f.unlink()
        return None

    # Clean up downloaded sub files (we'll write our own audio.srt/txt)
    for f in sub_files:
        f.unlink()

    _log(f"Subtitles ready ({source}, {len(txt_lines)} lines)", t0)
    return {
        "srt": raw,
        "txt": "\n".join(txt_lines),
        "source": source,
    }


def _transcribe_faster_whisper(
    audio_path: Path, model_size: str, t0: float
) -> dict | None:
    """Transcribe using faster-whisper (4x faster than OpenAI whisper)."""
    try:
        import os

        from faster_whisper import WhisperModel

        # Use all available CPU threads
        cpu_threads = os.cpu_count() or 4
        _log(f"  Loading model ({model_size}, {cpu_threads} threads)...", t0)

        # int8 compute for speed on CPU
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
        )

        _log("  Transcribing (batched)...", t0)
        # Batched transcription is faster on multi-core CPUs
        from faster_whisper import BatchedInferencePipeline

        batched_model = BatchedInferencePipeline(model=model)
        segments, info = batched_model.transcribe(
            str(audio_path),
            language="en",
            batch_size=16,  # Process 16 segments in parallel
        )

        # Build SRT and TXT
        srt_lines = []
        txt_lines = []
        for i, seg in enumerate(segments, 1):
            start = _format_srt_time(seg.start)
            end = _format_srt_time(seg.end)
            text = seg.text.strip()

            srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
            txt_lines.append(text)

            # Progress
            print(
                f"  [{start}] {text[:60]}{'...' if len(text) > 60 else ''}", flush=True
            )

        return {
            "srt": "\n".join(srt_lines),
            "txt": "\n".join(txt_lines),
        }

    except ImportError:
        _log("  faster-whisper not available", t0)
        return None
    except Exception as e:
        _log(f"  faster-whisper error: {e}", t0)
        return None


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _get_metadata(url: str) -> dict:
    """Fetch video metadata without downloading."""
    cmd = [_find_tool("yt-dlp"), "--dump-json", "--no-download", "--no-warnings", url]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=30
        )
        return json.loads(result.stdout)
    except Exception:
        return {}


def _extract_frames(
    video_path: Path,
    frames_dir: Path,
    interval: int,
    t0: float,
) -> list[Path]:
    """Extract frames at intervals."""
    frames_dir.mkdir(exist_ok=True)

    if list(frames_dir.glob("*.jpg")):
        return sorted(frames_dir.glob("*.jpg"))

    _log(f"Extracting frames every {interval}s...", t0)

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval},scale=480:-1",
        "-q:v",
        "8",
        "-vsync",
        "vfr",
        str(frames_dir / "frame_%03d.jpg"),
    ]
    subprocess.run(cmd, capture_output=True)

    # Rename with timestamps
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    for i, f in enumerate(frames):
        ts = i * interval
        ts_str = f"{int(ts // 60):02d}-{int(ts % 60):02d}"
        new_path = frames_dir / f"frame_{ts_str}.jpg"
        f.rename(new_path)

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    _log(f"Extracted {len(frames)} frames", t0)
    return frames
