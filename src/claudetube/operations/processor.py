"""
Main video processor - orchestrates download, transcription, and caching.
"""

from __future__ import annotations

import time
from pathlib import Path

from claudetube.cache.manager import CacheManager
from claudetube.config.loader import get_cache_dir
from claudetube.models.local_file import LocalFile, LocalFileError
from claudetube.models.state import VideoState
from claudetube.models.video_result import VideoResult
from claudetube.operations.download import (
    download_audio,
    download_thumbnail,
    extract_audio_local,
    fetch_metadata,
    fetch_subtitles,
)
from claudetube.operations.transcribe import transcribe_audio
from claudetube.parsing.utils import extract_url_context
from claudetube.tools.ffmpeg import FFmpegTool
from claudetube.tools.ffprobe import FFprobeTool
from claudetube.utils.formatting import format_duration
from claudetube.utils.logging import log_timed


def process_video(
    url: str,
    output_base: Path | None = None,
    whisper_model: str = "tiny",
    extract_frames: bool = False,
    frame_interval: int = 30,
) -> VideoResult:
    """Process a video - transcript first, frames optional.

    Args:
        url: Video URL
        output_base: Base directory for cache
        whisper_model: tiny|base|small|medium|large
        extract_frames: Whether to extract frames (default: False for speed)
        frame_interval: Seconds between frames if extracting

    Returns:
        VideoResult with transcript and optional frames
    """
    t0 = time.time()
    log_timed("Starting video processing", t0)

    cache = CacheManager(output_base or get_cache_dir())

    # Extract context from URL (video_id, playlist_id, etc.)
    url_context = extract_url_context(url)
    video_id = url_context["video_id"]
    playlist_id = url_context["playlist_id"]

    cache_dir = cache.get_cache_dir(video_id)

    # Check cache
    if cache.is_transcript_complete(video_id):
        log_timed(f"Cache hit for {video_id}", t0)
        state = cache.get_state(video_id)
        thumb = cache.get_thumbnail_path(video_id)
        srt, txt = cache.get_transcript_paths(video_id)

        return VideoResult(
            success=True,
            video_id=video_id,
            output_dir=cache_dir,
            transcript_srt=srt if srt.exists() else None,
            transcript_txt=txt if txt.exists() else None,
            thumbnail=thumb if thumb.exists() else None,
            frames=sorted(cache_dir.glob("frames/*.jpg")) if extract_frames else [],
            metadata=state.to_dict() if state else {},
        )

    cache.ensure_cache_dir(video_id)
    audio_path = cache.get_audio_path(video_id)
    srt_path, txt_path = cache.get_transcript_paths(video_id)
    thumbnail_path = cache.get_thumbnail_path(video_id)

    # STEP 1: Fetch metadata
    log_timed("Fetching video metadata...", t0)
    try:
        meta = fetch_metadata(url)
    except Exception as e:
        return VideoResult(
            success=False,
            video_id=video_id,
            output_dir=cache_dir,
            error=str(e),
        )

    state = VideoState.from_metadata(video_id, url, meta)
    state.playlist_id = playlist_id
    cache.save_state(video_id, state)
    log_timed(f"Metadata: '{state.title}' ({state.duration_string})", t0)

    # STEP 1b: Download thumbnail
    if not thumbnail_path.exists():
        log_timed("Downloading thumbnail...", t0)
        thumb = download_thumbnail(url, cache_dir)
        if thumb:
            state.has_thumbnail = True
            cache.save_state(video_id, state)
            log_timed("Thumbnail saved", t0)
        else:
            log_timed("No thumbnail available", t0)

    # STEP 2: Try subtitles first (fast)
    log_timed("Checking for subtitles...", t0)
    sub_result = fetch_subtitles(url, cache_dir)
    if sub_result:
        srt_path.write_text(sub_result["srt"])
        txt_path.write_text(sub_result["txt"])
        state.transcript_complete = True
        state.transcript_source = sub_result["source"]
        cache.save_state(video_id, state)
        log_timed(f"DONE via subtitles ({sub_result['source']}) in {time.time() - t0:.1f}s", t0)
        return VideoResult(
            success=True,
            video_id=video_id,
            output_dir=cache_dir,
            transcript_srt=srt_path,
            transcript_txt=txt_path,
            thumbnail=thumbnail_path if thumbnail_path.exists() else None,
            metadata=state.to_dict(),
        )

    # STEP 3: Download audio for whisper
    log_timed("No subtitles available, falling back to whisper...", t0)
    if not audio_path.exists():
        log_timed("Downloading audio...", t0)
        try:
            download_audio(url, audio_path)
            size_mb = audio_path.stat().st_size / 1024 / 1024
            log_timed(f"Audio downloaded: {size_mb:.1f}MB", t0)
        except Exception as e:
            return VideoResult(
                success=False,
                video_id=video_id,
                output_dir=cache_dir,
                error=str(e),
                metadata=state.to_dict(),
            )

    # STEP 4: Transcribe with whisper
    if not srt_path.exists():
        log_timed(f"Transcribing with faster-whisper ({whisper_model})...", t0)
        try:
            transcript = transcribe_audio(audio_path, model_size=whisper_model)
            txt_path.write_text(transcript["txt"])
            srt_path.write_text(transcript["srt"])
            log_timed("Transcription complete", t0)
        except Exception as e:
            log_timed(f"Transcription failed: {e}", t0)
            # Continue without transcript

    # STEP 5: Optional frames
    frames = []
    if extract_frames:
        video_path = cache_dir / "video.mp4"
        if not video_path.exists():
            log_timed("Downloading video for frames...", t0)
            from claudetube.operations.download import download_video_segment
            download_video_segment(
                url=url,
                output_path=video_path,
                start_time=0,
                end_time=state.duration or 3600,
                quality_sort="+size,+br",
                concurrent_fragments=1,
            )
        if video_path.exists():
            ffmpeg = FFmpegTool()
            frames = ffmpeg.extract_frames_interval(
                video_path, cache_dir / "frames", frame_interval
            )
            video_path.unlink()
            log_timed("Cleaned up video file", t0)

    # Update state
    state.transcript_complete = True
    state.transcript_source = "whisper"
    state.whisper_model = whisper_model
    if frames:
        state.frames_count = len(frames)
        state.frame_interval = frame_interval
    cache.save_state(video_id, state)

    log_timed(f"DONE in {time.time() - t0:.1f}s", t0)

    return VideoResult(
        success=True,
        video_id=video_id,
        output_dir=cache_dir,
        transcript_srt=srt_path if srt_path.exists() else None,
        transcript_txt=txt_path if txt_path.exists() else None,
        thumbnail=thumbnail_path if thumbnail_path.exists() else None,
        frames=frames,
        metadata=state.to_dict(),
    )


def process_local_video(
    path: str,
    output_base: Path | None = None,
    whisper_model: str = "tiny",
    copy: bool = False,
) -> VideoResult:
    """Process a local video file - cache, transcribe, and extract thumbnail.

    Args:
        path: Path to local video/audio file (absolute, relative, ~, or file://)
        output_base: Base directory for cache
        whisper_model: tiny|base|small|medium|large
        copy: If True, copy the file to cache; if False (default), create symlink

    Returns:
        VideoResult with transcript and metadata
    """
    t0 = time.time()
    log_timed("Starting local video processing", t0)

    cache = CacheManager(output_base or get_cache_dir())

    # Parse and validate the local file path
    try:
        local_file = LocalFile.parse(path)
    except LocalFileError as e:
        return VideoResult(
            success=False,
            video_id="",
            output_dir=Path("."),
            error=str(e),
        )

    video_id = local_file.video_id
    cache_dir = cache.get_cache_dir(video_id)

    # Check cache
    if cache.is_transcript_complete(video_id):
        # Verify source file still valid (symlink not broken)
        is_valid, warning = cache.check_source_valid(video_id)
        if is_valid:
            log_timed(f"Cache hit for {video_id}", t0)
            state = cache.get_state(video_id)
            thumb = cache.get_thumbnail_path(video_id)
            srt, txt = cache.get_transcript_paths(video_id)

            return VideoResult(
                success=True,
                video_id=video_id,
                output_dir=cache_dir,
                transcript_srt=srt if srt.exists() else None,
                transcript_txt=txt if txt.exists() else None,
                thumbnail=thumb if thumb.exists() else None,
                frames=[],
                metadata=state.to_dict() if state else {},
            )
        else:
            log_timed(f"Cache invalid: {warning}", t0)
            # Clear invalid cache and re-process
            cache.clear(video_id)

    cache.ensure_cache_dir(video_id)
    srt_path, txt_path = cache.get_transcript_paths(video_id)
    thumbnail_path = cache.get_thumbnail_path(video_id)

    # STEP 1: Cache the local file (symlink or copy)
    log_timed("Caching local file...", t0)
    try:
        cached_path, cache_mode = cache.cache_local_file(video_id, local_file.path, copy=copy)
        log_timed(f"Cached via {cache_mode}: {cached_path.name}", t0)
    except Exception as e:
        return VideoResult(
            success=False,
            video_id=video_id,
            output_dir=cache_dir,
            error=f"Failed to cache local file: {e}",
        )

    # STEP 2: Extract metadata via ffprobe
    log_timed("Extracting metadata via ffprobe...", t0)
    ffprobe = FFprobeTool()
    metadata = ffprobe.get_metadata(local_file.path)

    state = VideoState.from_local_file(
        video_id=video_id,
        source_path=str(local_file.path),
        title=local_file.stem,
        duration=metadata.duration,
        duration_string=format_duration(metadata.duration),
        width=metadata.width,
        height=metadata.height,
        fps=metadata.fps,
        codec=metadata.codec,
        creation_time=metadata.creation_time,
    )
    state.cache_mode = cache_mode
    state.cached_file = cached_path.name
    cache.save_state(video_id, state)
    log_timed(f"Metadata: '{state.title}' ({state.duration_string or 'unknown duration'})", t0)

    # STEP 3: Generate thumbnail from video
    if not thumbnail_path.exists() and local_file.is_video:
        log_timed("Generating thumbnail...", t0)
        ffmpeg = FFmpegTool()
        # Extract frame at 10% into the video, or 5 seconds, whichever is less
        thumb_time = min((metadata.duration or 60) * 0.1, 5.0)
        frame = ffmpeg.extract_frame(
            video_path=cached_path,
            output_path=thumbnail_path,
            timestamp=thumb_time,
            width=480,
            jpeg_quality=5,
        )
        if frame:
            state.has_thumbnail = True
            cache.save_state(video_id, state)
            log_timed("Thumbnail generated", t0)
        else:
            log_timed("Thumbnail generation failed", t0)

    # STEP 4: Check for embedded/sidecar subtitles (faster than whisper)
    transcript_source = None
    if not srt_path.exists():
        log_timed("Checking for existing subtitles...", t0)
        from claudetube.operations.subtitles import fetch_local_subtitles
        sub_result = fetch_local_subtitles(local_file.path, cache_dir)
        if sub_result:
            srt_path.write_text(sub_result["srt"])
            txt_path.write_text(sub_result["txt"])
            transcript_source = sub_result["source"]
            log_timed(f"DONE via {transcript_source} subtitles in {time.time() - t0:.1f}s", t0)
            state.transcript_complete = True
            state.transcript_source = transcript_source
            cache.save_state(video_id, state)
            return VideoResult(
                success=True,
                video_id=video_id,
                output_dir=cache_dir,
                transcript_srt=srt_path,
                transcript_txt=txt_path,
                thumbnail=thumbnail_path if thumbnail_path.exists() else None,
                frames=[],
                metadata=state.to_dict(),
            )

    # STEP 5: No existing subtitles - extract audio and transcribe with whisper
    if not srt_path.exists():
        log_timed("No existing subtitles, falling back to whisper...", t0)
        audio_path = cache.get_audio_path(video_id)
        log_timed("Extracting audio...", t0)
        try:
            audio_path = extract_audio_local(cached_path, cache_dir)
            size_mb = audio_path.stat().st_size / 1024 / 1024
            log_timed(f"Audio extracted: {size_mb:.1f}MB", t0)
        except Exception as e:
            return VideoResult(
                success=False,
                video_id=video_id,
                output_dir=cache_dir,
                error=f"Audio extraction failed: {e}",
                metadata=state.to_dict(),
            )

        log_timed(f"Transcribing with faster-whisper ({whisper_model})...", t0)
        try:
            transcript = transcribe_audio(audio_path, model_size=whisper_model)
            txt_path.write_text(transcript["txt"])
            srt_path.write_text(transcript["srt"])
            transcript_source = "whisper"
            log_timed("Transcription complete", t0)
        except Exception as e:
            log_timed(f"Transcription failed: {e}", t0)
            # Continue without transcript - still return success for metadata/thumbnail

    # Update state
    state.transcript_complete = srt_path.exists()
    state.transcript_source = transcript_source
    state.whisper_model = whisper_model if transcript_source == "whisper" else None
    cache.save_state(video_id, state)

    log_timed(f"DONE in {time.time() - t0:.1f}s", t0)

    return VideoResult(
        success=True,
        video_id=video_id,
        output_dir=cache_dir,
        transcript_srt=srt_path if srt_path.exists() else None,
        transcript_txt=txt_path if txt_path.exists() else None,
        thumbnail=thumbnail_path if thumbnail_path.exists() else None,
        frames=[],
        metadata=state.to_dict(),
    )
