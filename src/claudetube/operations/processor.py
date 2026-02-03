"""
Main video processor - orchestrates download, transcription, and caching.
"""

from __future__ import annotations

import time
from pathlib import Path

from claudetube.cache.manager import CacheManager
from claudetube.cache.storage import load_state, save_state
from claudetube.config.loader import get_cache_dir
from claudetube.models.local_file import LocalFile, LocalFileError
from claudetube.models.state import VideoState
from claudetube.models.video_path import VideoPath
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
    playlist_id: str | None = None,
) -> VideoResult:
    """Process a video - transcript first, frames optional.

    Args:
        url: Video URL
        output_base: Base directory for cache
        whisper_model: tiny|base|small|medium|large
        extract_frames: Whether to extract frames (default: False for speed)
        frame_interval: Seconds between frames if extracting
        playlist_id: Optional playlist ID to use in hierarchical path

    Returns:
        VideoResult with transcript and optional frames
    """
    from claudetube.models.video_path import VideoPath

    t0 = time.time()
    log_timed("Starting video processing", t0)

    cache_base = output_base or get_cache_dir()
    cache = CacheManager(cache_base)

    # Extract context from URL (video_id, playlist_id, etc.)
    url_context = extract_url_context(url)
    video_id = url_context["video_id"]
    # Use explicit playlist_id if provided, otherwise extract from URL
    url_playlist_id = playlist_id or url_context.get("playlist_id")

    # Construct VideoPath for hierarchical caching
    # At this stage, we may not have channel info (comes from yt-dlp metadata later)
    video_path = VideoPath.from_url(url)

    # Override playlist if explicitly provided
    if url_playlist_id and not video_path.playlist:
        video_path = VideoPath(
            domain=video_path.domain,
            channel=video_path.channel,
            playlist=url_playlist_id,
            video_id=video_path.video_id,
        )

    # Get cache directory - uses hierarchical path for new videos
    # First check if video already exists (SQLite -> flat -> glob resolution)
    existing_cache_dir = cache.get_cache_dir(video_id)
    if existing_cache_dir.exists() and (existing_cache_dir / "state.json").exists():
        # Video already cached, use existing location
        cache_dir = existing_cache_dir
    else:
        # New video - use hierarchical path
        cache_dir = cache.get_cache_dir_for_path(video_path)

    # Check cache - video may be at existing location
    if cache_dir.exists() and (cache_dir / "state.json").exists():
        state = load_state(cache_dir / "state.json")
        if state and state.transcript_complete:
            log_timed(f"Cache hit for {video_id}", t0)
            thumb = cache_dir / "thumbnail.jpg"
            srt = cache_dir / "audio.srt"
            txt = cache_dir / "audio.txt"

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

    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path = cache_dir / "audio.mp3"
    srt_path = cache_dir / "audio.srt"
    txt_path = cache_dir / "audio.txt"
    thumbnail_path = cache_dir / "thumbnail.jpg"

    # STEP 1: Fetch metadata
    log_timed("Fetching video metadata...", t0)

    # Record download pipeline step as running
    try:
        from claudetube.db.sync import record_pipeline_step, safe_update_pipeline_step

        download_step_id = record_pipeline_step(
            video_id, "download", "running", provider="yt-dlp"
        )
    except Exception:
        download_step_id = None

    try:
        meta = fetch_metadata(url)
    except Exception as e:
        # Record download failure
        safe_update_pipeline_step(download_step_id, "failed", error_message=str(e))
        return VideoResult(
            success=False,
            video_id=video_id,
            output_dir=cache_dir,
            error=str(e),
        )

    state = VideoState.from_metadata(video_id, url, meta)
    state.playlist_id = url_playlist_id
    # Store hierarchical path info in state for database sync
    state.domain = video_path.domain
    state.channel_id = (
        meta.get("channel_id") or meta.get("uploader_id") or video_path.channel
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_state(state, cache_dir / "state.json")
    log_timed(f"Metadata: '{state.title}' ({state.duration_string})", t0)

    # Sync video to SQLite with hierarchical cache_path
    try:
        from claudetube.db.sync import sync_video

        # Build cache_path relative to cache_base
        cache_path_rel = str(cache_dir.relative_to(cache_base))
        sync_video(state, cache_path_rel)
    except Exception:
        pass  # Fire-and-forget

    # Progressive enrichment: check if yt-dlp metadata provides better path info
    # than what we extracted from URL alone. If so, move the cache directory.
    cache_dir = _try_progressive_enrichment(
        video_id=video_id,
        video_path=video_path,
        meta=meta,
        cache_dir=cache_dir,
        cache_base=cache_base,
        state=state,
    )

    # Update audio and transcript paths to use the (possibly moved) cache_dir
    audio_path = cache_dir / "audio.mp3"
    srt_path = cache_dir / "audio.srt"
    txt_path = cache_dir / "audio.txt"
    thumbnail_path = cache_dir / "thumbnail.jpg"

    # Record download step completed
    safe_update_pipeline_step(download_step_id, "completed")

    # STEP 1b: Download thumbnail
    if not thumbnail_path.exists():
        log_timed("Downloading thumbnail...", t0)
        thumb = download_thumbnail(url, cache_dir)
        if thumb:
            state.has_thumbnail = True
            save_state(state, cache_dir / "state.json")
            log_timed("Thumbnail saved", t0)

            # Dual-write: sync thumbnail to SQLite (fire-and-forget)
            try:
                from claudetube.db.sync import get_video_uuid, sync_frame

                video_uuid = get_video_uuid(video_id)
                if video_uuid:
                    # Get relative path from cache_dir
                    relative_path = "thumbnail.jpg"

                    sync_frame(
                        video_uuid=video_uuid,
                        timestamp=0.0,
                        extraction_type="thumbnail",
                        file_path=relative_path,
                        is_thumbnail=True,
                    )
            except Exception:
                # Fire-and-forget: don't disrupt thumbnail download
                pass
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
        save_state(state, cache_dir / "state.json")
        log_timed(
            f"DONE via subtitles ({sub_result['source']}) in {time.time() - t0:.1f}s",
            t0,
        )

        # Fire-and-forget: sync transcription from subtitles to database
        try:
            from claudetube.db.sync import (
                get_video_uuid,
                record_pipeline_step,
                sync_transcription,
            )

            video_uuid = get_video_uuid(video_id)
            if video_uuid:
                # Read full transcript text for FTS indexing
                full_text = sub_result["txt"]
                word_count = len(full_text.split()) if full_text else None

                sync_transcription(
                    video_uuid=video_uuid,
                    provider="youtube_subtitles",
                    format_="txt",
                    file_path="audio.txt",
                    full_text=full_text,
                    word_count=word_count,
                    duration=state.duration,
                    file_size_bytes=int(txt_path.stat().st_size)
                    if txt_path.exists()
                    else None,
                    is_primary=True,
                )

                # Record transcribe step as completed (subtitles path)
                record_pipeline_step(
                    video_id,
                    "transcribe",
                    "completed",
                    provider="youtube_subtitles",
                )
        except Exception:
            pass  # Fire-and-forget

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

        # Record audio_extract pipeline step as running
        try:
            from claudetube.db.sync import (
                record_pipeline_step,
                safe_update_pipeline_step,
            )

            audio_step_id = record_pipeline_step(
                video_id, "audio_extract", "running", provider="yt-dlp"
            )
        except Exception:
            audio_step_id = None

        try:
            download_audio(url, audio_path)
            size_mb = audio_path.stat().st_size / 1024 / 1024
            log_timed(f"Audio downloaded: {size_mb:.1f}MB", t0)

            # Fire-and-forget: sync audio track to database
            try:
                from claudetube.db.sync import get_video_uuid, sync_audio_track

                video_uuid = get_video_uuid(video_id)
                if video_uuid:
                    sync_audio_track(
                        video_uuid=video_uuid,
                        format_="mp3",
                        file_path="audio.mp3",
                        file_size_bytes=int(audio_path.stat().st_size),
                        duration=state.duration,
                    )
            except Exception:
                pass  # Fire-and-forget

            # Record audio_extract step completed
            safe_update_pipeline_step(audio_step_id, "completed")

        except Exception as e:
            # Record audio_extract failure
            safe_update_pipeline_step(audio_step_id, "failed", error_message=str(e))
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

        # Record transcribe pipeline step as running
        try:
            from claudetube.db.sync import (
                record_pipeline_step,
                safe_update_pipeline_step,
            )

            transcribe_step_id = record_pipeline_step(
                video_id,
                "transcribe",
                "running",
                provider="whisper",
                model=whisper_model,
            )
        except Exception:
            transcribe_step_id = None

        try:
            transcript = transcribe_audio(audio_path, model_size=whisper_model)
            txt_path.write_text(transcript["txt"])
            srt_path.write_text(transcript["srt"])
            log_timed("Transcription complete", t0)

            # Fire-and-forget: sync transcription to database
            try:
                from claudetube.db.sync import get_video_uuid, sync_transcription

                video_uuid = get_video_uuid(video_id)
                if video_uuid:
                    # Read full transcript text for FTS indexing
                    full_text = transcript["txt"]
                    word_count = len(full_text.split()) if full_text else None

                    sync_transcription(
                        video_uuid=video_uuid,
                        provider="whisper",
                        format_="txt",
                        file_path="audio.txt",
                        model=whisper_model,
                        full_text=full_text,
                        word_count=word_count,
                        duration=state.duration,
                        file_size_bytes=int(txt_path.stat().st_size)
                        if txt_path.exists()
                        else None,
                        is_primary=True,
                    )
            except Exception:
                pass  # Fire-and-forget

            # Record transcribe step completed
            safe_update_pipeline_step(transcribe_step_id, "completed")

        except Exception as e:
            log_timed(f"Transcription failed: {e}", t0)
            # Record transcribe failure
            safe_update_pipeline_step(
                transcribe_step_id, "failed", error_message=str(e)
            )
            # Continue without transcript

    # STEP 5: Optional frames
    frames = []
    if extract_frames:
        video_file_path = cache_dir / "video.mp4"
        if not video_file_path.exists():
            log_timed("Downloading video for frames...", t0)
            from claudetube.operations.download import download_video_segment

            download_video_segment(
                url=url,
                output_path=video_file_path,
                start_time=0,
                end_time=state.duration or 3600,
                quality_sort="+size,+br",
                concurrent_fragments=1,
            )
        if video_file_path.exists():
            ffmpeg = FFmpegTool()
            frames = ffmpeg.extract_frames_interval(
                video_file_path, cache_dir / "frames", frame_interval
            )
            video_file_path.unlink()
            log_timed("Cleaned up video file", t0)

    # Update state
    state.transcript_complete = True
    state.transcript_source = "whisper"
    state.whisper_model = whisper_model
    if frames:
        state.frames_count = len(frames)
        state.frame_interval = frame_interval
    save_state(state, cache_dir / "state.json")

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
        cached_path, cache_mode = cache.cache_local_file(
            video_id, local_file.path, copy=copy
        )
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
    log_timed(
        f"Metadata: '{state.title}' ({state.duration_string or 'unknown duration'})", t0
    )

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

            # Dual-write: sync thumbnail to SQLite (fire-and-forget)
            try:
                from claudetube.db.sync import get_video_uuid, sync_frame

                video_uuid = get_video_uuid(video_id)
                if video_uuid:
                    # Get relative path from cache_dir
                    relative_path = "thumbnail.jpg"

                    sync_frame(
                        video_uuid=video_uuid,
                        timestamp=thumb_time,
                        extraction_type="thumbnail",
                        file_path=relative_path,
                        is_thumbnail=True,
                        width=480,
                    )
            except Exception:
                # Fire-and-forget: don't disrupt thumbnail generation
                pass
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
            log_timed(
                f"DONE via {transcript_source} subtitles in {time.time() - t0:.1f}s", t0
            )
            state.transcript_complete = True
            state.transcript_source = transcript_source
            cache.save_state(video_id, state)

            # Fire-and-forget: sync transcription from local subtitles to database
            try:
                from claudetube.db.sync import (
                    get_video_uuid,
                    record_pipeline_step,
                    sync_transcription,
                )

                video_uuid = get_video_uuid(video_id)
                if video_uuid:
                    # Read full transcript text for FTS indexing
                    full_text = sub_result["txt"]
                    word_count = len(full_text.split()) if full_text else None

                    sync_transcription(
                        video_uuid=video_uuid,
                        provider="manual",  # local subtitles treated as manual
                        format_="txt",
                        file_path="audio.txt",
                        full_text=full_text,
                        word_count=word_count,
                        duration=state.duration,
                        file_size_bytes=int(txt_path.stat().st_size)
                        if txt_path.exists()
                        else None,
                        is_primary=True,
                    )

                    # Record transcribe step as completed
                    record_pipeline_step(
                        video_id,
                        "transcribe",
                        "completed",
                        provider="manual",
                    )
            except Exception:
                pass  # Fire-and-forget

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

        # Record audio_extract pipeline step as running
        try:
            from claudetube.db.sync import (
                record_pipeline_step,
                safe_update_pipeline_step,
            )

            local_audio_step_id = record_pipeline_step(
                video_id, "audio_extract", "running", provider="ffmpeg"
            )
        except Exception:
            local_audio_step_id = None

        try:
            audio_path = extract_audio_local(cached_path, cache_dir)
            size_mb = audio_path.stat().st_size / 1024 / 1024
            log_timed(f"Audio extracted: {size_mb:.1f}MB", t0)

            # Fire-and-forget: sync audio track to database
            try:
                from claudetube.db.sync import get_video_uuid, sync_audio_track

                video_uuid = get_video_uuid(video_id)
                if video_uuid:
                    sync_audio_track(
                        video_uuid=video_uuid,
                        format_="mp3",
                        file_path="audio.mp3",
                        file_size_bytes=int(audio_path.stat().st_size),
                        duration=state.duration,
                    )
            except Exception:
                pass  # Fire-and-forget

            # Record audio_extract step completed
            safe_update_pipeline_step(local_audio_step_id, "completed")

        except Exception as e:
            # Record audio_extract failure
            safe_update_pipeline_step(
                local_audio_step_id, "failed", error_message=str(e)
            )
            return VideoResult(
                success=False,
                video_id=video_id,
                output_dir=cache_dir,
                error=f"Audio extraction failed: {e}",
                metadata=state.to_dict(),
            )

        log_timed(f"Transcribing with faster-whisper ({whisper_model})...", t0)

        # Record transcribe pipeline step as running
        try:
            from claudetube.db.sync import (
                record_pipeline_step,
                safe_update_pipeline_step,
            )

            local_transcribe_step_id = record_pipeline_step(
                video_id,
                "transcribe",
                "running",
                provider="whisper",
                model=whisper_model,
            )
        except Exception:
            local_transcribe_step_id = None

        try:
            transcript = transcribe_audio(audio_path, model_size=whisper_model)
            txt_path.write_text(transcript["txt"])
            srt_path.write_text(transcript["srt"])
            transcript_source = "whisper"
            log_timed("Transcription complete", t0)

            # Fire-and-forget: sync transcription to database
            try:
                from claudetube.db.sync import get_video_uuid, sync_transcription

                video_uuid = get_video_uuid(video_id)
                if video_uuid:
                    # Read full transcript text for FTS indexing
                    full_text = transcript["txt"]
                    word_count = len(full_text.split()) if full_text else None

                    sync_transcription(
                        video_uuid=video_uuid,
                        provider="whisper",
                        format_="txt",
                        file_path="audio.txt",
                        model=whisper_model,
                        full_text=full_text,
                        word_count=word_count,
                        duration=state.duration,
                        file_size_bytes=int(txt_path.stat().st_size)
                        if txt_path.exists()
                        else None,
                        is_primary=True,
                    )
            except Exception:
                pass  # Fire-and-forget

            # Record transcribe step completed
            safe_update_pipeline_step(local_transcribe_step_id, "completed")

        except Exception as e:
            log_timed(f"Transcription failed: {e}", t0)
            # Record transcribe failure
            safe_update_pipeline_step(
                local_transcribe_step_id, "failed", error_message=str(e)
            )
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


def _try_progressive_enrichment(
    video_id: str,
    video_path: VideoPath,
    meta: dict,
    cache_dir: Path,
    cache_base: Path,
    state: VideoState,
) -> Path:
    """Attempt to move cache directory if metadata provides richer path info.

    Compares the initial VideoPath (from URL parsing) with what we'd get from
    yt-dlp metadata. If metadata provides channel/playlist info we didn't have,
    triggers directory move via enrich_video().

    This implements progressive enrichment: initial download uses URL-derived
    path (may be no_channel/no_playlist), then enriched with yt-dlp metadata.

    Args:
        video_id: The video's natural key.
        video_path: Initial VideoPath from URL parsing.
        meta: yt-dlp metadata dict with channel_id, playlist_id, etc.
        cache_dir: Current cache directory (before potential move).
        cache_base: Base cache directory.
        state: VideoState to update if path changes.

    Returns:
        The cache directory (may be new location if moved).
    """
    try:
        # Build enriched path using yt-dlp metadata
        enriched_path = VideoPath.from_url(state.url or "", metadata=meta)

        # Override playlist if it was explicitly provided in video_path
        if video_path.playlist and not enriched_path.playlist:
            enriched_path = VideoPath(
                domain=enriched_path.domain,
                channel=enriched_path.channel,
                playlist=video_path.playlist,
                video_id=enriched_path.video_id,
            )

        # Check if path improved
        old_rel = str(video_path.relative_path())
        new_rel = str(enriched_path.relative_path())

        if old_rel == new_rel:
            # No improvement, return original cache_dir
            return cache_dir

        # Path improved - attempt enrichment via db/sync
        from claudetube.db.sync import enrich_video

        enrich_video(video_id, meta, cache_base)

        # After enrich_video, directory may have moved
        # Calculate new cache_dir from enriched path
        new_cache_dir = enriched_path.cache_dir(cache_base)

        if new_cache_dir.exists() and (new_cache_dir / "state.json").exists():
            # Move succeeded - update state with new location info
            state.domain = enriched_path.domain
            state.channel_id = enriched_path.channel
            state.playlist_id = enriched_path.playlist
            save_state(state, new_cache_dir / "state.json")
            log_timed(f"Enriched path: {old_rel} -> {new_rel}", 0)
            return new_cache_dir

        # Move may have failed - return original
        return cache_dir

    except Exception:
        # Fire-and-forget: any error returns original cache_dir
        import logging

        logging.getLogger(__name__).debug(
            "Progressive enrichment failed (ignored)", exc_info=True
        )
        return cache_dir
