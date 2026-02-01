"""
Transcription operations.
"""

from __future__ import annotations

import logging
from pathlib import Path

from claudetube.cache.manager import CacheManager
from claudetube.config.defaults import CACHE_DIR
from claudetube.exceptions import TranscriptionError
from claudetube.operations.download import download_audio
from claudetube.tools.whisper import WhisperTool
from claudetube.utils.logging import log_timed

logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_path: Path,
    model_size: str = "tiny",
) -> dict:
    """Transcribe audio file with Whisper.

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny/base/small/medium/large)

    Returns:
        Dict with 'srt' and 'txt' transcript text

    Raises:
        TranscriptionError: If transcription fails
    """
    tool = WhisperTool(model_size=model_size)
    return tool.transcribe(audio_path)


def transcribe_video(
    video_id_or_url: str,
    whisper_model: str = "small",
    force: bool = False,
    output_base: Path | None = None,
) -> dict:
    """Transcribe a video's audio using Whisper.

    Cache-first: returns existing transcript immediately unless force=True.
    If no transcript exists, downloads audio (if needed) and runs Whisper.

    Args:
        video_id_or_url: Video ID or URL
        whisper_model: Whisper model size
        force: Re-transcribe even if cached transcript exists
        output_base: Cache directory (default: ~/.claude/video_cache)

    Returns:
        Dict with success, video_id, transcript paths, source, message
    """
    import time

    from claudetube.parsing.utils import extract_video_id

    t0 = time.time()
    cache = CacheManager(output_base or CACHE_DIR)

    video_id = extract_video_id(video_id_or_url)
    srt_path, txt_path = cache.get_transcript_paths(video_id)
    audio_path = cache.get_audio_path(video_id)

    # Cache check: return existing transcript if available and not forced
    if not force and srt_path.exists() and txt_path.exists():
        log_timed(f"Returning cached transcript for {video_id}", t0)
        return {
            "success": True,
            "video_id": video_id,
            "transcript_srt": str(srt_path),
            "transcript_txt": str(txt_path),
            "source": "cached",
            "whisper_model": None,
            "message": "Returned cached transcript.",
        }

    # Need to run Whisper - ensure we have audio
    cache.ensure_cache_dir(video_id)

    if not audio_path.exists():
        # Get URL from state or input
        state = cache.get_state(video_id)
        url = state.url if state else None

        if not url and "://" in video_id_or_url:
            url = video_id_or_url

        if not url:
            return {
                "success": False,
                "video_id": video_id,
                "transcript_srt": None,
                "transcript_txt": None,
                "source": None,
                "whisper_model": None,
                "message": "No audio file and no URL available. Process the video first.",
            }

        log_timed("Downloading audio for transcription...", t0)
        try:
            download_audio(url, audio_path)
        except Exception as e:
            return {
                "success": False,
                "video_id": video_id,
                "transcript_srt": None,
                "transcript_txt": None,
                "source": None,
                "whisper_model": None,
                "message": f"Audio download failed: {e}",
            }

    # Run Whisper transcription
    log_timed(f"Transcribing with Whisper ({whisper_model})...", t0)
    try:
        transcript = transcribe_audio(audio_path, model_size=whisper_model)
    except TranscriptionError as e:
        return {
            "success": False,
            "video_id": video_id,
            "transcript_srt": None,
            "transcript_txt": None,
            "source": None,
            "whisper_model": whisper_model,
            "message": str(e),
        }

    srt_path.write_text(transcript["srt"])
    txt_path.write_text(transcript["txt"])

    # Update state
    state = cache.get_state(video_id)
    if state:
        state.transcript_complete = True
        state.transcript_source = "whisper"
        state.whisper_model = whisper_model
        cache.save_state(video_id, state)

    log_timed(f"Transcription complete in {time.time() - t0:.1f}s", t0)

    return {
        "success": True,
        "video_id": video_id,
        "transcript_srt": str(srt_path),
        "transcript_txt": str(txt_path),
        "source": "whisper",
        "whisper_model": whisper_model,
        "message": f"Transcribed with Whisper ({whisper_model}).",
    }
