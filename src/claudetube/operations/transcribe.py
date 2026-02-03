"""
Transcription operations.

Provides TranscribeOperation class for provider-based transcription,
plus backward-compatible function wrappers.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from claudetube.cache.manager import CacheManager
from claudetube.config.loader import get_cache_dir
from claudetube.exceptions import TranscriptionError
from claudetube.operations.download import download_audio
from claudetube.tools.whisper import WhisperTool
from claudetube.utils.logging import log_timed

if TYPE_CHECKING:
    from pathlib import Path

    from claudetube.providers.base import Transcriber

logger = logging.getLogger(__name__)


def _map_provider_to_db(provider: str) -> str:
    """Map provider name to valid database provider enum value.

    The database expects specific provider values. This maps various
    provider names to their canonical DB values.

    Args:
        provider: Provider name from transcription result.

    Returns:
        Valid provider value for database.
    """
    # Map known providers to DB enum values
    mapping = {
        "whisper": "whisper",
        "whisper-local": "whisper",
        "faster-whisper": "whisper",
        "openai": "openai",
        "openai-whisper": "openai",
        "deepgram": "deepgram",
        "youtube": "youtube_subtitles",
        "youtube_subtitles": "youtube_subtitles",
    }
    return mapping.get(provider.lower(), "manual")


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


class TranscribeOperation:
    """Transcribe video audio using a configurable Transcriber provider.

    Accepts any provider implementing the Transcriber protocol via constructor
    injection. The execute() method runs the transcription, saves results to
    cache, and updates video state.

    Args:
        transcriber: Provider implementing the Transcriber protocol.

    Example:
        >>> from claudetube.providers import get_provider
        >>> transcriber = get_provider("whisper-local", model_size="small")
        >>> op = TranscribeOperation(transcriber)
        >>> result = await op.execute("abc123", Path("audio.mp3"))
    """

    def __init__(self, transcriber: Transcriber):
        self.transcriber = transcriber

    async def execute(
        self,
        video_id: str,
        audio_path: Path,
        language: str | None = None,
        cache_dir: Path | None = None,
    ) -> dict:
        """Execute transcription and save results.

        Args:
            video_id: Video identifier
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "es")
            cache_dir: Optional cache base directory

        Returns:
            Dict with success status, paths, and metadata
        """
        cache = CacheManager(cache_dir or get_cache_dir())
        srt_path, txt_path = cache.get_transcript_paths(video_id)

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
                provider=getattr(self.transcriber, "provider_name", "unknown"),
            )
        except Exception:
            transcribe_step_id = None

        try:
            result = await self.transcriber.transcribe(audio_path, language=language)

            srt_path.write_text(result.to_srt())
            txt_path.write_text(result.text)

            state = cache.get_state(video_id)
            if state:
                state.transcript_complete = True
                state.transcript_source = result.provider
                cache.save_state(video_id, state)

            # Fire-and-forget: sync transcription to database
            try:
                from claudetube.db.sync import get_video_uuid, sync_transcription

                video_uuid = get_video_uuid(video_id)
                if video_uuid:
                    # Map provider to valid DB provider
                    db_provider = _map_provider_to_db(result.provider)

                    sync_transcription(
                        video_uuid=video_uuid,
                        provider=db_provider,
                        format_="txt",
                        file_path="audio.txt",
                        language=result.language,
                        full_text=result.text,
                        word_count=len(result.text.split()) if result.text else None,
                        duration=result.duration,
                        file_size_bytes=int(txt_path.stat().st_size) if txt_path.exists() else None,
                        is_primary=True,
                    )
            except Exception:
                pass  # Fire-and-forget

            # Record transcribe step completed
            safe_update_pipeline_step(transcribe_step_id, "completed")

            return {
                "success": True,
                "video_id": video_id,
                "transcript_srt": str(srt_path),
                "transcript_txt": str(txt_path),
                "source": result.provider,
                "whisper_model": None,
                "segments": len(result.segments),
                "duration": result.duration,
                "message": f"Transcribed with {result.provider}.",
            }

        except Exception as e:
            # Record transcribe failure
            safe_update_pipeline_step(transcribe_step_id, "failed", error_message=str(e))
            raise


async def transcribe_video(
    video_id_or_url: str,
    whisper_model: str = "small",
    force: bool = False,
    output_base: Path | None = None,
    transcriber: Transcriber | None = None,
) -> dict:
    """Transcribe a video's audio using a configurable provider.

    Cache-first: returns existing transcript immediately unless force=True.
    If no transcript exists, downloads audio (if needed) and runs transcription.

    This is a backward-compatible wrapper around TranscribeOperation.

    Args:
        video_id_or_url: Video ID or URL
        whisper_model: Whisper model size (used when transcriber is None)
        force: Re-transcribe even if cached transcript exists
        output_base: Cache directory (default: ~/.claude/video_cache)
        transcriber: Optional Transcriber provider. If None, uses whisper-local.

    Returns:
        Dict with success, video_id, transcript paths, source, message
    """
    from claudetube.parsing.utils import extract_video_id

    t0 = time.time()
    cache = CacheManager(output_base or get_cache_dir())

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

    # Ensure we have audio
    cache.ensure_cache_dir(video_id)

    if not audio_path.exists():
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

    # Get or create transcriber
    if transcriber is None:
        from claudetube.providers import get_provider

        transcriber = get_provider("whisper-local", model_size=whisper_model)

    # Execute operation
    log_timed("Transcribing...", t0)
    try:
        op = TranscribeOperation(transcriber)
        result = await op.execute(video_id, audio_path, cache_dir=output_base)
    except (TranscriptionError, FileNotFoundError) as e:
        return {
            "success": False,
            "video_id": video_id,
            "transcript_srt": None,
            "transcript_txt": None,
            "source": None,
            "whisper_model": whisper_model,
            "message": str(e),
        }

    log_timed(f"Transcription complete in {time.time() - t0:.1f}s", t0)
    return result
