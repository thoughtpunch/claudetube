"""
Whisper tool wrapper for audio transcription.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from claudetube.config.defaults import MIN_TRANSCRIPT_COVERAGE, WHISPER_BATCH_SIZE
from claudetube.exceptions import TranscriptionError
from claudetube.tools.base import VideoTool
from claudetube.utils.formatting import format_srt_time

logger = logging.getLogger(__name__)


class WhisperTool(VideoTool):
    """Wrapper for faster-whisper transcription."""

    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self._model = None

    @property
    def name(self) -> str:
        return "faster-whisper"

    def is_available(self) -> bool:
        """Check if faster-whisper is installed."""
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def get_path(self) -> str:
        """Not applicable for Python library."""
        return "faster-whisper"

    def _get_model(self):
        """Lazy-load the Whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel

            cpu_threads = os.cpu_count() or 4
            logger.info(
                f"Loading Whisper model ({self.model_size}, {cpu_threads} threads)"
            )

            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=cpu_threads,
            )

        return self._model

    def _collect_segments(self, segments) -> dict:
        """Collect transcription segments into SRT and TXT format."""
        srt_lines = []
        txt_lines = []
        last_end = 0.0

        for i, seg in enumerate(segments, 1):
            start = format_srt_time(seg.start)
            end = format_srt_time(seg.end)
            text = seg.text.strip()
            last_end = seg.end

            srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
            txt_lines.append(text)

            logger.debug(f"[{start}] {text[:60]}{'...' if len(text) > 60 else ''}")

        return {
            "srt": "\n".join(srt_lines),
            "txt": "\n".join(txt_lines),
            "last_end": last_end,
        }

    def transcribe(
        self,
        audio_path: Path,
        language: str = "en",
        use_batched: bool = True,
    ) -> dict:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file
            language: Language code
            use_batched: Use batched inference (faster on multi-core)

        Returns:
            Dict with 'srt' and 'txt' transcript text

        Raises:
            TranscriptionError: If transcription fails
        """
        if not self.is_available():
            raise TranscriptionError("faster-whisper not installed")

        try:
            model = self._get_model()

            if use_batched:
                result = self._transcribe_batched(model, audio_path, language)
            else:
                result = self._transcribe_standard(model, audio_path, language)

            return {"srt": result["srt"], "txt": result["txt"]}

        except ImportError as e:
            raise TranscriptionError(f"Missing dependency: {e}") from e
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

    def _transcribe_batched(
        self,
        model,
        audio_path: Path,
        language: str,
    ) -> dict:
        """Transcribe using batched inference (faster)."""
        from faster_whisper import BatchedInferencePipeline

        logger.info("Transcribing (batched)...")
        batched_model = BatchedInferencePipeline(model=model)

        segments, info = batched_model.transcribe(
            str(audio_path),
            language=language,
            batch_size=WHISPER_BATCH_SIZE,
        )

        result = self._collect_segments(segments)

        # Check if batched result covers enough of the audio
        audio_duration = info.duration or 0
        last_end = result["last_end"]

        if audio_duration > 30 and last_end < audio_duration * MIN_TRANSCRIPT_COVERAGE:
            logger.info(
                f"Batched result only covered {last_end:.0f}s of "
                f"{audio_duration:.0f}s, retrying without batching"
            )
            return self._transcribe_standard(model, audio_path, language)

        return result

    def _transcribe_standard(
        self,
        model,
        audio_path: Path,
        language: str,
    ) -> dict:
        """Transcribe using standard (non-batched) inference."""
        logger.info("Transcribing (standard)...")

        segments, info = model.transcribe(
            str(audio_path),
            language=language,
        )

        return self._collect_segments(segments)
