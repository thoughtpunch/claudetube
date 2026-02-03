"""
Whisper tool wrapper for audio transcription.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from claudetube.config.defaults import (
    MIN_TRANSCRIPT_COVERAGE,
    SRT_MAX_SEGMENT_DURATION,
    SRT_TARGET_SEGMENT_DURATION,
    WHISPER_BATCH_SIZE,
)
from claudetube.exceptions import TranscriptionError
from claudetube.tools.base import VideoTool
from claudetube.utils.formatting import format_srt_time

if TYPE_CHECKING:
    from pathlib import Path

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

    def _split_long_segment(self, seg) -> list[dict]:
        """Split a long segment into shorter chunks using word timestamps.

        Args:
            seg: Whisper segment with words attribute

        Returns:
            List of dicts with 'start', 'end', 'text' keys
        """
        # If no word-level timestamps, return as-is
        if not hasattr(seg, "words") or not seg.words:
            return [{"start": seg.start, "end": seg.end, "text": seg.text.strip()}]

        duration = seg.end - seg.start
        if duration <= SRT_MAX_SEGMENT_DURATION:
            return [{"start": seg.start, "end": seg.end, "text": seg.text.strip()}]

        # Split using word timestamps
        chunks = []
        current_words = []
        chunk_start = None

        for word in seg.words:
            word_text = word.word.strip() if hasattr(word, "word") else str(word)
            word_start = word.start if hasattr(word, "start") else seg.start
            word_end = word.end if hasattr(word, "end") else seg.end

            if chunk_start is None:
                chunk_start = word_start

            current_words.append(word_text)
            chunk_duration = word_end - chunk_start

            # Split when we hit target duration (at word boundary)
            if chunk_duration >= SRT_TARGET_SEGMENT_DURATION and current_words:
                chunk_text = " ".join(current_words).strip()
                if chunk_text:
                    chunks.append({
                        "start": chunk_start,
                        "end": word_end,
                        "text": chunk_text,
                    })
                current_words = []
                chunk_start = None

        # Don't forget the last chunk
        if current_words:
            last_word = seg.words[-1]
            chunk_text = " ".join(current_words).strip()
            if chunk_text:
                chunks.append({
                    "start": chunk_start or seg.start,
                    "end": last_word.end if hasattr(last_word, "end") else seg.end,
                    "text": chunk_text,
                })

        return chunks if chunks else [{"start": seg.start, "end": seg.end, "text": seg.text.strip()}]

    def _collect_segments(self, segments) -> dict:
        """Collect transcription segments into SRT and TXT format.

        Splits segments longer than SRT_MAX_SEGMENT_DURATION using word timestamps.
        """
        srt_lines = []
        txt_lines = []
        last_end = 0.0
        segment_num = 0

        for seg in segments:
            # Split long segments using word timestamps
            chunks = self._split_long_segment(seg)

            for chunk in chunks:
                segment_num += 1
                start = format_srt_time(chunk["start"])
                end = format_srt_time(chunk["end"])
                text = chunk["text"]
                last_end = chunk["end"]

                srt_lines.append(f"{segment_num}\n{start} --> {end}\n{text}\n")
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
            word_timestamps=True,  # Enable word-level timestamps for segment splitting
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
            word_timestamps=True,  # Enable word-level timestamps for segment splitting
        )

        return self._collect_segments(segments)
