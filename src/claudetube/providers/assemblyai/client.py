"""
claudetube.providers.assemblyai.client - AssemblyAIProvider implementation.

Uses the AssemblyAI SDK for audio transcription with optional speaker
diarization, auto-chapters, and sentiment analysis. The SDK is synchronous
so transcription calls are wrapped in asyncio.run_in_executor().

Example:
    >>> provider = AssemblyAIProvider()
    >>> result = await provider.transcribe(
    ...     Path("audio.mp3"),
    ...     diarize=True,
    ...     auto_chapters=True,
    ... )
    >>> for seg in result.segments:
    ...     print(f"[{seg.speaker}] {seg.text}")
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import partial
from typing import TYPE_CHECKING, Any

from claudetube.providers.base import Provider, Transcriber
from claudetube.providers.capabilities import PROVIDER_INFO, ProviderInfo
from claudetube.providers.types import TranscriptionResult, TranscriptionSegment

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class AssemblyAIProvider(Provider, Transcriber):
    """AssemblyAI transcription provider with chapters and sentiment support.

    Uses the AssemblyAI transcription API with support for:
    - Multiple languages (auto-detect or specified)
    - Speaker diarization (identify who is speaking)
    - Auto-chapters (automatic topic segmentation with summaries)
    - Sentiment analysis (per-sentence sentiment)

    The AssemblyAI SDK is synchronous, so transcription calls are wrapped
    in asyncio.run_in_executor() to avoid blocking the event loop.

    Args:
        api_key: AssemblyAI API key. Defaults to ASSEMBLYAI_API_KEY env var.
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key
        self._client = None

    @property
    def info(self) -> ProviderInfo:
        return PROVIDER_INFO["assemblyai"]

    def _resolve_api_key(self) -> str | None:
        """Resolve API key from init arg or environment."""
        return self._api_key or os.environ.get("ASSEMBLYAI_API_KEY")

    def is_available(self) -> bool:
        """Check if the AssemblyAI SDK is installed and API key is set."""
        try:
            import assemblyai  # noqa: F401
        except ImportError:
            return False
        return self._resolve_api_key() is not None

    def _get_client(self) -> Any:
        """Lazy-load the AssemblyAI Transcriber client.

        Returns:
            An assemblyai.Transcriber instance.

        Raises:
            ValueError: If ASSEMBLYAI_API_KEY is not set.
        """
        if self._client is None:
            import assemblyai as aai

            api_key = self._resolve_api_key()
            if not api_key:
                raise ValueError(
                    "ASSEMBLYAI_API_KEY not set. "
                    "Set the environment variable or pass api_key to the provider."
                )
            aai.settings.api_key = api_key
            self._client = aai.Transcriber()
        return self._client

    async def transcribe(
        self,
        audio: Path,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio file using the AssemblyAI API.

        The synchronous AssemblyAI SDK call is wrapped in run_in_executor
        to avoid blocking the event loop.

        Args:
            audio: Path to audio file (mp3, wav, etc.).
            language: Language code (e.g., "en", "es"). If None,
                language is auto-detected by AssemblyAI.
            **kwargs: Additional options:
                diarize (bool): Enable speaker diarization. Default False.
                auto_chapters (bool): Enable auto-chapters. Default False.
                sentiment_analysis (bool): Enable sentiment analysis. Default False.

        Returns:
            TranscriptionResult with full text and timed segments.
            When diarize=True, segments include speaker labels.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If transcription fails.
        """
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        diarize = kwargs.pop("diarize", False)
        auto_chapters = kwargs.pop("auto_chapters", False)
        sentiment_analysis = kwargs.pop("sentiment_analysis", False)

        client = self._get_client()

        import assemblyai as aai

        config_kwargs: dict[str, Any] = {
            "speaker_labels": diarize,
            "auto_chapters": auto_chapters,
            "sentiment_analysis": sentiment_analysis,
        }
        if language is not None:
            config_kwargs["language_code"] = language

        config = aai.TranscriptionConfig(**config_kwargs)

        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(
            None,
            partial(client.transcribe, str(audio), config=config),
        )

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

        return self._parse_transcript(transcript, language, diarize)

    def _parse_transcript(
        self,
        transcript: Any,
        language: str | None,
        diarize: bool,
    ) -> TranscriptionResult:
        """Parse AssemblyAI transcript into TranscriptionResult.

        Uses utterances when diarization is enabled (sentence-level segments
        with speaker labels), otherwise groups words into segments.

        Args:
            transcript: AssemblyAI Transcript object.
            language: Language code passed to the API.
            diarize: Whether diarization was requested.

        Returns:
            TranscriptionResult with segments and metadata.
        """
        full_text = transcript.text or ""
        segments = []

        # Prefer utterances when diarization is enabled
        utterances = getattr(transcript, "utterances", None)
        if diarize and utterances:
            for utt in utterances:
                speaker = f"SPEAKER_{utt.speaker}" if hasattr(utt, "speaker") else None
                segments.append(
                    TranscriptionSegment(
                        start=utt.start / 1000.0,
                        end=utt.end / 1000.0,
                        text=utt.text,
                        confidence=getattr(utt, "confidence", None),
                        speaker=speaker,
                    )
                )
        elif hasattr(transcript, "words") and transcript.words:
            segments = self._group_words_into_segments(transcript.words, diarize)

        duration = segments[-1].end if segments else None

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=language,
            duration=duration,
            provider="assemblyai",
        )

    @staticmethod
    def _group_words_into_segments(
        words: list,
        diarize: bool,
    ) -> list[TranscriptionSegment]:
        """Group individual words into sentence-like segments.

        Groups words by speaker (if diarizing) or by pauses between words.
        A new segment starts when there's a gap > 1 second or the speaker
        changes. AssemblyAI word timestamps are in milliseconds.

        Args:
            words: List of word objects from AssemblyAI transcript.
            diarize: Whether diarization was enabled.

        Returns:
            List of TranscriptionSegment objects.
        """
        if not words:
            return []

        segments = []
        current_words = [words[0]]
        current_speaker = getattr(words[0], "speaker", None) if diarize else None

        for word in words[1:]:
            word_speaker = getattr(word, "speaker", None) if diarize else None
            # AssemblyAI timestamps are in milliseconds
            gap = (word.start - current_words[-1].end) / 1000.0

            # Start new segment on speaker change or large gap
            if (diarize and word_speaker != current_speaker) or gap > 1.0:
                speaker_label = (
                    f"SPEAKER_{current_speaker}"
                    if diarize and current_speaker is not None
                    else None
                )
                segments.append(
                    TranscriptionSegment(
                        start=current_words[0].start / 1000.0,
                        end=current_words[-1].end / 1000.0,
                        text=" ".join(w.text for w in current_words),
                        speaker=speaker_label,
                    )
                )
                current_words = [word]
                current_speaker = word_speaker
            else:
                current_words.append(word)

        # Flush final segment
        if current_words:
            speaker_label = (
                f"SPEAKER_{current_speaker}"
                if diarize and current_speaker is not None
                else None
            )
            segments.append(
                TranscriptionSegment(
                    start=current_words[0].start / 1000.0,
                    end=current_words[-1].end / 1000.0,
                    text=" ".join(w.text for w in current_words),
                    speaker=speaker_label,
                )
            )

        return segments
