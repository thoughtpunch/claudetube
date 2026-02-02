"""
claudetube.providers.deepgram.client - DeepgramProvider implementation.

Uses the Deepgram SDK for audio transcription with optional speaker
diarization support. Wraps the prerecorded (batch) transcription API.

Example:
    >>> provider = DeepgramProvider()
    >>> result = await provider.transcribe(
    ...     Path("audio.mp3"),
    ...     diarize=True,
    ... )
    >>> for seg in result.segments:
    ...     print(f"[{seg.speaker}] {seg.text}")
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from claudetube.providers.base import Provider, StreamingTranscriber, Transcriber
from claudetube.providers.capabilities import PROVIDER_INFO, ProviderInfo
from claudetube.providers.types import (
    StreamingEventType,
    StreamingTranscriptionEvent,
    TranscriptionResult,
    TranscriptionSegment,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

logger = logging.getLogger(__name__)

# Default model for Deepgram transcription
DEFAULT_MODEL = "nova-2"


class DeepgramProvider(Provider, Transcriber, StreamingTranscriber):
    """Deepgram transcription provider with speaker diarization and streaming.

    Uses the Deepgram prerecorded (batch) and live (streaming) transcription
    APIs with support for:
    - Multiple languages (auto-detect or specified)
    - Speaker diarization (identify who is speaking)
    - Utterance-level segmentation for natural sentence boundaries
    - Real-time streaming transcription via WebSocket

    Args:
        model: Deepgram model identifier. Defaults to "nova-2".
        api_key: Deepgram API key. Defaults to DEEPGRAM_API_KEY env var.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ):
        self._model = model
        self._api_key = api_key
        self._client = None

    @property
    def info(self) -> ProviderInfo:
        return PROVIDER_INFO["deepgram"]

    def _resolve_api_key(self) -> str | None:
        """Resolve API key from init arg or environment."""
        return self._api_key or os.environ.get("DEEPGRAM_API_KEY")

    def is_available(self) -> bool:
        """Check if the Deepgram SDK is installed and API key is set."""
        try:
            import deepgram  # noqa: F401
        except ImportError:
            return False
        return self._resolve_api_key() is not None

    def _get_client(self) -> Any:
        """Lazy-load the Deepgram client."""
        if self._client is None:
            from deepgram import DeepgramClient

            api_key = self._resolve_api_key()
            if not api_key:
                raise ValueError(
                    "DEEPGRAM_API_KEY not set. "
                    "Set the environment variable or pass api_key to the provider."
                )
            self._client = DeepgramClient(api_key)
        return self._client

    async def transcribe(
        self,
        audio: Path,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio file using the Deepgram prerecorded API.

        Args:
            audio: Path to audio file (mp3, wav, etc.).
            language: Language code (e.g., "en", "es"). If None,
                language is auto-detected by Deepgram.
            **kwargs: Additional options:
                diarize (bool): Enable speaker diarization. Default False.
                model (str): Override the default model.

        Returns:
            TranscriptionResult with full text and timed segments.
            When diarize=True, segments include speaker labels.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
        """
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        model = kwargs.pop("model", self._model)
        diarize = kwargs.pop("diarize", False)

        client = self._get_client()

        # Read audio file
        with open(audio, "rb") as f:
            audio_data = f.read()

        from deepgram import PrerecordedOptions

        options = PrerecordedOptions(
            model=model,
            diarize=diarize,
            utterances=True,
        )
        if language is not None:
            options.language = language

        payload = {"buffer": audio_data}

        response = await client.listen.asyncprerecorded.v("1").transcribe_file(
            payload, options
        )

        return self._parse_response(response, language, diarize)

    async def stream_transcribe(
        self,
        audio: Path,
        language: str | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamingTranscriptionEvent]:
        """Stream transcription events from an audio file via Deepgram Live API.

        Reads the audio file and sends it in chunks over a WebSocket to
        Deepgram's live transcription API, yielding events as segments
        are recognized.

        Args:
            audio: Path to audio file (mp3, wav, etc.).
            language: Language code (e.g., "en", "es"). If None, auto-detected.
            **kwargs: Additional options:
                model (str): Override the default model.
                diarize (bool): Enable speaker diarization. Default False.
                chunk_size (int): Bytes per chunk. Default 8192.

        Yields:
            StreamingTranscriptionEvent with PARTIAL, FINAL, COMPLETE, or
            ERROR event types.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
        """
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        model = kwargs.pop("model", self._model)
        diarize = kwargs.pop("diarize", False)
        chunk_size = kwargs.pop("chunk_size", 8192)

        client = self._get_client()
        accumulated_text = ""
        event_queue: asyncio.Queue[StreamingTranscriptionEvent] = asyncio.Queue()

        from deepgram import LiveOptions, LiveTranscriptionEvents

        def _on_transcript(_, result, **kw):
            """Handle transcript events from Deepgram WebSocket."""
            channel = result.channel
            alt = channel.alternatives[0] if channel.alternatives else None
            if alt is None or not alt.transcript:
                return

            is_final = result.is_final
            speaker = None
            if diarize and hasattr(alt, "words") and alt.words:
                first_word = alt.words[0]
                if hasattr(first_word, "speaker"):
                    speaker = f"SPEAKER_{first_word.speaker}"

            segment = TranscriptionSegment(
                start=result.start,
                end=result.start + result.duration,
                text=alt.transcript,
                confidence=alt.confidence if hasattr(alt, "confidence") else None,
                speaker=speaker,
            )

            event_type = (
                StreamingEventType.FINAL if is_final else StreamingEventType.PARTIAL
            )
            event_queue.put_nowait(
                StreamingTranscriptionEvent(
                    event_type=event_type,
                    segment=segment,
                    is_final=is_final,
                )
            )

        def _on_error(_, error, **kw):
            """Handle error events from Deepgram WebSocket."""
            event_queue.put_nowait(
                StreamingTranscriptionEvent(
                    event_type=StreamingEventType.ERROR,
                    error=str(error),
                )
            )

        # Set up live connection
        connection = client.listen.asynclive.v("1")
        connection.on(LiveTranscriptionEvents.Transcript, _on_transcript)
        connection.on(LiveTranscriptionEvents.Error, _on_error)

        options = LiveOptions(
            model=model,
            diarize=diarize,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
        )
        if language is not None:
            options.language = language

        started = await connection.start(options)
        if not started:
            yield StreamingTranscriptionEvent(
                event_type=StreamingEventType.ERROR,
                error="Failed to start Deepgram live connection",
            )
            return

        # Send audio in chunks
        try:
            with open(audio, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    await connection.send(chunk)
                    # Small delay to avoid overwhelming the WebSocket
                    await asyncio.sleep(0.01)

                    # Yield any queued events
                    while not event_queue.empty():
                        event = event_queue.get_nowait()
                        if (
                            event.event_type == StreamingEventType.FINAL
                            and event.segment
                        ):
                            accumulated_text += (
                                " " if accumulated_text else ""
                            ) + event.segment.text
                            event.accumulated_text = accumulated_text
                        yield event

            # Signal end of audio
            await connection.finish()

            # Brief wait for final events
            await asyncio.sleep(0.5)

            # Drain remaining events
            while not event_queue.empty():
                event = event_queue.get_nowait()
                if event.event_type == StreamingEventType.FINAL and event.segment:
                    accumulated_text += (
                        " " if accumulated_text else ""
                    ) + event.segment.text
                    event.accumulated_text = accumulated_text
                yield event

        except Exception as exc:
            yield StreamingTranscriptionEvent(
                event_type=StreamingEventType.ERROR,
                error=str(exc),
            )

        # Emit completion event
        yield StreamingTranscriptionEvent(
            event_type=StreamingEventType.COMPLETE,
            accumulated_text=accumulated_text,
        )

    def _parse_response(
        self,
        response: Any,
        language: str | None,
        diarize: bool,
    ) -> TranscriptionResult:
        """Parse Deepgram API response into TranscriptionResult.

        Uses utterances when available (sentence-level segments with natural
        boundaries), falls back to grouping individual words.

        Args:
            response: Deepgram prerecorded API response.
            language: Language code passed to the API.
            diarize: Whether diarization was requested.

        Returns:
            TranscriptionResult with segments and metadata.
        """
        channel = response.results.channels[0]
        alternative = channel.alternatives[0]
        full_text = alternative.transcript

        segments = []

        # Prefer utterances (sentence-level segments)
        utterances = getattr(response.results, "utterances", None)
        if utterances:
            for utt in utterances:
                speaker = None
                if diarize and hasattr(utt, "speaker"):
                    speaker = f"SPEAKER_{utt.speaker}"

                segments.append(
                    TranscriptionSegment(
                        start=utt.start,
                        end=utt.end,
                        text=utt.transcript,
                        confidence=getattr(utt, "confidence", None),
                        speaker=speaker,
                    )
                )
        elif hasattr(alternative, "words") and alternative.words:
            # Fall back to grouping words into segments
            segments = self._group_words_into_segments(alternative.words, diarize)

        duration = segments[-1].end if segments else None

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=language,
            duration=duration,
            provider="deepgram",
        )

    @staticmethod
    def _group_words_into_segments(
        words: list,
        diarize: bool,
    ) -> list[TranscriptionSegment]:
        """Group individual words into sentence-like segments.

        Groups words by speaker (if diarizing) or by pauses between words.
        A new segment starts when there's a gap > 1 second or the speaker
        changes.

        Args:
            words: List of word objects from Deepgram response.
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
            gap = word.start - current_words[-1].end

            # Start new segment on speaker change or large gap
            if (diarize and word_speaker != current_speaker) or gap > 1.0:
                # Flush current segment
                speaker_label = (
                    f"SPEAKER_{current_speaker}"
                    if diarize and current_speaker is not None
                    else None
                )
                segments.append(
                    TranscriptionSegment(
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        text=" ".join(w.word for w in current_words),
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
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    text=" ".join(w.word for w in current_words),
                    speaker=speaker_label,
                )
            )

        return segments
