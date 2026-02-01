"""
claudetube.providers.whisper_local - Local Whisper transcription provider.

Wraps the existing WhisperTool (faster-whisper) in the Provider/Transcriber
interface for use with the provider registry.

Example:
    >>> from claudetube.providers.registry import get_provider
    >>> provider = get_provider("whisper-local", model_size="small")
    >>> result = await provider.transcribe(Path("audio.mp3"))
    >>> print(result.text)
"""

from __future__ import annotations

import asyncio
import logging
import re
from functools import partial
from typing import TYPE_CHECKING

from claudetube.providers.base import Provider, Transcriber

if TYPE_CHECKING:
    from pathlib import Path
from claudetube.providers.capabilities import PROVIDER_INFO, ProviderInfo
from claudetube.providers.types import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


def _parse_srt_time(time_str: str) -> float:
    """Parse SRT timestamp to seconds.

    Args:
        time_str: SRT timestamp like "00:01:23,456".

    Returns:
        Time in seconds.
    """
    match = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", time_str.strip())
    if not match:
        return 0.0
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _parse_srt(srt_text: str) -> list[TranscriptionSegment]:
    """Parse SRT text into TranscriptionSegment list.

    Args:
        srt_text: SRT-formatted subtitle string.

    Returns:
        List of TranscriptionSegment objects.
    """
    segments = []
    blocks = re.split(r"\n\n+", srt_text.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # Line 0: sequence number (skip)
        # Line 1: timestamps
        time_match = re.match(
            r"(.+?)\s*-->\s*(.+)",
            lines[1].strip(),
        )
        if not time_match:
            continue

        start = _parse_srt_time(time_match.group(1))
        end = _parse_srt_time(time_match.group(2))
        text = "\n".join(lines[2:]).strip()

        if text:
            segments.append(TranscriptionSegment(start=start, end=end, text=text))

    return segments


class WhisperLocalProvider(Provider, Transcriber):
    """Local faster-whisper transcription provider.

    Wraps the existing WhisperTool for compatibility with the provider
    architecture. The underlying faster-whisper model is lazy-loaded on
    first use.

    Args:
        model_size: Whisper model size ("tiny", "base", "small", "medium", "large").
            Defaults to "small" for good quality/speed balance.
        language: Default language code. Defaults to "en".
        use_batched: Use batched inference for faster multi-core transcription.
    """

    def __init__(
        self,
        model_size: str = "small",
        language: str = "en",
        use_batched: bool = True,
    ):
        self._model_size = model_size
        self._language = language
        self._use_batched = use_batched
        self._tool = None

    @property
    def info(self) -> ProviderInfo:
        return PROVIDER_INFO["whisper-local"]

    def is_available(self) -> bool:
        """Check if faster-whisper is installed."""
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_tool(self):
        """Lazy-load the WhisperTool instance."""
        if self._tool is None:
            from claudetube.tools.whisper import WhisperTool

            self._tool = WhisperTool(model_size=self._model_size)
        return self._tool

    async def transcribe(
        self,
        audio: Path,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio file to text with timestamps.

        Runs the synchronous WhisperTool in a thread executor to avoid
        blocking the event loop.

        Args:
            audio: Path to audio file (mp3, wav, etc.).
            language: Language code (e.g., "en", "es"). Uses instance default if None.
            **kwargs: Additional options passed to WhisperTool.transcribe().
                Supported: use_batched (bool).

        Returns:
            TranscriptionResult with full text and timed segments.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            TranscriptionError: If transcription fails.
        """
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        lang = language or self._language
        use_batched = kwargs.pop("use_batched", self._use_batched)
        tool = self._get_tool()

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(tool.transcribe, audio, language=lang, use_batched=use_batched),
        )

        srt_text = result["srt"]
        txt_text = result["txt"]
        segments = _parse_srt(srt_text)

        duration = segments[-1].end if segments else None

        return TranscriptionResult(
            text=txt_text,
            segments=segments,
            language=lang,
            duration=duration,
            provider="whisper-local",
        )
