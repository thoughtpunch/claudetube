"""
claudetube.providers.deepgram - Deepgram transcription provider.

Provides audio transcription with speaker diarization via the Deepgram API.
Uses the Nova-2 model by default for high-quality transcription.

Example:
    >>> from claudetube.providers.registry import get_provider
    >>> provider = get_provider("deepgram")
    >>> result = await provider.transcribe(Path("audio.mp3"))
"""

from claudetube.providers.deepgram.client import DeepgramProvider

__all__ = ["DeepgramProvider"]
