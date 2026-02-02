"""
claudetube.providers.assemblyai - AssemblyAI transcription provider.

Provides audio transcription with speaker diarization, auto-chapters,
and sentiment analysis via the AssemblyAI API.

Example:
    >>> from claudetube.providers.registry import get_provider
    >>> provider = get_provider("assemblyai")
    >>> result = await provider.transcribe(Path("audio.mp3"))
"""

from claudetube.providers.assemblyai.client import AssemblyAIProvider

__all__ = ["AssemblyAIProvider"]
