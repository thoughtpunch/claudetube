"""
claudetube.providers.openai - OpenAI provider.

Provides transcription (Whisper API), vision analysis (GPT-4o), and
reasoning via the OpenAI API. Handles automatic chunking for audio
files exceeding the 25MB Whisper API limit.

Example:
    >>> from claudetube.providers.registry import get_provider
    >>> provider = get_provider("openai")
    >>> result = await provider.transcribe(Path("audio.mp3"))
    >>> print(result.text)
"""

from claudetube.providers.openai.client import OpenaiProvider

__all__ = ["OpenaiProvider"]
