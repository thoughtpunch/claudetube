"""
claudetube.providers.google - Google Gemini provider.

Provides vision analysis, native video analysis, and reasoning via the
Gemini API. Supports structured output via response_schema.

Example:
    >>> from claudetube.providers.registry import get_provider
    >>> provider = get_provider("google")
    >>> result = await provider.analyze_video(
    ...     Path("video.mp4"),
    ...     prompt="What happens at 2:30?",
    ...     start_time=140.0,
    ...     end_time=160.0,
    ... )
"""

from claudetube.providers.google.client import GoogleProvider

__all__ = ["GoogleProvider"]
