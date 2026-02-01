"""
claudetube.providers.claude_code - Host Claude Code provider.

Formats content for the host Claude instance to process in-conversation,
rather than making external API calls. Always available when running
inside Claude Code.

Example:
    >>> from claudetube.providers.registry import get_provider
    >>> provider = get_provider("claude-code")
    >>> result = await provider.analyze_images(
    ...     [Path("frame1.jpg")],
    ...     prompt="What is shown in this frame?",
    ... )
"""

from claudetube.providers.claude_code.client import ClaudeCodeProvider

__all__ = ["ClaudeCodeProvider"]
