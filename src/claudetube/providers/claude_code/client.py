"""
claudetube.providers.claude_code.client - ClaudeCodeProvider implementation.

This provider is unique: it doesn't make external API calls. Instead, it
formats content (images, messages) for the host Claude instance to process
within the ongoing conversation context.

Example:
    >>> provider = ClaudeCodeProvider()
    >>> provider.is_available()
    True
    >>> result = await provider.analyze_images(
    ...     [Path("frame.jpg")],
    ...     prompt="Describe this frame",
    ... )
    >>> print(result)
    [Image: /abs/path/to/frame.jpg]
    <BLANKLINE>
    Describe this frame
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from claudetube.providers.base import Provider, Reasoner, VisionAnalyzer
from claudetube.providers.capabilities import PROVIDER_INFO, ProviderInfo

if TYPE_CHECKING:
    from pathlib import Path


class ClaudeCodeProvider(Provider, VisionAnalyzer, Reasoner):
    """Provider that uses the host Claude instance in Claude Code.

    This provider doesn't make API calls - instead, it formats content
    for the host AI to process in the conversation context.

    Always available when running inside Claude Code, requires no API key.
    """

    @property
    def info(self) -> ProviderInfo:
        return PROVIDER_INFO["claude-code"]

    def is_available(self) -> bool:
        """Check if running inside Claude Code.

        Detection methods:
        1. MCP_SERVER env var (set by claudetube MCP server)
        2. CLAUDE_CODE env var
        3. Default to True (safe fallback)
        """
        if os.environ.get("MCP_SERVER") == "1":
            return True
        if os.environ.get("CLAUDE_CODE") == "1":
            return True
        # Fallback: assume available (safe default)
        return True

    async def analyze_images(
        self,
        images: list[Path],
        prompt: str,
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Format images for host AI to analyze.

        Returns a formatted string with image references that Claude Code
        will render and analyze in the conversation.

        Args:
            images: List of image file paths. Paths must be absolute and exist.
            prompt: Question or instruction about the images.
            schema: Optional Pydantic model for structured output.
            **kwargs: Ignored (no provider-specific options).

        Returns:
            Formatted string with image references and prompt.
        """
        image_refs = []
        for img in images:
            abs_path = img.resolve()
            if abs_path.exists():
                image_refs.append(f"[Image: {abs_path}]")
            else:
                image_refs.append(f"[Image not found: {abs_path}]")

        content = "\n".join(image_refs)

        if schema:
            schema_json = self._get_schema_json(schema)
            content += (
                f"\n\n{prompt}"
                f"\n\nRespond with JSON matching this schema:\n```json\n{schema_json}\n```"
            )
        else:
            content += f"\n\n{prompt}"

        return content

    async def reason(
        self,
        messages: list[dict],
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Format messages for host AI to process.

        Returns a formatted prompt that the host Claude will respond to.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            schema: Optional Pydantic model for structured output.
            **kwargs: Ignored (no provider-specific options).

        Returns:
            Formatted string with role-labeled messages.
        """
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted_parts.append(f"[System]: {content}")
            elif role == "assistant":
                formatted_parts.append(f"[Previous response]: {content}")
            else:
                formatted_parts.append(content)

        content = "\n\n".join(formatted_parts)

        if schema:
            schema_json = self._get_schema_json(schema)
            content += f"\n\nRespond with JSON matching this schema:\n```json\n{schema_json}\n```"

        return content

    @staticmethod
    def _get_schema_json(schema: type) -> str:
        """Extract JSON schema from a Pydantic model or type.

        Args:
            schema: Pydantic BaseModel subclass or other type.

        Returns:
            JSON string of the schema.
        """
        if hasattr(schema, "model_json_schema"):
            return json.dumps(schema.model_json_schema(), indent=2)
        return str(schema)
