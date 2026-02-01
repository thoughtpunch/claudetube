"""
claudetube.providers.anthropic.client - AnthropicProvider implementation.

Uses the Anthropic Messages API for vision analysis and text reasoning.
Structured output is achieved via tool_choice (Anthropic's approach).

Example:
    >>> provider = AnthropicProvider()
    >>> result = await provider.analyze_images(
    ...     [Path("frame.jpg")],
    ...     prompt="Describe this scene",
    ... )
    >>> print(result)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import TYPE_CHECKING, Any

from claudetube.providers.base import Provider, Reasoner, VisionAnalyzer
from claudetube.providers.capabilities import PROVIDER_INFO, ProviderInfo

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# File extension -> MIME type mapping for images
_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Default model for vision and reasoning tasks
DEFAULT_MODEL = "claude-sonnet-4-20250514"


def _detect_media_type(path: Path) -> str:
    """Detect image media type from file extension.

    Args:
        path: Path to image file.

    Returns:
        MIME type string. Defaults to "image/jpeg" for unknown extensions.
    """
    suffix = path.suffix.lower()
    return _MEDIA_TYPES.get(suffix, "image/jpeg")


def _load_image_base64(path: Path) -> str:
    """Load image file as base64 string.

    Args:
        path: Path to image file.

    Returns:
        Base64-encoded image data.

    Raises:
        FileNotFoundError: If image file doesn't exist.
    """
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _schema_to_tool(schema: type) -> dict:
    """Convert a Pydantic model to an Anthropic tool definition.

    Anthropic uses tool_choice to force structured output. The schema
    is wrapped as a tool's input_schema.

    Args:
        schema: Pydantic BaseModel subclass with model_json_schema().

    Returns:
        Tool definition dict for the Anthropic API.
    """
    if hasattr(schema, "model_json_schema"):
        json_schema = schema.model_json_schema()
    else:
        json_schema = {"type": "object"}

    return {
        "name": "structured_output",
        "description": "Return structured data matching the schema.",
        "input_schema": json_schema,
    }


class AnthropicProvider(Provider, VisionAnalyzer, Reasoner):
    """Anthropic Claude provider for vision analysis and reasoning.

    Uses the Anthropic Messages API with support for:
    - Multi-image analysis (up to 20 images per request)
    - Structured output via tool_choice
    - Multiple model selection (haiku, sonnet, opus)

    Args:
        model: Model identifier. Defaults to claude-sonnet-4-20250514.
        api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
        max_tokens: Default max tokens for responses.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client = None

    @property
    def info(self) -> ProviderInfo:
        return PROVIDER_INFO["anthropic"]

    def _resolve_api_key(self) -> str | None:
        """Resolve API key from init arg or environment."""
        return self._api_key or os.environ.get("ANTHROPIC_API_KEY")

    def is_available(self) -> bool:
        """Check if the Anthropic SDK is installed and API key is set."""
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return self._resolve_api_key() is not None

    def _get_client(self) -> Any:
        """Lazy-load the async Anthropic client."""
        if self._client is None:
            from anthropic import AsyncAnthropic

            api_key = self._resolve_api_key()
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. "
                    "Set the environment variable or pass api_key to the provider."
                )
            self._client = AsyncAnthropic(api_key=api_key)
        return self._client

    def _build_image_blocks(self, images: list[Path]) -> list[dict]:
        """Build Anthropic image content blocks from file paths.

        Args:
            images: List of image file paths.

        Returns:
            List of image content block dicts for the Messages API.

        Raises:
            FileNotFoundError: If any image file doesn't exist.
        """
        blocks = []
        for img in images:
            if not img.exists():
                raise FileNotFoundError(f"Image file not found: {img}")
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _detect_media_type(img),
                        "data": _load_image_base64(img),
                    },
                }
            )
        return blocks

    async def analyze_images(
        self,
        images: list[Path],
        prompt: str,
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Analyze images using the Anthropic Messages API.

        Args:
            images: List of image file paths (max 20).
            prompt: Question or instruction about the images.
            schema: Optional Pydantic model for structured output via tool_choice.
            **kwargs: Additional options:
                model (str): Override the default model.
                max_tokens (int): Override default max tokens.

        Returns:
            str if no schema, dict if schema provided.

        Raises:
            FileNotFoundError: If any image file doesn't exist.
            anthropic.APIError: If the API call fails.
        """
        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        client = self._get_client()

        content = self._build_image_blocks(images)
        content.append({"type": "text", "text": prompt})

        if schema:
            return await self._call_with_tool(
                client, model, max_tokens, content, schema
            )

        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )

        return response.content[0].text

    async def reason(
        self,
        messages: list[dict],
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Generate text response using the Anthropic Messages API.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                A "system" role message is extracted as the system parameter.
            schema: Optional Pydantic model for structured output via tool_choice.
            **kwargs: Additional options:
                model (str): Override the default model.
                max_tokens (int): Override default max tokens.

        Returns:
            str if no schema, dict if schema provided.

        Raises:
            anthropic.APIError: If the API call fails.
        """
        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        client = self._get_client()

        # Extract system message (Anthropic uses a separate system parameter)
        system_prompt = None
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg["content"]
            else:
                api_messages.append(msg)

        api_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if system_prompt:
            api_kwargs["system"] = system_prompt

        if schema:
            tool = _schema_to_tool(schema)
            api_kwargs["tools"] = [tool]
            api_kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}

            response = await client.messages.create(**api_kwargs)
            return self._extract_tool_result(response)

        response = await client.messages.create(**api_kwargs)
        return response.content[0].text

    async def _call_with_tool(
        self,
        client: Any,
        model: str,
        max_tokens: int,
        content: list[dict],
        schema: type,
    ) -> dict:
        """Force structured output via tool_choice.

        Wraps the schema as a tool definition and forces Claude to call it,
        extracting the structured result from the tool_use block.

        Args:
            client: AsyncAnthropic client.
            model: Model identifier.
            max_tokens: Max tokens for response.
            content: Message content blocks.
            schema: Pydantic model for the output schema.

        Returns:
            Dictionary matching the schema.
        """
        tool = _schema_to_tool(schema)

        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "structured_output"},
        )

        return self._extract_tool_result(response)

    @staticmethod
    def _extract_tool_result(response: Any) -> dict:
        """Extract structured data from a tool_use response.

        The Anthropic API returns tool_use blocks with the structured
        data in the `input` field.

        Args:
            response: Anthropic Messages API response.

        Returns:
            Dictionary with the structured output.

        Raises:
            ValueError: If no tool_use block is found in the response.
        """
        for block in response.content:
            if block.type == "tool_use":
                return block.input

        # Fallback: try to parse text content as JSON
        for block in response.content:
            if block.type == "text":
                try:
                    return json.loads(block.text)
                except json.JSONDecodeError:
                    pass

        raise ValueError(
            "No structured output found in response. "
            "Expected a tool_use block but got: "
            f"{[b.type for b in response.content]}"
        )
