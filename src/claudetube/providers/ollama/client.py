"""
claudetube.providers.ollama.client - OllamaProvider implementation.

Uses a local Ollama server for vision analysis (LLaVA) and text reasoning
(Llama). Fully offline, no API keys required.

Example:
    >>> provider = OllamaProvider()
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

# Default models for vision and reasoning tasks
DEFAULT_VISION_MODEL = "llava:13b"
DEFAULT_REASON_MODEL = "llama3.2"


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
        return base64.b64encode(f.read()).decode("utf-8")


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


class OllamaProvider(Provider, VisionAnalyzer, Reasoner):
    """Ollama local LLM provider for vision analysis and reasoning.

    Uses a local Ollama server with support for:
    - Vision analysis via LLaVA (single image at a time)
    - Text reasoning via Llama or other local models
    - Structured output via prompt-based JSON requests

    Note: LLaVA only supports one image per request. If multiple images
    are provided, only the first is sent with the image data and the
    prompt is updated to mention additional images.

    Args:
        vision_model: Model for vision tasks. Defaults to "llava:13b".
        reason_model: Model for reasoning tasks. Defaults to "llama3.2".
        host: Ollama server URL. Defaults to OLLAMA_HOST env or localhost:11434.
    """

    def __init__(
        self,
        vision_model: str = DEFAULT_VISION_MODEL,
        reason_model: str = DEFAULT_REASON_MODEL,
        host: str | None = None,
    ):
        self._vision_model = vision_model
        self._reason_model = reason_model
        self._host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client: Any = None

    @property
    def info(self) -> ProviderInfo:
        return PROVIDER_INFO["ollama"]

    def is_available(self) -> bool:
        """Check if the Ollama SDK is installed and server may be reachable.

        Does a quick sync check: verifies the ollama package is importable
        and that the host is configured (via constructor, env var, or default).
        Does NOT actually ping the server to keep this fast and synchronous.
        """
        try:
            import ollama  # noqa: F401
        except ImportError:
            return False
        # SDK is installed and we have a host configured (always true since
        # we default to localhost:11434). Return True to indicate the provider
        # is potentially available.
        return True

    def _get_client(self) -> Any:
        """Lazy-load the async Ollama client."""
        if self._client is None:
            from ollama import AsyncClient

            self._client = AsyncClient(host=self._host)
        return self._client

    async def analyze_images(
        self,
        images: list[Path],
        prompt: str,
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Analyze images using the Ollama API with a vision model.

        LLaVA only supports a single image per request. If multiple images
        are provided, only the first is analyzed and the prompt is updated
        to note additional images were provided.

        Args:
            images: List of image file paths. Only the first is sent to the model.
            prompt: Question or instruction about the images.
            schema: Optional Pydantic model for structured output. If provided,
                the prompt is augmented to request JSON matching the schema.
            **kwargs: Additional options:
                model (str): Override the default vision model.

        Returns:
            str if no schema, dict if schema provided and JSON parseable.

        Raises:
            FileNotFoundError: If the first image file doesn't exist.
        """
        model = kwargs.pop("model", self._vision_model)
        client = self._get_client()

        # LLaVA only supports 1 image — use the first, mention others in prompt
        if not images:
            raise ValueError("At least one image is required for analyze_images()")

        first_image = images[0]
        if not first_image.exists():
            raise FileNotFoundError(f"Image file not found: {first_image}")

        image_data = _load_image_base64(first_image)

        # Update prompt if multiple images were provided
        effective_prompt = prompt
        if len(images) > 1:
            extra_count = len(images) - 1
            extra_names = ", ".join(str(img.name) for img in images[1:])
            effective_prompt = (
                f"{prompt}\n\n"
                f"Note: {extra_count} additional image(s) were provided "
                f"({extra_names}) but only the first image is being analyzed "
                f"due to model limitations."
            )

        # Append schema request to prompt if provided
        if schema:
            schema_json = _get_schema_json(schema)
            effective_prompt += (
                f"\n\nRespond with JSON matching this schema:\n"
                f"```json\n{schema_json}\n```"
            )

        response = await client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": effective_prompt,
                    "images": [image_data],
                },
            ],
        )

        text = response["message"]["content"]

        if schema:
            return self._try_parse_json(text)

        return text

    async def reason(
        self,
        messages: list[dict],
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Generate text response using the Ollama API.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                A "system" role message is passed through to Ollama directly.
            schema: Optional Pydantic model for structured output. If provided,
                the last user message is augmented to request JSON.
            **kwargs: Additional options:
                model (str): Override the default reasoning model.

        Returns:
            str if no schema, dict if schema provided and JSON parseable.
        """
        model = kwargs.pop("model", self._reason_model)
        client = self._get_client()

        # Build messages list, keeping system messages as-is (Ollama supports them)
        api_messages = []
        for msg in messages:
            api_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        # Append schema request to the last user message if provided
        if schema and api_messages:
            schema_json = _get_schema_json(schema)
            schema_suffix = (
                f"\n\nRespond with JSON matching this schema:\n"
                f"```json\n{schema_json}\n```"
            )
            # Find and augment the last user message
            for i in range(len(api_messages) - 1, -1, -1):
                if api_messages[i]["role"] == "user":
                    api_messages[i] = {
                        "role": "user",
                        "content": api_messages[i]["content"] + schema_suffix,
                    }
                    break

        response = await client.chat(
            model=model,
            messages=api_messages,
        )

        text = response["message"]["content"]

        if schema:
            return self._try_parse_json(text)

        return text

    @staticmethod
    def _try_parse_json(text: str) -> str | dict:
        """Try to parse text as JSON, return raw text on failure.

        Handles cases where the model wraps JSON in markdown code blocks.

        Args:
            text: Response text that may contain JSON.

        Returns:
            Parsed dict if valid JSON, original text otherwise.
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if len(lines) >= 3 and lines[-1].strip() == "```":
                inner = "\n".join(lines[1:-1])
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    pass

        return text
