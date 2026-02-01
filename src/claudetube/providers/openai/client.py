"""
claudetube.providers.openai.client - OpenAI provider implementation.

Implements Transcriber (Whisper API), VisionAnalyzer (GPT-4o), and
Reasoner (chat completions) protocols. Handles the 25MB Whisper API
file limit with automatic audio chunking.

Example:
    >>> provider = OpenaiProvider()
    >>> result = await provider.transcribe(Path("audio.mp3"))
    >>> print(result.text)
    >>> result = await provider.analyze_images(
    ...     [Path("frame.jpg")],
    ...     prompt="Describe this scene",
    ... )
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import TYPE_CHECKING, Any

from claudetube.providers.base import Provider, Reasoner, Transcriber, VisionAnalyzer
from claudetube.providers.capabilities import PROVIDER_INFO, ProviderInfo
from claudetube.providers.types import TranscriptionResult, TranscriptionSegment

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

# Default models
DEFAULT_CHAT_MODEL = "gpt-4o"
DEFAULT_WHISPER_MODEL = "whisper-1"

# Maximum images per request (OpenAI limit)
MAX_IMAGES_PER_REQUEST = 10


def _detect_media_type(path: Path) -> str:
    """Detect image media type from file extension.

    Args:
        path: Path to image file.

    Returns:
        MIME type string. Defaults to "image/jpeg" for unknown extensions.
    """
    suffix = path.suffix.lower()
    return _MEDIA_TYPES.get(suffix, "image/jpeg")


def _encode_image(path: Path) -> str:
    """Encode image file as base64 string.

    Args:
        path: Path to image file.

    Returns:
        Base64-encoded image data.

    Raises:
        FileNotFoundError: If image file doesn't exist.
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class OpenaiProvider(Provider, Transcriber, VisionAnalyzer, Reasoner):
    """OpenAI provider for transcription, vision analysis, and reasoning.

    Supports:
    - Audio transcription via Whisper API with auto-chunking for large files
    - Image analysis via GPT-4o vision
    - Text reasoning/chat completion with structured output via response_format

    Args:
        model: Chat/vision model identifier. Defaults to "gpt-4o".
        whisper_model: Whisper model for transcription. Defaults to "whisper-1".
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        max_tokens: Default max tokens for chat responses.
    """

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._whisper_model = whisper_model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client = None

    @property
    def info(self) -> ProviderInfo:
        return PROVIDER_INFO["openai"]

    def _resolve_api_key(self) -> str | None:
        """Resolve API key from init arg or environment."""
        return self._api_key or os.environ.get("OPENAI_API_KEY")

    def is_available(self) -> bool:
        """Check if the OpenAI SDK is installed and API key is set."""
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return self._resolve_api_key() is not None

    def _get_client(self) -> Any:
        """Lazy-load the async OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            api_key = self._resolve_api_key()
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. "
                    "Set the environment variable or pass api_key to the provider."
                )
            self._client = AsyncOpenAI(api_key=api_key)
        return self._client

    async def transcribe(
        self,
        audio: Path,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio using the Whisper API.

        Automatically chunks files exceeding 25MB and merges results
        with corrected timestamps.

        Args:
            audio: Path to audio file (mp3, wav, etc.).
            language: Optional language code (e.g., "en", "es").
            **kwargs: Additional options:
                whisper_model (str): Override the default Whisper model.

        Returns:
            TranscriptionResult with full text and timed segments.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
        """
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        from claudetube.providers.openai.chunker import chunk_audio_if_needed

        whisper_model = kwargs.pop("whisper_model", self._whisper_model)
        client = self._get_client()

        chunks = await chunk_audio_if_needed(audio, max_size_mb=25)
        all_segments: list[TranscriptionSegment] = []
        full_text_parts: list[str] = []
        detected_language = language
        total_duration = None

        for chunk in chunks:
            with open(chunk.path, "rb") as f:
                api_kwargs: dict[str, Any] = {
                    "file": f,
                    "model": whisper_model,
                    "response_format": "verbose_json",
                }
                if language:
                    api_kwargs["language"] = language

                response = await client.audio.transcriptions.create(**api_kwargs)

            # Parse segments with offset correction
            for seg in response.segments or []:
                all_segments.append(
                    TranscriptionSegment(
                        start=seg["start"] + chunk.offset,
                        end=seg["end"] + chunk.offset,
                        text=seg["text"].strip(),
                    )
                )
            full_text_parts.append(response.text)

            # Capture language and duration from the response
            if not detected_language and hasattr(response, "language"):
                detected_language = response.language
            if hasattr(response, "duration") and response.duration is not None:
                total_duration = chunk.offset + response.duration

        return TranscriptionResult(
            text=" ".join(full_text_parts),
            segments=all_segments,
            language=detected_language,
            duration=total_duration,
            provider="openai",
        )

    async def analyze_images(
        self,
        images: list[Path],
        prompt: str,
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Analyze images using GPT-4o vision.

        Args:
            images: List of image file paths (max 10).
            prompt: Question or instruction about the images.
            schema: Optional Pydantic model for structured output via response_format.
            **kwargs: Additional options:
                model (str): Override the default model.
                max_tokens (int): Override default max tokens.

        Returns:
            str if no schema, dict if schema provided.

        Raises:
            FileNotFoundError: If any image file doesn't exist.
        """
        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        client = self._get_client()

        content: list[dict] = []
        for img in images[:MAX_IMAGES_PER_REQUEST]:
            if not img.exists():
                raise FileNotFoundError(f"Image file not found: {img}")
            media_type = _detect_media_type(img)
            b64 = _encode_image(img)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"},
                }
            )
        content.append({"type": "text", "text": prompt})

        api_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }

        if schema:
            api_kwargs["response_format"] = _build_response_format(schema)

        response = await client.chat.completions.create(**api_kwargs)
        result = response.choices[0].message.content

        if schema:
            return json.loads(result)
        return result

    async def reason(
        self,
        messages: list[dict],
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Generate text response using chat completions.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            schema: Optional Pydantic model for structured output via response_format.
            **kwargs: Additional options:
                model (str): Override the default model.
                max_tokens (int): Override default max tokens.

        Returns:
            str if no schema, dict if schema provided.
        """
        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        client = self._get_client()

        api_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if schema:
            api_kwargs["response_format"] = _build_response_format(schema)

        response = await client.chat.completions.create(**api_kwargs)
        result = response.choices[0].message.content

        if schema:
            return json.loads(result)
        return result


def _build_response_format(schema: type) -> dict:
    """Build OpenAI response_format from a Pydantic model.

    Args:
        schema: Pydantic BaseModel subclass with model_json_schema().

    Returns:
        Dict suitable for the response_format parameter.
    """
    if hasattr(schema, "model_json_schema"):
        json_schema = schema.model_json_schema()
    else:
        json_schema = {"type": "object"}

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": json_schema,
        },
    }
