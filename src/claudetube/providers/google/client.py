"""
claudetube.providers.google.client - GoogleProvider implementation.

Uses the Google Generative AI SDK (google-generativeai) for vision analysis,
native video analysis, and text reasoning. The Gemini SDK is mostly synchronous,
so all API calls are wrapped in asyncio.run_in_executor().

Key differences from other providers:
- Only provider implementing VideoAnalyzer (native video upload)
- Uses PIL Image.open() for image loading (not base64)
- Uses response_schema with Pydantic models for structured output
- Video upload via File API requires polling for ACTIVE state

Example:
    >>> provider = GoogleProvider()
    >>> result = await provider.analyze_images(
    ...     [Path("frame.jpg")],
    ...     prompt="Describe this scene",
    ... )
    >>> print(result)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Any

from claudetube.providers.base import Provider, Reasoner, VideoAnalyzer, VisionAnalyzer
from claudetube.providers.capabilities import PROVIDER_INFO, ProviderInfo

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Default model for all tasks
DEFAULT_MODEL = "gemini-2.0-flash"


class GoogleProvider(Provider, VisionAnalyzer, VideoAnalyzer, Reasoner):
    """Google Gemini provider for vision, video analysis, and reasoning.

    Supports:
    - Image analysis via Gemini vision (PIL-based image loading)
    - Native video analysis via File API upload (up to 2GB)
    - Text reasoning/chat completion
    - Structured output via response_schema with Pydantic models

    Args:
        model: Model identifier. Defaults to "gemini-2.0-flash".
        api_key: Google API key. Defaults to GOOGLE_API_KEY env var.
        max_tokens: Default max output tokens for responses.
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
        self._genai = None

    @property
    def info(self) -> ProviderInfo:
        return PROVIDER_INFO["google"]

    def _resolve_api_key(self) -> str | None:
        """Resolve API key from init arg or environment."""
        return self._api_key or os.environ.get("GOOGLE_API_KEY")

    def is_available(self) -> bool:
        """Check if the google-generativeai SDK is installed and API key is set."""
        try:
            import google.generativeai  # noqa: F401
        except ImportError:
            return False
        return self._resolve_api_key() is not None

    def _get_genai(self) -> Any:
        """Lazy-load and configure the Google GenAI module."""
        if self._genai is None:
            import google.generativeai as genai

            api_key = self._resolve_api_key()
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not set. "
                    "Set the environment variable or pass api_key to the provider."
                )
            genai.configure(api_key=api_key)
            self._genai = genai
        return self._genai

    def _get_model(
        self,
        model_name: str,
        schema: type | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Get a Gemini GenerativeModel with optional structured output config.

        Args:
            model_name: Gemini model identifier.
            schema: Optional Pydantic model for structured output.
            max_tokens: Optional max output tokens.

        Returns:
            A GenerativeModel instance.
        """
        genai = self._get_genai()
        config_kwargs: dict[str, Any] = {}

        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        if schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schema

        if config_kwargs:
            config = genai.GenerationConfig(**config_kwargs)
            return genai.GenerativeModel(model_name, generation_config=config)
        return genai.GenerativeModel(model_name)

    async def _generate_async(self, model: Any, content: list) -> Any:
        """Run synchronous generate_content in executor.

        Args:
            model: GenerativeModel instance.
            content: List of content parts (images, text, video files).

        Returns:
            Gemini API response.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: model.generate_content(content),
        )

    async def _send_message_async(self, chat: Any, message: str) -> Any:
        """Run synchronous send_message in executor.

        Args:
            chat: Gemini ChatSession instance.
            message: Message text to send.

        Returns:
            Gemini API response.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: chat.send_message(message),
        )

    async def _upload_video(self, video_path: Path) -> Any:
        """Upload video to Gemini File API and wait for processing.

        Uploads the video file, then polls until the file state becomes
        ACTIVE. Raises RuntimeError if processing fails.

        Args:
            video_path: Path to video file.

        Returns:
            Gemini File object in ACTIVE state.

        Raises:
            RuntimeError: If video processing fails (state != ACTIVE).
            FileNotFoundError: If video file doesn't exist.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        genai = self._get_genai()
        loop = asyncio.get_running_loop()

        # Upload is synchronous in the SDK
        video_file = await loop.run_in_executor(
            None,
            lambda: genai.upload_file(str(video_path)),
        )

        # Poll until processing completes
        while video_file.state.name == "PROCESSING":
            await asyncio.sleep(2)
            file_name = video_file.name
            video_file = await loop.run_in_executor(
                None,
                lambda f=file_name: genai.get_file(f),  # type: ignore[misc]
            )

        if video_file.state.name != "ACTIVE":
            raise RuntimeError(
                f"Video processing failed with state: {video_file.state.name}"
            )

        return video_file

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS.

        Args:
            seconds: Time in seconds.

        Returns:
            Formatted time string (e.g., "02:30").
        """
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    async def analyze_images(
        self,
        images: list[Path],
        prompt: str,
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Analyze images using Gemini vision.

        Uses PIL Image.open() to load images, which Gemini accepts natively.

        Args:
            images: List of image file paths.
            prompt: Question or instruction about the images.
            schema: Optional Pydantic model for structured output via response_schema.
            **kwargs: Additional options:
                model (str): Override the default model.
                max_tokens (int): Override default max tokens.

        Returns:
            str if no schema, dict if schema provided.

        Raises:
            FileNotFoundError: If any image file doesn't exist.
        """
        from PIL import Image

        model_name = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)

        content: list[Any] = []
        for img in images:
            if not img.exists():
                raise FileNotFoundError(f"Image file not found: {img}")
            pil_img = Image.open(img)
            content.append(pil_img)
        content.append(prompt)

        model = self._get_model(model_name, schema=schema, max_tokens=max_tokens)
        response = await self._generate_async(model, content)

        if schema:
            return json.loads(response.text)
        return response.text

    async def analyze_video(
        self,
        video: Path,
        prompt: str,
        schema: type | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        **kwargs,
    ) -> str | dict:
        """Analyze video using Gemini native video understanding.

        Uploads the video to the File API, waits for processing, then
        sends the video reference with the prompt. Time ranges are
        included in the prompt text.

        Args:
            video: Path to video file (mp4, webm, etc.).
            prompt: Question or instruction about the video.
            schema: Optional Pydantic model for structured output via response_schema.
            start_time: Optional start time in seconds to analyze from.
            end_time: Optional end time in seconds to analyze until.
            **kwargs: Additional options:
                model (str): Override the default model.
                max_tokens (int): Override default max tokens.

        Returns:
            str if no schema, dict if schema provided.

        Raises:
            FileNotFoundError: If video file doesn't exist.
            RuntimeError: If video processing fails on File API.
            ValueError: If start_time > end_time.
        """
        model_name = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)

        if start_time is not None and end_time is not None and start_time > end_time:
            raise ValueError(
                f"start_time ({start_time}) must be <= end_time ({end_time})"
            )

        video_file = await self._upload_video(video)

        # Build prompt with optional time specification
        final_prompt = prompt
        if start_time is not None or end_time is not None:
            time_spec = ""
            if start_time is not None:
                time_spec += f"Starting from {self._format_time(start_time)}. "
            if end_time is not None:
                time_spec += f"Until {self._format_time(end_time)}. "
            final_prompt = time_spec + prompt

        content: list[Any] = [video_file, final_prompt]

        model = self._get_model(model_name, schema=schema, max_tokens=max_tokens)
        response = await self._generate_async(model, content)

        if schema:
            return json.loads(response.text)
        return response.text

    async def reason(
        self,
        messages: list[dict],
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Generate text response using Gemini chat.

        Converts standard message format to Gemini's role convention
        ("user"/"model" instead of "user"/"assistant"). System messages
        are prepended to the first user message.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                Roles: "system", "user", "assistant".
            schema: Optional Pydantic model for structured output via response_schema.
            **kwargs: Additional options:
                model (str): Override the default model.
                max_tokens (int): Override default max tokens.

        Returns:
            str if no schema, dict if schema provided.
        """
        model_name = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)

        # Convert messages to Gemini format
        # Gemini uses "user" and "model" roles; system messages get prepended
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg["content"]

            if role == "system":
                system_parts.append(text)
            elif role == "assistant":
                contents.append({"role": "model", "parts": [text]})
            else:
                contents.append({"role": "user", "parts": [text]})

        # Prepend system message to first user message
        if system_parts and contents:
            system_text = "\n".join(system_parts)
            first = contents[0]
            first["parts"] = [system_text + "\n\n" + first["parts"][0]]

        model = self._get_model(model_name, schema=schema, max_tokens=max_tokens)

        if len(contents) == 1:
            # Single message: use generate_content directly
            response = await self._generate_async(model, contents[0]["parts"])
        else:
            # Multi-turn: use chat
            chat = model.start_chat(history=contents[:-1])
            last_message = contents[-1]["parts"][0]
            response = await self._send_message_async(chat, last_message)

        if schema:
            return json.loads(response.text)
        return response.text
