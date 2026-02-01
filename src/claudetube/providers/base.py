"""
claudetube.providers.base - Abstract base classes and protocols for providers.

This module defines the contracts that all AI providers must implement.
Providers handle tasks like transcription, vision analysis, video analysis,
reasoning, and embedding generation.

Classes:
    Provider: Abstract base class for all AI providers.

Protocols:
    Transcriber: Protocol for audio transcription providers.
    VisionAnalyzer: Protocol for image analysis providers.
    VideoAnalyzer: Protocol for native video analysis (e.g., Gemini).
    Reasoner: Protocol for text reasoning/chat completion.
    Embedder: Protocol for embedding generation.

Example:
    >>> from claudetube.providers.base import Provider, Transcriber
    >>> class MyProvider(Provider, Transcriber):
    ...     @property
    ...     def info(self) -> ProviderInfo:
    ...         return ProviderInfo(name="my-provider", capabilities=frozenset())
    ...     def is_available(self) -> bool:
    ...         return True
    ...     async def transcribe(self, audio, language=None, **kwargs):
    ...         ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from claudetube.providers.capabilities import ProviderInfo
    from claudetube.providers.types import TranscriptionResult


class Provider(ABC):
    """Abstract base class for all AI providers.

    All providers must implement this base class to define their capabilities
    and availability status. Providers may also implement one or more protocol
    interfaces (Transcriber, VisionAnalyzer, etc.) to declare specific
    capabilities.

    Attributes:
        info: Provider metadata including capabilities and limits.

    Example:
        >>> class OpenAIProvider(Provider, Transcriber, VisionAnalyzer):
        ...     @property
        ...     def info(self) -> ProviderInfo:
        ...         return ProviderInfo(
        ...             name="openai",
        ...             capabilities=frozenset({Capability.TRANSCRIBE, Capability.VISION}),
        ...         )
        ...     def is_available(self) -> bool:
        ...         return os.environ.get("OPENAI_API_KEY") is not None
    """

    @property
    @abstractmethod
    def info(self) -> ProviderInfo:
        """Return provider capabilities and limits.

        This property provides metadata about the provider including its name,
        supported capabilities, feature flags, and operational limits.

        Returns:
            ProviderInfo instance with provider metadata.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and ready.

        This method verifies that all prerequisites are met for the provider
        to function, such as API keys being set, dependencies installed, or
        services reachable.

        Returns:
            True if the provider can be used, False otherwise.

        Example:
            >>> if provider.is_available():
            ...     result = await provider.transcribe(audio_path)
        """
        ...


@runtime_checkable
class Transcriber(Protocol):
    """Protocol for audio transcription providers.

    Providers implementing this protocol can convert audio to text with
    timestamps. Results are returned as TranscriptionResult with segments.

    This protocol is implemented by:
    - whisper-local: Local faster-whisper transcription
    - openai: OpenAI Whisper API
    - deepgram: Deepgram transcription service
    - assemblyai: AssemblyAI transcription service

    Example:
        >>> transcriber: Transcriber = get_provider("whisper-local")
        >>> result = await transcriber.transcribe(Path("audio.mp3"))
        >>> print(result.text)
        >>> for segment in result.segments:
        ...     print(f"{segment.start:.1f}s: {segment.text}")
    """

    async def transcribe(
        self,
        audio: Path,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio file to text with timestamps.

        Args:
            audio: Path to audio file (mp3, wav, etc.).
            language: Optional language code (e.g., "en", "es"). If None,
                language is auto-detected.
            **kwargs: Provider-specific options (e.g., model, word_timestamps).

        Returns:
            TranscriptionResult containing full text and timed segments.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            TranscriptionError: If transcription fails.
        """
        ...


@runtime_checkable
class VisionAnalyzer(Protocol):
    """Protocol for image analysis providers.

    Providers implementing this protocol can analyze one or more images
    and respond to prompts about their content. Supports optional structured
    output via Pydantic schemas.

    This protocol is implemented by:
    - openai: GPT-4o vision
    - anthropic: Claude vision
    - google: Gemini vision
    - claude-code: Host Claude instance
    - ollama: Local LLaVA/Moondream

    Example:
        >>> analyzer: VisionAnalyzer = get_provider("anthropic")
        >>> result = await analyzer.analyze_images(
        ...     [Path("frame1.jpg"), Path("frame2.jpg")],
        ...     prompt="What is happening in these frames?",
        ... )
        >>> print(result)
    """

    async def analyze_images(
        self,
        images: list[Path],
        prompt: str,
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Analyze one or more images with a prompt.

        Args:
            images: List of paths to image files (JPEG, PNG, etc.).
            prompt: Question or instruction about the images.
            schema: Optional Pydantic model for structured output. If provided,
                response will be validated against this schema. Providers that
                don't support structured output should ignore this parameter.
            **kwargs: Provider-specific options (e.g., model, max_tokens).

        Returns:
            str: Free-form text response if no schema provided.
            dict: Validated response matching schema if schema provided.

        Raises:
            FileNotFoundError: If any image file doesn't exist.
            AnalysisError: If analysis fails.
        """
        ...


@runtime_checkable
class VideoAnalyzer(Protocol):
    """Protocol for native video analysis.

    Providers implementing this protocol can analyze video content directly
    without requiring frame extraction. This is more efficient for long videos
    and captures temporal relationships.

    Currently only implemented by:
    - google: Gemini with native video support

    Example:
        >>> analyzer: VideoAnalyzer = get_provider("google")
        >>> result = await analyzer.analyze_video(
        ...     Path("video.mp4"),
        ...     prompt="What happens at 2:30?",
        ...     start_time=140.0,
        ...     end_time=160.0,
        ... )
        >>> print(result)
    """

    async def analyze_video(
        self,
        video: Path,
        prompt: str,
        schema: type | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        **kwargs,
    ) -> str | dict:
        """Analyze video content directly without frame extraction.

        Args:
            video: Path to video file (mp4, webm, etc.).
            prompt: Question or instruction about the video.
            schema: Optional Pydantic model for structured output.
            start_time: Optional start time in seconds to analyze from.
            end_time: Optional end time in seconds to analyze until.
            **kwargs: Provider-specific options (e.g., model).

        Returns:
            str: Free-form text response if no schema provided.
            dict: Validated response matching schema if schema provided.

        Raises:
            FileNotFoundError: If video file doesn't exist.
            AnalysisError: If analysis fails.
            ValueError: If start_time > end_time.
        """
        ...


@runtime_checkable
class Reasoner(Protocol):
    """Protocol for text reasoning/chat completion.

    Providers implementing this protocol can generate text responses based
    on message history. Supports multi-turn conversations and optional
    structured output.

    This protocol is implemented by:
    - openai: GPT-4o, GPT-4o-mini
    - anthropic: Claude models
    - google: Gemini models
    - claude-code: Host Claude instance
    - ollama: Local LLMs

    Example:
        >>> reasoner: Reasoner = get_provider("openai")
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Summarize this video transcript..."},
        ... ]
        >>> result = await reasoner.reason(messages)
        >>> print(result)
    """

    async def reason(
        self,
        messages: list[dict],
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Generate text response, optionally with structured output.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                Roles are typically "system", "user", "assistant".
            schema: Optional Pydantic model for structured output.
            **kwargs: Provider-specific options (e.g., model, max_tokens,
                temperature).

        Returns:
            str: Free-form text response if no schema provided.
            dict: Validated response matching schema if schema provided.

        Raises:
            ReasoningError: If generation fails.
            ValidationError: If response doesn't match schema.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding generation.

    Providers implementing this protocol can convert text (and optionally
    images) into dense vector representations for semantic search.

    This protocol is implemented by:
    - voyage: Voyage AI embeddings
    - openai: OpenAI embeddings (text-embedding-3-small)
    - Local models via sentence-transformers

    Example:
        >>> embedder: Embedder = get_provider("voyage")
        >>> vector = await embedder.embed("What is the main topic?")
        >>> print(f"Embedding dimension: {len(vector)}")
    """

    async def embed(
        self,
        text: str,
        images: list[Path] | None = None,
        **kwargs,
    ) -> list[float]:
        """Generate embedding vector for text and optionally images.

        Args:
            text: Text content to embed.
            images: Optional list of image paths for multi-modal embeddings.
                Not all providers support image embeddings.
            **kwargs: Provider-specific options (e.g., model, truncation).

        Returns:
            List of floats representing the embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        ...


__all__ = [
    # Abstract base class
    "Provider",
    # Protocols
    "Transcriber",
    "VisionAnalyzer",
    "VideoAnalyzer",
    "Reasoner",
    "Embedder",
]
