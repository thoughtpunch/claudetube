"""
claudetube.providers.types - Result types and data schemas for providers.

This module defines the data structures returned by provider operations.
All providers return these common types for interoperability.

Types:
    TranscriptionSegment: A single segment of transcribed audio.
    TranscriptionResult: Complete transcription result with segments.

Pydantic Models (for structured output):
    Structured output schemas are defined in ``providers.schemas`` and
    re-exported here for backwards compatibility. Prefer importing from
    ``claudetube.providers.schemas`` for new code.

Example:
    >>> from claudetube.providers.types import TranscriptionResult, TranscriptionSegment
    >>> result = TranscriptionResult(
    ...     text="Hello world",
    ...     segments=[TranscriptionSegment(start=0.0, end=1.5, text="Hello world")],
    ...     language="en",
    ...     provider="whisper-local",
    ... )
    >>> print(result.to_srt())
    1
    00:00:00,000 --> 00:00:01,500
    Hello world
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Lazy import for pydantic to avoid hard dependency
_pydantic_available: bool | None = None


def _check_pydantic() -> bool:
    """Check if pydantic is available."""
    global _pydantic_available
    if _pydantic_available is None:
        try:
            import pydantic  # noqa: F401

            _pydantic_available = True
        except ImportError:
            _pydantic_available = False
    return _pydantic_available


# =============================================================================
# Transcription Types (Dataclasses - simple data containers)
# =============================================================================


@dataclass
class TranscriptionSegment:
    """A single segment of transcribed audio.

    Represents a time-bounded portion of a transcription with optional
    confidence score and speaker identification.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        text: Transcribed text for this segment.
        confidence: Optional confidence score (0.0-1.0).
        speaker: Optional speaker identifier for diarization.

    Example:
        >>> segment = TranscriptionSegment(
        ...     start=10.5,
        ...     end=15.2,
        ...     text="Welcome to the video",
        ...     confidence=0.95,
        ...     speaker="SPEAKER_00",
        ... )
        >>> segment.to_dict()
        {'start': 10.5, 'end': 15.2, 'text': 'Welcome to the video', ...}
    """

    start: float  # seconds
    end: float  # seconds
    text: str
    confidence: float | None = None
    speaker: str | None = None  # For diarization

    def to_dict(self) -> dict:
        """Convert segment to dictionary.

        Returns:
            Dictionary with all segment fields.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TranscriptionSegment:
        """Create segment from dictionary.

        Args:
            data: Dictionary with segment fields.

        Returns:
            TranscriptionSegment instance.
        """
        return cls(
            start=data["start"],
            end=data["end"],
            text=data["text"],
            confidence=data.get("confidence"),
            speaker=data.get("speaker"),
        )


@dataclass
class TranscriptionResult:
    """Complete transcription result.

    Contains the full transcript text, timed segments, and metadata about
    the transcription process.

    Attributes:
        text: Full transcript as plain text.
        segments: List of timed transcript segments.
        language: Detected or specified language code.
        duration: Total audio duration in seconds.
        provider: Name of the provider that generated this result.

    Example:
        >>> result = TranscriptionResult(
        ...     text="Hello world. How are you?",
        ...     segments=[
        ...         TranscriptionSegment(0.0, 1.5, "Hello world."),
        ...         TranscriptionSegment(2.0, 3.5, "How are you?"),
        ...     ],
        ...     language="en",
        ...     duration=4.0,
        ...     provider="whisper-local",
        ... )
        >>> print(result.to_srt())
        >>> print(result.to_vtt())
    """

    text: str  # Full transcript
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str | None = None
    duration: float | None = None
    provider: str = ""

    def to_srt(self) -> str:
        """Convert to SRT subtitle format.

        SRT format uses comma for milliseconds (HH:MM:SS,mmm).

        Returns:
            String in SRT format.

        Example:
            >>> result.to_srt()
            '1\\n00:00:00,000 --> 00:00:01,500\\nHello world\\n\\n...'
        """
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start = self._format_srt_time(seg.start)
            end = self._format_srt_time(seg.end)
            lines.append(f"{i}\n{start} --> {end}\n{seg.text}\n")
        return "\n".join(lines)

    def to_vtt(self) -> str:
        """Convert to WebVTT format.

        WebVTT format uses period for milliseconds (HH:MM:SS.mmm).

        Returns:
            String in WebVTT format with WEBVTT header.

        Example:
            >>> result.to_vtt()
            'WEBVTT\\n\\n00:00:00.000 --> 00:00:01.500\\nHello world\\n\\n...'
        """
        lines = ["WEBVTT\n"]
        for seg in self.segments:
            start = self._format_vtt_time(seg.start)
            end = self._format_vtt_time(seg.end)
            lines.append(f"{start} --> {end}\n{seg.text}\n")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dictionary with all fields, segments as list of dicts.
        """
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "duration": self.duration,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TranscriptionResult:
        """Create result from dictionary.

        Args:
            data: Dictionary with result fields.

        Returns:
            TranscriptionResult instance.

        Example:
            >>> data = result.to_dict()
            >>> reconstructed = TranscriptionResult.from_dict(data)
            >>> assert reconstructed.text == result.text
        """
        segments = [
            TranscriptionSegment.from_dict(s) for s in data.get("segments", [])
        ]
        return cls(
            text=data["text"],
            segments=segments,
            language=data.get("language"),
            duration=data.get("duration"),
            provider=data.get("provider", ""),
        )

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm).

        Args:
            seconds: Time in seconds.

        Returns:
            SRT-formatted timestamp string.
        """
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        """Format seconds as WebVTT timestamp (HH:MM:SS.mmm).

        Args:
            seconds: Time in seconds.

        Returns:
            WebVTT-formatted timestamp string.
        """
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================
#
# Canonical definitions live in providers.schemas. This module provides
# lazy-loading accessor functions and wrapper classes for backwards
# compatibility. New code should import directly from providers.schemas.

_PYDANTIC_IMPORT_ERROR = (
    "pydantic is required for structured output schemas. "
    "Install with: pip install pydantic"
)


def _get_schema_model(name: str):
    """Import a model class from providers.schemas, with a helpful error."""
    if not _check_pydantic():
        raise ImportError(_PYDANTIC_IMPORT_ERROR)
    from claudetube.providers import schemas

    model = getattr(schemas, name, None)
    if model is None:
        raise ImportError(f"Schema model {name!r} not found in providers.schemas")
    return model


# Accessor functions (backwards-compatible API)


def get_visual_entity_model():
    """Get the VisualEntity Pydantic model.

    Returns:
        VisualEntity Pydantic model class from providers.schemas.

    Raises:
        ImportError: If pydantic is not installed.
    """
    return _get_schema_model("VisualEntity")


def get_semantic_concept_model():
    """Get the SemanticConcept Pydantic model.

    Returns:
        SemanticConcept Pydantic model class from providers.schemas.

    Raises:
        ImportError: If pydantic is not installed.
    """
    return _get_schema_model("SemanticConcept")


def get_entity_extraction_result_model():
    """Get the EntityExtractionResult Pydantic model.

    Returns:
        EntityExtractionResult Pydantic model class from providers.schemas.

    Raises:
        ImportError: If pydantic is not installed.
    """
    return _get_schema_model("EntityExtractionResult")


def get_visual_description_model():
    """Get the VisualDescription Pydantic model.

    Returns:
        VisualDescription Pydantic model class from providers.schemas.

    Raises:
        ImportError: If pydantic is not installed.
    """
    return _get_schema_model("VisualDescription")


def get_person_tracking_result_model():
    """Get the PersonTrackingResult Pydantic model.

    Returns:
        PersonTrackingResult Pydantic model class from providers.schemas.

    Raises:
        ImportError: If pydantic is not installed.
    """
    return _get_schema_model("PersonTrackingResult")


# Lazy-loading wrapper classes for backwards-compatible class-style access.
# These delegate to the accessor functions above so that pydantic is only
# imported when actually used.


class _LazyModelMeta(type):
    """Metaclass for lazy-loading Pydantic models."""

    def __new__(mcs, name, bases, namespace, model_getter=None):
        namespace["_model_getter"] = staticmethod(model_getter) if model_getter else None
        return super().__new__(mcs, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        if cls._model_getter is not None:
            model = cls._model_getter()
            return model(*args, **kwargs)
        return super().__call__(*args, **kwargs)

    def __getattr__(cls, name):
        if cls._model_getter is not None:
            model = cls._model_getter()
            return getattr(model, name)
        raise AttributeError(f"type object '{cls.__name__}' has no attribute '{name}'")


class VisualEntity(metaclass=_LazyModelMeta, model_getter=get_visual_entity_model):
    """A visual entity detected in frame/video.

    This is a lazy-loading wrapper. The actual model is defined in
    ``providers.schemas.VisualEntity``.
    """

    pass


class SemanticConcept(metaclass=_LazyModelMeta, model_getter=get_semantic_concept_model):
    """A concept discussed in the content.

    This is a lazy-loading wrapper. The actual model is defined in
    ``providers.schemas.SemanticConcept``.
    """

    pass


class EntityExtractionResult(
    metaclass=_LazyModelMeta, model_getter=get_entity_extraction_result_model
):
    """Complete entity extraction result - schema for structured output.

    This is a lazy-loading wrapper. The actual model is defined in
    ``providers.schemas.EntityExtractionResult``.
    """

    pass


class VisualDescription(
    metaclass=_LazyModelMeta, model_getter=get_visual_description_model
):
    """Visual description of a scene - for visual_transcript.

    This is a lazy-loading wrapper. The actual model is defined in
    ``providers.schemas.VisualDescription``.
    """

    pass


class PersonTrackingResult(
    metaclass=_LazyModelMeta, model_getter=get_person_tracking_result_model
):
    """Complete person tracking result - schema for structured output.

    This is a lazy-loading wrapper. The actual model is defined in
    ``providers.schemas.PersonTrackingResult``.
    """

    pass


__all__ = [
    # Transcription types (always available)
    "TranscriptionSegment",
    "TranscriptionResult",
    # Pydantic model accessors (require pydantic)
    "get_visual_entity_model",
    "get_semantic_concept_model",
    "get_entity_extraction_result_model",
    "get_visual_description_model",
    "get_person_tracking_result_model",
    # Lazy-loading model wrappers (require pydantic when used)
    "VisualEntity",
    "SemanticConcept",
    "EntityExtractionResult",
    "VisualDescription",
    "PersonTrackingResult",
]
