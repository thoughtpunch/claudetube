"""
claudetube.providers.types - Result types and data schemas for providers.

This module defines the data structures returned by provider operations.
All providers return these common types for interoperability.

Types:
    TranscriptionSegment: A single segment of transcribed audio.
    TranscriptionResult: Complete transcription result with segments.

Pydantic Models (for structured output):
    VisualEntity: A visual entity detected in frame/video.
    SemanticConcept: A concept discussed in the content.
    EntityExtractionResult: Complete entity extraction result.
    VisualDescription: Visual description of a scene.

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
from typing import Literal

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
# Pydantic Models for Structured Output (Entity Extraction)
# =============================================================================

# These models are used with LLM structured output features.
# They use Pydantic for JSON Schema generation and validation.


def _get_base_model():
    """Get Pydantic BaseModel, raising helpful error if not available."""
    if not _check_pydantic():
        raise ImportError(
            "pydantic is required for structured output schemas. "
            "Install with: pip install pydantic"
        )
    from pydantic import BaseModel

    return BaseModel


# Define Pydantic models only if pydantic is available
# Use lazy class definitions to avoid import errors

_VisualEntity = None
_SemanticConcept = None
_EntityExtractionResult = None
_VisualDescription = None
_PersonTrackingResult = None


def _ensure_pydantic_models():
    """Ensure Pydantic models are defined."""
    global _VisualEntity, _SemanticConcept, _EntityExtractionResult, _VisualDescription, _PersonTrackingResult

    if _VisualEntity is not None:
        return  # Already defined

    if not _check_pydantic():
        return  # Pydantic not available

    from pydantic import BaseModel, Field

    class VisualEntityModel(BaseModel):
        """A visual entity detected in frame/video.

        Used for structured output when extracting visual elements from video
        frames. Categories help organize entities and improve LLM extraction
        accuracy.

        Attributes:
            name: Name or description of the entity.
            category: Type of entity (object, person, text, code, ui_element).
            first_seen_sec: Timestamp when entity first appears.
            last_seen_sec: Timestamp when entity last appears (optional).
            confidence: Confidence score for the detection (0.0-1.0).
            attributes: Additional key-value attributes for the entity.

        Example:
            >>> entity = VisualEntity(
            ...     name="Python logo",
            ...     category="object",
            ...     first_seen_sec=12.5,
            ...     confidence=0.9,
            ...     attributes={"color": "blue and yellow"},
            ... )
        """

        name: str = Field(description="Name or description of the entity")
        category: Literal["object", "person", "text", "code", "ui_element"] = Field(
            description="Type of entity"
        )
        first_seen_sec: float = Field(description="Timestamp when entity first appears")
        last_seen_sec: float | None = Field(
            default=None, description="Timestamp when entity last appears"
        )
        confidence: float = Field(
            default=1.0, description="Confidence score (0.0-1.0)"
        )
        attributes: dict[str, str] = Field(
            default_factory=dict, description="Additional attributes"
        )

    class SemanticConceptModel(BaseModel):
        """A concept discussed in the content.

        Used for extracting and categorizing concepts from video transcripts
        and visual content. Importance levels help prioritize concepts.

        Attributes:
            term: The concept term or phrase.
            definition: Brief definition or explanation.
            importance: How central this concept is to the content.
            first_mention_sec: Timestamp of first mention.
            related_terms: Other terms related to this concept.

        Example:
            >>> concept = SemanticConcept(
            ...     term="Machine Learning",
            ...     definition="A subset of AI that learns from data",
            ...     importance="primary",
            ...     first_mention_sec=45.0,
            ...     related_terms=["AI", "neural networks", "deep learning"],
            ... )
        """

        term: str = Field(description="The concept term or phrase")
        definition: str = Field(description="Brief definition or explanation")
        importance: Literal["primary", "secondary", "mentioned"] = Field(
            description="How central this concept is"
        )
        first_mention_sec: float = Field(description="Timestamp of first mention")
        related_terms: list[str] = Field(
            default_factory=list, description="Related terms"
        )

    class EntityExtractionResultModel(BaseModel):
        """Complete entity extraction result - schema for structured output.

        This is the main schema used when asking LLMs to extract all entities
        from video content. It organizes entities by type for easier processing.

        Attributes:
            objects: Visual objects detected in frames.
            people: People identified in the video.
            text_on_screen: Text visible in frames.
            concepts: Semantic concepts from content.
            code_snippets: Code shown or discussed in video.

        Example:
            >>> result = EntityExtractionResult(
            ...     objects=[VisualEntity(...)],
            ...     people=[VisualEntity(...)],
            ...     concepts=[SemanticConcept(...)],
            ... )
        """

        objects: list[VisualEntityModel] = Field(
            default_factory=list, description="Visual objects detected"
        )
        people: list[VisualEntityModel] = Field(
            default_factory=list, description="People identified"
        )
        text_on_screen: list[VisualEntityModel] = Field(
            default_factory=list, description="Text visible in frames"
        )
        concepts: list[SemanticConceptModel] = Field(
            default_factory=list, description="Semantic concepts"
        )
        code_snippets: list[dict] = Field(
            default_factory=list, description="Code shown or discussed"
        )

    class VisualDescriptionModel(BaseModel):
        """Visual description of a scene - for visual_transcript.

        Used when generating visual descriptions for scenes in a video.
        Keeps the schema flat to avoid confusing LLMs with deep nesting.

        Attributes:
            description: Natural language description of the scene.
            objects: List of objects visible in the scene.
            people: List of people visible in the scene.
            text_on_screen: List of text visible on screen.
            actions: List of actions happening in the scene.
            setting: Description of the setting/environment.

        Example:
            >>> desc = VisualDescription(
            ...     description="A presenter at a whiteboard",
            ...     objects=["whiteboard", "marker", "laptop"],
            ...     people=["presenter"],
            ...     text_on_screen=["Chapter 1: Introduction"],
            ...     actions=["writing on whiteboard", "gesturing"],
            ...     setting="Conference room",
            ... )
        """

        description: str = Field(description="Natural language description of the scene")
        objects: list[str] = Field(
            default_factory=list, description="Objects visible in scene"
        )
        people: list[str] = Field(
            default_factory=list, description="People visible in scene"
        )
        text_on_screen: list[str] = Field(
            default_factory=list, description="Text visible on screen"
        )
        actions: list[str] = Field(
            default_factory=list, description="Actions happening in scene"
        )
        setting: str | None = Field(
            default=None, description="Description of setting/environment"
        )

    class PersonAppearanceModel(BaseModel):
        """A single appearance of a person in a scene.

        Attributes:
            scene_id: Scene identifier where person appears.
            timestamp: Timestamp in seconds from video start.
            action: What the person is doing (optional).
            confidence: Confidence score for the detection (0.0-1.0).
        """

        scene_id: int = Field(description="Scene identifier")
        timestamp: float = Field(description="Timestamp in seconds from video start")
        action: str | None = Field(
            default=None, description="What the person is doing"
        )
        confidence: float = Field(
            default=1.0, description="Confidence score (0.0-1.0)"
        )

    class PersonTrackModel(BaseModel):
        """Track of a single person across the video.

        Attributes:
            person_id: Unique identifier for this person.
            description: Visual description of the person.
            appearances: List of scene appearances.
        """

        person_id: str = Field(description="Unique identifier for the person")
        description: str = Field(
            description="Visual description (e.g., 'man in blue shirt')"
        )
        appearances: list[PersonAppearanceModel] = Field(
            default_factory=list, description="Scene appearances"
        )

    class PersonTrackingResultModel(BaseModel):
        """Complete person tracking result - schema for structured output.

        Used when asking LLMs to identify and track people across video scenes.

        Attributes:
            people: List of tracked people with their appearances.
        """

        people: list[PersonTrackModel] = Field(
            default_factory=list, description="People tracked across scenes"
        )

    # Store the models
    _VisualEntity = VisualEntityModel
    _SemanticConcept = SemanticConceptModel
    _EntityExtractionResult = EntityExtractionResultModel
    _VisualDescription = VisualDescriptionModel
    _PersonTrackingResult = PersonTrackingResultModel


# Accessor functions that lazy-load the models


def get_visual_entity_model():
    """Get the VisualEntity Pydantic model.

    Returns:
        VisualEntity Pydantic model class.

    Raises:
        ImportError: If pydantic is not installed.
    """
    _ensure_pydantic_models()
    if _VisualEntity is None:
        raise ImportError(
            "pydantic is required for structured output schemas. "
            "Install with: pip install pydantic"
        )
    return _VisualEntity


def get_semantic_concept_model():
    """Get the SemanticConcept Pydantic model.

    Returns:
        SemanticConcept Pydantic model class.

    Raises:
        ImportError: If pydantic is not installed.
    """
    _ensure_pydantic_models()
    if _SemanticConcept is None:
        raise ImportError(
            "pydantic is required for structured output schemas. "
            "Install with: pip install pydantic"
        )
    return _SemanticConcept


def get_entity_extraction_result_model():
    """Get the EntityExtractionResult Pydantic model.

    Returns:
        EntityExtractionResult Pydantic model class.

    Raises:
        ImportError: If pydantic is not installed.
    """
    _ensure_pydantic_models()
    if _EntityExtractionResult is None:
        raise ImportError(
            "pydantic is required for structured output schemas. "
            "Install with: pip install pydantic"
        )
    return _EntityExtractionResult


def get_visual_description_model():
    """Get the VisualDescription Pydantic model.

    Returns:
        VisualDescription Pydantic model class.

    Raises:
        ImportError: If pydantic is not installed.
    """
    _ensure_pydantic_models()
    if _VisualDescription is None:
        raise ImportError(
            "pydantic is required for structured output schemas. "
            "Install with: pip install pydantic"
        )
    return _VisualDescription


def get_person_tracking_result_model():
    """Get the PersonTrackingResult Pydantic model.

    Returns:
        PersonTrackingResult Pydantic model class.

    Raises:
        ImportError: If pydantic is not installed.
    """
    _ensure_pydantic_models()
    if _PersonTrackingResult is None:
        raise ImportError(
            "pydantic is required for structured output schemas. "
            "Install with: pip install pydantic"
        )
    return _PersonTrackingResult


# For backwards compatibility and type hints, also provide class-style access
# These will fail at import time if pydantic is not installed and the class is accessed


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

    This is a lazy-loading wrapper around the Pydantic model.
    The actual model is loaded when first accessed.
    """

    pass


class SemanticConcept(metaclass=_LazyModelMeta, model_getter=get_semantic_concept_model):
    """A concept discussed in the content.

    This is a lazy-loading wrapper around the Pydantic model.
    The actual model is loaded when first accessed.
    """

    pass


class EntityExtractionResult(
    metaclass=_LazyModelMeta, model_getter=get_entity_extraction_result_model
):
    """Complete entity extraction result - schema for structured output.

    This is a lazy-loading wrapper around the Pydantic model.
    The actual model is loaded when first accessed.
    """

    pass


class VisualDescription(
    metaclass=_LazyModelMeta, model_getter=get_visual_description_model
):
    """Visual description of a scene - for visual_transcript.

    This is a lazy-loading wrapper around the Pydantic model.
    The actual model is loaded when first accessed.
    """

    pass


class PersonTrackingResult(
    metaclass=_LazyModelMeta, model_getter=get_person_tracking_result_model
):
    """Complete person tracking result - schema for structured output.

    This is a lazy-loading wrapper around the Pydantic model.
    The actual model is loaded when first accessed.
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
