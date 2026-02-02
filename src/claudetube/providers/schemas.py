"""
claudetube.providers.schemas - Structured output schemas for AI providers.

Pydantic models that define the JSON schemas enforced by provider structured
output features. These schemas are passed to VisionAnalyzer.analyze_images(),
VideoAnalyzer.analyze_video(), and Reasoner.reason() via the ``schema``
parameter. Providers convert them to their native format:

- OpenAI: ``response_format`` with ``json_schema``
- Anthropic: ``tool_choice`` + tool definition
- Google/Gemini: ``response_schema``

All schemas are defined here once and shared across all providers, ensuring
consistent structured output regardless of which provider is used.

Usage:
    >>> from claudetube.providers.schemas import EntityExtractionResult
    >>> result = await vision.analyze_images(frames, prompt, schema=EntityExtractionResult)

Models:
    VisualEntity: A visual entity detected in frame/video.
    SemanticConcept: A concept discussed in the content.
    EntityExtractionResult: Complete entity extraction result.
    VisualDescription: Visual description of a scene.
    PersonAppearance: A single appearance of a person in a scene.
    PersonTrack: Track of a single person across the video.
    PersonTrackingResult: Complete person tracking result.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# =============================================================================
# Entity Extraction Schemas
# =============================================================================


class VisualEntity(BaseModel):
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
    first_seen_sec: float = Field(
        description="Timestamp when entity first appears"
    )
    last_seen_sec: float | None = Field(
        default=None, description="Timestamp when entity last appears"
    )
    confidence: float = Field(
        default=1.0, description="Confidence score (0.0-1.0)"
    )
    attributes: dict[str, str] = Field(
        default_factory=dict, description="Additional attributes"
    )


class SemanticConcept(BaseModel):
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
    first_mention_sec: float = Field(
        description="Timestamp of first mention"
    )
    related_terms: list[str] = Field(
        default_factory=list, description="Related terms"
    )


class EntityExtractionResult(BaseModel):
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
        ...     objects=[VisualEntity(name="laptop", category="object", first_seen_sec=0.0)],
        ...     people=[VisualEntity(name="presenter", category="person", first_seen_sec=0.0)],
        ...     concepts=[SemanticConcept(term="Python", definition="Programming language",
        ...         importance="primary", first_mention_sec=10.0)],
        ... )
    """

    objects: list[VisualEntity] = Field(
        default_factory=list, description="Visual objects detected"
    )
    people: list[VisualEntity] = Field(
        default_factory=list, description="People identified"
    )
    text_on_screen: list[VisualEntity] = Field(
        default_factory=list, description="Text visible in frames"
    )
    concepts: list[SemanticConcept] = Field(
        default_factory=list, description="Semantic concepts"
    )
    code_snippets: list[dict] = Field(
        default_factory=list, description="Code shown or discussed"
    )


# =============================================================================
# Visual Description Schema
# =============================================================================


class VisualDescription(BaseModel):
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

    description: str = Field(
        description="Natural language description of the scene"
    )
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


# =============================================================================
# Person Tracking Schemas
# =============================================================================


class PersonAppearance(BaseModel):
    """A single appearance of a person in a scene.

    Attributes:
        scene_id: Scene identifier where person appears.
        timestamp: Timestamp in seconds from video start.
        action: What the person is doing (optional).
        confidence: Confidence score for the detection (0.0-1.0).
    """

    scene_id: int = Field(description="Scene identifier")
    timestamp: float = Field(
        description="Timestamp in seconds from video start"
    )
    action: str | None = Field(
        default=None, description="What the person is doing"
    )
    confidence: float = Field(
        default=1.0, description="Confidence score (0.0-1.0)"
    )


class PersonTrack(BaseModel):
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
    appearances: list[PersonAppearance] = Field(
        default_factory=list, description="Scene appearances"
    )


class PersonTrackingResult(BaseModel):
    """Complete person tracking result - schema for structured output.

    Used when asking LLMs to identify and track people across video scenes.

    Attributes:
        people: List of tracked people with their appearances.
    """

    people: list[PersonTrack] = Field(
        default_factory=list, description="People tracked across scenes"
    )


__all__ = [
    # Entity extraction
    "VisualEntity",
    "SemanticConcept",
    "EntityExtractionResult",
    # Visual description
    "VisualDescription",
    # Person tracking
    "PersonAppearance",
    "PersonTrack",
    "PersonTrackingResult",
]
