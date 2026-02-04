"""
Visual criticality assessment for video content.

Uses LLM-based assessment to determine how critical visual analysis is for
understanding video content. This is more nuanced than keyword-based heuristics
because the LLM can understand context, channel reputation, and task requirements.

Architecture: Cheap First, Expensive Last
1. CACHE - Return cached assessment instantly if available
2. LLM - Fast/cheap model assessment (~$0.0001 per call with Haiku)

Example:
    >>> from claudetube.analysis.visual_criticality import assess_visual_criticality
    >>> assessment = await assess_visual_criticality(
    ...     title="Attention in transformers, visually explained",
    ...     channel="3Blue1Brown",
    ...     description="A visual walkthrough of the attention mechanism",
    ...     transcript_excerpt="Today we'll visualize how attention works...",
    ...     task="understand the math behind transformers",
    ... )
    >>> print(assessment.score)  # 10
    >>> print(assessment.recommended_action)  # "required"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from claudetube.providers.base import Reasoner

logger = logging.getLogger(__name__)

# Assessment prompt template - instructs the LLM to score visual criticality
VISUAL_ASSESSMENT_PROMPT = """\
Given this video:
- Title: {title}
- Channel: {channel}
- Description: {description}
- Transcript excerpt: {transcript_excerpt}
- Task: {task}

How critical are the visuals for understanding this content, given the task?

Consider:
- Is this educational/technical content with diagrams, animations, or demos?
- Does the speaker reference things on screen ("as you can see", "look at this")?
- Are there likely code snippets, charts, or mathematical visualizations?
- Would someone listening without video miss significant information for THIS task?
- For coding tutorials: code syntax IS visual information
- For music videos: depends on task (lyrics analysis vs. experiencing the art)

Respond with JSON:
{{
  "visual_criticality": <0-10>,
  "confidence": "low|medium|high",
  "reasoning": "<1-2 sentences explaining why>",
  "likely_visual_elements": ["<list of expected visual content>"],
  "recommended_action": "none|suggested|recommended|required"
}}"""


@dataclass
class VisualAssessment:
    """Result of LLM-based visual criticality assessment.

    Attributes:
        score: Visual criticality score from 0-10.
            0-3 = visuals not needed
            4-6 = visuals suggested
            7-8 = visuals recommended
            9-10 = visuals required
        confidence: LLM's confidence in the assessment.
        reasoning: Human-readable explanation of the score.
        likely_elements: Expected visual elements in the video.
        recommended_action: Action recommendation based on score.
    """

    score: int
    confidence: Literal["low", "medium", "high"]
    reasoning: str
    likely_elements: list[str]
    recommended_action: Literal["none", "suggested", "recommended", "required"]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "score": self.score,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "likely_elements": self.likely_elements,
            "recommended_action": self.recommended_action,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VisualAssessment:
        """Create from dictionary."""
        return cls(
            score=data.get("score", 0),
            confidence=data.get("confidence", "low"),
            reasoning=data.get("reasoning", ""),
            likely_elements=data.get("likely_elements", []),
            recommended_action=data.get("recommended_action", "none"),
        )

    @property
    def visuals_needed(self) -> bool:
        """Whether visual analysis is recommended or required."""
        return self.recommended_action in ("recommended", "required")

    @property
    def visuals_required(self) -> bool:
        """Whether visual analysis is required (score 9-10)."""
        return self.recommended_action == "required"


def _get_visual_assessment_schema():
    """Get the Pydantic schema for visual assessment structured output.

    Returns:
        Pydantic model class for VisualAssessment JSON output.
    """
    from typing import Literal as LiteralType

    from pydantic import BaseModel, Field

    class VisualAssessmentSchema(BaseModel):
        """Schema for visual criticality assessment response."""

        visual_criticality: int = Field(
            ge=0, le=10, description="Visual criticality score from 0-10"
        )
        confidence: LiteralType["low", "medium", "high"] = Field(
            description="Confidence in the assessment"
        )
        reasoning: str = Field(
            description="1-2 sentences explaining the score"
        )
        likely_visual_elements: list[str] = Field(
            default_factory=list,
            description="Expected visual elements in the video",
        )
        recommended_action: LiteralType[
            "none", "suggested", "recommended", "required"
        ] = Field(description="Action recommendation based on score")

    return VisualAssessmentSchema


async def assess_visual_criticality(
    title: str,
    channel: str,
    description: str,
    transcript_excerpt: str,
    task: str = "general understanding",
    reasoner: Reasoner | None = None,
) -> VisualAssessment:
    """Assess how critical visuals are for understanding video content.

    Uses an LLM to analyze the video metadata and transcript to determine
    whether visual analysis (frame extraction) should be performed.

    Args:
        title: Video title.
        channel: Channel/creator name.
        description: Video description (can be truncated).
        transcript_excerpt: First ~2000 chars of transcript.
        task: User's goal or task with the video.
        reasoner: Optional Reasoner provider. If None, uses default from router.

    Returns:
        VisualAssessment with score, reasoning, and recommendations.

    Example:
        >>> assessment = await assess_visual_criticality(
        ...     title="Neural Networks Explained",
        ...     channel="3Blue1Brown",
        ...     description="A visual guide to neural networks",
        ...     transcript_excerpt="Today we'll visualize...",
        ...     task="learn how neural networks work",
        ... )
        >>> if assessment.visuals_required:
        ...     frames = await extract_frames(video_id)
    """
    # Get reasoner from router if not provided
    if reasoner is None:
        from claudetube.providers.router import ProviderRouter

        router = ProviderRouter()
        try:
            reasoner = router.get_reasoner_for_structured_output()
        except Exception as e:
            logger.warning(f"No structured output reasoner available: {e}")
            # Fall back to heuristic assessment
            return _heuristic_assessment(
                title, channel, description, transcript_excerpt, task
            )

    # Build prompt
    prompt = VISUAL_ASSESSMENT_PROMPT.format(
        title=title,
        channel=channel,
        description=description[:500] if description else "",
        transcript_excerpt=transcript_excerpt[:2000] if transcript_excerpt else "",
        task=task,
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        schema = _get_visual_assessment_schema()
        result = await reasoner.reason(messages, schema=schema)

        # Parse response
        data = result if isinstance(result, dict) else json.loads(result)

        return VisualAssessment(
            score=data.get("visual_criticality", 5),
            confidence=data.get("confidence", "medium"),
            reasoning=data.get("reasoning", ""),
            likely_elements=data.get("likely_visual_elements", []),
            recommended_action=data.get("recommended_action", "suggested"),
        )
    except Exception as e:
        logger.warning(f"LLM assessment failed, falling back to heuristics: {e}")
        return _heuristic_assessment(
            title, channel, description, transcript_excerpt, task
        )


def _heuristic_assessment(
    title: str,
    channel: str,
    description: str,
    transcript_excerpt: str,
    task: str,
) -> VisualAssessment:
    """Fallback heuristic assessment when LLM is unavailable.

    Uses keyword matching and channel recognition as a fallback.

    Args:
        title: Video title.
        channel: Channel/creator name.
        description: Video description.
        transcript_excerpt: Transcript excerpt.
        task: User's task.

    Returns:
        VisualAssessment based on heuristic rules.
    """
    score = 5  # Default to middle
    reasons = []
    elements = []

    # Combine all text for analysis
    all_text = f"{title} {description} {transcript_excerpt}".lower()
    channel_lower = channel.lower()

    # Channel signals (high confidence)
    always_visual_channels = {
        "3blue1brown", "veritasium", "numberphile", "welch labs",
        "computerphile", "two minute papers", "kurzgesagt",
        "cgp grey", "smartereveryday", "primer", "reducible",
        "the coding train", "sebastian lague",
    }
    high_visual_channels = {
        "fireship", "theo", "theprimeagen", "traversy media",
        "tech with tim", "web dev simplified",
    }

    if any(ch in channel_lower for ch in always_visual_channels):
        score = 10
        reasons.append(f"Channel '{channel}' is known for visual-first content")
        elements.extend(["animations", "diagrams", "visualizations"])

    elif any(ch in channel_lower for ch in high_visual_channels):
        score = max(score, 8)
        reasons.append(f"Channel '{channel}' typically has visual content")
        elements.extend(["code", "diagrams"])

    # Visual reference phrases
    visual_phrases = [
        "as you can see", "look at this", "notice how", "watch what happens",
        "in this animation", "the diagram shows", "shown here", "visualize",
        "on screen", "here we have", "take a look", "you'll see",
    ]
    phrase_count = sum(1 for phrase in visual_phrases if phrase in all_text)

    if phrase_count >= 10:
        score = max(score, 9)
        reasons.append(f"High density of visual references ({phrase_count} phrases)")
    elif phrase_count >= 5:
        score = max(score, 7)
        reasons.append(f"Multiple visual references ({phrase_count} phrases)")
    elif phrase_count >= 2:
        score = max(score, 6)
        reasons.append(f"Some visual references ({phrase_count} phrases)")

    # Educational/technical keywords
    educational_keywords = [
        "explained", "tutorial", "how to", "learn", "introduction",
        "basics", "understanding", "guide", "course", "lesson",
    ]
    if any(kw in all_text for kw in educational_keywords):
        score = max(score, 7)
        reasons.append("Educational content detected")
        elements.append("educational diagrams")

    # Code/technical keywords
    code_keywords = ["code", "programming", "javascript", "python", "react", "api"]
    if any(kw in all_text for kw in code_keywords):
        score = max(score, 8)
        reasons.append("Coding/technical content detected")
        elements.append("code snippets")

    # Math/science keywords
    math_keywords = ["vector", "dimension", "gradient", "equation", "graph", "plot"]
    if any(kw in all_text for kw in math_keywords):
        score = max(score, 8)
        reasons.append("Mathematical/scientific content detected")
        elements.extend(["graphs", "equations", "visualizations"])

    # Task signals
    task_lower = task.lower()
    if any(kw in task_lower for kw in ["learn", "understand", "explain", "teach"]):
        score = max(score, 6)
        reasons.append("Educational task")
    if any(kw in task_lower for kw in ["code", "implement", "syntax"]):
        score = max(score, 8)
        reasons.append("Code-focused task")

    # Determine action based on score
    if score >= 9:
        action = "required"
    elif score >= 7:
        action = "recommended"
    elif score >= 4:
        action = "suggested"
    else:
        action = "none"

    return VisualAssessment(
        score=score,
        confidence="medium",  # Heuristics are less confident than LLM
        reasoning="; ".join(reasons) if reasons else "Default assessment",
        likely_elements=list(set(elements)),
        recommended_action=action,
    )


__all__ = [
    "VisualAssessment",
    "assess_visual_criticality",
    "VISUAL_ASSESSMENT_PROMPT",
]
