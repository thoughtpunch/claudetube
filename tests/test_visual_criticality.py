"""Tests for visual criticality assessment."""

from __future__ import annotations

import pytest

from claudetube.analysis.visual_criticality import (
    VisualAssessment,
    _heuristic_assessment,
    assess_visual_criticality,
)


class TestVisualAssessment:
    """Tests for VisualAssessment dataclass."""

    def test_to_dict(self):
        """Test serialization to dict."""
        assessment = VisualAssessment(
            score=8,
            confidence="high",
            reasoning="Educational content with animations",
            likely_elements=["animations", "diagrams"],
            recommended_action="recommended",
        )
        result = assessment.to_dict()

        assert result["score"] == 8
        assert result["confidence"] == "high"
        assert result["reasoning"] == "Educational content with animations"
        assert result["likely_elements"] == ["animations", "diagrams"]
        assert result["recommended_action"] == "recommended"

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "score": 10,
            "confidence": "high",
            "reasoning": "3Blue1Brown is purely visual",
            "likely_elements": ["math visualizations"],
            "recommended_action": "required",
        }
        assessment = VisualAssessment.from_dict(data)

        assert assessment.score == 10
        assert assessment.confidence == "high"
        assert assessment.reasoning == "3Blue1Brown is purely visual"
        assert assessment.likely_elements == ["math visualizations"]
        assert assessment.recommended_action == "required"

    def test_from_dict_defaults(self):
        """Test from_dict with missing keys uses defaults."""
        data = {}
        assessment = VisualAssessment.from_dict(data)

        assert assessment.score == 0
        assert assessment.confidence == "low"
        assert assessment.reasoning == ""
        assert assessment.likely_elements == []
        assert assessment.recommended_action == "none"

    def test_visuals_needed_property(self):
        """Test visuals_needed property."""
        required = VisualAssessment(
            score=10, confidence="high", reasoning="",
            likely_elements=[], recommended_action="required"
        )
        recommended = VisualAssessment(
            score=7, confidence="medium", reasoning="",
            likely_elements=[], recommended_action="recommended"
        )
        suggested = VisualAssessment(
            score=5, confidence="medium", reasoning="",
            likely_elements=[], recommended_action="suggested"
        )
        none_needed = VisualAssessment(
            score=2, confidence="high", reasoning="",
            likely_elements=[], recommended_action="none"
        )

        assert required.visuals_needed is True
        assert recommended.visuals_needed is True
        assert suggested.visuals_needed is False
        assert none_needed.visuals_needed is False

    def test_visuals_required_property(self):
        """Test visuals_required property."""
        required = VisualAssessment(
            score=10, confidence="high", reasoning="",
            likely_elements=[], recommended_action="required"
        )
        recommended = VisualAssessment(
            score=7, confidence="medium", reasoning="",
            likely_elements=[], recommended_action="recommended"
        )

        assert required.visuals_required is True
        assert recommended.visuals_required is False


class TestHeuristicAssessment:
    """Tests for the fallback heuristic assessment."""

    def test_3blue1brown_channel(self):
        """Test that 3Blue1Brown is always rated 10."""
        assessment = _heuristic_assessment(
            title="Attention in transformers",
            channel="3Blue1Brown",
            description="A visual explanation",
            transcript_excerpt="Today we'll look at attention",
            task="understand transformers",
        )

        assert assessment.score == 10
        assert assessment.recommended_action == "required"
        assert "3Blue1Brown" in assessment.reasoning

    def test_fireship_channel(self):
        """Test that Fireship is rated high."""
        assessment = _heuristic_assessment(
            title="React in 100 Seconds",
            channel="Fireship",
            description="Quick overview of React",
            transcript_excerpt="React is a JavaScript library",
            task="learn React basics",
        )

        assert assessment.score >= 8
        assert assessment.recommended_action in ("recommended", "required")

    def test_visual_reference_phrases(self):
        """Test detection of visual reference phrases."""
        assessment = _heuristic_assessment(
            title="Some Video",
            channel="Unknown",
            description="",
            transcript_excerpt=(
                "As you can see here, the diagram shows how data flows. "
                "Look at this animation - notice how the values change. "
                "Watch what happens when we visualize the gradient."
            ),
            task="understand the concept",
        )

        # Multiple visual references should boost score
        assert assessment.score >= 7
        assert "visual references" in assessment.reasoning.lower()

    def test_educational_keywords(self):
        """Test detection of educational content."""
        assessment = _heuristic_assessment(
            title="Introduction to Machine Learning - Tutorial for Beginners",
            channel="Some Channel",
            description="Learn the basics of ML",
            transcript_excerpt="In this tutorial, we'll explain how ML works",
            task="learn ML",
        )

        assert assessment.score >= 7
        assert "educational" in assessment.reasoning.lower()

    def test_coding_content(self):
        """Test detection of coding content."""
        assessment = _heuristic_assessment(
            title="Building an API with Python",
            channel="Coding Channel",
            description="Learn to build REST APIs",
            transcript_excerpt="Let's write some Python code for our API",
            task="learn API development",
        )

        assert assessment.score >= 8
        assert "coding" in assessment.reasoning.lower() or "code" in assessment.reasoning.lower()

    def test_math_content(self):
        """Test detection of mathematical content."""
        assessment = _heuristic_assessment(
            title="Understanding Gradients",
            channel="Math Channel",
            description="Vector calculus explained",
            transcript_excerpt=(
                "The gradient points in the direction of steepest ascent. "
                "Let's plot this vector field."
            ),
            task="understand gradients",
        )

        assert assessment.score >= 8
        assert "math" in assessment.reasoning.lower()

    def test_interview_low_score(self):
        """Test that simple interviews get low scores."""
        assessment = _heuristic_assessment(
            title="Interview with CEO",
            channel="Business Channel",
            description="We talk about company strategy",
            transcript_excerpt=(
                "So tell me about your vision for the company. "
                "Well, I think we need to focus on customer satisfaction."
            ),
            task="understand the CEO's vision",
        )

        # No visual cues, just talking
        assert assessment.score <= 6

    def test_task_influences_score(self):
        """Test that task keywords influence the score."""
        # Educational task
        assessment1 = _heuristic_assessment(
            title="Some Video",
            channel="Unknown",
            description="",
            transcript_excerpt="Some content",
            task="learn and understand the concepts",
        )

        # Code-focused task
        assessment2 = _heuristic_assessment(
            title="Some Video",
            channel="Unknown",
            description="",
            transcript_excerpt="Some content",
            task="implement this code syntax",
        )

        assert assessment1.score >= 6
        assert assessment2.score >= 8

    def test_action_thresholds(self):
        """Test that action is correctly assigned based on score."""
        # Score 9-10 -> required
        high = _heuristic_assessment(
            title="Video", channel="3Blue1Brown",
            description="", transcript_excerpt="", task=""
        )
        assert high.recommended_action == "required"

        # Score < 4 -> none (interview with no visual cues)
        low = _heuristic_assessment(
            title="Quick Chat",
            channel="Podcast",
            description="We discuss things",
            transcript_excerpt="Yeah so I think...",
            task="listen",
        )
        # This might be "suggested" due to default score of 5
        assert low.recommended_action in ("none", "suggested")


class TestAssessVisualCriticality:
    """Tests for the main assess_visual_criticality function."""

    @pytest.mark.asyncio
    async def test_returns_visual_assessment_type(self):
        """Test that function returns a VisualAssessment regardless of reasoner availability."""
        # Don't provide a reasoner - will try router, may fallback to heuristics
        assessment = await assess_visual_criticality(
            title="Attention explained",
            channel="3Blue1Brown",
            description="Visual math",
            transcript_excerpt="As you can see the animation shows",
            task="learn attention",
            reasoner=None,
        )

        # Should always get a VisualAssessment back
        assert isinstance(assessment, VisualAssessment)
        assert 0 <= assessment.score <= 10
        assert assessment.confidence in ("low", "medium", "high")
        assert assessment.recommended_action in ("none", "suggested", "recommended", "required")

    def test_heuristic_assessment_for_visual_channel(self):
        """Test that heuristic assessment handles visual channels correctly."""
        # Direct test of heuristic for 3Blue1Brown
        assessment = _heuristic_assessment(
            title="Attention explained",
            channel="3Blue1Brown",
            description="Visual math",
            transcript_excerpt="As you can see",
            task="learn attention",
        )

        assert assessment.score == 10
        assert assessment.recommended_action == "required"
        assert "3Blue1Brown" in assessment.reasoning


class TestRoundTrip:
    """Test serialization round-trips."""

    def test_dict_round_trip(self):
        """Test to_dict/from_dict round trip preserves data."""
        original = VisualAssessment(
            score=7,
            confidence="medium",
            reasoning="Some reasoning here",
            likely_elements=["diagrams", "code"],
            recommended_action="recommended",
        )

        reconstructed = VisualAssessment.from_dict(original.to_dict())

        assert reconstructed.score == original.score
        assert reconstructed.confidence == original.confidence
        assert reconstructed.reasoning == original.reasoning
        assert reconstructed.likely_elements == original.likely_elements
        assert reconstructed.recommended_action == original.recommended_action
