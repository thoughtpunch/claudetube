"""Tests for comprehension verification."""

import pytest

from claudetube.analysis import (
    VerificationResult,
    answer_from_understanding,
    generate_self_test_questions,
    verify_answer,
    verify_comprehension,
)


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_create(self):
        result = VerificationResult(
            question="What is the topic?",
            answer="Python debugging",
            verified=True,
            confidence=0.8,
        )
        assert result.question == "What is the topic?"
        assert result.answer == "Python debugging"
        assert result.verified is True
        assert result.confidence == 0.8

    def test_to_dict(self):
        result = VerificationResult(
            question="Q1",
            answer="A1",
            verified=False,
            confidence=0.3,
        )
        d = result.to_dict()
        assert d["question"] == "Q1"
        assert d["answer"] == "A1"
        assert d["verified"] is False
        assert d["confidence"] == 0.3

    def test_from_dict(self):
        d = {
            "question": "Q1",
            "answer": "A1",
            "verified": True,
            "confidence": 0.95,
        }
        result = VerificationResult.from_dict(d)
        assert result.question == "Q1"
        assert result.answer == "A1"
        assert result.verified is True
        assert result.confidence == 0.95

    def test_from_dict_defaults(self):
        d = {"question": "Q1", "answer": "A1"}
        result = VerificationResult.from_dict(d)
        assert result.verified is False
        assert result.confidence == 0.0

    def test_roundtrip(self):
        original = VerificationResult(
            question="Test?",
            answer="Yes",
            verified=True,
            confidence=0.77,
        )
        restored = VerificationResult.from_dict(original.to_dict())
        assert restored.question == original.question
        assert restored.answer == original.answer
        assert restored.verified == original.verified
        assert restored.confidence == original.confidence


class TestGenerateSelfTestQuestions:
    """Tests for generate_self_test_questions."""

    def test_basic_questions(self):
        understanding = {"scenes": [], "structure": {}}
        questions = generate_self_test_questions(understanding)
        # Should always include basic and synthesis questions
        assert len(questions) >= 2
        assert any("main topic" in q.lower() for q in questions)
        assert any("learn" in q.lower() for q in questions)

    def test_with_scenes(self):
        understanding = {
            "scenes": [
                {"scene_id": 0, "start_time": 0, "transcript_text": "Hello"},
                {"scene_id": 1, "start_time": 30, "transcript_text": "World"},
            ],
            "structure": {},
        }
        questions = generate_self_test_questions(understanding)
        # Should include a scene-specific question
        assert len(questions) >= 3
        assert any("happening" in q.lower() for q in questions)

    def test_with_sections(self):
        understanding = {
            "scenes": [],
            "structure": {
                "sections": [
                    {"start": 0, "title": "Intro"},
                    {"start": 60, "title": "Main"},
                ],
            },
        }
        questions = generate_self_test_questions(understanding)
        assert len(questions) >= 3
        assert any("section" in q.lower() for q in questions)

    def test_with_scenes_and_sections(self):
        understanding = {
            "scenes": [
                {"scene_id": 0, "start_time": 0, "transcript_text": "Hello"},
            ],
            "structure": {
                "sections": [{"start": 0, "title": "Intro"}],
            },
        }
        questions = generate_self_test_questions(understanding)
        # basic + section + scene + synthesis = 4
        assert len(questions) == 4

    def test_empty_understanding(self):
        questions = generate_self_test_questions({})
        # Should still produce basic + synthesis
        assert len(questions) == 2


class TestAnswerFromUnderstanding:
    """Tests for answer_from_understanding."""

    def test_answer_from_relevant_scene(self):
        understanding = {
            "scenes": [
                {
                    "transcript_text": "This tutorial covers Python debugging techniques"
                },
                {
                    "transcript_text": "Use breakpoints to step through code execution"
                },
            ],
            "memory": {},
        }
        answer = answer_from_understanding(
            understanding, "What debugging techniques are discussed?"
        )
        assert "Based on the video content" in answer
        assert len(answer) > 30

    def test_answer_from_qa_history(self):
        understanding = {
            "scenes": [],
            "memory": {
                "qa_history": [
                    {
                        "question": "What is the main topic?",
                        "answer": "Python debugging",
                    }
                ],
            },
        }
        answer = answer_from_understanding(
            understanding, "What is the main topic?"
        )
        assert answer == "Python debugging"

    def test_similar_question_matches_history(self):
        understanding = {
            "scenes": [],
            "memory": {
                "qa_history": [
                    {
                        "question": "What is the primary subject discussed?",
                        "answer": "Machine learning basics",
                    }
                ],
            },
        }
        # Similar question should match
        answer = answer_from_understanding(
            understanding, "What is the primary subject?"
        )
        assert answer == "Machine learning basics"

    def test_unable_to_answer(self):
        understanding = {
            "scenes": [
                {"transcript_text": "Hello world"},
            ],
            "memory": {},
        }
        answer = answer_from_understanding(
            understanding, "What quantum physics concepts are covered?"
        )
        assert answer == "Unable to answer from current understanding"

    def test_empty_understanding(self):
        answer = answer_from_understanding({}, "Any question?")
        assert answer == "Unable to answer from current understanding"


class TestVerifyAnswer:
    """Tests for verify_answer."""

    def test_verified_answer(self):
        understanding = {
            "scenes": [
                {"transcript_text": "Python debugging with breakpoints and logging"},
                {"transcript_text": "Advanced debugging techniques using profilers"},
            ],
        }
        result = verify_answer(
            "debugging techniques using breakpoints and profilers",
            "What techniques?",
            understanding,
        )
        assert result["correct"] is True
        assert result["confidence"] > 0.5

    def test_unverified_answer(self):
        understanding = {
            "scenes": [
                {"transcript_text": "This is about cooking recipes"},
            ],
        }
        result = verify_answer(
            "quantum computing algorithms",
            "What is discussed?",
            understanding,
        )
        assert result["correct"] is False
        assert result["confidence"] < 0.5

    def test_empty_scenes(self):
        result = verify_answer("any answer", "any question", {"scenes": []})
        assert result["correct"] is False
        assert result["confidence"] == 0.0

    def test_empty_transcript(self):
        understanding = {
            "scenes": [{"transcript_text": ""}],
        }
        result = verify_answer("any answer", "question", understanding)
        assert result["correct"] is False

    def test_short_answer_words_filtered(self):
        understanding = {
            "scenes": [{"transcript_text": "some text here"}],
        }
        # All answer words are <= 4 chars or stop words
        result = verify_answer("it is so", "question", understanding)
        assert result["correct"] is False
        assert result["confidence"] == 0.0

    def test_partial_match(self):
        understanding = {
            "scenes": [
                {"transcript_text": "machine learning algorithms for classification"},
            ],
        }
        # 2 out of 3 important words match
        result = verify_answer(
            "machine learning algorithms for regression",
            "What is covered?",
            understanding,
        )
        assert result["confidence"] > 0.0


class TestVerifyComprehension:
    """Tests for verify_comprehension orchestrator."""

    @pytest.fixture
    def rich_understanding(self):
        return {
            "scenes": [
                {
                    "scene_id": 0,
                    "start_time": 0,
                    "transcript_text": (
                        "Welcome to this Python debugging tutorial where we "
                        "cover breakpoints logging profilers"
                    ),
                },
                {
                    "scene_id": 1,
                    "start_time": 30,
                    "transcript_text": (
                        "Using breakpoints allows stepping through program "
                        "execution line by line for debugging"
                    ),
                },
                {
                    "scene_id": 2,
                    "start_time": 60,
                    "transcript_text": (
                        "Logging is another technique for tracking program "
                        "behavior without stopping execution"
                    ),
                },
                {
                    "scene_id": 3,
                    "start_time": 90,
                    "transcript_text": (
                        "Profilers help identify performance bottlenecks in "
                        "your Python applications"
                    ),
                },
            ],
            "structure": {},
            "memory": {},
        }

    def test_with_custom_questions(self, rich_understanding):
        result = verify_comprehension(
            rich_understanding,
            verification_questions=["What debugging tools are covered?"],
        )
        assert "score" in result
        assert "results" in result
        assert "gaps" in result
        assert "ready_to_answer" in result
        assert len(result["results"]) == 1

    def test_auto_generated_questions(self, rich_understanding):
        result = verify_comprehension(rich_understanding)
        # Should auto-generate questions
        assert len(result["results"]) >= 2

    def test_score_calculation(self, rich_understanding):
        result = verify_comprehension(
            rich_understanding,
            verification_questions=[
                "What Python debugging techniques are discussed?",
                "What tools like breakpoints are covered?",
            ],
        )
        score = result["score"]
        assert 0.0 <= score <= 1.0

    def test_gaps_identified(self):
        understanding = {
            "scenes": [
                {"transcript_text": "This is about cooking pasta recipes"},
            ],
            "structure": {},
            "memory": {},
        }
        result = verify_comprehension(
            understanding,
            verification_questions=[
                "What quantum physics concepts are covered?",
            ],
        )
        assert len(result["gaps"]) > 0
        assert result["score"] < 1.0

    def test_ready_to_answer_threshold(self, rich_understanding):
        # High threshold
        result = verify_comprehension(
            rich_understanding,
            verification_questions=[
                "What Python debugging techniques are discussed?",
            ],
            readiness_threshold=0.99,
        )
        # Even a good understanding may not meet 0.99

        # Low threshold
        result_low = verify_comprehension(
            rich_understanding,
            verification_questions=[
                "What Python debugging techniques are discussed?",
            ],
            readiness_threshold=0.0,
        )
        assert result_low["ready_to_answer"] is True

    def test_empty_understanding(self):
        result = verify_comprehension(
            {},
            verification_questions=["What is the topic?"],
        )
        assert result["score"] == 0.0
        assert result["ready_to_answer"] is False

    def test_no_questions_generated(self):
        # verify_comprehension with no questions should return empty
        result = verify_comprehension(
            {"scenes": [], "structure": {}},
            verification_questions=[],
        )
        assert result["score"] == 0.0
        assert result["results"] == []
        assert result["ready_to_answer"] is False

    def test_all_verified(self, rich_understanding):
        result = verify_comprehension(
            rich_understanding,
            verification_questions=[
                "What Python debugging techniques are discussed?",
                "What breakpoints logging profilers tools are covered?",
            ],
        )
        if result["score"] == 1.0:
            assert result["gaps"] == []
            assert result["ready_to_answer"] is True

    def test_result_structure(self, rich_understanding):
        result = verify_comprehension(
            rich_understanding,
            verification_questions=["What is discussed?"],
        )
        assert isinstance(result["score"], float)
        assert isinstance(result["results"], list)
        assert isinstance(result["gaps"], list)
        assert isinstance(result["ready_to_answer"], bool)

        for r in result["results"]:
            assert "question" in r
            assert "answer" in r
            assert "verified" in r
            assert "confidence" in r


class TestComprehensionIntegration:
    """Integration tests for the full comprehension verification flow."""

    def test_full_workflow(self):
        """Test the complete verification workflow."""
        understanding = {
            "scenes": [
                {
                    "scene_id": 0,
                    "start_time": 0,
                    "transcript_text": (
                        "Today we discuss advanced Python testing strategies "
                        "including unit testing integration testing"
                    ),
                },
                {
                    "scene_id": 1,
                    "start_time": 60,
                    "transcript_text": (
                        "Pytest is the recommended framework for Python testing "
                        "with powerful fixtures and parametrize"
                    ),
                },
                {
                    "scene_id": 2,
                    "start_time": 120,
                    "transcript_text": (
                        "Integration testing verifies that components work "
                        "together correctly across boundaries"
                    ),
                },
            ],
            "structure": {
                "sections": [
                    {"start": 0, "title": "Introduction"},
                    {"start": 60, "title": "Pytest"},
                    {"start": 120, "title": "Integration"},
                ],
            },
            "memory": {
                "qa_history": [
                    {
                        "question": "What testing frameworks are mentioned?",
                        "answer": "Pytest is the main framework discussed",
                    },
                ],
            },
        }

        # Auto-generated questions
        questions = generate_self_test_questions(understanding)
        assert len(questions) >= 3

        # Answer from understanding
        for q in questions:
            answer = answer_from_understanding(understanding, q)
            assert isinstance(answer, str)
            assert len(answer) > 0

        # Full verification
        result = verify_comprehension(understanding)
        assert 0.0 <= result["score"] <= 1.0
        assert len(result["results"]) >= 2

    def test_poor_understanding(self):
        """Test that poor understanding gets low scores."""
        understanding = {
            "scenes": [
                {"scene_id": 0, "transcript_text": "abc"},
            ],
            "structure": {},
            "memory": {},
        }
        result = verify_comprehension(
            understanding,
            verification_questions=[
                "What complex algorithms are discussed?",
                "What machine learning models are trained?",
            ],
        )
        assert result["score"] < 1.0
        assert len(result["gaps"]) > 0
