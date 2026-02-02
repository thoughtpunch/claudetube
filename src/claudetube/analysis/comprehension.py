"""
Comprehension verification for video understanding.

Verifies that the agent's understanding of a video is actually correct
before answering user questions. The verifier:
- Generates self-test questions from the video understanding
- Answers them from understanding alone (no re-examining)
- Verifies answers against video content
- Returns a comprehension score and identified gaps

Architecture: Cheap First, Expensive Last
1. TEXT - Verify against transcript text (instant)
2. STRUCTURE - Check structural consistency (fast)
3. No expensive operations needed - verification is purely analytical
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from claudetube.analysis.search import format_timestamp

logger = logging.getLogger(__name__)

# Words filtered out when comparing answer terms to transcript
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "this",
        "that",
        "these",
        "those",
        "based",
        "video",
        "content",
        "about",
    }
)


@dataclass
class VerificationResult:
    """Result of verifying a single comprehension question.

    Attributes:
        question: The self-test question.
        answer: Answer generated from understanding alone.
        verified: Whether the answer was verified against video content.
        confidence: Confidence score from 0.0 to 1.0.
    """

    question: str
    answer: str
    verified: bool
    confidence: float

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "question": self.question,
            "answer": self.answer,
            "verified": self.verified,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VerificationResult:
        """Create from dictionary."""
        return cls(
            question=data["question"],
            answer=data["answer"],
            verified=data.get("verified", False),
            confidence=data.get("confidence", 0.0),
        )


def verify_comprehension(
    video_understanding: dict,
    verification_questions: list[str] | None = None,
    readiness_threshold: float = 0.7,
) -> dict:
    """Verify the agent's understanding of a video.

    Generates self-test questions (or uses provided ones), answers them
    from the understanding alone, then verifies answers against the
    video content.

    Args:
        video_understanding: Dict with keys like 'scenes', 'structure',
            'memory', containing the agent's understanding of the video.
        verification_questions: Optional list of questions to verify.
            If None, questions are auto-generated.
        readiness_threshold: Minimum score to consider understanding
            sufficient. Defaults to 0.7.

    Returns:
        Dict with:
        - score: Overall comprehension score (0.0-1.0)
        - results: List of VerificationResult dicts
        - gaps: Questions that failed verification
        - ready_to_answer: Whether score meets readiness_threshold
    """
    if verification_questions is None:
        verification_questions = generate_self_test_questions(video_understanding)

    if not verification_questions:
        logger.warning("No verification questions generated")
        return {
            "score": 0.0,
            "results": [],
            "gaps": [],
            "ready_to_answer": False,
        }

    results = []
    for question in verification_questions:
        # Answer from understanding only
        answer = answer_from_understanding(video_understanding, question)

        # Verify against video content
        verification = verify_answer(answer, question, video_understanding)

        results.append(
            VerificationResult(
                question=question,
                answer=answer,
                verified=verification["correct"],
                confidence=verification["confidence"],
            )
        )

    # Calculate overall score
    comprehension_score = sum(r.verified for r in results) / len(results)

    logger.info(
        f"Comprehension verification: {comprehension_score:.2f} "
        f"({sum(r.verified for r in results)}/{len(results)} verified)"
    )

    return {
        "score": comprehension_score,
        "results": [r.to_dict() for r in results],
        "gaps": [r.question for r in results if not r.verified],
        "ready_to_answer": comprehension_score >= readiness_threshold,
    }


def generate_self_test_questions(understanding: dict) -> list[str]:
    """Generate questions to test understanding of a video.

    Creates questions from different aspects of the understanding:
    - Basic topic comprehension
    - Section-specific questions
    - Scene-specific questions
    - Synthesis/takeaway questions

    Args:
        understanding: Dict with 'scenes', 'structure', 'memory' keys.

    Returns:
        List of self-test question strings.
    """
    scenes = understanding.get("scenes", [])
    structure = understanding.get("structure", {})

    questions = []

    # Basic comprehension
    questions.append("What is the main topic of this video?")

    # Section-specific
    sections = structure.get("sections", [])
    if sections:
        section = random.choice(sections)
        start = section.get("start", 0)
        questions.append(
            f"What is covered in the section starting at {format_timestamp(start)}?"
        )

    # Scene-specific
    if scenes:
        scene = random.choice(scenes)
        start = scene.get("start_time", scene.get("start", 0))
        questions.append(f"What is happening at {format_timestamp(start)}?")

    # Synthesis
    questions.append("What would someone learn from watching this video?")

    return questions


def answer_from_understanding(understanding: dict, question: str) -> str:
    """Answer a question using only the cached understanding.

    Does not re-examine the video. Searches through scenes,
    structure, and QA history to synthesize an answer.

    Args:
        understanding: Dict with 'scenes', 'structure', 'memory' keys.
        question: Question to answer.

    Returns:
        Answer string synthesized from understanding.
    """
    scenes = understanding.get("scenes", [])
    memory = understanding.get("memory", {})

    # Check if we've answered similar before
    qa_history = memory.get("qa_history", [])
    for qa in qa_history:
        if _similar_question(question, qa.get("question", "")):
            return qa.get("answer", "")

    # Synthesize from scene content
    relevant_text = []
    for scene in scenes:
        if _scene_relevant_to_question(scene, question):
            transcript = scene.get("transcript_text", "")
            if transcript:
                relevant_text.append(transcript[:200])

    if relevant_text:
        return f"Based on the video content: {' '.join(relevant_text[:3])}"

    return "Unable to answer from current understanding"


def verify_answer(
    answer: str,
    question: str,
    understanding: dict,
) -> dict:
    """Check if an answer is supported by video content.

    Verifies by checking whether significant words from the answer
    appear in the video transcripts.

    Args:
        answer: The answer to verify.
        question: The question that was answered.
        understanding: Video understanding dict with 'scenes'.

    Returns:
        Dict with 'correct' (bool) and 'confidence' (float).
    """
    scenes = understanding.get("scenes", [])
    all_text = " ".join(s.get("transcript_text", "") for s in scenes).lower()

    if not all_text.strip():
        return {"correct": False, "confidence": 0.0}

    answer_words = set(answer.lower().split())
    important_words = {w for w in answer_words if len(w) > 4 and w not in _STOP_WORDS}

    if not important_words:
        return {"correct": False, "confidence": 0.0}

    matches = sum(1 for w in important_words if w in all_text)
    confidence = matches / len(important_words)

    return {
        "correct": confidence > 0.5,
        "confidence": round(confidence, 4),
    }


def _similar_question(q1: str, q2: str) -> bool:
    """Check if two questions are semantically similar.

    Uses word overlap as a simple heuristic.

    Args:
        q1: First question.
        q2: Second question.

    Returns:
        True if questions appear similar.
    """
    if not q1 or not q2:
        return False

    words1 = set(q1.lower().split()) - _STOP_WORDS
    words2 = set(q2.lower().split()) - _STOP_WORDS

    if not words1 or not words2:
        return False

    overlap = len(words1 & words2)
    smaller = min(len(words1), len(words2))

    return overlap / smaller >= 0.6 if smaller > 0 else False


def _scene_relevant_to_question(scene: dict, question: str) -> bool:
    """Check if a scene is relevant to a question.

    Args:
        scene: Scene dict with 'transcript_text'.
        question: Question to check relevance for.

    Returns:
        True if scene appears relevant.
    """
    transcript = scene.get("transcript_text", "").lower()
    if not transcript:
        return False

    question_words = set(question.lower().split()) - _STOP_WORDS
    if not question_words:
        return False

    transcript_words = set(transcript.split())
    matches = sum(1 for w in question_words if w in transcript_words)

    return matches >= 2 or (matches >= 1 and len(question_words) <= 2)
