"""
Active video watching agent for human-like video comprehension.

Implements an agent that actively decides what to examine in a video
rather than passively analyzing everything. The watcher:
- Ranks unexplored scenes by expected information gain
- Builds and updates hypotheses from findings
- Stops when confidence is sufficient
- Formulates answers with evidence

Architecture: Cheap First, Expensive Last
1. TEXT - Use transcript matching first for relevance
2. EMBEDDINGS - Use vector similarity only if text fails
3. VISUAL - Deep examination only for high-relevance scenes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from claudetube.analysis.attention import calculate_attention_priority

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from claudetube.analysis.embeddings import SceneEmbedding
    from claudetube.cache.scenes import SceneBoundary

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """A hypothesis about video content being built from evidence.

    Attributes:
        claim: The hypothesis statement.
        evidence: List of evidence dicts supporting this hypothesis.
        confidence: Confidence score from 0.0 to 1.0.
    """

    claim: str
    evidence: list[dict] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "claim": self.claim,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Hypothesis:
        """Create from dictionary."""
        return cls(
            claim=data["claim"],
            evidence=data.get("evidence", []),
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class WatcherAction:
    """An action the watcher decides to take.

    Attributes:
        action: Type of action - 'examine_quick', 'examine_deep', or 'answer'.
        scene_id: Scene to examine (for examine_* actions).
        content: Answer content (for answer action).
    """

    action: str  # 'examine_quick', 'examine_deep', 'answer'
    scene_id: int | None = None
    content: dict | None = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "action": self.action,
            "scene_id": self.scene_id,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict) -> WatcherAction:
        """Create from dictionary."""
        return cls(
            action=data["action"],
            scene_id=data.get("scene_id"),
            content=data.get("content"),
        )


class ActiveVideoWatcher:
    """Agent that actively decides what to examine in a video.

    Rather than analyzing all content, the watcher strategically
    explores based on the user's goal, building hypotheses and
    stopping when sufficient confidence is reached.

    Example usage:
        watcher = ActiveVideoWatcher(
            video_id="abc123",
            user_goal="When do they fix the bug?",
            scenes=scenes_list,
            cache_dir=cache_dir,
        )

        while True:
            action = watcher.decide_next_action()
            if action.action == 'answer':
                return action.content

            # Examine the scene (quick or deep based on action.action)
            findings = examine_scene(action.scene_id, action.action)
            watcher.update_understanding(action.scene_id, findings)
    """

    def __init__(
        self,
        video_id: str,
        user_goal: str,
        scenes: list[dict | SceneBoundary],
        cache_dir: Path | None = None,
        confidence_threshold: float = 0.8,
        max_examinations: int = 10,
        video_type: str = "unknown",
        video_duration: float = 0.0,
        scene_embeddings: list[SceneEmbedding] | None = None,
        goal_embedding: np.ndarray | None = None,
    ):
        """Initialize the active video watcher.

        Args:
            video_id: Video identifier.
            user_goal: User's question or goal for watching.
            scenes: List of scene data (dicts or SceneBoundary objects).
            cache_dir: Optional video cache directory for memory/embeddings.
            confidence_threshold: Stop when hypothesis confidence reaches this.
            max_examinations: Maximum scenes to examine before answering.
            video_type: Video category for attention weighting (e.g.,
                'coding_tutorial', 'lecture'). Defaults to 'unknown'.
            video_duration: Total video duration in seconds. Used for
                structural importance scoring.
            scene_embeddings: Pre-loaded scene embeddings. If None, attempts
                to load from cache_dir.
            goal_embedding: Pre-computed goal embedding. If None, attention
                scoring falls back to keyword matching.
        """
        self.video_id = video_id
        self.user_goal = user_goal
        self.scenes = scenes
        self.cache_dir = cache_dir
        self.examined: set[int] = set()
        self.hypotheses: list[Hypothesis] = []
        self.confidence_threshold = confidence_threshold
        self.max_examinations = max_examinations
        self.video_type = video_type
        self.video_duration = video_duration

        # Pre-compute goal words for text matching
        self._goal_words = set(user_goal.lower().split())

        # Embeddings for semantic attention scoring
        self._scene_embeddings: list[SceneEmbedding] | None = scene_embeddings
        self._goal_embedding: np.ndarray | None = goal_embedding
        self._embedding_lookup: dict[int, SceneEmbedding] | None = None
        self._embeddings_loaded = scene_embeddings is not None

        # Try loading cached embeddings if not provided
        if self._scene_embeddings is None and cache_dir is not None:
            self._load_embeddings()

        # Build lookup for fast access
        if self._scene_embeddings is not None:
            self._embedding_lookup = {e.scene_id: e for e in self._scene_embeddings}

        logger.info(
            f"ActiveVideoWatcher initialized for '{user_goal}' with "
            f"{len(scenes)} scenes (threshold={confidence_threshold}, "
            f"embeddings={'yes' if self._scene_embeddings else 'no'})"
        )

    def decide_next_action(self) -> WatcherAction:
        """Decide what to examine next.

        Returns:
            WatcherAction indicating what to do next:
            - examine_quick: Quick look at a moderately relevant scene
            - examine_deep: Deep analysis of a highly relevant scene
            - answer: Stop examining and return the answer
        """
        # Check if we have sufficient confidence
        if self.has_sufficient_confidence():
            logger.info(
                f"Sufficient confidence ({self._get_max_confidence():.2f}) - "
                "formulating answer"
            )
            return WatcherAction("answer", content=self.formulate_answer())

        # Check examination budget
        if len(self.examined) >= self.max_examinations:
            logger.info(
                f"Reached max examinations ({self.max_examinations}) - "
                "formulating answer"
            )
            return WatcherAction("answer", content=self.formulate_answer())

        # Rank unexplored scenes
        candidates = self.rank_unexplored_scenes()

        if not candidates:
            logger.info("No more scenes to examine - formulating answer")
            return WatcherAction("answer", content=self.formulate_answer())

        best = candidates[0]
        scene_id = best["scene_id"]
        priority = best["priority"]

        # Decide examination depth based on attention priority
        if priority > 0.6:
            logger.info(
                f"High priority ({priority:.2f}) - examining scene {scene_id} deeply"
            )
            return WatcherAction("examine_deep", scene_id=scene_id)
        else:
            logger.info(
                f"Moderate priority ({priority:.2f}) - quick examination of "
                f"scene {scene_id}"
            )
            return WatcherAction("examine_quick", scene_id=scene_id)

    def _load_embeddings(self) -> None:
        """Try to load cached scene embeddings from cache_dir.

        Sets self._scene_embeddings if embeddings are found on disk.
        Silently does nothing if no embeddings are cached.
        """
        if self._embeddings_loaded or self.cache_dir is None:
            return

        self._embeddings_loaded = True

        try:
            from claudetube.analysis.embeddings import load_embeddings

            embeddings = load_embeddings(self.cache_dir)
            if embeddings:
                self._scene_embeddings = embeddings
                logger.info(
                    f"Loaded {len(embeddings)} cached scene embeddings "
                    "for attention scoring"
                )
        except Exception as e:
            logger.debug(f"Could not load cached embeddings: {e}")

    def rank_unexplored_scenes(self) -> list[dict]:
        """Rank scenes by attention priority (multi-factor model).

        Uses the attention priority model which scores scenes by:
        relevance, information density, novelty, visual salience,
        audio emphasis, and structural importance. When embeddings
        are available, uses semantic similarity for relevance and
        novelty scoring.

        Returns:
            List of dicts with scene_id, priority, and scene data,
            sorted by priority descending.
        """
        candidates = []
        previous_scenes = [
            s for s in self.scenes if self._get_scene_id(s) in self.examined
        ]

        # Build previous embeddings list for novelty calculation
        previous_embeddings: list[SceneEmbedding] | None = None
        if self._embedding_lookup and previous_scenes:
            previous_embeddings = [
                self._embedding_lookup[self._get_scene_id(s)]
                for s in previous_scenes
                if self._get_scene_id(s) in self._embedding_lookup
            ]
            if not previous_embeddings:
                previous_embeddings = None

        for scene in self.scenes:
            scene_id = self._get_scene_id(scene)
            if scene_id in self.examined:
                continue

            # Look up scene embedding if available
            scene_embedding = (
                self._embedding_lookup.get(scene_id) if self._embedding_lookup else None
            )

            priority = calculate_attention_priority(
                scene=scene,
                user_goal=self.user_goal,
                video_type=self.video_type,
                previous_scenes=previous_scenes,
                total_scenes=len(self.scenes),
                video_duration=self.video_duration,
                scene_embedding=scene_embedding,
                goal_embedding=self._goal_embedding,
                previous_embeddings=previous_embeddings,
            )

            candidates.append(
                {
                    "scene_id": scene_id,
                    "priority": priority,
                    "scene": scene,
                }
            )

        return sorted(candidates, key=lambda x: x["priority"], reverse=True)

    def update_understanding(self, scene_id: int, findings: list[dict]) -> None:
        """Update hypotheses based on examination findings.

        Args:
            scene_id: Scene that was examined.
            findings: List of finding dicts, each with at least:
                - description: What was observed
                - claim (optional): Hypothesis this supports
                - initial_confidence (optional): Confidence boost
                - timestamp (optional): When in the scene
        """
        self.examined.add(scene_id)
        logger.info(
            f"Updating understanding from scene {scene_id}: {len(findings)} findings"
        )

        for finding in findings:
            matched = False

            # Try to match to existing hypothesis
            for hyp in self.hypotheses:
                if self._finding_supports_hypothesis(finding, hyp):
                    hyp.evidence.append(finding)
                    hyp.confidence = self._calculate_confidence(hyp)
                    matched = True
                    logger.debug(
                        f"Finding supports existing hypothesis '{hyp.claim}' "
                        f"(confidence now {hyp.confidence:.2f})"
                    )
                    break

            if not matched:
                # Create new hypothesis
                claim = finding.get("claim") or finding.get("description", "")
                if claim:
                    new_hyp = Hypothesis(
                        claim=claim,
                        evidence=[finding],
                        confidence=finding.get("initial_confidence", 0.3),
                    )
                    self.hypotheses.append(new_hyp)
                    logger.debug(
                        f"Created new hypothesis '{claim}' "
                        f"(confidence {new_hyp.confidence:.2f})"
                    )

    def _finding_supports_hypothesis(self, finding: dict, hyp: Hypothesis) -> bool:
        """Check if a finding supports an existing hypothesis.

        Uses simple keyword overlap between finding description
        and hypothesis claim.

        Args:
            finding: Finding dict with at least 'description'.
            hyp: Hypothesis to check against.

        Returns:
            True if finding appears to support the hypothesis.
        """
        description = finding.get("description", "").lower()
        claim = hyp.claim.lower()

        if not description or not claim:
            return False

        # Check for significant word overlap
        desc_words = set(description.split())
        claim_words = set(claim.split())

        # Filter common words
        stop_words = {
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
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
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
        }
        desc_words -= stop_words
        claim_words -= stop_words

        if not desc_words or not claim_words:
            return False

        overlap = len(desc_words & claim_words)
        return overlap >= 2 or (overlap >= 1 and len(claim_words) <= 3)

    def _calculate_confidence(self, hyp: Hypothesis) -> float:
        """Calculate confidence score for a hypothesis.

        Confidence increases with:
        - More pieces of evidence
        - Evidence from different scenes
        - High individual evidence confidence

        Args:
            hyp: Hypothesis to evaluate.

        Returns:
            Confidence score from 0.0 to 1.0.
        """
        if not hyp.evidence:
            return 0.0

        # Base confidence from evidence count (diminishing returns)
        evidence_count = len(hyp.evidence)
        base_confidence = min(0.6, evidence_count * 0.15)

        # Bonus for evidence from multiple scenes
        scenes_with_evidence = set()
        for e in hyp.evidence:
            if "scene_id" in e:
                scenes_with_evidence.add(e["scene_id"])
            elif "timestamp" in e:
                # Try to infer scene from timestamp
                scenes_with_evidence.add(e.get("timestamp", 0))

        multi_scene_bonus = min(0.2, len(scenes_with_evidence) * 0.05)

        # Average individual confidence scores
        avg_individual = (
            sum(e.get("initial_confidence", 0.3) for e in hyp.evidence) / evidence_count
            if evidence_count
            else 0
        )

        confidence = base_confidence + multi_scene_bonus + (avg_individual * 0.3)

        return min(1.0, confidence)

    def has_sufficient_confidence(self) -> bool:
        """Check if any hypothesis has reached confidence threshold.

        Returns:
            True if confident enough to answer.
        """
        if not self.hypotheses:
            return False
        return self._get_max_confidence() >= self.confidence_threshold

    def _get_max_confidence(self) -> float:
        """Get the maximum confidence among all hypotheses."""
        if not self.hypotheses:
            return 0.0
        return max(h.confidence for h in self.hypotheses)

    def formulate_answer(self) -> dict:
        """Generate answer from hypotheses.

        Ranks hypotheses by a combined score of confidence AND relevance
        to the original question. This prevents high-confidence but
        irrelevant hypotheses from winning over lower-confidence but
        more relevant ones.

        Returns:
            Dict with:
            - main_answer: Best hypothesis claim
            - confidence: Confidence score
            - evidence: List of supporting evidence
            - alternative_interpretations: Other plausible hypotheses
            - scenes_examined: Number of scenes looked at
        """
        if not self.hypotheses:
            return {
                "main_answer": "Unable to determine from video content",
                "confidence": 0.0,
                "evidence": [],
                "alternative_interpretations": [],
                "scenes_examined": len(self.examined),
            }

        # Score hypotheses by combined confidence and relevance
        scored = []
        for hyp in self.hypotheses:
            relevance = self._calculate_relevance(hyp.claim)
            # Combined score: weight relevance heavily to prefer answers
            # that actually address the question
            combined = (hyp.confidence * 0.4) + (relevance * 0.6)
            scored.append((combined, relevance, hyp))

        # Sort by combined score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = [item[2] for item in scored]

        best = ranked[0]
        return {
            "main_answer": best.claim,
            "confidence": best.confidence,
            "evidence": [
                {
                    "timestamp": e.get("timestamp"),
                    "observation": e.get("description"),
                    "scene_id": e.get("scene_id"),
                }
                for e in best.evidence
            ],
            "alternative_interpretations": [h.claim for h in ranked[1:3]],
            "scenes_examined": len(self.examined),
        }

    def _calculate_relevance(self, claim: str) -> float:
        """Calculate how relevant a claim is to the user's goal.

        Uses keyword overlap and phrase matching.

        Args:
            claim: The hypothesis claim text.

        Returns:
            Relevance score from 0.0 to 1.0.
        """
        if not claim:
            return 0.0

        claim_lower = claim.lower()
        claim_words = set(claim_lower.split())

        # Filter common words for better matching
        stop_words = {
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
            "and",
            "or",
            "but",
            "if",
            "it",
            "this",
            "that",
            "what",
            "how",
            "when",
            "where",
            "who",
            "which",
            "why",
        }
        goal_words = self._goal_words - stop_words
        claim_words -= stop_words

        if not goal_words:
            return 0.5  # Neutral if no keywords

        # Calculate word overlap ratio
        overlap = len(goal_words & claim_words)
        overlap_ratio = overlap / len(goal_words) if goal_words else 0

        # Bonus for containing key phrases from the goal
        goal_lower = self.user_goal.lower()
        phrase_bonus = 0.0
        goal_words_list = goal_lower.split()
        for i in range(len(goal_words_list) - 1):
            phrase = f"{goal_words_list[i]} {goal_words_list[i + 1]}"
            if phrase in claim_lower:
                phrase_bonus += 0.15

        # Score from 0 to 1
        score = min(1.0, overlap_ratio * 0.7 + phrase_bonus)
        return score

    def get_state(self) -> dict:
        """Get current watcher state for serialization.

        Returns:
            Dict with video_id, goal, examined scenes, hypotheses,
            and attention model parameters.
        """
        return {
            "video_id": self.video_id,
            "user_goal": self.user_goal,
            "examined": list(self.examined),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "confidence_threshold": self.confidence_threshold,
            "max_examinations": self.max_examinations,
            "video_type": self.video_type,
            "video_duration": self.video_duration,
        }

    @classmethod
    def from_state(
        cls,
        state: dict,
        scenes: list[dict | SceneBoundary],
        cache_dir: Path | None = None,
        scene_embeddings: list[SceneEmbedding] | None = None,
        goal_embedding: np.ndarray | None = None,
    ) -> ActiveVideoWatcher:
        """Restore watcher from saved state.

        Args:
            state: State dict from get_state().
            scenes: Scene data (required, not serialized).
            cache_dir: Optional cache directory.
            scene_embeddings: Pre-loaded scene embeddings.
            goal_embedding: Pre-computed goal embedding.

        Returns:
            Restored ActiveVideoWatcher instance.
        """
        watcher = cls(
            video_id=state["video_id"],
            user_goal=state["user_goal"],
            scenes=scenes,
            cache_dir=cache_dir,
            confidence_threshold=state.get("confidence_threshold", 0.8),
            max_examinations=state.get("max_examinations", 10),
            video_type=state.get("video_type", "unknown"),
            video_duration=state.get("video_duration", 0.0),
            scene_embeddings=scene_embeddings,
            goal_embedding=goal_embedding,
        )

        watcher.examined = set(state.get("examined", []))
        watcher.hypotheses = [
            Hypothesis.from_dict(h) for h in state.get("hypotheses", [])
        ]

        return watcher

    def _get_scene_id(self, scene: dict | SceneBoundary) -> int:
        """Extract scene_id from scene data."""
        if hasattr(scene, "scene_id"):
            return scene.scene_id
        return scene.get("scene_id", 0)
