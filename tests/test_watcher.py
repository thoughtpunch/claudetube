"""Tests for ActiveVideoWatcher and related classes."""

import pytest

from claudetube.analysis import (
    ActiveVideoWatcher,
    Hypothesis,
    WatcherAction,
)
from claudetube.cache.scenes import SceneBoundary


class TestHypothesis:
    """Tests for Hypothesis dataclass."""

    def test_create_hypothesis(self):
        hyp = Hypothesis(
            claim="The bug is in the auth module",
            evidence=[{"description": "Error at line 42"}],
            confidence=0.5,
        )
        assert hyp.claim == "The bug is in the auth module"
        assert len(hyp.evidence) == 1
        assert hyp.confidence == 0.5

    def test_default_values(self):
        hyp = Hypothesis(claim="Test claim")
        assert hyp.evidence == []
        assert hyp.confidence == 0.0

    def test_to_dict(self):
        hyp = Hypothesis(
            claim="Testing",
            evidence=[{"desc": "evidence 1"}],
            confidence=0.75,
        )
        d = hyp.to_dict()
        assert d["claim"] == "Testing"
        assert d["evidence"] == [{"desc": "evidence 1"}]
        assert d["confidence"] == 0.75

    def test_from_dict(self):
        d = {
            "claim": "Restored hypothesis",
            "evidence": [{"desc": "e1"}, {"desc": "e2"}],
            "confidence": 0.9,
        }
        hyp = Hypothesis.from_dict(d)
        assert hyp.claim == "Restored hypothesis"
        assert len(hyp.evidence) == 2
        assert hyp.confidence == 0.9

    def test_from_dict_missing_fields(self):
        d = {"claim": "Minimal"}
        hyp = Hypothesis.from_dict(d)
        assert hyp.claim == "Minimal"
        assert hyp.evidence == []
        assert hyp.confidence == 0.0


class TestWatcherAction:
    """Tests for WatcherAction dataclass."""

    def test_create_examine_quick(self):
        action = WatcherAction("examine_quick", scene_id=5)
        assert action.action == "examine_quick"
        assert action.scene_id == 5
        assert action.content is None

    def test_create_examine_deep(self):
        action = WatcherAction("examine_deep", scene_id=10)
        assert action.action == "examine_deep"
        assert action.scene_id == 10

    def test_create_answer(self):
        answer = {"main_answer": "The bug is fixed at 5:30"}
        action = WatcherAction("answer", content=answer)
        assert action.action == "answer"
        assert action.scene_id is None
        assert action.content["main_answer"] == "The bug is fixed at 5:30"

    def test_to_dict(self):
        action = WatcherAction("examine_deep", scene_id=7)
        d = action.to_dict()
        assert d["action"] == "examine_deep"
        assert d["scene_id"] == 7
        assert d["content"] is None

    def test_from_dict(self):
        d = {
            "action": "answer",
            "scene_id": None,
            "content": {"answer": "Test"},
        }
        action = WatcherAction.from_dict(d)
        assert action.action == "answer"
        assert action.scene_id is None
        assert action.content["answer"] == "Test"


class TestActiveVideoWatcher:
    """Tests for ActiveVideoWatcher class."""

    @pytest.fixture
    def sample_scenes(self):
        """Create sample scenes for testing."""
        return [
            {
                "scene_id": 0,
                "start_time": 0.0,
                "end_time": 30.0,
                "transcript_text": "Welcome to this tutorial about Python",
            },
            {
                "scene_id": 1,
                "start_time": 30.0,
                "end_time": 60.0,
                "transcript_text": "Here we have a bug in the authentication code",
            },
            {
                "scene_id": 2,
                "start_time": 60.0,
                "end_time": 90.0,
                "transcript_text": "Let me show you how to fix this bug",
            },
            {
                "scene_id": 3,
                "start_time": 90.0,
                "end_time": 120.0,
                "transcript_text": "Now the authentication is working correctly",
            },
            {
                "scene_id": 4,
                "start_time": 120.0,
                "end_time": 150.0,
                "transcript_text": "Thanks for watching this video",
            },
        ]

    def test_init(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="When do they fix the bug?",
            scenes=sample_scenes,
        )
        assert watcher.video_id == "test123"
        assert watcher.user_goal == "When do they fix the bug?"
        assert len(watcher.scenes) == 5
        assert watcher.examined == set()
        assert watcher.hypotheses == []
        assert watcher.confidence_threshold == 0.8
        assert watcher.max_examinations == 10

    def test_init_custom_thresholds(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="Test",
            scenes=sample_scenes,
            confidence_threshold=0.9,
            max_examinations=5,
        )
        assert watcher.confidence_threshold == 0.9
        assert watcher.max_examinations == 5

    def test_rank_unexplored_scenes(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="fix the bug",
            scenes=sample_scenes,
            video_duration=150.0,
        )
        ranked = watcher.rank_unexplored_scenes()

        # Should have all 5 scenes
        assert len(ranked) == 5

        # Each result should have a priority score
        for r in ranked:
            assert "priority" in r
            assert 0.0 <= r["priority"] <= 1.0

        # Scene 2 ("fix this bug") should be highly ranked
        top_ids = [r["scene_id"] for r in ranked[:2]]
        assert 2 in top_ids  # "how to fix this bug"

    def test_rank_unexplored_scenes_excludes_examined(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="fix the bug",
            scenes=sample_scenes,
            video_duration=150.0,
        )
        watcher.examined.add(2)  # Mark scene 2 as examined

        ranked = watcher.rank_unexplored_scenes()
        scene_ids = [r["scene_id"] for r in ranked]
        assert 2 not in scene_ids
        assert len(ranked) == 4

    def test_decide_next_action_examines_relevant_scene(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="fix the bug",
            scenes=sample_scenes,
        )
        action = watcher.decide_next_action()

        # Should decide to examine (not answer yet)
        assert action.action in ("examine_quick", "examine_deep")
        assert action.scene_id is not None

    def test_decide_next_action_answers_at_max_examinations(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="test",
            scenes=sample_scenes,
            max_examinations=2,
        )
        # Mark 2 scenes as examined
        watcher.examined = {0, 1}

        action = watcher.decide_next_action()
        assert action.action == "answer"
        assert action.content is not None

    def test_decide_next_action_answers_at_sufficient_confidence(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="test",
            scenes=sample_scenes,
            confidence_threshold=0.5,
        )
        # Add a high-confidence hypothesis
        watcher.hypotheses.append(Hypothesis(
            claim="Found it!",
            evidence=[{"description": "Evidence"}],
            confidence=0.6,
        ))

        action = watcher.decide_next_action()
        assert action.action == "answer"

    def test_decide_next_action_answers_when_no_scenes_left(self):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="test",
            scenes=[],  # No scenes
        )
        action = watcher.decide_next_action()
        assert action.action == "answer"

    def test_update_understanding_adds_findings(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="test",
            scenes=sample_scenes,
        )

        findings = [
            {"description": "Found a bug at line 42", "claim": "Bug at line 42"},
            {"description": "Authentication error"},
        ]
        watcher.update_understanding(scene_id=1, findings=findings)

        assert 1 in watcher.examined
        assert len(watcher.hypotheses) == 2

    def test_update_understanding_supports_existing_hypothesis(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="test",
            scenes=sample_scenes,
        )

        # Create initial hypothesis
        watcher.hypotheses.append(Hypothesis(
            claim="Bug in authentication code",
            evidence=[],
            confidence=0.0,  # Start with 0 confidence
        ))

        # Add supporting finding
        findings = [{"description": "authentication bug confirmed", "scene_id": 1}]
        watcher.update_understanding(scene_id=1, findings=findings)

        # Should have added to existing hypothesis
        assert len(watcher.hypotheses) == 1
        assert len(watcher.hypotheses[0].evidence) == 1
        # Confidence should be calculated based on evidence (not 0 anymore)
        assert watcher.hypotheses[0].confidence > 0.0

    def test_has_sufficient_confidence_false_no_hypotheses(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="test",
            scenes=sample_scenes,
        )
        assert watcher.has_sufficient_confidence() is False

    def test_has_sufficient_confidence_false_low_confidence(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="test",
            scenes=sample_scenes,
            confidence_threshold=0.8,
        )
        watcher.hypotheses.append(Hypothesis(
            claim="Maybe this",
            confidence=0.5,
        ))
        assert watcher.has_sufficient_confidence() is False

    def test_has_sufficient_confidence_true(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="test",
            scenes=sample_scenes,
            confidence_threshold=0.7,
        )
        watcher.hypotheses.append(Hypothesis(
            claim="Found it!",
            confidence=0.75,
        ))
        assert watcher.has_sufficient_confidence() is True

    def test_formulate_answer_no_hypotheses(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="test",
            scenes=sample_scenes,
        )
        watcher.examined = {0, 1, 2}

        answer = watcher.formulate_answer()
        assert answer["main_answer"] == "Unable to determine from video content"
        assert answer["confidence"] == 0.0
        assert answer["evidence"] == []
        assert answer["scenes_examined"] == 3

    def test_formulate_answer_with_hypotheses(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="test",
            scenes=sample_scenes,
        )
        watcher.examined = {0, 1}

        watcher.hypotheses.append(Hypothesis(
            claim="The bug is fixed at 1:00",
            evidence=[
                {"description": "Bug fix at 1:00", "timestamp": 60.0, "scene_id": 2}
            ],
            confidence=0.8,
        ))
        watcher.hypotheses.append(Hypothesis(
            claim="Alternative interpretation",
            confidence=0.4,
        ))

        answer = watcher.formulate_answer()
        assert answer["main_answer"] == "The bug is fixed at 1:00"
        assert answer["confidence"] == 0.8
        assert len(answer["evidence"]) == 1
        assert answer["alternative_interpretations"] == ["Alternative interpretation"]
        assert answer["scenes_examined"] == 2

    def test_get_state(self, sample_scenes):
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="fix the bug",
            scenes=sample_scenes,
            confidence_threshold=0.85,
            max_examinations=15,
            video_type="coding_tutorial",
            video_duration=150.0,
        )
        watcher.examined = {0, 2}
        watcher.hypotheses.append(Hypothesis(
            claim="Test hypothesis",
            confidence=0.5,
        ))

        state = watcher.get_state()
        assert state["video_id"] == "test123"
        assert state["user_goal"] == "fix the bug"
        assert set(state["examined"]) == {0, 2}
        assert len(state["hypotheses"]) == 1
        assert state["confidence_threshold"] == 0.85
        assert state["max_examinations"] == 15
        assert state["video_type"] == "coding_tutorial"
        assert state["video_duration"] == 150.0

    def test_from_state(self, sample_scenes):
        state = {
            "video_id": "test123",
            "user_goal": "fix the bug",
            "examined": [1, 3],
            "hypotheses": [
                {"claim": "Hypothesis A", "evidence": [], "confidence": 0.6}
            ],
            "confidence_threshold": 0.9,
            "max_examinations": 20,
            "video_type": "lecture",
            "video_duration": 300.0,
        }

        watcher = ActiveVideoWatcher.from_state(state, scenes=sample_scenes)

        assert watcher.video_id == "test123"
        assert watcher.user_goal == "fix the bug"
        assert watcher.examined == {1, 3}
        assert len(watcher.hypotheses) == 1
        assert watcher.hypotheses[0].claim == "Hypothesis A"
        assert watcher.confidence_threshold == 0.9
        assert watcher.max_examinations == 20
        assert watcher.video_type == "lecture"
        assert watcher.video_duration == 300.0

    def test_from_state_defaults_missing_fields(self, sample_scenes):
        """Test from_state with missing video_type/video_duration defaults."""
        state = {
            "video_id": "test123",
            "user_goal": "test defaults",
            "examined": [],
            "hypotheses": [],
        }
        watcher = ActiveVideoWatcher.from_state(state, scenes=sample_scenes)
        assert watcher.video_type == "unknown"
        assert watcher.video_duration == 0.0

    def test_state_roundtrip(self, sample_scenes):
        """Test saving and restoring state."""
        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="test roundtrip",
            scenes=sample_scenes,
            video_type="demo",
            video_duration=200.0,
        )
        watcher.examined = {0, 1, 2}
        watcher.hypotheses.append(Hypothesis(
            claim="Original hypothesis",
            evidence=[{"desc": "e1"}],
            confidence=0.7,
        ))

        state = watcher.get_state()
        restored = ActiveVideoWatcher.from_state(state, scenes=sample_scenes)

        assert restored.video_id == watcher.video_id
        assert restored.user_goal == watcher.user_goal
        assert restored.examined == watcher.examined
        assert restored.video_type == watcher.video_type
        assert restored.video_duration == watcher.video_duration
        assert len(restored.hypotheses) == len(watcher.hypotheses)
        assert restored.hypotheses[0].claim == watcher.hypotheses[0].claim
        assert restored.hypotheses[0].confidence == watcher.hypotheses[0].confidence

    def test_works_with_scene_boundary_objects(self):
        """Test that watcher works with SceneBoundary dataclass instances."""
        scenes = [
            SceneBoundary(
                scene_id=0,
                start_time=0.0,
                end_time=30.0,
                transcript_text="Introduction to the topic",
            ),
            SceneBoundary(
                scene_id=1,
                start_time=30.0,
                end_time=60.0,
                transcript_text="Here is the bug fix we need",
            ),
        ]

        watcher = ActiveVideoWatcher(
            video_id="test123",
            user_goal="bug fix",
            scenes=scenes,
            video_duration=60.0,
        )

        ranked = watcher.rank_unexplored_scenes()
        assert len(ranked) == 2

        # Each result should have priority
        for r in ranked:
            assert "priority" in r

        # Scene 1 should be higher priority (mentions "bug fix")
        assert ranked[0]["scene_id"] == 1

        action = watcher.decide_next_action()
        assert action.action in ("examine_quick", "examine_deep")


class TestWatcherIntegration:
    """Integration tests for ActiveVideoWatcher workflow."""

    def test_full_watching_workflow(self):
        """Test complete workflow from start to answer."""
        scenes = [
            {"scene_id": 0, "transcript_text": "Welcome to today's debugging session"},
            {"scene_id": 1, "transcript_text": "We have a null pointer bug here"},
            {"scene_id": 2, "transcript_text": "The fix is to add a null check"},
            {"scene_id": 3, "transcript_text": "Now it's working, bug is fixed"},
            {"scene_id": 4, "transcript_text": "Thanks for watching"},
        ]

        watcher = ActiveVideoWatcher(
            video_id="debug_video",
            user_goal="When is the bug fixed?",
            scenes=scenes,
            confidence_threshold=0.5,
            max_examinations=3,
        )

        # Simulate watching loop
        actions_taken = []
        for _ in range(5):  # Safety limit
            action = watcher.decide_next_action()
            actions_taken.append(action)

            if action.action == "answer":
                break

            # Simulate examining and finding evidence
            if action.scene_id == 3:
                # Found the fix
                watcher.update_understanding(action.scene_id, [
                    {
                        "description": "Bug is fixed at this point",
                        "claim": "Bug fixed in scene 3",
                        "initial_confidence": 0.7,
                        "scene_id": 3,
                    }
                ])
            elif action.scene_id == 2:
                # Found the fix approach
                watcher.update_understanding(action.scene_id, [
                    {
                        "description": "Fix involves adding null check",
                        "claim": "Null check is the fix",
                        "initial_confidence": 0.5,
                        "scene_id": 2,
                    }
                ])
            else:
                # Less relevant scene
                watcher.update_understanding(action.scene_id, [])

        # Should have produced an answer
        assert actions_taken[-1].action == "answer"
        answer = actions_taken[-1].content
        assert answer is not None
        assert answer["scenes_examined"] > 0

    def test_stops_at_max_examinations(self):
        """Test that watcher stops after max examinations."""
        scenes = [
            {"scene_id": i, "transcript_text": f"Scene {i} content"}
            for i in range(20)
        ]

        watcher = ActiveVideoWatcher(
            video_id="test",
            user_goal="find something",
            scenes=scenes,
            max_examinations=5,
        )

        # Examine until we get an answer
        for _ in range(10):
            action = watcher.decide_next_action()
            if action.action == "answer":
                break
            # Mark as examined with no findings
            watcher.update_understanding(action.scene_id, [])

        assert action.action == "answer"
        assert len(watcher.examined) <= 5
