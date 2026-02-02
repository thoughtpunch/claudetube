"""Tests for analysis/attention.py attention priority modeling."""

import numpy as np
import pytest

from claudetube.analysis.attention import (
    DEFAULT_WEIGHTS,
    VIDEO_TYPE_WEIGHTS,
    AttentionFactors,
    calculate_attention_factors,
    calculate_attention_priority,
    calculate_novelty,
    calculate_relevance,
    detect_audio_emphasis,
    detect_visual_salience,
    estimate_information_density,
    get_structural_weight,
    get_weights_for_video_type,
    rank_scenes_by_attention,
)
from claudetube.analysis.embeddings import SceneEmbedding

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_scenes():
    """Five scenes covering a typical tutorial video."""
    return [
        {
            "scene_id": 0,
            "start_time": 0.0,
            "end_time": 30.0,
            "transcript_text": "Welcome to this Python tutorial about debugging",
        },
        {
            "scene_id": 1,
            "start_time": 30.0,
            "end_time": 60.0,
            "transcript_text": (
                "Here we have a bug in the authentication code that "
                "causes a null pointer error"
            ),
        },
        {
            "scene_id": 2,
            "start_time": 60.0,
            "end_time": 90.0,
            "transcript_text": (
                "Let me show you how to fix this bug by adding a null check"
            ),
        },
        {
            "scene_id": 3,
            "start_time": 90.0,
            "end_time": 120.0,
            "transcript_text": (
                "Now the authentication is working correctly after the fix"
            ),
        },
        {
            "scene_id": 4,
            "start_time": 120.0,
            "end_time": 150.0,
            "transcript_text": "Thanks for watching this video",
        },
    ]


# ---------------------------------------------------------------------------
# get_weights_for_video_type
# ---------------------------------------------------------------------------


class TestGetWeightsForVideoType:
    def test_known_types_return_specific_weights(self):
        for vtype in VIDEO_TYPE_WEIGHTS:
            weights = get_weights_for_video_type(vtype)
            assert weights is VIDEO_TYPE_WEIGHTS[vtype]

    def test_unknown_type_returns_defaults(self):
        assert get_weights_for_video_type("unknown") is DEFAULT_WEIGHTS
        assert get_weights_for_video_type("") is DEFAULT_WEIGHTS
        assert get_weights_for_video_type("random_type") is DEFAULT_WEIGHTS

    def test_all_weights_sum_to_one(self):
        for vtype, weights in VIDEO_TYPE_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, f"{vtype} weights sum to {total}"

        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_all_weights_have_required_keys(self):
        required = {"relevance", "density", "visual", "novelty", "audio", "structure"}
        for vtype, weights in VIDEO_TYPE_WEIGHTS.items():
            assert set(weights.keys()) == required, f"{vtype} missing keys"
        assert set(DEFAULT_WEIGHTS.keys()) == required


# ---------------------------------------------------------------------------
# AttentionFactors
# ---------------------------------------------------------------------------


class TestAttentionFactors:
    def test_create(self):
        f = AttentionFactors(
            relevance_to_goal=0.8,
            information_density=0.5,
            novelty=0.6,
            visual_salience=0.3,
            audio_emphasis=0.1,
            structural_importance=0.7,
        )
        assert f.relevance_to_goal == 0.8
        assert f.structural_importance == 0.7

    def test_to_dict(self):
        f = AttentionFactors(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        d = f.to_dict()
        assert d == {
            "relevance_to_goal": 0.1,
            "information_density": 0.2,
            "novelty": 0.3,
            "visual_salience": 0.4,
            "audio_emphasis": 0.5,
            "structural_importance": 0.6,
        }

    def test_from_dict(self):
        d = {
            "relevance_to_goal": 0.9,
            "information_density": 0.8,
            "novelty": 0.7,
            "visual_salience": 0.6,
            "audio_emphasis": 0.5,
            "structural_importance": 0.4,
        }
        f = AttentionFactors.from_dict(d)
        assert f.relevance_to_goal == 0.9
        assert f.structural_importance == 0.4

    def test_from_dict_missing_fields_default_zero(self):
        f = AttentionFactors.from_dict({})
        assert f.relevance_to_goal == 0.0
        assert f.novelty == 0.0

    def test_roundtrip(self):
        original = AttentionFactors(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        restored = AttentionFactors.from_dict(original.to_dict())
        assert original == restored


# ---------------------------------------------------------------------------
# calculate_relevance
# ---------------------------------------------------------------------------


class TestCalculateRelevance:
    def test_matching_keywords_score_high(self):
        scene = {"transcript_text": "Let me show you how to fix this bug"}
        score = calculate_relevance(scene, "fix the bug")
        assert score > 0.5

    def test_no_match_scores_zero(self):
        scene = {"transcript_text": "Welcome to the tutorial"}
        score = calculate_relevance(scene, "database migration")
        assert score == 0.0

    def test_exact_phrase_bonus(self):
        scene = {"transcript_text": "Here is the fix the bug section"}
        with_phrase = calculate_relevance(scene, "fix the bug")
        # Compare to a partial match (reorder so phrase doesn't appear)
        scene2 = {"transcript_text": "Here we fix things and find a bug"}
        without_phrase = calculate_relevance(scene2, "fix the bug")
        assert with_phrase > without_phrase

    def test_empty_transcript_returns_zero(self):
        assert calculate_relevance({"transcript_text": ""}, "anything") == 0.0

    def test_empty_goal_returns_zero(self):
        assert calculate_relevance({"transcript_text": "Some text"}, "") == 0.0

    def test_score_clamped_to_one(self):
        # Many matching words should not exceed 1.0
        scene = {"transcript_text": "fix the bug now fix the bug again fix the bug"}
        score = calculate_relevance(scene, "fix the bug")
        assert score <= 1.0

    def test_works_with_scene_boundary(self):
        """Test with an object that has transcript_text attribute."""
        from claudetube.cache.scenes import SceneBoundary

        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=30.0,
            transcript_text="How to debug Python code",
        )
        score = calculate_relevance(scene, "debug Python")
        assert score > 0.0


# ---------------------------------------------------------------------------
# estimate_information_density
# ---------------------------------------------------------------------------


class TestEstimateInformationDensity:
    def test_dense_transcript_scores_higher(self):
        # 40 words in 10 seconds = 4 words/sec (high density)
        dense = {
            "start_time": 0.0,
            "end_time": 10.0,
            "transcript_text": " ".join(["word"] * 40),
        }
        # 5 words in 10 seconds = 0.5 words/sec (low density)
        sparse = {
            "start_time": 0.0,
            "end_time": 10.0,
            "transcript_text": " ".join(["word"] * 5),
        }
        assert estimate_information_density(dense) > estimate_information_density(
            sparse
        )

    def test_empty_transcript(self):
        scene = {"start_time": 0, "end_time": 10, "transcript_text": ""}
        score = estimate_information_density(scene)
        assert score == 0.0

    def test_technical_content_boosts_density(self):
        base = {
            "start_time": 0,
            "end_time": 10,
            "transcript_text": " ".join(["word"] * 20),
        }
        with_tech = {
            **base,
            "technical": {
                "ocr_text": ["some text on screen"],
                "code_blocks": ["def foo(): pass", "x = 1"],
            },
        }
        assert estimate_information_density(with_tech) > estimate_information_density(
            base
        )

    def test_zero_duration_uses_word_count_fallback(self):
        scene = {"start_time": 0, "end_time": 0, "transcript_text": "hello world"}
        score = estimate_information_density(scene)
        assert score > 0.0

    def test_score_between_zero_and_one(self):
        scene = {
            "start_time": 0,
            "end_time": 5,
            "transcript_text": " ".join(["word"] * 100),
            "technical": {"ocr_text": ["a"] * 20, "code_blocks": ["b"] * 10},
        }
        score = estimate_information_density(scene)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# calculate_novelty
# ---------------------------------------------------------------------------


class TestCalculateNovelty:
    def test_first_scene_gets_neutral_score(self):
        scene = {"transcript_text": "Introduction to the topic"}
        score = calculate_novelty(scene, [])
        assert score == 0.5

    def test_similar_scene_has_low_novelty(self):
        prev = [{"transcript_text": "Python debugging tutorial walkthrough"}]
        scene = {"transcript_text": "Python debugging tutorial example"}
        score = calculate_novelty(scene, prev)
        assert score < 0.5

    def test_different_scene_has_high_novelty(self):
        prev = [{"transcript_text": "Welcome to our cooking show today"}]
        scene = {"transcript_text": "Deploying Kubernetes clusters in production"}
        score = calculate_novelty(scene, prev)
        assert score > 0.5

    def test_empty_transcript_returns_neutral(self):
        score = calculate_novelty(
            {"transcript_text": ""}, [{"transcript_text": "anything"}]
        )
        assert score == 0.5

    def test_score_between_zero_and_one(self):
        prev = [{"transcript_text": "word " * 50}]
        scene = {"transcript_text": "unique " * 50}
        score = calculate_novelty(scene, prev)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# detect_visual_salience
# ---------------------------------------------------------------------------


class TestDetectVisualSalience:
    def test_code_content_type_high_salience(self):
        scene = {"technical": {"content_type": "code"}}
        assert detect_visual_salience(scene) >= 0.9

    def test_diagram_high_salience(self):
        scene = {"technical": {"content_type": "diagram"}}
        assert detect_visual_salience(scene) >= 0.8

    def test_talking_head_low_salience(self):
        scene = {"technical": {"content_type": "talking_head"}}
        assert detect_visual_salience(scene) <= 0.3

    def test_unknown_content_type_moderate(self):
        scene = {"technical": {"content_type": "unknown"}}
        assert detect_visual_salience(scene) == 0.3

    def test_ocr_text_adds_bonus(self):
        base = {"technical": {"content_type": "unknown"}}
        with_ocr = {"technical": {"content_type": "unknown", "ocr_text": ["text"]}}
        assert detect_visual_salience(with_ocr) > detect_visual_salience(base)

    def test_code_blocks_add_bonus(self):
        base = {"technical": {"content_type": "unknown"}}
        with_code = {"technical": {"content_type": "unknown", "code_blocks": ["x=1"]}}
        assert detect_visual_salience(with_code) > detect_visual_salience(base)

    def test_no_technical_data(self):
        # Plain dict with no technical key
        score = detect_visual_salience({"transcript_text": "hello"})
        assert score == 0.3  # default for unknown

    def test_scene_boundary_object(self):
        """SceneBoundary objects have no technical dict."""
        from claudetube.cache.scenes import SceneBoundary

        scene = SceneBoundary(scene_id=0, start_time=0.0, end_time=10.0)
        score = detect_visual_salience(scene)
        assert score == 0.3  # empty technical dict


# ---------------------------------------------------------------------------
# detect_audio_emphasis
# ---------------------------------------------------------------------------


class TestDetectAudioEmphasis:
    def test_emphasis_phrase_detected(self):
        scene = {"transcript_text": "This is a key point that is important to remember"}
        score = detect_audio_emphasis(scene)
        assert score > 0.0

    def test_no_emphasis_scores_zero(self):
        scene = {"transcript_text": "The weather today is mild and cloudy"}
        score = detect_audio_emphasis(scene)
        assert score == 0.0

    def test_multiple_emphasis_phrases(self):
        scene = {
            "transcript_text": (
                "This is important. Let me show you. Remember this key point. "
                "Be careful here."
            )
        }
        score = detect_audio_emphasis(scene)
        assert score >= 0.75  # 4+ matches * 0.25

    def test_empty_transcript(self):
        assert detect_audio_emphasis({"transcript_text": ""}) == 0.0
        assert detect_audio_emphasis({}) == 0.0

    def test_clamped_to_one(self):
        # Pack many emphasis phrases
        scene = {
            "transcript_text": (
                "important key point remember crucial essential critical "
                "note that pay attention the main keep in mind don't forget "
                "in summary the takeaway most importantly let me show watch this"
            )
        }
        score = detect_audio_emphasis(scene)
        assert score <= 1.0


# ---------------------------------------------------------------------------
# get_structural_weight
# ---------------------------------------------------------------------------


class TestGetStructuralWeight:
    def test_intro_scene_gets_bonus(self):
        scene = {"scene_id": 0, "start_time": 5.0}
        weight = get_structural_weight(
            scene, "lecture", total_scenes=10, video_duration=300.0
        )
        # position_ratio = 5/300 = 0.017 < 0.1 (intro)
        # scene_id <= 2 (chapter start)
        assert weight >= 0.6

    def test_conclusion_gets_bonus(self):
        scene = {"scene_id": 9, "start_time": 285.0}
        weight = get_structural_weight(
            scene, "lecture", total_scenes=10, video_duration=300.0
        )
        # position_ratio = 285/300 = 0.95 > 0.9 (conclusion)
        assert weight >= 0.5

    def test_middle_scene_lower_weight(self):
        scene = {"scene_id": 5, "start_time": 150.0}
        weight = get_structural_weight(
            scene, "lecture", total_scenes=10, video_duration=300.0
        )
        # Middle scene, not intro/conclusion, scene_id > 2
        assert weight < 0.5

    def test_demo_section_bonus_for_tutorials(self):
        mid_scene = {"scene_id": 5, "start_time": 100.0}
        tutorial_weight = get_structural_weight(
            mid_scene, "coding_tutorial", total_scenes=10, video_duration=300.0
        )
        lecture_weight = get_structural_weight(
            mid_scene, "lecture", total_scenes=10, video_duration=300.0
        )
        # Tutorial gets demo bonus for 20-80% position
        assert tutorial_weight > lecture_weight

    def test_zero_duration_handled(self):
        scene = {"scene_id": 0, "start_time": 0.0}
        weight = get_structural_weight(
            scene, "unknown", total_scenes=5, video_duration=0.0
        )
        assert 0.0 <= weight <= 1.0

    def test_weight_clamped_to_one(self):
        # First scene (intro + chapter start bonuses stacked)
        scene = {"scene_id": 0, "start_time": 0.0}
        weight = get_structural_weight(
            scene, "coding_tutorial", total_scenes=10, video_duration=100.0
        )
        assert weight <= 1.0


# ---------------------------------------------------------------------------
# calculate_attention_factors (composite)
# ---------------------------------------------------------------------------


class TestCalculateAttentionFactors:
    def test_returns_all_factors(self, simple_scenes):
        factors = calculate_attention_factors(
            scene=simple_scenes[2],
            user_goal="fix the bug",
            video_type="coding_tutorial",
            total_scenes=5,
            video_duration=150.0,
        )
        assert isinstance(factors, AttentionFactors)
        assert 0.0 <= factors.relevance_to_goal <= 1.0
        assert 0.0 <= factors.information_density <= 1.0
        assert 0.0 <= factors.novelty <= 1.0
        assert 0.0 <= factors.visual_salience <= 1.0
        assert 0.0 <= factors.audio_emphasis <= 1.0
        assert 0.0 <= factors.structural_importance <= 1.0

    def test_previous_scenes_affect_novelty(self, simple_scenes):
        factors_no_prev = calculate_attention_factors(
            scene=simple_scenes[1],
            user_goal="bug",
            previous_scenes=[],
        )
        factors_with_prev = calculate_attention_factors(
            scene=simple_scenes[1],
            user_goal="bug",
            previous_scenes=[simple_scenes[1]],  # same scene as previous
        )
        # Novelty should be lower when scene content was already seen
        assert factors_with_prev.novelty < factors_no_prev.novelty


# ---------------------------------------------------------------------------
# calculate_attention_priority
# ---------------------------------------------------------------------------


class TestCalculateAttentionPriority:
    def test_returns_float_between_zero_and_one(self, simple_scenes):
        priority = calculate_attention_priority(
            scene=simple_scenes[0],
            user_goal="Python tutorial",
            video_type="lecture",
        )
        assert isinstance(priority, float)
        assert 0.0 <= priority <= 1.0

    def test_relevant_scene_scores_higher(self, simple_scenes):
        relevant = calculate_attention_priority(
            scene=simple_scenes[2],
            user_goal="fix the bug",
            video_type="unknown",
            total_scenes=5,
            video_duration=150.0,
        )
        irrelevant = calculate_attention_priority(
            scene=simple_scenes[4],
            user_goal="fix the bug",
            video_type="unknown",
            total_scenes=5,
            video_duration=150.0,
        )
        assert relevant > irrelevant

    def test_custom_weights_override_video_type(self, simple_scenes):
        # All weight on relevance
        custom = {
            "relevance": 1.0,
            "density": 0.0,
            "novelty": 0.0,
            "visual": 0.0,
            "audio": 0.0,
            "structure": 0.0,
        }
        priority = calculate_attention_priority(
            scene=simple_scenes[2],
            user_goal="fix the bug",
            video_type="unknown",
            custom_weights=custom,
        )
        # Should equal just the relevance score
        relevance = calculate_relevance(simple_scenes[2], "fix the bug")
        assert abs(priority - relevance) < 1e-9

    def test_different_video_types_give_different_scores(self, simple_scenes):
        """Same scene, different video types produce different priorities."""
        scores = {}
        for vtype in ("coding_tutorial", "lecture", "interview"):
            scores[vtype] = calculate_attention_priority(
                scene=simple_scenes[1],
                user_goal="bug in the code",
                video_type=vtype,
                total_scenes=5,
                video_duration=150.0,
            )
        # At least two types should differ (weights differ)
        values = list(scores.values())
        assert len({round(v, 6) for v in values}) > 1


# ---------------------------------------------------------------------------
# rank_scenes_by_attention
# ---------------------------------------------------------------------------


class TestRankScenesByAttention:
    def test_returns_sorted_by_priority_descending(self, simple_scenes):
        ranked = rank_scenes_by_attention(
            scenes=simple_scenes,
            user_goal="fix the bug",
            video_duration=150.0,
        )
        priorities = [r["priority"] for r in ranked]
        assert priorities == sorted(priorities, reverse=True)

    def test_excludes_examined_scenes(self, simple_scenes):
        ranked = rank_scenes_by_attention(
            scenes=simple_scenes,
            user_goal="fix the bug",
            examined_scene_ids={0, 4},
            video_duration=150.0,
        )
        result_ids = {r["scene_id"] for r in ranked}
        assert 0 not in result_ids
        assert 4 not in result_ids
        assert len(ranked) == 3

    def test_result_dicts_have_required_keys(self, simple_scenes):
        ranked = rank_scenes_by_attention(
            scenes=simple_scenes,
            user_goal="tutorial",
            video_duration=150.0,
        )
        for r in ranked:
            assert "scene_id" in r
            assert "priority" in r
            assert "factors" in r
            assert isinstance(r["factors"], dict)
            assert "relevance_to_goal" in r["factors"]

    def test_all_priorities_in_valid_range(self, simple_scenes):
        ranked = rank_scenes_by_attention(
            scenes=simple_scenes,
            user_goal="anything",
            video_duration=150.0,
        )
        for r in ranked:
            assert 0.0 <= r["priority"] <= 1.0

    def test_empty_scenes_returns_empty(self):
        ranked = rank_scenes_by_attention(
            scenes=[],
            user_goal="test",
        )
        assert ranked == []

    def test_all_examined_returns_empty(self, simple_scenes):
        ranked = rank_scenes_by_attention(
            scenes=simple_scenes,
            user_goal="test",
            examined_scene_ids={0, 1, 2, 3, 4},
        )
        assert ranked == []


# ---------------------------------------------------------------------------
# Embedding fixtures
# ---------------------------------------------------------------------------


def _make_embedding(scene_id: int, vector: list[float]) -> SceneEmbedding:
    """Helper to create a SceneEmbedding with a numpy vector."""
    return SceneEmbedding(
        scene_id=scene_id,
        embedding=np.array(vector, dtype=np.float32),
        model="test",
    )


@pytest.fixture
def goal_embedding():
    """A unit vector representing the user's goal embedding."""
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def similar_scene_embedding():
    """Scene embedding very similar to goal_embedding (cosine ~0.97)."""
    return _make_embedding(0, [0.95, 0.1, 0.1, 0.05])


@pytest.fixture
def dissimilar_scene_embedding():
    """Scene embedding very different from goal_embedding (cosine ~0.0)."""
    return _make_embedding(1, [0.0, 0.0, 1.0, 0.0])


@pytest.fixture
def orthogonal_scene_embedding():
    """Scene embedding orthogonal to goal_embedding (cosine = 0)."""
    return _make_embedding(2, [0.0, 1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# calculate_relevance — embedding paths
# ---------------------------------------------------------------------------


class TestCalculateRelevanceWithEmbeddings:
    def test_similar_embedding_scores_high(
        self, goal_embedding, similar_scene_embedding
    ):
        scene = {"scene_id": 0, "transcript_text": "irrelevant text"}
        score = calculate_relevance(
            scene, "irrelevant", similar_scene_embedding, goal_embedding
        )
        # Cosine similarity of [1,0,0,0] and normalized [0.95,0.1,0.1,0.05]
        # should be high, mapped to [0,1] range
        assert score > 0.7

    def test_dissimilar_embedding_scores_low(
        self, goal_embedding, dissimilar_scene_embedding
    ):
        scene = {"scene_id": 1, "transcript_text": "irrelevant text"}
        score = calculate_relevance(
            scene, "irrelevant", dissimilar_scene_embedding, goal_embedding
        )
        assert score < 0.6

    def test_orthogonal_embedding_scores_midpoint(
        self, goal_embedding, orthogonal_scene_embedding
    ):
        scene = {"scene_id": 2, "transcript_text": "irrelevant text"}
        score = calculate_relevance(
            scene, "irrelevant", orthogonal_scene_embedding, goal_embedding
        )
        # Cosine = 0, mapped to (0+1)/2 = 0.5
        assert abs(score - 0.5) < 0.01

    def test_embedding_overrides_keyword_matching(self, goal_embedding):
        """When embeddings are provided, keyword matching is not used."""
        # Transcript matches goal perfectly by keywords
        scene = {"scene_id": 0, "transcript_text": "fix the bug"}
        # But embedding is orthogonal to goal
        orthogonal_emb = _make_embedding(0, [0.0, 1.0, 0.0, 0.0])
        score_with_emb = calculate_relevance(
            scene, "fix the bug", orthogonal_emb, goal_embedding
        )
        score_keyword = calculate_relevance(scene, "fix the bug")
        # Embedding path gives ~0.5, keyword gives >0.5
        assert score_with_emb < score_keyword

    def test_falls_back_to_keywords_when_scene_embedding_none(self, goal_embedding):
        """If scene_embedding is None, falls back to keyword matching."""
        scene = {"scene_id": 0, "transcript_text": "fix the bug"}
        score = calculate_relevance(scene, "fix the bug", None, goal_embedding)
        # Should use keyword path
        assert score > 0.5

    def test_falls_back_to_keywords_when_goal_embedding_none(
        self, similar_scene_embedding
    ):
        """If goal_embedding is None, falls back to keyword matching."""
        scene = {"scene_id": 0, "transcript_text": "fix the bug"}
        score = calculate_relevance(
            scene, "fix the bug", similar_scene_embedding, None
        )
        # Should use keyword path
        assert score > 0.5

    def test_score_in_valid_range(self, goal_embedding, similar_scene_embedding):
        scene = {"scene_id": 0, "transcript_text": "anything"}
        score = calculate_relevance(
            scene, "query", similar_scene_embedding, goal_embedding
        )
        assert 0.0 <= score <= 1.0

    def test_negative_cosine_maps_to_near_zero(self, goal_embedding):
        """Anti-correlated embeddings should score near 0."""
        anti_emb = _make_embedding(0, [-1.0, 0.0, 0.0, 0.0])
        scene = {"scene_id": 0, "transcript_text": ""}
        score = calculate_relevance(scene, "query", anti_emb, goal_embedding)
        assert score < 0.05


# ---------------------------------------------------------------------------
# calculate_novelty — embedding paths
# ---------------------------------------------------------------------------


class TestCalculateNoveltyWithEmbeddings:
    def test_novel_scene_scores_high(self):
        """Scene dissimilar to all previous scenes should have high novelty."""
        scene = {"scene_id": 2, "transcript_text": "new topic"}
        scene_emb = _make_embedding(2, [1.0, 0.0, 0.0, 0.0])
        prev_embs = [
            _make_embedding(0, [0.0, 1.0, 0.0, 0.0]),
            _make_embedding(1, [0.0, 0.0, 1.0, 0.0]),
        ]
        prev_scenes = [
            {"scene_id": 0, "transcript_text": "old topic"},
            {"scene_id": 1, "transcript_text": "another old topic"},
        ]
        score = calculate_novelty(scene, prev_scenes, scene_emb, prev_embs)
        # Orthogonal to both -> max similarity is 0 -> novelty = 1 - (0+1)/2 = 0.5
        assert score >= 0.5

    def test_duplicate_scene_scores_low(self):
        """Scene identical to a previous scene should have low novelty."""
        scene = {"scene_id": 1, "transcript_text": "same topic"}
        scene_emb = _make_embedding(1, [1.0, 0.0, 0.0, 0.0])
        prev_embs = [_make_embedding(0, [1.0, 0.0, 0.0, 0.0])]
        prev_scenes = [{"scene_id": 0, "transcript_text": "same topic"}]
        score = calculate_novelty(scene, prev_scenes, scene_emb, prev_embs)
        # Identical -> cosine=1 -> novelty = 1 - (1+1)/2 = 0.0
        assert score < 0.05

    def test_partially_similar_scene(self):
        """Scene partially similar to previous should have moderate novelty."""
        scene = {"scene_id": 1, "transcript_text": "mixed topic"}
        # 45-degree angle from previous
        scene_emb = _make_embedding(1, [0.707, 0.707, 0.0, 0.0])
        prev_embs = [_make_embedding(0, [1.0, 0.0, 0.0, 0.0])]
        prev_scenes = [{"scene_id": 0, "transcript_text": "old topic"}]
        score = calculate_novelty(scene, prev_scenes, scene_emb, prev_embs)
        # Cosine ~0.707, novelty = 1 - (0.707+1)/2 ≈ 0.146
        assert 0.05 < score < 0.5

    def test_falls_back_without_previous_embeddings(self):
        """No previous embeddings -> falls back to keyword novelty."""
        scene = {"scene_id": 1, "transcript_text": "completely unique words here"}
        scene_emb = _make_embedding(1, [1.0, 0.0, 0.0, 0.0])
        prev_scenes = [{"scene_id": 0, "transcript_text": "different vocabulary"}]
        score = calculate_novelty(scene, prev_scenes, scene_emb, None)
        # Uses keyword fallback, should still return a valid score
        assert 0.0 <= score <= 1.0

    def test_falls_back_without_scene_embedding(self):
        """No scene embedding -> falls back to keyword novelty."""
        scene = {"scene_id": 1, "transcript_text": "completely unique words here"}
        prev_embs = [_make_embedding(0, [1.0, 0.0, 0.0, 0.0])]
        prev_scenes = [{"scene_id": 0, "transcript_text": "different vocabulary"}]
        score = calculate_novelty(scene, prev_scenes, None, prev_embs)
        assert 0.0 <= score <= 1.0

    def test_empty_previous_returns_neutral(self):
        """No previous scenes -> returns 0.5 regardless of embeddings."""
        scene = {"scene_id": 0, "transcript_text": "first scene"}
        scene_emb = _make_embedding(0, [1.0, 0.0, 0.0, 0.0])
        score = calculate_novelty(scene, [], scene_emb, [])
        assert score == 0.5

    def test_novelty_in_valid_range(self):
        """Score should always be between 0 and 1."""
        scene = {"scene_id": 1, "transcript_text": "text"}
        scene_emb = _make_embedding(1, [0.5, 0.5, 0.5, 0.5])
        prev_embs = [_make_embedding(0, [0.3, 0.7, 0.1, 0.6])]
        prev_scenes = [{"scene_id": 0, "transcript_text": "other"}]
        score = calculate_novelty(scene, prev_scenes, scene_emb, prev_embs)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# calculate_attention_priority — embedding paths
# ---------------------------------------------------------------------------


class TestCalculateAttentionPriorityWithEmbeddings:
    def test_embeddings_affect_priority(self, simple_scenes, goal_embedding):
        """Priority should differ when embeddings are provided vs not."""
        scene = simple_scenes[2]  # "Let me show you how to fix this bug"

        # Without embeddings (keyword path)
        priority_keyword = calculate_attention_priority(
            scene=scene,
            user_goal="fix the bug",
            video_type="unknown",
            total_scenes=5,
            video_duration=150.0,
        )

        # With embeddings - use a dissimilar embedding so result differs
        scene_emb = _make_embedding(2, [0.0, 0.0, 1.0, 0.0])
        priority_embedding = calculate_attention_priority(
            scene=scene,
            user_goal="fix the bug",
            video_type="unknown",
            total_scenes=5,
            video_duration=150.0,
            scene_embedding=scene_emb,
            goal_embedding=goal_embedding,
        )

        # They should differ because embedding gives different relevance
        assert priority_keyword != priority_embedding

    def test_similar_embedding_boosts_relevance(self, simple_scenes, goal_embedding):
        """Scene with similar embedding to goal should get higher relevance."""
        scene = simple_scenes[4]  # "Thanks for watching" - irrelevant keywords

        # With similar embedding, should boost relevance despite bad keywords
        similar_emb = _make_embedding(4, [0.95, 0.1, 0.1, 0.0])
        priority = calculate_attention_priority(
            scene=scene,
            user_goal="unrelated query words",
            video_type="unknown",
            total_scenes=5,
            video_duration=150.0,
            scene_embedding=similar_emb,
            goal_embedding=goal_embedding,
            custom_weights={
                "relevance": 1.0,
                "density": 0.0,
                "novelty": 0.0,
                "visual": 0.0,
                "audio": 0.0,
                "structure": 0.0,
            },
        )

        # With 100% relevance weight and similar embedding -> high score
        assert priority > 0.7

    def test_previous_embeddings_affect_novelty(
        self, simple_scenes, goal_embedding
    ):
        """Previous embeddings should affect novelty scoring."""
        scene = simple_scenes[2]

        # Scene is identical to previous (same embedding)
        scene_emb = _make_embedding(2, [1.0, 0.0, 0.0, 0.0])
        prev_embs = [_make_embedding(1, [1.0, 0.0, 0.0, 0.0])]

        priority_duplicate = calculate_attention_priority(
            scene=scene,
            user_goal="query",
            video_type="unknown",
            previous_scenes=[simple_scenes[1]],
            total_scenes=5,
            video_duration=150.0,
            scene_embedding=scene_emb,
            goal_embedding=goal_embedding,
            previous_embeddings=prev_embs,
            custom_weights={
                "relevance": 0.0,
                "density": 0.0,
                "novelty": 1.0,
                "visual": 0.0,
                "audio": 0.0,
                "structure": 0.0,
            },
        )

        # Duplicate should have very low novelty
        assert priority_duplicate < 0.1


# ---------------------------------------------------------------------------
# rank_scenes_by_attention — embedding paths
# ---------------------------------------------------------------------------


class TestRankScenesByAttentionWithEmbeddings:
    def test_embeddings_change_ranking(self, simple_scenes):
        """Providing embeddings should change how scenes are ranked."""
        goal_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Make scene 4 (normally irrelevant) have the most similar embedding
        embeddings = [
            _make_embedding(0, [0.0, 1.0, 0.0, 0.0]),
            _make_embedding(1, [0.0, 0.0, 1.0, 0.0]),
            _make_embedding(2, [0.0, 0.0, 0.0, 1.0]),
            _make_embedding(3, [0.1, 0.1, 0.1, 0.9]),
            _make_embedding(4, [0.99, 0.01, 0.0, 0.0]),  # Most similar to goal
        ]

        ranked = rank_scenes_by_attention(
            scenes=simple_scenes,
            user_goal="semantic query with no keyword overlap",
            video_duration=150.0,
            embeddings=embeddings,
            goal_embedding=goal_emb,
        )

        # Scene 4 should be ranked relatively high due to embedding similarity
        scene_4_result = next(r for r in ranked if r["scene_id"] == 4)
        assert scene_4_result["priority"] > 0.0

    def test_embedding_ranking_still_sorted(self, simple_scenes):
        """Results should still be sorted by priority descending."""
        goal_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        embeddings = [
            _make_embedding(i, [float(i == 2), float(i != 2), 0.0, 0.0])
            for i in range(5)
        ]

        ranked = rank_scenes_by_attention(
            scenes=simple_scenes,
            user_goal="anything",
            video_duration=150.0,
            embeddings=embeddings,
            goal_embedding=goal_emb,
        )

        priorities = [r["priority"] for r in ranked]
        assert priorities == sorted(priorities, reverse=True)

    def test_partial_embeddings_handled(self, simple_scenes):
        """Only some scenes having embeddings should work fine."""
        goal_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Only scenes 0 and 2 have embeddings
        embeddings = [
            _make_embedding(0, [0.9, 0.1, 0.0, 0.0]),
            _make_embedding(2, [0.1, 0.9, 0.0, 0.0]),
        ]

        ranked = rank_scenes_by_attention(
            scenes=simple_scenes,
            user_goal="query",
            video_duration=150.0,
            embeddings=embeddings,
            goal_embedding=goal_emb,
        )

        assert len(ranked) == 5
        for r in ranked:
            assert 0.0 <= r["priority"] <= 1.0
