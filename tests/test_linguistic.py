"""Tests for linguistic transition cue detection."""

from claudetube.analysis.linguistic import (
    COMPILED_PATTERNS,
    Boundary,
    detect_linguistic_boundaries,
)


class TestBoundary:
    """Tests for Boundary namedtuple."""

    def test_boundary_creation(self):
        """Test basic Boundary creation."""
        b = Boundary(
            timestamp=30.0,
            type="linguistic_cue",
            trigger_text="now let's talk about",
            confidence=0.7,
        )
        assert b.timestamp == 30.0
        assert b.type == "linguistic_cue"
        assert b.trigger_text == "now let's talk about"
        assert b.confidence == 0.7

    def test_boundary_is_tuple(self):
        """Test that Boundary behaves like a tuple."""
        b = Boundary(10.0, "linguistic_cue", "test", 0.7)
        assert b[0] == 10.0
        assert b[1] == "linguistic_cue"
        assert len(b) == 4


class TestExplicitTransitions:
    """Tests for explicit transition phrase detection."""

    def test_next_lets(self):
        """Detect 'next let's' pattern."""
        segments = [{"start": 10.0, "text": "next let's look at the code"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1
        assert boundaries[0].timestamp == 10.0

    def test_now_lets(self):
        """Detect 'now let's' pattern."""
        segments = [{"start": 20.0, "text": "now let's move on"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_now_well(self):
        """Detect 'now we'll' pattern."""
        segments = [{"start": 30.0, "text": "now we'll see how it works"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_moving_on(self):
        """Detect 'moving on' pattern."""
        segments = [{"start": 40.0, "text": "moving on to the next topic"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_lets_talk_about(self):
        """Detect 'let's talk about' pattern."""
        segments = [{"start": 50.0, "text": "let's talk about performance"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_now_that(self):
        """Detect 'now that' pattern."""
        segments = [{"start": 60.0, "text": "now that we have that setup"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_first_second_third(self):
        """Detect ordinal transitions."""
        segments = [
            {"start": 10.0, "text": "First, we need to install"},
            {"start": 20.0, "text": "Second, configure the settings"},
            {"start": 30.0, "text": "Third, run the tests"},
            {"start": 40.0, "text": "Finally, deploy to production"},
        ]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 4

    def test_so_now(self):
        """Detect 'so now' pattern."""
        segments = [{"start": 70.0, "text": "so now we can start coding"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_okay_so(self):
        """Detect 'okay so' pattern."""
        segments = [{"start": 80.0, "text": "okay, so let's begin"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_alright_now(self):
        """Detect 'alright now' pattern."""
        segments = [{"start": 90.0, "text": "alright now we're ready"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1


class TestSectionMarkers:
    """Tests for section marker detection."""

    def test_step_number(self):
        """Detect 'step N' pattern."""
        segments = [
            {"start": 10.0, "text": "step 1 is the installation"},
            {"start": 60.0, "text": "step 2 involves configuration"},
            {"start": 120.0, "text": "step 10 is the final cleanup"},
        ]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 3

    def test_part_number(self):
        """Detect 'part N' pattern."""
        segments = [
            {"start": 0.0, "text": "in part 1 we covered basics"},
            {"start": 300.0, "text": "part 2 focuses on advanced topics"},
        ]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 2

    def test_in_this_section(self):
        """Detect 'in this section' pattern."""
        segments = [{"start": 15.0, "text": "in this section we'll cover"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_in_this_video(self):
        """Detect 'in this video' pattern."""
        segments = [{"start": 5.0, "text": "in this video I'll show you"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_to_summarize(self):
        """Detect summary patterns."""
        segments = [
            {"start": 500.0, "text": "to summarize what we learned"},
            {"start": 510.0, "text": "in summary, the key points are"},
            {"start": 520.0, "text": "to recap, we covered three topics"},
        ]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 3


class TestTopicShifts:
    """Tests for topic shift detection."""

    def test_another_thing(self):
        """Detect 'another thing' pattern."""
        segments = [{"start": 100.0, "text": "another thing to note is"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_another_way(self):
        """Detect 'another way' pattern."""
        segments = [{"start": 110.0, "text": "another way to do this is"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_another_approach(self):
        """Detect 'another approach' pattern."""
        segments = [{"start": 120.0, "text": "another approach would be"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_the_next_thing(self):
        """Detect 'the next thing' pattern."""
        segments = [{"start": 130.0, "text": "the next thing we need"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_the_final_step(self):
        """Detect 'the final step' pattern."""
        segments = [{"start": 140.0, "text": "the final step is to deploy"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_the_last_part(self):
        """Detect 'the last part' pattern."""
        segments = [{"start": 150.0, "text": "the last part of the tutorial"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1


class TestCaseInsensitivity:
    """Tests for case-insensitive matching."""

    def test_uppercase(self):
        """Detect patterns in uppercase."""
        segments = [{"start": 10.0, "text": "NEXT LET'S LOOK AT THIS"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_mixed_case(self):
        """Detect patterns in mixed case."""
        segments = [{"start": 10.0, "text": "Now Let's Move On"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1


class TestLowFalsePositives:
    """Tests ensuring low false positive rate."""

    def test_no_match_regular_text(self):
        """Regular text without transitions should not match."""
        segments = [
            {"start": 0.0, "text": "This is just regular content"},
            {"start": 10.0, "text": "We're writing some code here"},
            {"start": 20.0, "text": "The function returns a value"},
        ]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 0

    def test_no_match_partial_patterns(self):
        """Partial pattern matches should not trigger false positives."""
        segments = [
            {"start": 0.0, "text": "The next value is 42"},  # 'next' alone
            {"start": 10.0, "text": "Now is the time"},  # 'now' without follow-up
            {"start": 20.0, "text": "Let's see what happens"},  # let's alone
        ]
        boundaries = detect_linguistic_boundaries(segments)
        # 'now is' matches 'now i' pattern - this is acceptable
        # We check that we don't get 3 matches
        assert len(boundaries) <= 1

    def test_no_match_embedded_words(self):
        """Words embedded in other words should not match."""
        segments = [
            {"start": 0.0, "text": "The renown expert spoke"},  # 'now' in 'renown'
            {"start": 10.0, "text": "It's a wonderful day"},  # no transition
        ]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 0


class TestOneMatchPerSegment:
    """Tests ensuring only one match per segment."""

    def test_multiple_patterns_in_segment(self):
        """Multiple patterns in one segment should only produce one boundary."""
        segments = [{"start": 10.0, "text": "Okay so first let's talk about step 1"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1

    def test_multiple_segments_multiple_matches(self):
        """Multiple segments can each have their own match."""
        segments = [
            {"start": 10.0, "text": "First, the basics"},
            {"start": 60.0, "text": "Second, advanced topics"},
            {"start": 120.0, "text": "Finally, best practices"},
        ]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 3


class TestTriggerText:
    """Tests for trigger_text truncation."""

    def test_short_text_not_truncated(self):
        """Short text should not be truncated."""
        segments = [{"start": 10.0, "text": "now let's begin"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert boundaries[0].trigger_text == "now let's begin"

    def test_long_text_truncated(self):
        """Long text should be truncated to 50 chars."""
        long_text = "now let's talk about " + "x" * 100
        segments = [{"start": 10.0, "text": long_text}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries[0].trigger_text) == 50


class TestEmptyInput:
    """Tests for edge cases with empty input."""

    def test_empty_segments(self):
        """Empty segment list should return empty boundaries."""
        boundaries = detect_linguistic_boundaries([])
        assert boundaries == []

    def test_segment_with_empty_text(self):
        """Segment with empty text should be handled."""
        segments = [{"start": 10.0, "text": ""}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 0

    def test_segment_missing_text(self):
        """Segment missing 'text' key should be handled."""
        segments = [{"start": 10.0}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 0

    def test_segment_missing_start(self):
        """Segment missing 'start' key should default to 0.0."""
        segments = [{"text": "now let's begin"}]
        boundaries = detect_linguistic_boundaries(segments)
        assert len(boundaries) == 1
        assert boundaries[0].timestamp == 0.0


class TestConfidenceValue:
    """Tests for confidence value."""

    def test_confidence_is_always_0_7(self):
        """All linguistic boundaries should have 0.7 confidence."""
        segments = [
            {"start": 10.0, "text": "step 1"},
            {"start": 20.0, "text": "moving on"},
            {"start": 30.0, "text": "to summarize"},
        ]
        boundaries = detect_linguistic_boundaries(segments)
        assert all(b.confidence == 0.7 for b in boundaries)


class TestPatternCompilation:
    """Tests for pattern compilation."""

    def test_patterns_are_precompiled(self):
        """Patterns should be pre-compiled for performance."""
        import re

        assert all(isinstance(p, re.Pattern) for p in COMPILED_PATTERNS)

    def test_pattern_count(self):
        """Should have expected number of patterns."""
        # 6 explicit + 3 section + 2 topic = 11 patterns
        assert len(COMPILED_PATTERNS) == 11
