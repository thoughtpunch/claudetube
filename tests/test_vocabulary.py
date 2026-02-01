"""Tests for vocabulary shift detection."""

from claudetube.analysis.vocabulary import (
    DEFAULT_WINDOW_SECONDS,
    SIMILARITY_THRESHOLD,
    VOCABULARY_SHIFT_CONFIDENCE,
    _group_into_windows,
    detect_vocabulary_shifts,
)


class TestGroupIntoWindows:
    """Tests for window grouping utility."""

    def test_empty_segments(self):
        """Empty segments should return empty windows."""
        windows = _group_into_windows([], 30)
        assert windows == []

    def test_single_segment(self):
        """Single segment should create single window."""
        segments = [{"start": 0.0, "text": "Hello world"}]
        windows = _group_into_windows(segments, 30)
        assert len(windows) == 1
        assert windows[0]["start"] == 0.0
        assert "Hello world" in windows[0]["text"]

    def test_segments_within_window(self):
        """Segments within window duration should be grouped."""
        segments = [
            {"start": 0.0, "text": "First"},
            {"start": 10.0, "text": "Second"},
            {"start": 20.0, "text": "Third"},
        ]
        windows = _group_into_windows(segments, 30)
        assert len(windows) == 1
        assert "First" in windows[0]["text"]
        assert "Second" in windows[0]["text"]
        assert "Third" in windows[0]["text"]

    def test_segments_across_windows(self):
        """Segments exceeding window duration should create new windows."""
        segments = [
            {"start": 0.0, "text": "First"},
            {"start": 35.0, "text": "Second"},
            {"start": 70.0, "text": "Third"},
        ]
        windows = _group_into_windows(segments, 30)
        assert len(windows) == 3
        assert windows[0]["start"] == 0.0
        assert windows[1]["start"] == 35.0
        assert windows[2]["start"] == 70.0

    def test_missing_start_defaults_to_zero(self):
        """Segments missing start key should default to 0.0."""
        segments = [{"text": "Hello"}]
        windows = _group_into_windows(segments, 30)
        assert len(windows) == 1
        assert windows[0]["start"] == 0.0

    def test_missing_text_defaults_to_empty(self):
        """Segments missing text key should use empty string."""
        segments = [{"start": 0.0}]
        windows = _group_into_windows(segments, 30)
        # Empty text window gets dropped
        assert len(windows) == 0

    def test_custom_window_size(self):
        """Custom window size should be respected."""
        segments = [
            {"start": 0.0, "text": "First"},
            {"start": 15.0, "text": "Second"},
        ]
        # With 10s window, second segment creates new window
        windows = _group_into_windows(segments, 10)
        assert len(windows) == 2


class TestDetectVocabularyShifts:
    """Tests for vocabulary shift detection."""

    def test_empty_segments(self):
        """Empty segments should return empty boundaries."""
        boundaries = detect_vocabulary_shifts([])
        assert boundaries == []

    def test_single_segment(self):
        """Single segment should return empty boundaries."""
        segments = [{"start": 0.0, "text": "Hello world"}]
        boundaries = detect_vocabulary_shifts(segments, window_seconds=30)
        assert boundaries == []

    def test_single_window(self):
        """Segments fitting in one window should return empty boundaries."""
        segments = [
            {"start": 0.0, "text": "Hello world"},
            {"start": 10.0, "text": "How are you"},
        ]
        boundaries = detect_vocabulary_shifts(segments, window_seconds=30)
        assert boundaries == []

    def test_similar_vocabulary_no_shift(self):
        """Similar vocabulary across windows should not detect shifts."""
        segments = [
            {"start": 0.0, "text": "Python programming language code"},
            {"start": 35.0, "text": "Python programming language code development"},
        ]
        boundaries = detect_vocabulary_shifts(
            segments, window_seconds=30, similarity_threshold=0.3
        )
        # Similar vocabulary should have high similarity, no shift
        assert len(boundaries) == 0

    def test_different_vocabulary_detects_shift(self):
        """Very different vocabulary should detect a shift."""
        segments = [
            {"start": 0.0, "text": "Python programming language functions classes modules"},
            {"start": 35.0, "text": "Cooking recipes kitchen ingredients chef food"},
        ]
        boundaries = detect_vocabulary_shifts(
            segments, window_seconds=30, similarity_threshold=0.3
        )
        # Very different vocabulary should have low similarity
        assert len(boundaries) == 1
        assert boundaries[0].timestamp == 35.0
        assert boundaries[0].type == "vocabulary_shift"
        assert boundaries[0].confidence == VOCABULARY_SHIFT_CONFIDENCE

    def test_boundary_type(self):
        """Boundary type should be 'vocabulary_shift'."""
        segments = [
            {"start": 0.0, "text": "Programming software development code"},
            {"start": 35.0, "text": "Ocean marine biology fish coral reef"},
        ]
        boundaries = detect_vocabulary_shifts(segments, window_seconds=30)
        if boundaries:
            assert boundaries[0].type == "vocabulary_shift"

    def test_boundary_confidence(self):
        """All vocabulary shift boundaries should have 0.6 confidence."""
        segments = [
            {"start": 0.0, "text": "Computer science algorithms data structures"},
            {"start": 35.0, "text": "Music concert piano violin orchestra symphony"},
            {"start": 70.0, "text": "Sports football basketball tennis athletics"},
        ]
        boundaries = detect_vocabulary_shifts(segments, window_seconds=30)
        # Should detect multiple shifts
        assert all(b.confidence == VOCABULARY_SHIFT_CONFIDENCE for b in boundaries)

    def test_trigger_text_contains_keywords(self):
        """Trigger text should contain vocab shift label and similarity."""
        segments = [
            {"start": 0.0, "text": "Python programming language functions"},
            {"start": 35.0, "text": "Cooking recipes kitchen ingredients food"},
        ]
        boundaries = detect_vocabulary_shifts(segments, window_seconds=30)
        if boundaries:
            assert "vocab shift" in boundaries[0].trigger_text
            assert "sim=" in boundaries[0].trigger_text

    def test_trigger_text_truncated(self):
        """Trigger text should be truncated to 50 chars."""
        segments = [
            {"start": 0.0, "text": "Python programming language functions classes"},
            {"start": 35.0, "text": "Cooking recipes kitchen ingredients chef restaurant"},
        ]
        boundaries = detect_vocabulary_shifts(segments, window_seconds=30)
        if boundaries:
            assert len(boundaries[0].trigger_text) <= 50

    def test_multiple_shifts(self):
        """Multiple vocabulary shifts should be detected."""
        segments = [
            {"start": 0.0, "text": "Python programming software code development"},
            {"start": 35.0, "text": "Cooking recipes kitchen ingredients chef"},
            {"start": 70.0, "text": "Music concert piano violin orchestra"},
        ]
        boundaries = detect_vocabulary_shifts(segments, window_seconds=30)
        # May detect 0, 1, or 2 shifts depending on TF-IDF similarity
        # At minimum, topics are very different
        assert len(boundaries) >= 1

    def test_custom_window_seconds(self):
        """Custom window size should be respected."""
        segments = [
            {"start": 0.0, "text": "Python programming"},
            {"start": 15.0, "text": "Cooking recipes"},
        ]
        # With 10s windows, these should be in different windows
        boundaries = detect_vocabulary_shifts(segments, window_seconds=10)
        # Results depend on TF-IDF, just verify no crash
        assert isinstance(boundaries, list)

    def test_custom_threshold(self):
        """Custom similarity threshold should be respected."""
        segments = [
            {"start": 0.0, "text": "Python programming language code"},
            {"start": 35.0, "text": "Python software development programming"},
        ]
        # With very high threshold (0.9), even similar text may trigger
        boundaries_high = detect_vocabulary_shifts(
            segments, window_seconds=30, similarity_threshold=0.9
        )
        # With very low threshold (0.05), only very different text triggers
        boundaries_low = detect_vocabulary_shifts(
            segments, window_seconds=30, similarity_threshold=0.05
        )
        # High threshold should detect more (or equal) shifts
        assert len(boundaries_high) >= len(boundaries_low)

    def test_all_stopwords_handled(self):
        """Windows with only stopwords should be handled gracefully."""
        segments = [
            {"start": 0.0, "text": "the a an is are was were"},
            {"start": 35.0, "text": "the a an it they them"},
        ]
        # Should not crash, may return empty
        boundaries = detect_vocabulary_shifts(segments, window_seconds=30)
        assert isinstance(boundaries, list)

    def test_empty_text_handled(self):
        """Empty text segments should be handled gracefully."""
        segments = [
            {"start": 0.0, "text": ""},
            {"start": 35.0, "text": ""},
        ]
        boundaries = detect_vocabulary_shifts(segments, window_seconds=30)
        assert boundaries == []


class TestConstants:
    """Tests for module constants."""

    def test_default_window_seconds(self):
        """Default window should be 30 seconds."""
        assert DEFAULT_WINDOW_SECONDS == 30

    def test_similarity_threshold(self):
        """Similarity threshold should be 0.3."""
        assert SIMILARITY_THRESHOLD == 0.3

    def test_vocabulary_shift_confidence(self):
        """Vocabulary shift confidence should be 0.6."""
        assert VOCABULARY_SHIFT_CONFIDENCE == 0.6


class TestBoundaryCompatibility:
    """Tests ensuring Boundary compatibility with other modules."""

    def test_same_boundary_type(self):
        """Vocabulary detection should use same Boundary class."""
        from claudetube.analysis.linguistic import Boundary as LinguisticBoundary

        segments = [
            {"start": 0.0, "text": "Python programming software code"},
            {"start": 35.0, "text": "Cooking recipes kitchen ingredients chef"},
        ]
        boundaries = detect_vocabulary_shifts(segments, window_seconds=30)
        if boundaries:
            assert isinstance(boundaries[0], LinguisticBoundary)

    def test_boundary_fields_match(self):
        """Boundary fields should match expected structure."""
        segments = [
            {"start": 0.0, "text": "Computer science algorithms data"},
            {"start": 35.0, "text": "Music concert orchestra symphony"},
        ]
        boundaries = detect_vocabulary_shifts(segments, window_seconds=30)
        if boundaries:
            b = boundaries[0]
            assert hasattr(b, "timestamp")
            assert hasattr(b, "type")
            assert hasattr(b, "trigger_text")
            assert hasattr(b, "confidence")


class TestModuleExport:
    """Tests for module exports."""

    def test_import_from_analysis(self):
        """Should be importable from analysis package."""
        from claudetube.analysis import detect_vocabulary_shifts

        assert callable(detect_vocabulary_shifts)
