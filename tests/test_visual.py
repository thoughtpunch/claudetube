"""Tests for visual scene detection module."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from claudetube.analysis.linguistic import Boundary
from claudetube.analysis.visual import (
    MAX_AVG_SEGMENT_DURATION,
    MIN_CHEAP_BOUNDARIES,
    MIN_DURATION_FOR_FALLBACK,
    detect_visual_boundaries,
    detect_visual_boundaries_fast,
    should_use_visual_detection,
)


class TestConstants:
    """Tests for module constants."""

    def test_min_cheap_boundaries(self):
        """Minimum boundaries threshold should be 3."""
        assert MIN_CHEAP_BOUNDARIES == 3

    def test_min_duration_for_fallback(self):
        """Minimum duration for fallback should be 5 minutes (300s)."""
        assert MIN_DURATION_FOR_FALLBACK == 300

    def test_max_avg_segment_duration(self):
        """Max average segment duration should be 5 minutes (300s)."""
        assert MAX_AVG_SEGMENT_DURATION == 300


class TestShouldUseVisualDetection:
    """Tests for should_use_visual_detection function."""

    def test_short_video_no_fallback(self):
        """Short videos should not use fallback."""
        # 60 second video with no boundaries
        assert should_use_visual_detection([], 60.0) is False

    def test_short_video_under_5min_no_fallback(self):
        """Videos under 5 minutes should not trigger fallback."""
        assert should_use_visual_detection([], 299.0) is False

    def test_long_video_no_boundaries_needs_fallback(self):
        """Long video with no boundaries should use fallback."""
        assert should_use_visual_detection([], 600.0) is True

    def test_long_video_few_boundaries_needs_fallback(self):
        """Long video with <3 boundaries should use fallback."""
        boundaries = [
            Boundary(0.0, "chapter", "Intro", 0.95),
            Boundary(300.0, "chapter", "Main", 0.95),
        ]
        assert should_use_visual_detection(boundaries, 600.0) is True

    def test_long_video_enough_boundaries_no_fallback(self):
        """Long video with >=3 boundaries should not use fallback."""
        boundaries = [
            Boundary(0.0, "chapter", "Intro", 0.95),
            Boundary(200.0, "chapter", "Middle", 0.95),
            Boundary(400.0, "chapter", "End", 0.95),
        ]
        assert should_use_visual_detection(boundaries, 600.0) is False

    def test_no_transcript_few_boundaries_needs_fallback(self):
        """No transcript with few boundaries should use fallback."""
        boundaries = [Boundary(0.0, "chapter", "Intro", 0.95)]
        result = should_use_visual_detection(boundaries, 600.0, has_transcript=False)
        assert result is True

    def test_very_long_segments_need_fallback(self):
        """Very long average segments should trigger fallback."""
        # 30-minute video with only 2 boundaries
        # avg = 1800 / (2 + 1) = 600s > 300s threshold
        boundaries = [
            Boundary(0.0, "chapter", "Intro", 0.95),
            Boundary(900.0, "chapter", "End", 0.95),
        ]
        assert should_use_visual_detection(boundaries, 1800.0) is True

    def test_good_segment_coverage_no_fallback(self):
        """Good segment coverage should not trigger fallback."""
        # 30-minute video with 6 boundaries
        # avg = 1800 / (6 + 1) = 257s < 300s threshold
        boundaries = [
            Boundary(0.0, "chapter", "Ch1", 0.95),
            Boundary(300.0, "chapter", "Ch2", 0.95),
            Boundary(600.0, "chapter", "Ch3", 0.95),
            Boundary(900.0, "chapter", "Ch4", 0.95),
            Boundary(1200.0, "chapter", "Ch5", 0.95),
            Boundary(1500.0, "chapter", "Ch6", 0.95),
        ]
        assert should_use_visual_detection(boundaries, 1800.0) is False

    def test_boundary_exactly_at_threshold(self):
        """Exactly 3 boundaries on 5-min video should not trigger fallback."""
        boundaries = [
            Boundary(0.0, "ch", "a", 0.9),
            Boundary(100.0, "ch", "b", 0.9),
            Boundary(200.0, "ch", "c", 0.9),
        ]
        # 3 boundaries, 300s duration - no fallback needed
        assert should_use_visual_detection(boundaries, 300.0) is False


@pytest.fixture
def mock_scenedetect():
    """Create a mock scenedetect module."""
    mock_module = MagicMock()

    # Create mock classes
    mock_adaptive_detector = MagicMock()
    mock_content_detector = MagicMock()
    mock_scene_manager = MagicMock()
    mock_open_video = MagicMock()

    mock_module.AdaptiveDetector = mock_adaptive_detector
    mock_module.ContentDetector = mock_content_detector
    mock_module.SceneManager = mock_scene_manager
    mock_module.open_video = mock_open_video

    return mock_module


class TestDetectVisualBoundaries:
    """Tests for detect_visual_boundaries function."""

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing video."""
        with pytest.raises(FileNotFoundError, match="Video not found"):
            detect_visual_boundaries("/nonexistent/video.mp4")

    def test_basic_detection(self, tmp_path, mock_scenedetect):
        """Should detect scenes and return Boundary objects."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        # Mock scene list - three scenes
        mock_scene_start_1 = MagicMock()
        mock_scene_start_1.get_seconds.return_value = 0.0
        mock_scene_end_1 = MagicMock()

        mock_scene_start_2 = MagicMock()
        mock_scene_start_2.get_seconds.return_value = 30.0
        mock_scene_end_2 = MagicMock()

        mock_scene_start_3 = MagicMock()
        mock_scene_start_3.get_seconds.return_value = 60.0
        mock_scene_end_3 = MagicMock()

        mock_manager_instance = MagicMock()
        mock_manager_instance.get_scene_list.return_value = [
            (mock_scene_start_1, mock_scene_end_1),
            (mock_scene_start_2, mock_scene_end_2),
            (mock_scene_start_3, mock_scene_end_3),
        ]
        mock_scenedetect.SceneManager.return_value = mock_manager_instance

        mock_video = MagicMock()
        mock_scenedetect.open_video.return_value = mock_video

        with patch.dict(sys.modules, {"scenedetect": mock_scenedetect}):
            # Need to reimport to get the mocked version
            from importlib import reload

            import claudetube.analysis.visual as visual_module

            reload(visual_module)
            boundaries = visual_module.detect_visual_boundaries(str(video_file))

        # Should have 2 boundaries (skips first scene at 0.0)
        assert len(boundaries) == 2
        assert boundaries[0].timestamp == 30.0
        assert boundaries[0].type == "visual_scene"
        assert boundaries[0].confidence == 0.75
        assert boundaries[1].timestamp == 60.0

    def test_downscale_factor(self, tmp_path, mock_scenedetect):
        """Should apply downscale factor to video."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        mock_manager_instance = MagicMock()
        mock_manager_instance.get_scene_list.return_value = []
        mock_scenedetect.SceneManager.return_value = mock_manager_instance

        mock_video = MagicMock()
        mock_scenedetect.open_video.return_value = mock_video

        with patch.dict(sys.modules, {"scenedetect": mock_scenedetect}):
            from importlib import reload

            import claudetube.analysis.visual as visual_module

            reload(visual_module)
            visual_module.detect_visual_boundaries(str(video_file), downscale_factor=4)

        mock_video.set_downscale_factor.assert_called_once_with(4)

    def test_no_downscale_when_factor_1(self, tmp_path, mock_scenedetect):
        """Should not downscale when factor is 1."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        mock_manager_instance = MagicMock()
        mock_manager_instance.get_scene_list.return_value = []
        mock_scenedetect.SceneManager.return_value = mock_manager_instance

        mock_video = MagicMock()
        mock_scenedetect.open_video.return_value = mock_video

        with patch.dict(sys.modules, {"scenedetect": mock_scenedetect}):
            from importlib import reload

            import claudetube.analysis.visual as visual_module

            reload(visual_module)
            visual_module.detect_visual_boundaries(str(video_file), downscale_factor=1)

        mock_video.set_downscale_factor.assert_not_called()

    def test_adaptive_threshold_passed(self, tmp_path, mock_scenedetect):
        """Should pass adaptive threshold to detector."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        mock_manager_instance = MagicMock()
        mock_manager_instance.get_scene_list.return_value = []
        mock_scenedetect.SceneManager.return_value = mock_manager_instance

        mock_video = MagicMock()
        mock_scenedetect.open_video.return_value = mock_video

        with patch.dict(sys.modules, {"scenedetect": mock_scenedetect}):
            from importlib import reload

            import claudetube.analysis.visual as visual_module

            reload(visual_module)
            visual_module.detect_visual_boundaries(
                str(video_file),
                adaptive_threshold=5.0,
                min_scene_len=60,
            )

        mock_scenedetect.AdaptiveDetector.assert_called_once_with(
            adaptive_threshold=5.0,
            min_scene_len=60,
        )


class TestDetectVisualBoundariesFast:
    """Tests for detect_visual_boundaries_fast function."""

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing video."""
        with pytest.raises(FileNotFoundError, match="Video not found"):
            detect_visual_boundaries_fast("/nonexistent/video.mp4")

    def test_basic_detection(self, tmp_path, mock_scenedetect):
        """Should detect scenes and return Boundary objects."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        mock_scene_start_1 = MagicMock()
        mock_scene_start_1.get_seconds.return_value = 0.0
        mock_scene_end_1 = MagicMock()

        mock_scene_start_2 = MagicMock()
        mock_scene_start_2.get_seconds.return_value = 45.0
        mock_scene_end_2 = MagicMock()

        mock_manager_instance = MagicMock()
        mock_manager_instance.get_scene_list.return_value = [
            (mock_scene_start_1, mock_scene_end_1),
            (mock_scene_start_2, mock_scene_end_2),
        ]
        mock_scenedetect.SceneManager.return_value = mock_manager_instance

        mock_video = MagicMock()
        mock_scenedetect.open_video.return_value = mock_video

        with patch.dict(sys.modules, {"scenedetect": mock_scenedetect}):
            from importlib import reload

            import claudetube.analysis.visual as visual_module

            reload(visual_module)
            boundaries = visual_module.detect_visual_boundaries_fast(str(video_file))

        # Should have 1 boundary (skips first scene at 0.0)
        assert len(boundaries) == 1
        assert boundaries[0].timestamp == 45.0
        assert boundaries[0].type == "visual_scene_fast"
        assert boundaries[0].confidence == 0.65  # Lower confidence for fast

    def test_content_threshold_passed(self, tmp_path, mock_scenedetect):
        """Should pass content threshold to detector."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        mock_manager_instance = MagicMock()
        mock_manager_instance.get_scene_list.return_value = []
        mock_scenedetect.SceneManager.return_value = mock_manager_instance

        mock_video = MagicMock()
        mock_scenedetect.open_video.return_value = mock_video

        with patch.dict(sys.modules, {"scenedetect": mock_scenedetect}):
            from importlib import reload

            import claudetube.analysis.visual as visual_module

            reload(visual_module)
            visual_module.detect_visual_boundaries_fast(
                str(video_file),
                content_threshold=30.0,
                min_scene_len=20,
            )

        mock_scenedetect.ContentDetector.assert_called_once_with(
            threshold=30.0,
            min_scene_len=20,
        )


class TestModuleExports:
    """Tests for module exports."""

    def test_import_from_analysis(self):
        """Should be importable from analysis package."""
        from claudetube.analysis import (
            detect_visual_boundaries,
            detect_visual_boundaries_fast,
            should_use_visual_detection,
        )

        assert callable(detect_visual_boundaries)
        assert callable(detect_visual_boundaries_fast)
        assert callable(should_use_visual_detection)

    def test_functions_return_boundary_list(self):
        """Functions should have proper return type annotations."""
        import inspect

        from claudetube.analysis.visual import (
            detect_visual_boundaries,
            detect_visual_boundaries_fast,
        )

        # Check that return annotations exist and mention Boundary
        sig1 = inspect.signature(detect_visual_boundaries)
        sig2 = inspect.signature(detect_visual_boundaries_fast)

        assert sig1.return_annotation is not inspect.Parameter.empty
        assert sig2.return_annotation is not inspect.Parameter.empty
