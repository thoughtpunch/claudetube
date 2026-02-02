"""
Tests for OCR extraction module.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest


class TestTextRegion:
    """Tests for TextRegion dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        from claudetube.analysis.ocr import TextRegion

        region = TextRegion(
            text="hello world",
            confidence=0.95,
            bbox={"x1": 10, "y1": 20, "x2": 100, "y2": 50},
        )

        result = region.to_dict()

        assert result["text"] == "hello world"
        assert result["confidence"] == 0.95
        assert result["bbox"]["x1"] == 10


class TestFrameOCRResult:
    """Tests for FrameOCRResult dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary with nested regions."""
        from claudetube.analysis.ocr import FrameOCRResult, TextRegion

        result = FrameOCRResult(
            frame_path="/path/to/frame.jpg",
            timestamp=120.5,
            regions=[
                TextRegion("hello", 0.9, {"x1": 0, "y1": 0, "x2": 50, "y2": 20}),
            ],
            content_type="code",
        )

        data = result.to_dict()

        assert data["frame_path"] == "/path/to/frame.jpg"
        assert data["timestamp"] == 120.5
        assert data["content_type"] == "code"
        assert len(data["regions"]) == 1
        assert data["regions"][0]["text"] == "hello"


class TestEstimateTextLikelihood:
    """Tests for text likelihood estimation."""

    def test_high_contrast_image(self, tmp_path):
        """High contrast images should have higher likelihood."""
        from claudetube.analysis.ocr import estimate_text_likelihood

        # Create a high-contrast test image
        try:
            import numpy as np
            from PIL import Image

            # Create black and white stripes (high contrast)
            arr = np.zeros((100, 100), dtype=np.uint8)
            arr[::2, :] = 255  # Alternating white rows
            img = Image.fromarray(arr)
            img_path = tmp_path / "high_contrast.png"
            img.save(img_path)

            likelihood = estimate_text_likelihood(img_path)
            assert likelihood > 0.3  # Should detect as likely text

        except ImportError:
            # Skip if PIL not available
            pass

    def test_uniform_image(self, tmp_path):
        """Uniform images should have lower likelihood."""
        from claudetube.analysis.ocr import estimate_text_likelihood

        try:
            import numpy as np
            from PIL import Image

            # Create uniform gray image
            arr = np.full((100, 100), 128, dtype=np.uint8)
            img = Image.fromarray(arr)
            img_path = tmp_path / "uniform.png"
            img.save(img_path)

            likelihood = estimate_text_likelihood(img_path)
            assert likelihood < 0.5  # Should be low likelihood

        except ImportError:
            pass


class TestClassifyContentType:
    """Tests for content type classification."""

    def test_code_detection(self, tmp_path):
        """Should detect code content."""
        from claudetube.analysis.ocr import TextRegion, classify_content_type

        try:
            import numpy as np
            from PIL import Image

            # Create simple test image
            arr = np.full((100, 100, 3), 255, dtype=np.uint8)
            img = Image.fromarray(arr)
            img_path = tmp_path / "test.png"
            img.save(img_path)

            regions = [
                TextRegion("def hello():", 0.9, {"x1": 0, "y1": 0, "x2": 80, "y2": 20}),
                TextRegion(
                    "    return True", 0.9, {"x1": 0, "y1": 20, "x2": 80, "y2": 40}
                ),
            ]

            content_type = classify_content_type(regions, img_path)
            assert content_type == "code"

        except ImportError:
            pass

    def test_talking_head_detection(self, tmp_path):
        """Should detect talking head when no text."""
        from claudetube.analysis.ocr import classify_content_type

        try:
            import numpy as np
            from PIL import Image

            arr = np.full((100, 100, 3), 128, dtype=np.uint8)
            img = Image.fromarray(arr)
            img_path = tmp_path / "test.png"
            img.save(img_path)

            content_type = classify_content_type([], img_path)
            assert content_type == "talking_head"

        except ImportError:
            pass


class TestSaveLoadOCRResults:
    """Tests for saving and loading OCR results."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Should save and load results correctly."""
        from claudetube.analysis.ocr import (
            FrameOCRResult,
            TextRegion,
            load_ocr_results,
            save_ocr_results,
        )

        results = [
            FrameOCRResult(
                frame_path="/path/to/frame1.jpg",
                timestamp=10.0,
                regions=[
                    TextRegion("test", 0.95, {"x1": 0, "y1": 0, "x2": 50, "y2": 20}),
                ],
                content_type="code",
            ),
            FrameOCRResult(
                frame_path="/path/to/frame2.jpg",
                timestamp=20.0,
                regions=[],
                content_type="talking_head",
            ),
        ]

        output_path = tmp_path / "technical.json"
        save_ocr_results(results, output_path)

        # Verify file was created
        assert output_path.exists()

        # Load and verify
        loaded = load_ocr_results(output_path)
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].frame_path == "/path/to/frame1.jpg"
        assert loaded[0].timestamp == 10.0
        assert loaded[0].content_type == "code"
        assert len(loaded[0].regions) == 1
        assert loaded[0].regions[0].text == "test"

    def test_load_nonexistent_file(self, tmp_path):
        """Should return None for nonexistent file."""
        from claudetube.analysis.ocr import load_ocr_results

        result = load_ocr_results(tmp_path / "nonexistent.json")
        assert result is None

    def test_save_creates_summary(self, tmp_path):
        """Should create summary in saved JSON."""
        from claudetube.analysis.ocr import (
            FrameOCRResult,
            TextRegion,
            save_ocr_results,
        )

        results = [
            FrameOCRResult(
                "/f1.jpg",
                0,
                [TextRegion("x", 0.9, {"x1": 0, "y1": 0, "x2": 1, "y2": 1})],
                "code",
            ),
            FrameOCRResult("/f2.jpg", 0, [], "talking_head"),
            FrameOCRResult(
                "/f3.jpg",
                0,
                [TextRegion("y", 0.9, {"x1": 0, "y1": 0, "x2": 1, "y2": 1})],
                "code",
            ),
        ]

        output_path = tmp_path / "technical.json"
        save_ocr_results(results, output_path)

        data = json.loads(output_path.read_text())
        assert data["summary"]["total_frames"] == 3
        assert data["summary"]["frames_with_text"] == 2
        assert data["summary"]["content_types"]["code"] == 2
        assert data["summary"]["content_types"]["talking_head"] == 1


class TestShouldUseVision:
    """Tests for _should_use_vision decision function."""

    def test_code_content_returns_true(self):
        """Should return True for code content type."""
        from claudetube.analysis.ocr import FrameOCRResult, _should_use_vision

        result = FrameOCRResult(
            frame_path="/path/frame.jpg",
            timestamp=0.0,
            regions=[],
            content_type="code",
        )
        assert _should_use_vision(result) is True

    def test_terminal_content_returns_true(self):
        """Should return True for terminal content type."""
        from claudetube.analysis.ocr import FrameOCRResult, _should_use_vision

        result = FrameOCRResult(
            frame_path="/path/frame.jpg",
            timestamp=0.0,
            regions=[],
            content_type="terminal",
        )
        assert _should_use_vision(result) is True

    def test_low_confidence_returns_true(self):
        """Should return True when average confidence is below threshold."""
        from claudetube.analysis.ocr import (
            FrameOCRResult,
            TextRegion,
            _should_use_vision,
        )

        result = FrameOCRResult(
            frame_path="/path/frame.jpg",
            timestamp=0.0,
            regions=[
                TextRegion("blurry", 0.3, {"x1": 0, "y1": 0, "x2": 50, "y2": 20}),
                TextRegion("text", 0.2, {"x1": 0, "y1": 20, "x2": 50, "y2": 40}),
            ],
            content_type="slides",
        )
        assert _should_use_vision(result) is True

    def test_high_confidence_slides_returns_false(self):
        """Should return False for high-confidence non-code content."""
        from claudetube.analysis.ocr import (
            FrameOCRResult,
            TextRegion,
            _should_use_vision,
        )

        result = FrameOCRResult(
            frame_path="/path/frame.jpg",
            timestamp=0.0,
            regions=[
                TextRegion("clear text", 0.95, {"x1": 0, "y1": 0, "x2": 50, "y2": 20}),
            ],
            content_type="slides",
        )
        assert _should_use_vision(result) is False

    def test_talking_head_no_regions_returns_false(self):
        """Should return False for talking head with no regions."""
        from claudetube.analysis.ocr import FrameOCRResult, _should_use_vision

        result = FrameOCRResult(
            frame_path="/path/frame.jpg",
            timestamp=0.0,
            regions=[],
            content_type="talking_head",
        )
        assert _should_use_vision(result) is False

    def test_custom_min_confidence_threshold(self):
        """Should respect custom min_confidence threshold."""
        from claudetube.analysis.ocr import (
            FrameOCRResult,
            TextRegion,
            _should_use_vision,
        )

        result = FrameOCRResult(
            frame_path="/path/frame.jpg",
            timestamp=0.0,
            regions=[
                TextRegion("text", 0.6, {"x1": 0, "y1": 0, "x2": 50, "y2": 20}),
            ],
            content_type="slides",
        )
        # Default threshold 0.5 -> 0.6 > 0.5, should be False
        assert _should_use_vision(result) is False
        # Higher threshold 0.8 -> 0.6 < 0.8, should be True
        assert _should_use_vision(result, min_confidence=0.8) is True


class TestExtractTextWithVision:
    """Tests for extract_text_with_vision async function."""

    def test_successful_extraction(self, tmp_path):
        """Should extract text using VisionAnalyzer."""
        from claudetube.analysis.ocr import extract_text_with_vision

        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            pytest.skip("PIL/numpy not available")

        # Create test image
        arr = np.full((100, 100, 3), 255, dtype=np.uint8)
        img = Image.fromarray(arr)
        img_path = tmp_path / "frame_42.5.jpg"
        img.save(img_path)

        # Mock VisionAnalyzer
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_images.return_value = "def hello():\n    return True"

        result = asyncio.run(extract_text_with_vision(img_path, mock_analyzer))

        assert len(result.regions) == 1
        assert "def hello():" in result.regions[0].text
        assert result.regions[0].confidence == 0.9
        assert result.timestamp == 42.5
        mock_analyzer.analyze_images.assert_called_once()

    def test_empty_vision_result(self, tmp_path):
        """Should return empty regions for blank vision output."""
        from claudetube.analysis.ocr import extract_text_with_vision

        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            pytest.skip("PIL/numpy not available")

        arr = np.full((100, 100, 3), 255, dtype=np.uint8)
        img = Image.fromarray(arr)
        img_path = tmp_path / "frame_10.0.jpg"
        img.save(img_path)

        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_images.return_value = "   "

        result = asyncio.run(extract_text_with_vision(img_path, mock_analyzer))

        assert len(result.regions) == 0

    def test_vision_failure_returns_empty(self, tmp_path):
        """Should return empty result when VisionAnalyzer raises."""
        from claudetube.analysis.ocr import extract_text_with_vision

        img_path = tmp_path / "frame_5.0.jpg"
        img_path.write_bytes(b"fake image data")

        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_images.side_effect = RuntimeError("API error")

        result = asyncio.run(extract_text_with_vision(img_path, mock_analyzer))

        assert len(result.regions) == 0
        assert result.content_type == "unknown"

    def test_timestamp_from_filename(self, tmp_path):
        """Should extract timestamp from frame filename."""
        from claudetube.analysis.ocr import extract_text_with_vision

        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            pytest.skip("PIL/numpy not available")

        arr = np.full((100, 100, 3), 255, dtype=np.uint8)
        img = Image.fromarray(arr)
        img_path = tmp_path / "frame_123.45.jpg"
        img.save(img_path)

        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_images.return_value = "some text"

        result = asyncio.run(extract_text_with_vision(img_path, mock_analyzer))

        assert result.timestamp == 123.45

    def test_non_string_result_converted(self, tmp_path):
        """Should handle non-string VisionAnalyzer results."""
        from claudetube.analysis.ocr import extract_text_with_vision

        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            pytest.skip("PIL/numpy not available")

        arr = np.full((100, 100, 3), 255, dtype=np.uint8)
        img = Image.fromarray(arr)
        img_path = tmp_path / "frame_1.0.jpg"
        img.save(img_path)

        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_images.return_value = ["line1", "line2"]

        result = asyncio.run(extract_text_with_vision(img_path, mock_analyzer))

        assert len(result.regions) == 1
        # Non-string gets str() conversion
        assert "line1" in result.regions[0].text


class TestExtractTextFromSceneVision:
    """Tests for extract_text_from_scene with vision_analyzer parameter."""

    def test_signature_accepts_vision_analyzer(self):
        """extract_text_from_scene should accept vision_analyzer kwarg."""
        import inspect

        from claudetube.analysis.ocr import extract_text_from_scene

        sig = inspect.signature(extract_text_from_scene)
        assert "vision_analyzer" in sig.parameters
        # Default should be None
        assert sig.parameters["vision_analyzer"].default is None

    def test_exports_extract_text_with_vision(self):
        """extract_text_with_vision should be exported from analysis package."""
        from claudetube.analysis import extract_text_with_vision

        assert callable(extract_text_with_vision)
