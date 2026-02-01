"""
Tests for OCR extraction module.
"""

import json


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
                TextRegion("    return True", 0.9, {"x1": 0, "y1": 20, "x2": 80, "y2": 40}),
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
            FrameOCRResult("/f1.jpg", 0, [TextRegion("x", 0.9, {"x1": 0, "y1": 0, "x2": 1, "y2": 1})], "code"),
            FrameOCRResult("/f2.jpg", 0, [], "talking_head"),
            FrameOCRResult("/f3.jpg", 0, [TextRegion("y", 0.9, {"x1": 0, "y1": 0, "x2": 1, "y2": 1})], "code"),
        ]

        output_path = tmp_path / "technical.json"
        save_ocr_results(results, output_path)

        data = json.loads(output_path.read_text())
        assert data["summary"]["total_frames"] == 3
        assert data["summary"]["frames_with_text"] == 2
        assert data["summary"]["content_types"]["code"] == 2
        assert data["summary"]["content_types"]["talking_head"] == 1
