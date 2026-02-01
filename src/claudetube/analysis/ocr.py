"""
OCR extraction for technical video content.

Extracts text from video frames including code, slides, and terminal output.
Uses EasyOCR for accurate text detection with bounding boxes.

Architecture: Cheap First, Expensive Last
1. CACHE - Check for cached OCR results first
2. DETECT - Skip frames unlikely to contain text
3. COMPUTE - Run OCR only when necessary
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """A detected text region in a frame."""

    text: str
    confidence: float
    bbox: dict  # {x1, y1, x2, y2}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class FrameOCRResult:
    """OCR results for a single frame."""

    frame_path: str
    timestamp: float
    regions: list[TextRegion]
    content_type: str  # 'code', 'slides', 'terminal', 'diagram', 'talking_head'

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "frame_path": self.frame_path,
            "timestamp": self.timestamp,
            "regions": [r.to_dict() for r in self.regions],
            "content_type": self.content_type,
        }


def _get_easyocr_reader():
    """Lazy-load EasyOCR reader to avoid import overhead."""
    try:
        import easyocr

        # Initialize with English, CPU mode for compatibility
        return easyocr.Reader(["en"], gpu=False, verbose=False)
    except ImportError as e:
        raise ImportError(
            "EasyOCR is required for OCR extraction. "
            "Install with: pip install easyocr"
        ) from e


# Module-level reader (lazy-initialized)
_reader = None


def get_reader():
    """Get or create the EasyOCR reader singleton."""
    global _reader
    if _reader is None:
        _reader = _get_easyocr_reader()
    return _reader


def estimate_text_likelihood(frame_path: str | Path) -> float:
    """Estimate likelihood that a frame contains significant text.

    Uses cheap heuristics before running expensive OCR.

    Args:
        frame_path: Path to the frame image.

    Returns:
        Score from 0.0 to 1.0 indicating text likelihood.
    """
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(frame_path)
        arr = np.array(img.convert("L"))  # Convert to grayscale

        # Heuristic 1: High contrast regions (text tends to have sharp edges)
        # Compute standard deviation of pixel values
        std_dev = np.std(arr)

        # Heuristic 2: Bimodal histogram (text = dark on light or light on dark)
        hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
        hist_norm = hist / hist.sum()

        # Check for peaks at extremes (bimodal)
        dark_mass = hist_norm[:64].sum()
        light_mass = hist_norm[192:].sum()
        bimodal_score = min(dark_mass, light_mass) * 2  # Higher if balanced extremes

        # Combine heuristics
        score = 0.0
        if std_dev > 50:  # High contrast
            score += 0.3
        if std_dev > 80:  # Very high contrast
            score += 0.2
        if bimodal_score > 0.2:  # Bimodal distribution
            score += 0.3
        if bimodal_score > 0.4:  # Strong bimodal
            score += 0.2

        return min(1.0, score)

    except Exception as e:
        logger.warning(f"Failed to estimate text likelihood: {e}")
        return 0.5  # Default to uncertain


def classify_content_type(regions: list[TextRegion], frame_path: str | Path) -> str:
    """Classify frame content based on OCR results and visual features.

    Args:
        regions: List of detected text regions.
        frame_path: Path to the frame image.

    Returns:
        Content type: 'code', 'slides', 'terminal', 'diagram', 'talking_head'
    """
    if not regions:
        return "talking_head"

    try:
        import numpy as np
        from PIL import Image

        img = Image.open(frame_path)
        width, height = img.size
        frame_area = width * height

        # Calculate text coverage
        total_text_area = 0
        for r in regions:
            bbox = r.bbox
            area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])
            total_text_area += area

        text_coverage = total_text_area / frame_area if frame_area > 0 else 0

        # Check for code-like patterns in text
        all_text = " ".join(r.text for r in regions)
        code_indicators = [
            "def ",
            "function ",
            "class ",
            "import ",
            "return ",
            "if (",
            "for (",
            "while (",
            "=>",
            "->",
            "==",
            "!=",
            "&&",
            "||",
            "{ }",
            "[]",
        ]
        code_matches = sum(1 for ind in code_indicators if ind in all_text)

        # Check for dark background (terminal)
        arr = np.array(img.convert("L"))
        mean_brightness = np.mean(arr)

        # Classification logic
        if code_matches >= 2 or (code_matches >= 1 and text_coverage > 0.3):
            return "code"
        elif mean_brightness < 80 and text_coverage > 0.2:
            return "terminal"
        elif text_coverage > 0.3 and any(len(r.text) > 50 for r in regions):
            return "slides"
        elif text_coverage < 0.1:
            return "talking_head"
        else:
            return "diagram"

    except Exception as e:
        logger.warning(f"Failed to classify content: {e}")
        return "unknown"


def extract_text_from_frame(
    frame_path: str | Path,
    min_confidence: float = 0.5,
) -> FrameOCRResult:
    """Extract all text from a frame using OCR.

    Args:
        frame_path: Path to the frame image.
        min_confidence: Minimum confidence threshold (0-1). Default 0.5.

    Returns:
        FrameOCRResult with detected regions and content type.
    """
    frame_path = Path(frame_path)

    # Get OCR results
    reader = get_reader()
    results = reader.readtext(str(frame_path))

    # Convert to TextRegion objects
    regions = []
    for bbox, text, conf in results:
        if conf >= min_confidence:
            # Convert bbox from [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] to dict
            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]
            regions.append(
                TextRegion(
                    text=text,
                    confidence=float(conf),
                    bbox={
                        "x1": int(min(x_coords)),
                        "y1": int(min(y_coords)),
                        "x2": int(max(x_coords)),
                        "y2": int(max(y_coords)),
                    },
                )
            )

    # Classify content type
    content_type = classify_content_type(regions, frame_path)

    # Extract timestamp from filename if present (e.g., "frame_120.5.jpg")
    timestamp = 0.0
    try:
        name = frame_path.stem
        if "_" in name:
            timestamp = float(name.split("_")[-1])
    except (ValueError, IndexError):
        pass

    return FrameOCRResult(
        frame_path=str(frame_path),
        timestamp=timestamp,
        regions=regions,
        content_type=content_type,
    )


def extract_text_from_scene(
    scene_dir: Path,
    keyframe_paths: list[Path] | None = None,
    skip_low_likelihood: bool = True,
    likelihood_threshold: float = 0.3,
) -> list[FrameOCRResult]:
    """Extract text from all keyframes in a scene.

    Args:
        scene_dir: Path to the scene directory.
        keyframe_paths: Optional list of specific keyframes to process.
            If None, finds all .jpg/.png files in scene_dir.
        skip_low_likelihood: If True, skip frames unlikely to contain text.
        likelihood_threshold: Minimum text likelihood score to process.

    Returns:
        List of FrameOCRResult for each processed frame.
    """
    # Find keyframes if not provided
    if keyframe_paths is None:
        keyframe_paths = list(scene_dir.glob("*.jpg")) + list(scene_dir.glob("*.png"))
        keyframe_paths.sort()

    results = []
    for frame_path in keyframe_paths:
        # Cheap first: check if likely to contain text
        if skip_low_likelihood:
            likelihood = estimate_text_likelihood(frame_path)
            if likelihood < likelihood_threshold:
                logger.debug(f"Skipping {frame_path.name} (text likelihood: {likelihood:.2f})")
                continue

        # Expensive: run OCR
        try:
            result = extract_text_from_frame(frame_path)
            results.append(result)
        except Exception as e:
            logger.warning(f"OCR failed for {frame_path}: {e}")

    return results


def save_ocr_results(
    results: list[FrameOCRResult],
    output_path: Path,
) -> None:
    """Save OCR results to JSON file.

    Args:
        results: List of FrameOCRResult objects.
        output_path: Path to output JSON file (e.g., technical.json).
    """
    data = {
        "version": 1,
        "frames": [r.to_dict() for r in results],
        "summary": {
            "total_frames": len(results),
            "frames_with_text": sum(1 for r in results if r.regions),
            "content_types": {},
        },
    }

    # Count content types
    for r in results:
        ct = r.content_type
        data["summary"]["content_types"][ct] = data["summary"]["content_types"].get(ct, 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    logger.info(f"Saved OCR results to {output_path}")


def load_ocr_results(input_path: Path) -> list[FrameOCRResult] | None:
    """Load cached OCR results from JSON file.

    Args:
        input_path: Path to the technical.json file.

    Returns:
        List of FrameOCRResult objects, or None if file doesn't exist.
    """
    if not input_path.exists():
        return None

    try:
        data = json.loads(input_path.read_text())
        results = []
        for frame_data in data.get("frames", []):
            regions = [
                TextRegion(**r) for r in frame_data.get("regions", [])
            ]
            results.append(
                FrameOCRResult(
                    frame_path=frame_data["frame_path"],
                    timestamp=frame_data.get("timestamp", 0.0),
                    regions=regions,
                    content_type=frame_data.get("content_type", "unknown"),
                )
            )
        return results
    except Exception as e:
        logger.warning(f"Failed to load OCR results from {input_path}: {e}")
        return None
