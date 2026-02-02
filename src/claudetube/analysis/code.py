"""
Code block detection and language identification.

Detects code regions in OCR results and identifies programming languages.
Uses pygments for language detection and heuristics for code region identification.

Architecture: Cheap First, Expensive Last
1. CACHE - Check for cached code analysis first
2. HEURISTICS - Use pattern matching to identify likely code
3. LANGUAGE - Run pygments lexer guessing only on confirmed code
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path  # noqa: TCH003

# These are used at runtime for type checking and function parameters
from claudetube.analysis.ocr import FrameOCRResult, TextRegion  # noqa: TCH001

logger = logging.getLogger(__name__)

# Patterns that indicate code (compiled once for efficiency)
CODE_PATTERNS = [
    re.compile(r"\bdef\s+\w+\s*\("),  # Python function
    re.compile(r"\bfunction\s+\w+\s*\("),  # JS function
    re.compile(r"\bclass\s+\w+"),  # Class definition
    re.compile(r"\bimport\s+\w+"),  # Import statement
    re.compile(r"\bfrom\s+\w+\s+import"),  # Python import
    re.compile(r"\breturn\s+"),  # Return statement
    re.compile(r"=>\s*[{\(]"),  # Arrow function
    re.compile(r"->\s*\w+"),  # Type annotation / arrow
    re.compile(r"[=!]=="),  # Strict equality
    re.compile(r"&&|\|\|"),  # Logical operators
    re.compile(r"\{\s*\}"),  # Empty braces
    re.compile(r"\[\s*\]"),  # Empty brackets
    re.compile(r"^\s*(//|#|/\*|\*/)"),  # Comments
    re.compile(r"\bif\s*\("),  # If statement
    re.compile(r"\bfor\s*\("),  # For loop
    re.compile(r"\bwhile\s*\("),  # While loop
    re.compile(r"\bconst\s+\w+"),  # JS const
    re.compile(r"\blet\s+\w+"),  # JS let
    re.compile(r"\bvar\s+\w+"),  # JS var
    re.compile(r"\bpub\s+(fn|struct|enum)"),  # Rust
    re.compile(r"\bfn\s+\w+"),  # Rust function
    re.compile(r"\bfunc\s+\w+"),  # Go function
    re.compile(r"<\w+>"),  # Generics
]

# Minimum patterns to match for code confidence
MIN_CODE_PATTERNS = 2

# Confidence for code detected via patterns
CODE_PATTERN_CONFIDENCE = 0.7


@dataclass
class CodeBlock:
    """A detected code block with language identification."""

    content: str
    language: str | None
    confidence: float
    bbox: dict  # {x1, y1, x2, y2} - merged bounding box

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class FrameCodeResult:
    """Code analysis results for a single frame."""

    frame_path: str
    timestamp: float
    code_blocks: list[CodeBlock]
    ide_detected: str | None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "frame_path": self.frame_path,
            "timestamp": self.timestamp,
            "code_blocks": [b.to_dict() for b in self.code_blocks],
            "ide_detected": self.ide_detected,
        }


def is_likely_code(text: str) -> tuple[bool, int]:
    """Check if text looks like code using pattern matching.

    Args:
        text: Text to analyze.

    Returns:
        Tuple of (is_code, pattern_count).
    """
    if not text or len(text) < 10:
        return False, 0

    pattern_count = sum(1 for p in CODE_PATTERNS if p.search(text))

    # Strong indicator: multiple patterns OR single pattern with length
    is_code = pattern_count >= MIN_CODE_PATTERNS or (
        pattern_count >= 1 and len(text) > 50
    )

    return is_code, pattern_count


def detect_language(text: str) -> str | None:
    """Detect programming language from code snippet.

    Uses pygments lexer guessing for accurate language detection.

    Args:
        text: Code text to analyze.

    Returns:
        Language name (lowercase) or None if unknown.
    """
    try:
        from pygments.lexers import guess_lexer
        from pygments.util import ClassNotFound

        # Clean up text before guessing
        clean_text = text.strip()
        if len(clean_text) < 10:
            return None

        try:
            lexer = guess_lexer(clean_text)
            return lexer.name.lower()
        except ClassNotFound:
            return None

    except ImportError:
        logger.debug("pygments not installed, using heuristics for language detection")
        return _detect_language_heuristic(text)


def _detect_language_heuristic(text: str) -> str | None:
    """Simple heuristic-based language detection when pygments unavailable."""
    text_lower = text.lower()

    # Python indicators
    if re.search(r"\bdef\s+\w+\s*\(", text) and "self" in text:
        return "python"
    if re.search(r"\bfrom\s+\w+\s+import", text):
        return "python"

    # JavaScript/TypeScript
    if re.search(r"\bconst\s+\w+\s*=", text) or re.search(r"=>\s*\{", text):
        if "interface" in text_lower or ": string" in text or ": number" in text:
            return "typescript"
        return "javascript"

    # Rust
    if re.search(r"\bfn\s+\w+", text) and ("mut" in text or "let" in text):
        return "rust"

    # Go
    if re.search(r"\bfunc\s+\w+", text) and "package" in text_lower:
        return "go"

    # Java
    if "public class" in text or "public static void main" in text:
        return "java"

    # C/C++
    if re.search(r"#include\s*<", text):
        return "c++" if "cout" in text or "std::" in text else "c"

    return None


def merge_bboxes(bboxes: list[dict]) -> dict:
    """Merge multiple bounding boxes into one encompassing box.

    Args:
        bboxes: List of bbox dicts with x1, y1, x2, y2 keys.

    Returns:
        Single bbox encompassing all input boxes.
    """
    if not bboxes:
        return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

    return {
        "x1": min(b["x1"] for b in bboxes),
        "y1": min(b["y1"] for b in bboxes),
        "x2": max(b["x2"] for b in bboxes),
        "y2": max(b["y2"] for b in bboxes),
    }


def cluster_regions_by_position(
    regions: list[TextRegion],
    vertical_threshold: int = 30,
) -> list[list[TextRegion]]:
    """Cluster text regions into groups by vertical proximity.

    Nearby regions (likely on same line or consecutive lines) are grouped.

    Args:
        regions: List of TextRegion objects.
        vertical_threshold: Max vertical distance to cluster together.

    Returns:
        List of region clusters (each cluster is a list of TextRegion).
    """
    if not regions:
        return []

    # Sort by y position
    sorted_regions = sorted(regions, key=lambda r: r.bbox["y1"])

    clusters: list[list[TextRegion]] = []
    current_cluster: list[TextRegion] = [sorted_regions[0]]

    for region in sorted_regions[1:]:
        # Check if this region is close to the previous cluster
        last_y2 = max(r.bbox["y2"] for r in current_cluster)
        if region.bbox["y1"] - last_y2 <= vertical_threshold:
            current_cluster.append(region)
        else:
            clusters.append(current_cluster)
            current_cluster = [region]

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def extract_code_blocks(ocr_result: FrameOCRResult) -> list[CodeBlock]:
    """Extract code blocks from OCR results.

    Groups OCR text regions, identifies code blocks, and detects languages.

    Args:
        ocr_result: OCR result from extract_text_from_frame().

    Returns:
        List of CodeBlock objects.
    """
    if not ocr_result.regions:
        return []

    # Cluster regions by position
    clusters = cluster_regions_by_position(ocr_result.regions)

    code_blocks = []
    for cluster in clusters:
        # Combine text from cluster
        text = "\n".join(r.text for r in cluster)

        # Check if it looks like code
        is_code, pattern_count = is_likely_code(text)
        if not is_code:
            continue

        # Detect language
        language = detect_language(text)

        # Calculate confidence based on pattern count
        confidence = min(1.0, CODE_PATTERN_CONFIDENCE + pattern_count * 0.1)

        # Merge bounding boxes
        bbox = merge_bboxes([r.bbox for r in cluster])

        code_blocks.append(
            CodeBlock(
                content=text,
                language=language,
                confidence=confidence,
                bbox=bbox,
            )
        )

    return code_blocks


# IDE signature detection
IDE_SIGNATURES = {
    "vscode": ["explorer", "terminal", "problems", "output", "debug console"],
    "intellij": ["project", "run", "debug", "terminal", "structure"],
    "xcode": ["navigator", "debug area", "utilities", "inspector"],
    "vim": ["normal", "insert", "visual", "command"],
    "sublime": ["goto anything", "command palette"],
}


def detect_ide(ocr_result: FrameOCRResult) -> str | None:
    """Detect IDE from OCR text.

    Args:
        ocr_result: OCR result to analyze.

    Returns:
        IDE name or None if not detected.
    """
    if not ocr_result.regions:
        return None

    # Combine all text
    all_text = " ".join(r.text.lower() for r in ocr_result.regions)

    for ide, signatures in IDE_SIGNATURES.items():
        matches = sum(1 for sig in signatures if sig in all_text)
        if matches >= 2:
            return ide

    return None


def analyze_frame_for_code(ocr_result: FrameOCRResult) -> FrameCodeResult:
    """Analyze a frame's OCR results for code blocks.

    Args:
        ocr_result: OCR result from the frame.

    Returns:
        FrameCodeResult with detected code blocks and IDE.
    """
    code_blocks = extract_code_blocks(ocr_result)
    ide = detect_ide(ocr_result)

    return FrameCodeResult(
        frame_path=ocr_result.frame_path,
        timestamp=ocr_result.timestamp,
        code_blocks=code_blocks,
        ide_detected=ide,
    )


def save_code_results(
    results: list[FrameCodeResult],
    output_path: Path,
) -> None:
    """Save code analysis results to JSON file.

    Args:
        results: List of FrameCodeResult objects.
        output_path: Path to output JSON file.
    """
    data = {
        "version": 1,
        "frames": [r.to_dict() for r in results],
        "summary": {
            "total_frames": len(results),
            "frames_with_code": sum(1 for r in results if r.code_blocks),
            "total_code_blocks": sum(len(r.code_blocks) for r in results),
            "languages": {},
            "ides": {},
        },
    }

    # Count languages
    for r in results:
        for block in r.code_blocks:
            lang = block.language or "unknown"
            data["summary"]["languages"][lang] = (
                data["summary"]["languages"].get(lang, 0) + 1
            )
        if r.ide_detected:
            data["summary"]["ides"][r.ide_detected] = (
                data["summary"]["ides"].get(r.ide_detected, 0) + 1
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    logger.info(f"Saved code results to {output_path}")


def load_code_results(input_path: Path) -> list[FrameCodeResult] | None:
    """Load cached code results from JSON file.

    Args:
        input_path: Path to the JSON file.

    Returns:
        List of FrameCodeResult objects, or None if file doesn't exist.
    """
    if not input_path.exists():
        return None

    try:
        data = json.loads(input_path.read_text())
        results = []
        for frame_data in data.get("frames", []):
            code_blocks = [CodeBlock(**b) for b in frame_data.get("code_blocks", [])]
            results.append(
                FrameCodeResult(
                    frame_path=frame_data["frame_path"],
                    timestamp=frame_data.get("timestamp", 0.0),
                    code_blocks=code_blocks,
                    ide_detected=frame_data.get("ide_detected"),
                )
            )
        return results
    except Exception as e:
        logger.warning(f"Failed to load code results from {input_path}: {e}")
        return None
