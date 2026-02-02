#!/usr/bin/env python3
"""Benchmark EasyOCR vs VisionAnalyzer OCR quality.

Generates synthetic test images for each content type and measures:
- Accuracy (character error rate)
- Timing (seconds per frame)
- Content type classification correctness

Usage:
    # EasyOCR only (no API keys needed)
    python scripts/benchmark_ocr.py

    # Include vision provider comparison (requires API key)
    python scripts/benchmark_ocr.py --vision

    # Save results to file
    python scripts/benchmark_ocr.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class BenchmarkResult:
    """Result from benchmarking a single test case."""

    test_case: str
    content_type: str
    ground_truth: str
    easyocr_text: str
    easyocr_cer: float
    easyocr_time: float
    easyocr_regions: int
    easyocr_content_type: str
    vision_text: str | None = None
    vision_cer: float | None = None
    vision_time: float | None = None
    vision_content_type: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) using edit distance.

    CER = edit_distance(reference, hypothesis) / len(reference)

    Args:
        reference: Ground truth text.
        hypothesis: OCR-extracted text.

    Returns:
        CER as a float (0.0 = perfect, 1.0 = completely wrong).
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0

    ref = reference.strip()
    hyp = hypothesis.strip()

    # Simple Levenshtein distance
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / len(ref)


def generate_test_images(output_dir: Path) -> list[dict]:
    """Generate synthetic test images for each content type.

    Creates images with known ground truth text for benchmarking.

    Args:
        output_dir: Directory to save generated images.

    Returns:
        List of dicts with keys: path, ground_truth, content_type, test_case
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("ERROR: Pillow is required. Install with: pip install Pillow")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to get a monospace font
    mono_font = None
    for font_name in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]:
        try:
            mono_font = ImageFont.truetype(font_name, 16)
            break
        except OSError:
            continue

    if mono_font is None:
        mono_font = ImageFont.load_default()

    # Try to get a proportional font for slides
    prop_font = None
    for font_name in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
    ]:
        try:
            prop_font = ImageFont.truetype(font_name, 24)
            break
        except OSError:
            continue

    if prop_font is None:
        prop_font = mono_font

    test_cases = []

    # 1. Code screenshot (dark theme)
    code_text = """def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b"""
    img = Image.new("RGB", (640, 300), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), code_text, fill=(200, 200, 200), font=mono_font)
    path = output_dir / "code_dark.png"
    img.save(path)
    test_cases.append(
        {
            "path": str(path),
            "ground_truth": code_text,
            "content_type": "code",
            "test_case": "code_dark_theme",
        }
    )

    # 2. Code screenshot (light theme)
    code_text_light = """class UserService:
    def __init__(self, db):
        self.db = db

    async def get_user(self, user_id: int):
        return await self.db.fetch_one(
            "SELECT * FROM users WHERE id = $1", user_id
        )"""
    img = Image.new("RGB", (640, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), code_text_light, fill=(30, 30, 30), font=mono_font)
    path = output_dir / "code_light.png"
    img.save(path)
    test_cases.append(
        {
            "path": str(path),
            "ground_truth": code_text_light,
            "content_type": "code",
            "test_case": "code_light_theme",
        }
    )

    # 3. Terminal output
    terminal_text = """$ python manage.py migrate
Operations to perform:
  Apply all migrations: auth, contenttypes, sessions
Running migrations:
  Applying auth.0001_initial... OK
  Applying auth.0002_alter_permission... OK
  Applying sessions.0001_initial... OK"""
    img = Image.new("RGB", (640, 260), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), terminal_text, fill=(0, 255, 0), font=mono_font)
    path = output_dir / "terminal.png"
    img.save(path)
    test_cases.append(
        {
            "path": str(path),
            "ground_truth": terminal_text,
            "content_type": "terminal",
            "test_case": "terminal_output",
        }
    )

    # 4. Slide with text
    slide_title = "Machine Learning Pipeline"
    slide_bullets = """1. Data Collection and Preprocessing
2. Feature Engineering
3. Model Training and Validation
4. Hyperparameter Tuning
5. Deployment and Monitoring"""
    slide_full = f"{slide_title}\n\n{slide_bullets}"
    img = Image.new("RGB", (640, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype(
            prop_font.path if hasattr(prop_font, "path") else "", 32
        )
    except (OSError, AttributeError):
        title_font = prop_font
    draw.text((40, 30), slide_title, fill=(0, 0, 0), font=title_font)
    draw.text((60, 100), slide_bullets, fill=(50, 50, 50), font=prop_font)
    path = output_dir / "slides.png"
    img.save(path)
    test_cases.append(
        {
            "path": str(path),
            "ground_truth": slide_full,
            "content_type": "slides",
            "test_case": "presentation_slide",
        }
    )

    # 5. Diagram with labels
    diagram_labels = "Input -> Process -> Output"
    img = Image.new("RGB", (640, 200), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    # Draw boxes
    for _i, (label, x) in enumerate([("Input", 50), ("Process", 250), ("Output", 450)]):
        draw.rectangle([x, 60, x + 120, 120], outline=(0, 0, 0), width=2)
        draw.text((x + 20, 75), label, fill=(0, 0, 0), font=prop_font)
    # Draw arrows
    draw.line([(170, 90), (250, 90)], fill=(0, 0, 0), width=2)
    draw.line([(370, 90), (450, 90)], fill=(0, 0, 0), width=2)
    path = output_dir / "diagram.png"
    img.save(path)
    test_cases.append(
        {
            "path": str(path),
            "ground_truth": diagram_labels,
            "content_type": "diagram",
            "test_case": "labeled_diagram",
        }
    )

    # 6. Lower thirds (talking head with text overlay)
    lower_text = "Dr. Jane Smith - AI Research Lead"
    img = Image.new("RGB", (640, 360), color=(180, 200, 220))
    draw = ImageDraw.Draw(img)
    # Simulated lower third bar
    draw.rectangle([0, 280, 640, 340], fill=(0, 0, 0))
    draw.text((20, 290), lower_text, fill=(255, 255, 255), font=prop_font)
    path = output_dir / "lower_thirds.png"
    img.save(path)
    test_cases.append(
        {
            "path": str(path),
            "ground_truth": lower_text,
            "content_type": "talking_head",
            "test_case": "lower_thirds",
        }
    )

    return test_cases


def run_easyocr_benchmark(test_cases: list[dict]) -> list[BenchmarkResult]:
    """Run EasyOCR on all test cases and measure accuracy/timing.

    Args:
        test_cases: List of test case dicts from generate_test_images().

    Returns:
        List of BenchmarkResult objects.
    """
    from claudetube.analysis.ocr import extract_text_from_frame

    results = []
    for tc in test_cases:
        print(f"  EasyOCR: {tc['test_case']}...", end=" ", flush=True)

        t0 = time.time()
        try:
            ocr_result = extract_text_from_frame(tc["path"], min_confidence=0.3)
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(
                BenchmarkResult(
                    test_case=tc["test_case"],
                    content_type=tc["content_type"],
                    ground_truth=tc["ground_truth"],
                    easyocr_text="",
                    easyocr_cer=1.0,
                    easyocr_time=time.time() - t0,
                    easyocr_regions=0,
                    easyocr_content_type="error",
                )
            )
            continue

        elapsed = time.time() - t0

        # Combine all extracted text
        extracted_text = "\n".join(r.text for r in ocr_result.regions)
        cer = character_error_rate(tc["ground_truth"], extracted_text)

        result = BenchmarkResult(
            test_case=tc["test_case"],
            content_type=tc["content_type"],
            ground_truth=tc["ground_truth"],
            easyocr_text=extracted_text,
            easyocr_cer=round(cer, 4),
            easyocr_time=round(elapsed, 3),
            easyocr_regions=len(ocr_result.regions),
            easyocr_content_type=ocr_result.content_type,
        )
        results.append(result)
        print(f"CER={cer:.2%}, {elapsed:.2f}s, type={ocr_result.content_type}")

    return results


def run_vision_benchmark(
    test_cases: list[dict], results: list[BenchmarkResult]
) -> list[BenchmarkResult]:
    """Run VisionAnalyzer OCR on all test cases.

    Args:
        test_cases: List of test case dicts.
        results: Existing results to augment with vision data.

    Returns:
        Updated list of BenchmarkResult objects.
    """
    import asyncio

    from claudetube.analysis.ocr import extract_text_with_vision
    from claudetube.operations.visual_transcript import _get_default_vision_analyzer

    try:
        vision = _get_default_vision_analyzer()
        print(f"  Using vision provider: {vision.info.name}")
    except RuntimeError as e:
        print(f"  Vision benchmark skipped: {e}")
        return results

    for tc, result in zip(test_cases, results, strict=True):
        print(f"  Vision: {tc['test_case']}...", end=" ", flush=True)

        t0 = time.time()
        try:
            vision_result = asyncio.run(
                extract_text_with_vision(Path(tc["path"]), vision)
            )
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        elapsed = time.time() - t0

        extracted_text = "\n".join(r.text for r in vision_result.regions)
        cer = character_error_rate(tc["ground_truth"], extracted_text)

        result.vision_text = extracted_text
        result.vision_cer = round(cer, 4)
        result.vision_time = round(elapsed, 3)
        result.vision_content_type = vision_result.content_type

        print(f"CER={cer:.2%}, {elapsed:.2f}s")

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    header = f"{'Test Case':<25} {'Type':<12} {'EasyOCR CER':<14} {'Vision CER':<14} {'Better':<10}"
    print(header)
    print("-" * 80)

    for r in results:
        vision_cer_str = f"{r.vision_cer:.2%}" if r.vision_cer is not None else "N/A"
        easyocr_cer_str = f"{r.easyocr_cer:.2%}"

        if r.vision_cer is not None:
            if r.vision_cer < r.easyocr_cer:
                better = "Vision"
            elif r.easyocr_cer < r.vision_cer:
                better = "EasyOCR"
            else:
                better = "Tie"
        else:
            better = "-"

        print(
            f"{r.test_case:<25} {r.content_type:<12} {easyocr_cer_str:<14} {vision_cer_str:<14} {better:<10}"
        )

    # Averages
    print("-" * 80)
    avg_easy_cer = sum(r.easyocr_cer for r in results) / len(results)
    avg_easy_time = sum(r.easyocr_time for r in results) / len(results)
    print(f"{'AVERAGE':<25} {'':12} {avg_easy_cer:<14.2%}", end="")

    vision_results = [r for r in results if r.vision_cer is not None]
    if vision_results:
        avg_vision_cer = sum(r.vision_cer for r in vision_results) / len(vision_results)
        print(f" {avg_vision_cer:<14.2%}")
    else:
        print(f" {'N/A':<14}")

    print(f"\nAvg EasyOCR time: {avg_easy_time:.2f}s per frame")
    if vision_results:
        avg_vision_time = sum(r.vision_time for r in vision_results) / len(
            vision_results
        )
        print(f"Avg Vision time:  {avg_vision_time:.2f}s per frame")

    # Content type classification accuracy
    print("\nContent Type Classification:")
    correct = sum(1 for r in results if r.easyocr_content_type == r.content_type)
    print(f"  EasyOCR: {correct}/{len(results)} correct ({correct / len(results):.0%})")

    # _should_use_vision assessment
    print("\n_should_use_vision() recommendations:")
    for r in results:
        recommended = r.content_type in ("code", "terminal") or r.easyocr_cer > 0.5
        actual_benefit = r.vision_cer is not None and r.vision_cer < r.easyocr_cer
        status = ""
        if r.vision_cer is not None:
            if recommended and actual_benefit:
                status = "CORRECT (recommended + beneficial)"
            elif recommended and not actual_benefit:
                status = "FALSE POSITIVE (recommended but not beneficial)"
            elif not recommended and actual_benefit:
                status = "MISSED (not recommended but beneficial)"
            else:
                status = "CORRECT (not recommended, not needed)"
        else:
            status = f"{'would recommend' if recommended else 'would skip'}"
        print(f"  {r.test_case:<25} {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark EasyOCR vs VisionAnalyzer OCR quality"
    )
    parser.add_argument(
        "--vision", action="store_true", help="Include vision provider comparison"
    )
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory with test images (default: generates synthetic images)",
    )
    args = parser.parse_args()

    # Generate test images
    if args.image_dir:
        image_dir = Path(args.image_dir)
        print(f"Using existing images from: {image_dir}")
        # TODO: Load test cases from directory with ground truth files
        print(
            "ERROR: Custom image directory not yet supported. Use default generation."
        )
        sys.exit(1)
    else:
        image_dir = Path(__file__).parent.parent / ".benchmark_frames"
        print(f"Generating synthetic test images in: {image_dir}")
        test_cases = generate_test_images(image_dir)
        print(f"Generated {len(test_cases)} test images.\n")

    # Run EasyOCR benchmark
    print("Running EasyOCR benchmark:")
    results = run_easyocr_benchmark(test_cases)

    # Optionally run vision benchmark
    if args.vision:
        print("\nRunning Vision benchmark:")
        results = run_vision_benchmark(test_cases, results)

    # Print summary
    print_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_cases": len(test_cases),
            "results": [r.to_dict() for r in results],
        }
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to: {output_path}")

    # Clean up generated images
    if not args.image_dir:
        import shutil

        shutil.rmtree(image_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
