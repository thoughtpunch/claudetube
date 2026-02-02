"""
Subtitle detection and extraction for local video files.

Checks for:
1. Embedded subtitle streams in the video container
2. Sidecar subtitle files (.srt, .vtt, .ass, .ssa)
"""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

import pysubs2

if TYPE_CHECKING:
    from pathlib import Path

    from claudetube.tools.ffprobe import FFprobeTool

logger = logging.getLogger(__name__)

# Supported sidecar subtitle extensions in preference order
SIDECAR_EXTENSIONS = [".srt", ".vtt", ".ass", ".ssa"]


def find_embedded_subtitles(
    video_path: Path, ffprobe: FFprobeTool | None = None
) -> list[dict]:
    """Find embedded subtitle streams in a video file.

    Args:
        video_path: Path to video file
        ffprobe: Optional FFprobeTool instance (created if not provided)

    Returns:
        List of subtitle stream dicts with codec_name, language, etc.
    """
    if ffprobe is None:
        from claudetube.tools.ffprobe import FFprobeTool

        ffprobe = FFprobeTool()

    probe_data = ffprobe.probe(video_path)
    if not probe_data:
        return []

    streams = probe_data.get("streams", [])
    return [s for s in streams if s.get("codec_type") == "subtitle"]


def find_sidecar_subtitles(video_path: Path) -> Path | None:
    """Find sidecar subtitle file for a video.

    Checks for files with same name as video but subtitle extension.

    Args:
        video_path: Path to video file

    Returns:
        Path to sidecar subtitle file, or None if not found
    """
    for ext in SIDECAR_EXTENSIONS:
        sidecar = video_path.with_suffix(ext)
        if sidecar.exists():
            logger.info(f"Found sidecar subtitle: {sidecar.name}")
            return sidecar
    return None


def extract_embedded_subtitles(
    video_path: Path,
    output_path: Path,
    stream_index: int = 0,
) -> bool:
    """Extract embedded subtitle stream to SRT file.

    Args:
        video_path: Path to video file
        output_path: Path to output .srt file
        stream_index: Subtitle stream index (default: first subtitle stream)

    Returns:
        True if extraction succeeded, False otherwise
    """
    try:
        from claudetube.utils.system import find_tool

        ffmpeg_path = find_tool("ffmpeg")

        result = subprocess.run(
            [
                ffmpeg_path,
                "-i",
                str(video_path),
                "-map",
                f"0:s:{stream_index}",
                "-c:s",
                "srt",
                "-y",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            logger.warning(f"Failed to extract embedded subtitles: {result.stderr}")
            return False

        # Verify output file has content
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"Extracted embedded subtitles to {output_path.name}")
            return True
        return False

    except subprocess.TimeoutExpired:
        logger.warning("Subtitle extraction timed out")
        return False
    except Exception as e:
        logger.warning(f"Subtitle extraction failed: {e}")
        return False


def convert_to_srt(input_path: Path, output_path: Path) -> bool:
    """Convert subtitle file to SRT format using pysubs2.

    Supports conversion from: ASS, SSA, VTT, SUB, SBV, etc.

    Args:
        input_path: Path to source subtitle file
        output_path: Path to output .srt file

    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        subs = pysubs2.load(str(input_path))
        subs.save(str(output_path))
        logger.info(f"Converted {input_path.suffix} to SRT")
        return True
    except Exception as e:
        logger.warning(f"Subtitle conversion failed: {e}")
        return False


def srt_to_txt(srt_path: Path, txt_path: Path) -> bool:
    """Convert SRT file to plain text (strip timing, merge lines).

    Args:
        srt_path: Path to SRT file
        txt_path: Path to output TXT file

    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        subs = pysubs2.load(str(srt_path))
        # Extract just the text, one line per subtitle entry
        lines = [event.plaintext.replace("\\N", " ") for event in subs]
        txt_path.write_text("\n".join(lines))
        return True
    except Exception as e:
        logger.warning(f"SRT to TXT conversion failed: {e}")
        return False


def fetch_local_subtitles(
    video_path: Path,
    output_dir: Path,
) -> dict | None:
    """Check for and extract subtitles from a local video file.

    Checks in order:
    1. Sidecar subtitle files (.srt, .vtt, .ass, .ssa)
    2. Embedded subtitle streams in video container

    Args:
        video_path: Path to video file
        output_dir: Cache directory to write subtitle files

    Returns:
        Dict with 'srt', 'txt', 'source' keys if found, None otherwise
    """
    srt_path = output_dir / "audio.srt"
    txt_path = output_dir / "audio.txt"

    # 1. Check for sidecar subtitles
    sidecar = find_sidecar_subtitles(video_path)
    if sidecar:
        source = "sidecar"
        if sidecar.suffix == ".srt":
            # Already SRT, just copy
            srt_content = sidecar.read_text()
            srt_path.write_text(srt_content)
        else:
            # Convert to SRT
            if not convert_to_srt(sidecar, srt_path):
                logger.warning(f"Could not convert sidecar {sidecar.suffix}")
                sidecar = None  # Fall through to embedded check

        if sidecar:
            # Generate plain text version
            srt_to_txt(srt_path, txt_path)
            return {
                "srt": srt_path.read_text(),
                "txt": txt_path.read_text() if txt_path.exists() else "",
                "source": source,
            }

    # 2. Check for embedded subtitles
    embedded = find_embedded_subtitles(video_path)
    if embedded:
        logger.info(f"Found {len(embedded)} embedded subtitle stream(s)")
        if extract_embedded_subtitles(video_path, srt_path):
            srt_to_txt(srt_path, txt_path)
            return {
                "srt": srt_path.read_text(),
                "txt": txt_path.read_text() if txt_path.exists() else "",
                "source": "embedded",
            }

    return None
