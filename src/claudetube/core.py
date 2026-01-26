"""
Core video processing functions.
"""

import json
import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Optional imports for frame deduplication
try:
    from PIL import Image
    import imagehash
    HASH_AVAILABLE = True
except ImportError:
    HASH_AVAILABLE = False


@dataclass
class ProcessingResult:
    """Result of video processing."""
    success: bool
    output_dir: Path | None = None
    frames_dir: Path | None = None
    frames_count: int = 0
    duplicates_removed: int = 0
    transcript_txt: Path | None = None
    transcript_srt: Path | None = None
    transcript_json: Path | None = None
    manifest_path: Path | None = None
    error: str | None = None


@dataclass
class VideoSummarizer:
    """
    Main class for processing YouTube videos for AI summarization.

    Attributes:
        output_dir: Directory to save output files
        frame_interval: Seconds between frame captures (default: 10)
        whisper_model: Whisper model size (default: "base")
        similarity_threshold: Frame deduplication threshold 0-1 (default: 0.90)
        keep_video: Whether to keep the video file after processing
        log: Optional callback for logging messages
    """
    output_dir: Path
    frame_interval: int = 10
    whisper_model: str = "base"
    similarity_threshold: float = 0.90
    keep_video: bool = False
    skip_dedup: bool = False
    log: Callable[[str], None] | None = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir).resolve()
        self.frames_dir = self.output_dir / "frames"
        self.video_path = self.output_dir / "video.mp4"
        self.audio_path = self.output_dir / "audio.mp3"

    def _log(self, msg: str):
        """Log a message if callback is set."""
        if self.log:
            self.log(msg)
        else:
            print(msg)

    def process(self, url: str) -> ProcessingResult:
        """
        Process a YouTube video: download, extract frames, transcribe.

        Args:
            url: YouTube video URL

        Returns:
            ProcessingResult with paths to all generated files
        """
        # Create directories
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"Output directory: {self.output_dir}")

        # Step 1: Download video
        self._log("Downloading video...")
        if not download_video(url, self.video_path, self._log):
            return ProcessingResult(success=False, error="Failed to download video")

        # Step 2: Extract frames
        self._log(f"Extracting frames (every {self.frame_interval}s)...")
        frames = extract_frames(self.video_path, self.frames_dir, self.frame_interval, self._log)

        # Step 3: Deduplicate frames
        if not self.skip_dedup:
            self._log(f"Deduplicating frames (threshold: {self.similarity_threshold:.0%})...")
            kept, removed = deduplicate_frames(self.frames_dir, self.similarity_threshold, self._log)
        else:
            kept = len(list(self.frames_dir.glob("*.jpg")))
            removed = 0

        # Step 4: Extract audio
        self._log("Extracting audio...")
        if not extract_audio(self.video_path, self.audio_path, self._log):
            return ProcessingResult(success=False, error="Failed to extract audio")

        # Step 5: Transcribe
        self._log(f"Transcribing with Whisper ({self.whisper_model})...")
        if not transcribe_audio(self.audio_path, self.output_dir, self.whisper_model, self._log):
            return ProcessingResult(success=False, error="Failed to transcribe audio")

        # Clean up video if not keeping
        if not self.keep_video and self.video_path.exists():
            self.video_path.unlink()
            self._log("Removed video file to save space")

        # Write manifest
        manifest_path = self._write_manifest(url, kept, removed)

        self._log("Processing complete!")

        return ProcessingResult(
            success=True,
            output_dir=self.output_dir,
            frames_dir=self.frames_dir,
            frames_count=kept,
            duplicates_removed=removed,
            transcript_txt=self.output_dir / "audio.txt",
            transcript_srt=self.output_dir / "audio.srt",
            transcript_json=self.output_dir / "audio.json",
            manifest_path=manifest_path,
        )

    def _write_manifest(self, url: str, frames_kept: int, frames_removed: int) -> Path:
        """Write a manifest file summarizing the output."""
        manifest_path = self.output_dir / "MANIFEST.md"
        with open(manifest_path, "w") as f:
            f.write(f"# Video Summary Materials\n\n")
            f.write(f"**Source URL:** {url}\n\n")
            f.write(f"**Frame interval:** {self.frame_interval} seconds\n\n")
            f.write(f"**Whisper model:** {self.whisper_model}\n\n")
            f.write(f"**Frames:** {frames_kept} unique frames ({frames_removed} duplicates removed)\n\n")
            f.write(f"## Files\n\n")
            f.write(f"- `audio.txt` - Full transcript (plain text)\n")
            f.write(f"- `audio.srt` - Transcript with timestamps (SRT format)\n")
            f.write(f"- `audio.json` - Transcript with detailed timing data\n")
            f.write(f"- `frames/` - Extracted frames (named by timestamp: frame_MM-SS.jpg)\n")
            if self.keep_video:
                f.write(f"- `video.mp4` - Original video\n")
            f.write(f"\n## Frame Timestamps\n\n")
            f.write(f"Frame filenames indicate their timestamp in the video:\n")
            f.write(f"- `frame_00-00.jpg` = 0:00\n")
            f.write(f"- `frame_01-30.jpg` = 1:30\n")
            f.write(f"- `frame_10-45.jpg` = 10:45\n")
            f.write(f"- `frame_01-05-30.jpg` = 1:05:30 (for videos over 1 hour)\n")
        return manifest_path


def run_command(cmd: list[str], capture_output: bool = False) -> subprocess.CompletedProcess | None:
    """Run a shell command and return result or None on failure."""
    try:
        return subprocess.run(cmd, check=True, capture_output=capture_output, text=capture_output)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_video_duration(video_path: Path) -> float | None:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError):
        return None


def format_timestamp(seconds: float) -> str:
    """Convert seconds to timestamp string for filename (HH-MM-SS or MM-SS)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}-{minutes:02d}-{secs:02d}"
    return f"{minutes:02d}-{secs:02d}"


def download_video(url: str, output_path: Path, log: Callable[[str], None] | None = None) -> bool:
    """
    Download a YouTube video.

    Args:
        url: YouTube video URL
        output_path: Path to save the video
        log: Optional logging callback

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        url
    ]

    result = run_command(cmd)
    if result is None:
        # Try youtube-dl fallback
        if shutil.which("youtube-dl"):
            cmd[0] = "youtube-dl"
            result = run_command(cmd)

    return result is not None


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    interval: int,
    log: Callable[[str], None] | None = None
) -> list[Path]:
    """
    Extract frames at regular intervals with timestamp-based filenames.

    Args:
        video_path: Path to video file
        frames_dir: Directory to save frames
        interval: Seconds between frame captures
        log: Optional logging callback

    Returns:
        List of paths to extracted frames
    """
    duration = get_video_duration(video_path)

    if duration is None:
        # Fallback: extract without knowing duration
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps=1/{interval}",
            "-q:v", "2",
            str(frames_dir / "frame_%04d.jpg")
        ]
        run_command(cmd)
        return list(frames_dir.glob("*.jpg"))

    extracted = []
    timestamp = 0.0

    while timestamp < duration:
        ts_str = format_timestamp(timestamp)
        output = frames_dir / f"frame_{ts_str}.jpg"

        cmd = [
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            "-y",
            str(output)
        ]

        if run_command(cmd, capture_output=True):
            extracted.append(output)
            if log:
                log(f"  Extracted frame at {ts_str}")

        timestamp += interval

    return extracted


def deduplicate_frames(
    frames_dir: Path,
    similarity_threshold: float = 0.90,
    log: Callable[[str], None] | None = None
) -> tuple[int, int]:
    """
    Remove frames that are too similar to previous frames using perceptual hashing.

    Args:
        frames_dir: Directory containing frames
        similarity_threshold: 0-1, higher = more frames removed (default: 0.90)
        log: Optional logging callback

    Returns:
        Tuple of (frames_kept, frames_removed)
    """
    if not HASH_AVAILABLE:
        if log:
            log("Pillow/imagehash not available, skipping deduplication")
        return len(list(frames_dir.glob("*.jpg"))), 0

    # Hash difference threshold (lower = more similar)
    max_hash_diff = int((1 - similarity_threshold) * 64)

    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return 0, 0

    removed = 0
    last_hash = None

    for frame_path in frames:
        try:
            img = Image.open(frame_path)
            current_hash = imagehash.phash(img)
        except Exception:
            continue

        if last_hash is None:
            last_hash = current_hash
            continue

        if current_hash - last_hash <= max_hash_diff:
            frame_path.unlink()
            removed += 1
        else:
            last_hash = current_hash

    kept = len(list(frames_dir.glob("*.jpg")))
    return kept, removed


def extract_audio(
    video_path: Path,
    audio_path: Path,
    log: Callable[[str], None] | None = None
) -> bool:
    """
    Extract audio from video as MP3.

    Args:
        video_path: Path to video file
        audio_path: Path to save audio
        log: Optional logging callback

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-acodec", "mp3",
        "-y",
        str(audio_path)
    ]
    return run_command(cmd, capture_output=True) is not None


def transcribe_audio(
    audio_path: Path,
    output_dir: Path,
    model: str = "base",
    log: Callable[[str], None] | None = None
) -> bool:
    """
    Transcribe audio using Whisper.

    Args:
        audio_path: Path to audio file
        output_dir: Directory to save transcripts
        model: Whisper model size (tiny, base, small, medium, large)
        log: Optional logging callback

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        "whisper",
        str(audio_path),
        "--model", model,
        "--output_format", "all",
        "--output_dir", str(output_dir)
    ]
    return run_command(cmd) is not None
