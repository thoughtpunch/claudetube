"""
Console progress reporters for download operations.

Provides ready-to-use ProgressCallback implementations for CLI usage.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from claudetube.tools.yt_dlp import DownloadProgress


class ConsoleProgressReporter:
    """Simple console progress reporter with optional bar display.

    Example usage:
        from claudetube.tools.progress import ConsoleProgressReporter
        from claudetube.operations import download_audio

        reporter = ConsoleProgressReporter()
        download_audio(url, output_path, on_progress=reporter)
    """

    def __init__(
        self,
        output: TextIO | None = None,
        show_bar: bool = True,
        bar_width: int = 30,
        show_speed: bool = True,
        show_eta: bool = True,
        prefix: str = "",
    ) -> None:
        """Initialize console progress reporter.

        Args:
            output: Output stream (default: sys.stderr)
            show_bar: Show progress bar visualization
            bar_width: Width of progress bar in characters
            show_speed: Show download speed
            show_eta: Show estimated time remaining
            prefix: Optional prefix before progress line
        """
        self.output = output or sys.stderr
        self.show_bar = show_bar
        self.bar_width = bar_width
        self.show_speed = show_speed
        self.show_eta = show_eta
        self.prefix = prefix
        self._last_line_len = 0

    def __call__(self, progress: DownloadProgress) -> None:
        """Handle progress update from download operation."""
        # Handle postprocessor progress
        if progress.postprocessor:
            self._handle_postprocess(progress)
            return

        # Handle download progress
        self._handle_download(progress)

    def _handle_download(self, progress: DownloadProgress) -> None:
        """Handle download phase progress."""
        status = progress.status

        if status == "downloading":
            parts = [self.prefix] if self.prefix else []

            # Progress bar
            if self.show_bar and progress.percent is not None:
                filled = int(self.bar_width * progress.percent / 100)
                bar = "█" * filled + "░" * (self.bar_width - filled)
                parts.append(f"[{bar}]")
                parts.append(f"{progress.percent:5.1f}%")

            # Speed
            if self.show_speed and progress.speed:
                speed = progress.speed.strip()
                if speed and speed != "N/A":
                    parts.append(f"@ {speed}")

            # ETA
            if self.show_eta and progress.eta:
                eta = progress.eta.strip()
                if eta and eta != "N/A":
                    parts.append(f"ETA: {eta}")

            # Fragment progress (for HLS/DASH)
            if progress.fragment_index and progress.fragment_count:
                parts.append(f"({progress.fragment_index}/{progress.fragment_count})")

            line = " ".join(parts)
            self._write_line(line, end="\r")

        elif status == "finished":
            # Clear the progress line and show completion
            self._clear_line()
            if progress.filename:
                filename = progress.filename.split("/")[-1]
                self._write_line(f"{self.prefix}Downloaded: {filename}")
            else:
                self._write_line(f"{self.prefix}Download complete")

    def _handle_postprocess(self, progress: DownloadProgress) -> None:
        """Handle postprocessor phase progress."""
        pp = progress.postprocessor or "Processing"
        status = progress.status

        # Map postprocessor names to user-friendly descriptions
        pp_names = {
            "FFmpegExtractAudio": "Extracting audio",
            "FFmpegVideoConvertor": "Converting video",
            "FFmpegVideoRemuxer": "Remuxing video",
            "FFmpegMetadata": "Writing metadata",
            "FFmpegFixupM3u8": "Fixing stream",
            "FFmpegMerger": "Merging formats",
        }
        friendly_name = pp_names.get(pp, pp)

        if status in ("started", "processing"):
            self._write_line(f"{self.prefix}{friendly_name}...", end="\r")
        elif status == "finished":
            self._clear_line()
            self._write_line(f"{self.prefix}{friendly_name}: done")

    def _write_line(self, text: str, end: str = "\n") -> None:
        """Write a line, tracking length for clearing."""
        self.output.write(text + end)
        self.output.flush()
        self._last_line_len = len(text)

    def _clear_line(self) -> None:
        """Clear the current line."""
        if self._last_line_len > 0:
            self.output.write("\r" + " " * self._last_line_len + "\r")
            self.output.flush()
            self._last_line_len = 0


class SilentProgressReporter:
    """Progress reporter that does nothing.

    Useful as a placeholder or for suppressing output.
    """

    def __call__(self, progress: DownloadProgress) -> None:
        """Ignore progress updates."""
        pass


def create_console_reporter(
    verbose: bool = True,
    prefix: str = "",
) -> ConsoleProgressReporter | SilentProgressReporter:
    """Create a progress reporter based on verbosity setting.

    Args:
        verbose: If True, return a ConsoleProgressReporter; if False, return SilentProgressReporter
        prefix: Optional prefix for progress lines

    Returns:
        Appropriate progress reporter
    """
    if verbose:
        return ConsoleProgressReporter(prefix=prefix)
    return SilentProgressReporter()
