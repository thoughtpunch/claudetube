"""Tests for progress tracking functionality."""

from __future__ import annotations

import io

from claudetube.tools.progress import (
    ConsoleProgressReporter,
    SilentProgressReporter,
    create_console_reporter,
)
from claudetube.tools.yt_dlp import DownloadProgress


class TestDownloadProgress:
    """Tests for DownloadProgress dataclass."""

    def test_basic_fields(self):
        """Test basic field initialization."""
        p = DownloadProgress(
            status="downloading",
            percent=50.0,
            speed="1.5MiB/s",
            eta="00:30",
            downloaded_bytes=1024000,
            total_bytes=2048000,
        )

        assert p.status == "downloading"
        assert p.percent == 50.0
        assert p.speed == "1.5MiB/s"
        assert p.eta == "00:30"
        assert p.downloaded_bytes == 1024000
        assert p.total_bytes == 2048000

    def test_fragment_fields(self):
        """Test HLS/DASH fragment tracking fields."""
        p = DownloadProgress(
            status="downloading",
            fragment_index=5,
            fragment_count=100,
        )

        assert p.fragment_index == 5
        assert p.fragment_count == 100

    def test_postprocessor_fields(self):
        """Test postprocessor-specific fields."""
        p = DownloadProgress(
            status="started",
            postprocessor="FFmpegExtractAudio",
            filename="video.mp4",
        )

        assert p.status == "started"
        assert p.postprocessor == "FFmpegExtractAudio"
        assert p.filename == "video.mp4"

    def test_from_json_line_download(self):
        """Test parsing download progress JSON."""
        line = '{"status":"downloading","percent":"  50.0%","speed":"1.5MiB/s"}'
        p = DownloadProgress.from_json_line(line)

        assert p is not None
        assert p.status == "downloading"
        assert p.percent == 50.0
        assert p.speed == "1.5MiB/s"

    def test_from_json_line_postprocess(self):
        """Test parsing postprocessor progress JSON."""
        line = '{"status":"started","postprocessor":"FFmpegExtractAudio"}'
        p = DownloadProgress.from_json_line(line)

        assert p is not None
        assert p.status == "started"
        assert p.postprocessor == "FFmpegExtractAudio"

    def test_from_json_line_invalid(self):
        """Test that invalid lines return None."""
        assert DownloadProgress.from_json_line("not json") is None
        assert DownloadProgress.from_json_line('{"no_status":"field"}') is None
        assert DownloadProgress.from_json_line("") is None

    def test_from_json_line_with_bytes(self):
        """Test parsing progress with byte counts."""
        line = '{"status":"downloading","downloaded_bytes":1024,"total_bytes":2048}'
        p = DownloadProgress.from_json_line(line)

        assert p is not None
        assert p.downloaded_bytes == 1024
        assert p.total_bytes == 2048


class TestConsoleProgressReporter:
    """Tests for ConsoleProgressReporter."""

    def test_download_progress(self):
        """Test download progress display."""
        output = io.StringIO()
        reporter = ConsoleProgressReporter(output=output, show_bar=True)

        progress = DownloadProgress(
            status="downloading",
            percent=50.0,
            speed="1.5MiB/s",
            eta="00:30",
        )

        reporter(progress)
        result = output.getvalue()

        assert "50.0%" in result
        assert "1.5MiB/s" in result
        assert "00:30" in result

    def test_download_finished(self):
        """Test download finished display."""
        output = io.StringIO()
        reporter = ConsoleProgressReporter(output=output)

        progress = DownloadProgress(
            status="finished",
            filename="/path/to/video.mp4",
        )

        reporter(progress)
        result = output.getvalue()

        assert "video.mp4" in result

    def test_postprocessor_progress(self):
        """Test postprocessor progress display."""
        output = io.StringIO()
        reporter = ConsoleProgressReporter(output=output)

        # Started
        reporter(DownloadProgress(status="started", postprocessor="FFmpegExtractAudio"))
        assert "Extracting audio" in output.getvalue()

        # Finished
        output.truncate(0)
        output.seek(0)
        reporter(
            DownloadProgress(status="finished", postprocessor="FFmpegExtractAudio")
        )
        assert "done" in output.getvalue()

    def test_prefix(self):
        """Test custom prefix."""
        output = io.StringIO()
        reporter = ConsoleProgressReporter(output=output, prefix="[Download] ")

        reporter(DownloadProgress(status="downloading", percent=25.0))
        result = output.getvalue()

        assert "[Download]" in result

    def test_no_bar(self):
        """Test without progress bar."""
        output = io.StringIO()
        reporter = ConsoleProgressReporter(output=output, show_bar=False)

        reporter(DownloadProgress(status="downloading", percent=50.0))
        result = output.getvalue()

        assert "█" not in result
        assert "░" not in result


class TestSilentProgressReporter:
    """Tests for SilentProgressReporter."""

    def test_does_nothing(self):
        """Test that silent reporter doesn't output anything."""
        reporter = SilentProgressReporter()

        # Should not raise
        reporter(DownloadProgress(status="downloading", percent=50.0))
        reporter(DownloadProgress(status="finished"))
        reporter(DownloadProgress(status="started", postprocessor="FFmpegExtractAudio"))


class TestCreateConsoleReporter:
    """Tests for create_console_reporter factory."""

    def test_verbose_true(self):
        """Test verbose=True returns ConsoleProgressReporter."""
        reporter = create_console_reporter(verbose=True)
        assert isinstance(reporter, ConsoleProgressReporter)

    def test_verbose_false(self):
        """Test verbose=False returns SilentProgressReporter."""
        reporter = create_console_reporter(verbose=False)
        assert isinstance(reporter, SilentProgressReporter)

    def test_prefix(self):
        """Test prefix is passed through."""
        reporter = create_console_reporter(verbose=True, prefix="[Test] ")
        assert reporter.prefix == "[Test] "
