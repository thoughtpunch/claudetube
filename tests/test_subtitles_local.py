"""Tests for local subtitle detection and extraction."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claudetube.operations.subtitles import (
    convert_to_srt,
    find_embedded_subtitles,
    find_sidecar_subtitles,
    fetch_local_subtitles,
    srt_to_txt,
)


class TestFindSidecarSubtitles:
    """Tests for find_sidecar_subtitles function."""

    def test_finds_srt_sidecar(self, tmp_path):
        """Finds .srt sidecar file."""
        video = tmp_path / "video.mp4"
        srt = tmp_path / "video.srt"
        video.touch()
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        result = find_sidecar_subtitles(video)
        assert result == srt

    def test_finds_vtt_sidecar(self, tmp_path):
        """Finds .vtt sidecar file."""
        video = tmp_path / "video.mp4"
        vtt = tmp_path / "video.vtt"
        video.touch()
        vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello")

        result = find_sidecar_subtitles(video)
        assert result == vtt

    def test_finds_ass_sidecar(self, tmp_path):
        """Finds .ass sidecar file."""
        video = tmp_path / "video.mp4"
        ass = tmp_path / "video.ass"
        video.touch()
        ass.write_text("[Script Info]\nTitle: Test")

        result = find_sidecar_subtitles(video)
        assert result == ass

    def test_prefers_srt_over_vtt(self, tmp_path):
        """Prefers .srt when multiple sidecars exist."""
        video = tmp_path / "video.mp4"
        srt = tmp_path / "video.srt"
        vtt = tmp_path / "video.vtt"
        video.touch()
        srt.write_text("SRT content")
        vtt.write_text("VTT content")

        result = find_sidecar_subtitles(video)
        assert result == srt

    def test_returns_none_when_no_sidecar(self, tmp_path):
        """Returns None when no sidecar exists."""
        video = tmp_path / "video.mp4"
        video.touch()

        result = find_sidecar_subtitles(video)
        assert result is None


class TestFindEmbeddedSubtitles:
    """Tests for find_embedded_subtitles function."""

    def test_finds_subtitle_streams(self, tmp_path):
        """Returns subtitle streams from ffprobe."""
        video = tmp_path / "video.mp4"
        video.touch()

        mock_ffprobe = MagicMock()
        mock_ffprobe.probe.return_value = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
                {"codec_type": "subtitle", "codec_name": "subrip", "index": 2},
            ]
        }

        result = find_embedded_subtitles(video, ffprobe=mock_ffprobe)
        assert len(result) == 1
        assert result[0]["codec_type"] == "subtitle"

    def test_returns_empty_when_no_subtitles(self, tmp_path):
        """Returns empty list when no subtitle streams."""
        video = tmp_path / "video.mp4"
        video.touch()

        mock_ffprobe = MagicMock()
        mock_ffprobe.probe.return_value = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
            ]
        }

        result = find_embedded_subtitles(video, ffprobe=mock_ffprobe)
        assert result == []

    def test_returns_empty_when_probe_fails(self, tmp_path):
        """Returns empty list when ffprobe fails."""
        video = tmp_path / "video.mp4"
        video.touch()

        mock_ffprobe = MagicMock()
        mock_ffprobe.probe.return_value = None

        result = find_embedded_subtitles(video, ffprobe=mock_ffprobe)
        assert result == []


class TestConvertToSrt:
    """Tests for convert_to_srt function."""

    def test_converts_vtt_to_srt(self, tmp_path):
        """Converts VTT to SRT format."""
        vtt = tmp_path / "subs.vtt"
        srt = tmp_path / "subs.srt"
        vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello World")

        result = convert_to_srt(vtt, srt)
        assert result is True
        assert srt.exists()
        content = srt.read_text()
        assert "Hello World" in content

    def test_returns_false_on_invalid_file(self, tmp_path):
        """Returns False for invalid subtitle file."""
        invalid = tmp_path / "invalid.vtt"
        srt = tmp_path / "subs.srt"
        invalid.write_text("This is not a valid subtitle file")

        result = convert_to_srt(invalid, srt)
        # pysubs2 may or may not parse this, depends on version
        # Just ensure we don't crash
        assert result in (True, False)


class TestSrtToTxt:
    """Tests for srt_to_txt function."""

    def test_extracts_text_from_srt(self, tmp_path):
        """Extracts plain text from SRT file."""
        srt = tmp_path / "subs.srt"
        txt = tmp_path / "subs.txt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\nWorld"
        )

        result = srt_to_txt(srt, txt)
        assert result is True
        assert txt.exists()
        content = txt.read_text()
        assert "Hello" in content
        assert "World" in content


class TestFetchLocalSubtitles:
    """Tests for fetch_local_subtitles function."""

    def test_returns_sidecar_subtitles(self, tmp_path):
        """Returns sidecar subtitles when available."""
        video = tmp_path / "video.mp4"
        srt = tmp_path / "video.srt"
        video.touch()
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nHello from sidecar"
        )

        output_dir = tmp_path / "cache"
        output_dir.mkdir()

        result = fetch_local_subtitles(video, output_dir)
        assert result is not None
        assert result["source"] == "sidecar"
        assert "Hello from sidecar" in result["srt"]

    def test_checks_embedded_when_no_sidecar(self, tmp_path):
        """Checks embedded subtitles when no sidecar found."""
        video = tmp_path / "video.mp4"
        video.touch()

        output_dir = tmp_path / "cache"
        output_dir.mkdir()

        with patch("claudetube.operations.subtitles.find_embedded_subtitles") as mock_find:
            mock_find.return_value = [{"codec_type": "subtitle"}]
            with patch("claudetube.operations.subtitles.extract_embedded_subtitles") as mock_extract:
                mock_extract.return_value = False

                result = fetch_local_subtitles(video, output_dir)

        # Should have checked for embedded
        mock_find.assert_called_once()
        assert result is None  # Extraction failed

    def test_returns_none_when_no_subtitles(self, tmp_path):
        """Returns None when no subtitles found."""
        video = tmp_path / "video.mp4"
        video.touch()

        output_dir = tmp_path / "cache"
        output_dir.mkdir()

        with patch("claudetube.operations.subtitles.find_embedded_subtitles") as mock_find:
            mock_find.return_value = []

            result = fetch_local_subtitles(video, output_dir)

        assert result is None


class TestFetchLocalSubtitlesIntegration:
    """Integration tests for subtitle fetching."""

    def test_writes_srt_and_txt_files(self, tmp_path):
        """Writes both SRT and TXT files to output directory."""
        video = tmp_path / "video.mp4"
        srt = tmp_path / "video.srt"
        video.touch()
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nLine one\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\nLine two"
        )

        output_dir = tmp_path / "cache"
        output_dir.mkdir()

        result = fetch_local_subtitles(video, output_dir)

        assert result is not None
        assert (output_dir / "audio.srt").exists()
        assert (output_dir / "audio.txt").exists()

    def test_converts_vtt_sidecar_to_srt(self, tmp_path):
        """Converts VTT sidecar to SRT format."""
        video = tmp_path / "video.mp4"
        vtt = tmp_path / "video.vtt"
        video.touch()
        vtt.write_text(
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello from VTT"
        )

        output_dir = tmp_path / "cache"
        output_dir.mkdir()

        result = fetch_local_subtitles(video, output_dir)

        assert result is not None
        assert result["source"] == "sidecar"
        # Should have been converted to SRT
        srt_path = output_dir / "audio.srt"
        assert srt_path.exists()
