"""Tests for operations/download.py module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from claudetube.exceptions import DownloadError, MetadataError
from claudetube.operations import download


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset module-level singletons between tests."""
    download._yt_dlp = None
    download._ffmpeg = None
    yield
    download._yt_dlp = None
    download._ffmpeg = None


# ---------------------------------------------------------------------------
# Singleton factories
# ---------------------------------------------------------------------------


class TestGetYtDlp:
    """Tests for _get_yt_dlp singleton."""

    @patch("claudetube.operations.download.YtDlpTool")
    def test_creates_instance_on_first_call(self, mock_cls):
        instance = download._get_yt_dlp()
        mock_cls.assert_called_once()
        assert instance is mock_cls.return_value

    @patch("claudetube.operations.download.YtDlpTool")
    def test_returns_same_instance_on_second_call(self, mock_cls):
        first = download._get_yt_dlp()
        second = download._get_yt_dlp()
        mock_cls.assert_called_once()
        assert first is second


class TestGetFfmpeg:
    """Tests for _get_ffmpeg singleton."""

    @patch("claudetube.operations.download.FFmpegTool")
    def test_creates_instance_on_first_call(self, mock_cls):
        instance = download._get_ffmpeg()
        mock_cls.assert_called_once()
        assert instance is mock_cls.return_value

    @patch("claudetube.operations.download.FFmpegTool")
    def test_returns_same_instance_on_second_call(self, mock_cls):
        first = download._get_ffmpeg()
        second = download._get_ffmpeg()
        mock_cls.assert_called_once()
        assert first is second


# ---------------------------------------------------------------------------
# fetch_metadata
# ---------------------------------------------------------------------------


class TestFetchMetadata:
    """Tests for fetch_metadata()."""

    @patch("claudetube.operations.download.YtDlpTool")
    def test_returns_metadata_dict(self, mock_cls):
        mock_tool = mock_cls.return_value
        mock_tool.get_metadata.return_value = {"title": "Test", "id": "abc123"}

        result = download.fetch_metadata("https://youtube.com/watch?v=abc123")

        mock_tool.get_metadata.assert_called_once_with(
            "https://youtube.com/watch?v=abc123", timeout=30
        )
        assert result == {"title": "Test", "id": "abc123"}

    @patch("claudetube.operations.download.YtDlpTool")
    def test_custom_timeout(self, mock_cls):
        mock_tool = mock_cls.return_value
        mock_tool.get_metadata.return_value = {}

        download.fetch_metadata("https://example.com/video", timeout=60)

        mock_tool.get_metadata.assert_called_once_with(
            "https://example.com/video", timeout=60
        )

    @patch("claudetube.operations.download.YtDlpTool")
    def test_raises_metadata_error(self, mock_cls):
        mock_tool = mock_cls.return_value
        mock_tool.get_metadata.side_effect = MetadataError("Failed to fetch")

        with pytest.raises(MetadataError, match="Failed to fetch"):
            download.fetch_metadata("https://example.com/bad")


# ---------------------------------------------------------------------------
# download_audio
# ---------------------------------------------------------------------------


class TestDownloadAudio:
    """Tests for download_audio()."""

    @patch("claudetube.operations.download.YtDlpTool")
    def test_returns_output_path(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        output = tmp_path / "audio.mp3"
        mock_tool.download_audio.return_value = output

        result = download.download_audio("https://example.com/v", output)

        mock_tool.download_audio.assert_called_once_with(
            "https://example.com/v", output, quality="64K"
        )
        assert result == output

    @patch("claudetube.operations.download.YtDlpTool")
    def test_custom_quality(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        output = tmp_path / "audio.mp3"
        mock_tool.download_audio.return_value = output

        download.download_audio("https://example.com/v", output, quality="128K")

        mock_tool.download_audio.assert_called_once_with(
            "https://example.com/v", output, quality="128K"
        )

    @patch("claudetube.operations.download.YtDlpTool")
    def test_raises_download_error(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        mock_tool.download_audio.side_effect = DownloadError("Audio download failed")

        with pytest.raises(DownloadError, match="Audio download failed"):
            download.download_audio("https://example.com/v", tmp_path / "audio.mp3")


# ---------------------------------------------------------------------------
# download_thumbnail
# ---------------------------------------------------------------------------


class TestDownloadThumbnail:
    """Tests for download_thumbnail()."""

    @patch("claudetube.operations.download.YtDlpTool")
    def test_returns_path_on_success(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        thumb = tmp_path / "thumbnail.jpg"
        mock_tool.download_thumbnail.return_value = thumb

        result = download.download_thumbnail("https://example.com/v", tmp_path)

        mock_tool.download_thumbnail.assert_called_once_with(
            "https://example.com/v", tmp_path, timeout=15
        )
        assert result == thumb

    @patch("claudetube.operations.download.YtDlpTool")
    def test_returns_none_when_unavailable(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        mock_tool.download_thumbnail.return_value = None

        result = download.download_thumbnail("https://example.com/v", tmp_path)

        assert result is None

    @patch("claudetube.operations.download.YtDlpTool")
    def test_custom_timeout(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        mock_tool.download_thumbnail.return_value = None

        download.download_thumbnail("https://example.com/v", tmp_path, timeout=30)

        mock_tool.download_thumbnail.assert_called_once_with(
            "https://example.com/v", tmp_path, timeout=30
        )


# ---------------------------------------------------------------------------
# fetch_subtitles
# ---------------------------------------------------------------------------


class TestFetchSubtitles:
    """Tests for fetch_subtitles()."""

    @patch("claudetube.operations.download.YtDlpTool")
    def test_returns_subtitle_dict(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        sub_data = {
            "srt": "1\n00:00:00,000 --> 00:00:05,000\nHello\n",
            "txt": "Hello",
            "source": "auto-generated",
        }
        mock_tool.fetch_subtitles.return_value = sub_data

        result = download.fetch_subtitles("https://example.com/v", tmp_path)

        mock_tool.fetch_subtitles.assert_called_once_with(
            "https://example.com/v", tmp_path, timeout=30
        )
        assert result == sub_data
        assert result["source"] == "auto-generated"

    @patch("claudetube.operations.download.YtDlpTool")
    def test_returns_none_when_no_subtitles(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        mock_tool.fetch_subtitles.return_value = None

        result = download.fetch_subtitles("https://example.com/v", tmp_path)

        assert result is None

    @patch("claudetube.operations.download.YtDlpTool")
    def test_custom_timeout(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        mock_tool.fetch_subtitles.return_value = None

        download.fetch_subtitles("https://example.com/v", tmp_path, timeout=60)

        mock_tool.fetch_subtitles.assert_called_once_with(
            "https://example.com/v", tmp_path, timeout=60
        )


# ---------------------------------------------------------------------------
# download_video_segment
# ---------------------------------------------------------------------------


class TestDownloadVideoSegment:
    """Tests for download_video_segment()."""

    @patch("claudetube.operations.download.YtDlpTool")
    def test_returns_path_on_success(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        output = tmp_path / "segment.mp4"
        mock_tool.download_video_segment.return_value = output

        result = download.download_video_segment(
            "https://example.com/v", output, start_time=10.0, end_time=20.0
        )

        mock_tool.download_video_segment.assert_called_once_with(
            url="https://example.com/v",
            output_path=output,
            start_time=10.0,
            end_time=20.0,
            quality_sort="+res,+size,+br,+fps",
            concurrent_fragments=1,
        )
        assert result == output

    @patch("claudetube.operations.download.YtDlpTool")
    def test_returns_none_on_failure(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        mock_tool.download_video_segment.return_value = None

        result = download.download_video_segment(
            "https://example.com/v",
            tmp_path / "segment.mp4",
            start_time=0,
            end_time=5,
        )

        assert result is None

    @patch("claudetube.operations.download.YtDlpTool")
    def test_custom_quality_sort_and_fragments(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        output = tmp_path / "segment.mp4"
        mock_tool.download_video_segment.return_value = output

        download.download_video_segment(
            "https://example.com/v",
            output,
            start_time=0,
            end_time=10,
            quality_sort="res:1080",
            concurrent_fragments=4,
        )

        mock_tool.download_video_segment.assert_called_once_with(
            url="https://example.com/v",
            output_path=output,
            start_time=0,
            end_time=10,
            quality_sort="res:1080",
            concurrent_fragments=4,
        )


# ---------------------------------------------------------------------------
# extract_audio_local
# ---------------------------------------------------------------------------


class TestExtractAudioLocal:
    """Tests for extract_audio_local()."""

    @patch("claudetube.operations.download.FFmpegTool")
    def test_returns_path_on_success(self, mock_cls, tmp_path):
        mock_tool = mock_cls.return_value
        output_path = tmp_path / "audio.mp3"
        mock_tool.extract_audio.return_value = output_path

        input_path = tmp_path / "video.mp4"
        input_path.write_bytes(b"fake video")

        result = download.extract_audio_local(input_path, tmp_path)

        mock_tool.extract_audio.assert_called_once_with(input_path, output_path)
        assert result == output_path

    @patch("claudetube.operations.download.FFmpegTool")
    def test_cache_hit_skips_extraction(self, mock_cls, tmp_path):
        """If audio.mp3 already exists, should return immediately without calling ffmpeg."""
        mock_tool = mock_cls.return_value

        # Pre-create the cached audio file
        output_path = tmp_path / "audio.mp3"
        output_path.write_bytes(b"cached audio")

        input_path = tmp_path / "video.mp4"
        input_path.write_bytes(b"fake video")

        result = download.extract_audio_local(input_path, tmp_path)

        mock_tool.extract_audio.assert_not_called()
        assert result == output_path

    @patch("claudetube.operations.download.FFmpegTool")
    def test_creates_output_dir(self, mock_cls, tmp_path):
        """Should create output directory if it doesn't exist."""
        mock_tool = mock_cls.return_value
        output_dir = tmp_path / "new_dir"
        output_path = output_dir / "audio.mp3"
        mock_tool.extract_audio.return_value = output_path

        input_path = tmp_path / "video.mp4"
        input_path.write_bytes(b"fake video")

        download.extract_audio_local(input_path, output_dir)

        assert output_dir.exists()

    @patch("claudetube.operations.download.FFmpegTool")
    def test_raises_runtime_error_on_failure(self, mock_cls, tmp_path):
        """Should raise RuntimeError when FFmpeg returns None."""
        mock_tool = mock_cls.return_value
        mock_tool.extract_audio.return_value = None

        input_path = tmp_path / "video.mp4"
        input_path.write_bytes(b"fake video")

        with pytest.raises(RuntimeError, match="Failed to extract audio"):
            download.extract_audio_local(input_path, tmp_path)

    @patch("claudetube.operations.download.FFmpegTool")
    def test_output_path_is_audio_mp3(self, mock_cls, tmp_path):
        """The output path passed to ffmpeg should always be output_dir/audio.mp3."""
        mock_tool = mock_cls.return_value
        expected_output = tmp_path / "audio.mp3"
        mock_tool.extract_audio.return_value = expected_output

        input_path = tmp_path / "video.mp4"
        input_path.write_bytes(b"fake video")

        download.extract_audio_local(input_path, tmp_path)

        call_args = mock_tool.extract_audio.call_args
        assert call_args[0][1] == expected_output
