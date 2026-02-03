"""Tests for process_video (URL-based processing pipeline)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from claudetube.models.video_result import VideoResult
from claudetube.operations.processor import process_video

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_VIDEO_ID = "dQw4w9WgXcQ"
FAKE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

FAKE_URL_CONTEXT = {
    "video_id": FAKE_VIDEO_ID,
    "playlist_id": None,
    "original_url": FAKE_URL,
    "clean_url": FAKE_URL,
    "provider": "YouTube",
    "provider_data": {},
}

FAKE_METADATA = {
    "title": "Test Video",
    "duration": 180.0,
    "duration_string": "3:00",
    "uploader": "TestUser",
    "channel": "TestChannel",
    "upload_date": "20240115",
    "description": "A test video",
    "categories": ["Education"],
    "tags": ["test"],
    "language": "en",
    "view_count": 1000,
    "like_count": 50,
    "thumbnail": "https://example.com/thumb.jpg",
}


@pytest.fixture
def cache_base(tmp_path):
    """Provide a temporary cache directory."""
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


def _patch_url_context(ctx=None):
    """Patch extract_url_context to return a fixed context dict."""
    return patch(
        "claudetube.operations.processor.extract_url_context",
        return_value=ctx or FAKE_URL_CONTEXT,
    )


# ---------------------------------------------------------------------------
# Subtitle-path tests (fast path – no audio download or whisper)
# ---------------------------------------------------------------------------


class TestProcessVideoSubtitlePath:
    """Tests for the subtitle (fast) processing path."""

    def test_returns_video_result(self, cache_base):
        """process_video always returns a VideoResult."""
        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value={
                    "srt": "1\n00:00:00,000 --> 00:00:02,000\nhello",
                    "txt": "hello",
                    "source": "auto-generated",
                },
            ),
        ):
            result = process_video(FAKE_URL, output_base=cache_base)

        assert isinstance(result, VideoResult)

    def test_success_via_subtitles(self, cache_base):
        """Returns success=True when subtitles are available."""
        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value={
                    "srt": "1\n00:00:00,000 --> 00:00:02,000\nhello",
                    "txt": "hello",
                    "source": "auto-generated",
                },
            ),
        ):
            result = process_video(FAKE_URL, output_base=cache_base)

        assert result.success is True
        assert result.error is None
        assert result.video_id == FAKE_VIDEO_ID

    def test_transcript_files_written(self, cache_base):
        """SRT and TXT transcript files are created from subtitles."""
        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value={
                    "srt": "1\n00:00:00,000 --> 00:00:02,000\nhello world",
                    "txt": "hello world",
                    "source": "uploaded",
                },
            ),
        ):
            result = process_video(FAKE_URL, output_base=cache_base)

        assert result.transcript_srt is not None
        assert result.transcript_srt.exists()
        assert result.transcript_txt is not None
        assert result.transcript_txt.exists()
        assert result.transcript_txt.read_text() == "hello world"

    def test_state_saved_with_subtitle_source(self, cache_base):
        """state.json records the subtitle source."""
        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value={
                    "srt": "1\n00:00:00,000 --> 00:00:02,000\nhello",
                    "txt": "hello",
                    "source": "auto-generated",
                },
            ),
        ):
            result = process_video(FAKE_URL, output_base=cache_base)

        state = json.loads((result.output_dir / "state.json").read_text())
        assert state["transcript_complete"] is True
        assert state["transcript_source"] == "auto-generated"

    def test_no_whisper_when_subtitles_found(self, cache_base):
        """Whisper is never called when subtitles are available (cheap first)."""
        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value={
                    "srt": "sub",
                    "txt": "sub",
                    "source": "auto-generated",
                },
            ),
            patch(
                "claudetube.operations.processor.download_audio",
            ) as mock_dl_audio,
            patch(
                "claudetube.operations.processor.transcribe_audio",
            ) as mock_transcribe,
        ):
            process_video(FAKE_URL, output_base=cache_base)

        mock_dl_audio.assert_not_called()
        mock_transcribe.assert_not_called()

    def test_metadata_populated(self, cache_base):
        """result.metadata contains values from yt-dlp metadata."""
        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value={
                    "srt": "s",
                    "txt": "t",
                    "source": "auto-generated",
                },
            ),
        ):
            result = process_video(FAKE_URL, output_base=cache_base)

        assert result.metadata["title"] == "Test Video"
        assert result.metadata["duration"] == 180.0
        assert result.metadata["uploader"] == "TestUser"


# ---------------------------------------------------------------------------
# Whisper-path tests (no subtitles available)
# ---------------------------------------------------------------------------


class TestProcessVideoWhisperPath:
    """Tests for the whisper fallback processing path."""

    def test_whisper_fallback_on_no_subtitles(self, cache_base):
        """Falls back to whisper when fetch_subtitles returns None."""
        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.download_audio",
            ) as mock_dl_audio,
            patch(
                "claudetube.operations.processor.transcribe_audio",
                return_value={
                    "srt": "1\n00:00:00,000 --> 00:00:03,000\nwhispered text",
                    "txt": "whispered text",
                },
            ) as mock_transcribe,
        ):

            def _fake_download(url, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fake-audio")

            mock_dl_audio.side_effect = _fake_download

            result = process_video(FAKE_URL, output_base=cache_base)

        assert result.success is True
        mock_dl_audio.assert_called_once()
        mock_transcribe.assert_called_once()
        assert result.transcript_txt.read_text() == "whispered text"

    def test_whisper_model_passed_through(self, cache_base):
        """whisper_model parameter is forwarded to transcribe_audio."""
        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.download_audio",
            ) as mock_dl_audio,
            patch(
                "claudetube.operations.processor.transcribe_audio",
                return_value={"srt": "s", "txt": "t"},
            ) as mock_transcribe,
        ):
            mock_dl_audio.side_effect = lambda url, path: path.parent.mkdir(
                parents=True, exist_ok=True
            ) or path.write_bytes(b"audio")

            process_video(FAKE_URL, output_base=cache_base, whisper_model="medium")

        _, kwargs = mock_transcribe.call_args
        assert kwargs["model_size"] == "medium"

    def test_state_saved_with_whisper_source(self, cache_base):
        """state.json records whisper as transcript source."""
        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.download_audio",
            ) as mock_dl_audio,
            patch(
                "claudetube.operations.processor.transcribe_audio",
                return_value={"srt": "s", "txt": "t"},
            ),
        ):
            mock_dl_audio.side_effect = lambda url, path: path.parent.mkdir(
                parents=True, exist_ok=True
            ) or path.write_bytes(b"audio")

            result = process_video(FAKE_URL, output_base=cache_base)

        state = json.loads((result.output_dir / "state.json").read_text())
        assert state["transcript_complete"] is True
        assert state["transcript_source"] == "whisper"
        assert state["whisper_model"] == "tiny"  # default

    def test_skips_audio_download_when_cached(self, cache_base):
        """Skips download_audio if audio.mp3 already exists."""
        # Pre-create the audio file at hierarchical path using channel from metadata
        # Since FAKE_METADATA has channel="TestChannel" (no channel_id), the enriched
        # path will be youtube/TestChannel/no_playlist/video_id
        audio_dir = (
            cache_base / "youtube" / "TestChannel" / "no_playlist" / FAKE_VIDEO_ID
        )
        audio_dir.mkdir(parents=True)
        (audio_dir / "audio.mp3").write_bytes(b"pre-existing-audio")

        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.download_audio",
            ) as mock_dl_audio,
            patch(
                "claudetube.operations.processor.transcribe_audio",
                return_value={"srt": "s", "txt": "t"},
            ),
        ):
            result = process_video(FAKE_URL, output_base=cache_base)

        mock_dl_audio.assert_not_called()
        assert result.success is True


# ---------------------------------------------------------------------------
# Cache-hit tests
# ---------------------------------------------------------------------------


class TestProcessVideoCacheHit:
    """Tests for the cache-hit (already processed) path."""

    def _populate_cache(self, cache_base, video_id=FAKE_VIDEO_ID):
        """Pre-populate the cache to simulate a previous run."""
        from claudetube.cache.manager import CacheManager
        from claudetube.models.state import VideoState

        cache = CacheManager(cache_base)
        state = VideoState.from_metadata(video_id, FAKE_URL, FAKE_METADATA)
        state.transcript_complete = True
        state.transcript_source = "auto-generated"
        cache.save_state(video_id, state)

        srt, txt = cache.get_transcript_paths(video_id)
        srt.write_text("1\n00:00:00,000 --> 00:00:02,000\ncached")
        txt.write_text("cached")

        thumb = cache.get_thumbnail_path(video_id)
        thumb.write_bytes(b"fake-thumb")

        return cache

    def test_cache_hit_returns_success(self, cache_base):
        """Returns success immediately on cache hit."""
        self._populate_cache(cache_base)

        with _patch_url_context():
            result = process_video(FAKE_URL, output_base=cache_base)

        assert result.success is True
        assert result.video_id == FAKE_VIDEO_ID

    def test_cache_hit_no_network_calls(self, cache_base):
        """No metadata/download calls are made on cache hit."""
        self._populate_cache(cache_base)

        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
            ) as mock_meta,
            patch(
                "claudetube.operations.processor.download_audio",
            ) as mock_dl,
            patch(
                "claudetube.operations.processor.fetch_subtitles",
            ) as mock_subs,
        ):
            process_video(FAKE_URL, output_base=cache_base)

        mock_meta.assert_not_called()
        mock_dl.assert_not_called()
        mock_subs.assert_not_called()

    def test_cache_hit_returns_transcript_paths(self, cache_base):
        """Cached transcripts are returned in the result."""
        self._populate_cache(cache_base)

        with _patch_url_context():
            result = process_video(FAKE_URL, output_base=cache_base)

        assert result.transcript_srt is not None
        assert result.transcript_srt.exists()
        assert result.transcript_txt is not None
        assert result.transcript_txt.read_text() == "cached"

    def test_cache_hit_returns_thumbnail(self, cache_base):
        """Cached thumbnail is returned in the result."""
        self._populate_cache(cache_base)

        with _patch_url_context():
            result = process_video(FAKE_URL, output_base=cache_base)

        assert result.thumbnail is not None
        assert result.thumbnail.exists()

    def test_cache_hit_returns_metadata(self, cache_base):
        """Cached metadata is returned in the result."""
        self._populate_cache(cache_base)

        with _patch_url_context():
            result = process_video(FAKE_URL, output_base=cache_base)

        assert result.metadata["title"] == "Test Video"
        assert result.metadata["transcript_complete"] is True


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestProcessVideoErrors:
    """Tests for error handling in the URL processing pipeline."""

    def test_metadata_fetch_failure(self, cache_base):
        """Returns error result when fetch_metadata raises."""
        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                side_effect=RuntimeError("Network unreachable"),
            ),
        ):
            result = process_video(FAKE_URL, output_base=cache_base)

        assert result.success is False
        assert "Network unreachable" in result.error

    def test_audio_download_failure(self, cache_base):
        """Returns error result when download_audio raises."""
        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.download_audio",
                side_effect=RuntimeError("403 Forbidden"),
            ),
        ):
            result = process_video(FAKE_URL, output_base=cache_base)

        assert result.success is False
        assert "403 Forbidden" in result.error

    def test_transcription_failure_continues(self, cache_base):
        """Transcription failure doesn't crash — returns success with no transcript."""
        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.download_audio",
            ) as mock_dl_audio,
            patch(
                "claudetube.operations.processor.transcribe_audio",
                side_effect=RuntimeError("Whisper OOM"),
            ),
        ):
            mock_dl_audio.side_effect = lambda url, path: path.parent.mkdir(
                parents=True, exist_ok=True
            ) or path.write_bytes(b"audio")

            result = process_video(FAKE_URL, output_base=cache_base)

        # Succeeds overall even though transcription failed
        assert result.success is True
        # No transcript files should exist
        assert result.transcript_srt is None or not result.transcript_srt.exists()


# ---------------------------------------------------------------------------
# Thumbnail handling
# ---------------------------------------------------------------------------


class TestProcessVideoThumbnail:
    """Tests for thumbnail download behavior."""

    def test_thumbnail_downloaded(self, cache_base):
        """Thumbnail is downloaded and state updated."""

        def _fake_thumb_download(url, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            thumb = output_dir / "thumbnail.jpg"
            thumb.write_bytes(b"thumb-data")
            return thumb

        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                side_effect=_fake_thumb_download,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value={
                    "srt": "s",
                    "txt": "t",
                    "source": "auto-generated",
                },
            ),
        ):
            result = process_video(FAKE_URL, output_base=cache_base)

        state = json.loads((result.output_dir / "state.json").read_text())
        assert state["has_thumbnail"] is True

    def test_no_thumbnail_available(self, cache_base):
        """Handles missing thumbnail gracefully."""
        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value={
                    "srt": "s",
                    "txt": "t",
                    "source": "auto-generated",
                },
            ),
        ):
            result = process_video(FAKE_URL, output_base=cache_base)

        assert result.success is True
        # Thumbnail path is returned as None if file doesn't exist
        assert result.thumbnail is None


# ---------------------------------------------------------------------------
# Playlist context
# ---------------------------------------------------------------------------


class TestProcessVideoPlaylistContext:
    """Tests for playlist-aware processing."""

    def test_playlist_id_stored_in_state(self, cache_base):
        """Playlist ID from URL is persisted in state.json."""
        ctx = {
            **FAKE_URL_CONTEXT,
            "playlist_id": "PLexample123",
        }
        with (
            _patch_url_context(ctx),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value={
                    "srt": "s",
                    "txt": "t",
                    "source": "auto-generated",
                },
            ),
        ):
            result = process_video(FAKE_URL, output_base=cache_base)

        state = json.loads((result.output_dir / "state.json").read_text())
        assert state["playlist_id"] == "PLexample123"


# ---------------------------------------------------------------------------
# Frame extraction (optional)
# ---------------------------------------------------------------------------


class TestProcessVideoFrames:
    """Tests for optional frame extraction."""

    def test_no_frames_by_default(self, cache_base):
        """Frames are not extracted when extract_frames=False (default)."""
        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.download_audio",
            ) as mock_dl_audio,
            patch(
                "claudetube.operations.processor.transcribe_audio",
                return_value={"srt": "s", "txt": "t"},
            ),
        ):
            mock_dl_audio.side_effect = lambda url, path: path.parent.mkdir(
                parents=True, exist_ok=True
            ) or path.write_bytes(b"audio")

            result = process_video(FAKE_URL, output_base=cache_base)

        assert result.frames == []

    def test_frames_extracted_when_requested(self, cache_base):
        """Frames are extracted when extract_frames=True."""
        fake_frames = [Path("/fake/frame_001.jpg"), Path("/fake/frame_002.jpg")]

        with (
            _patch_url_context(),
            patch(
                "claudetube.operations.processor.fetch_metadata",
                return_value=FAKE_METADATA,
            ),
            patch(
                "claudetube.operations.processor.download_thumbnail",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.fetch_subtitles",
                return_value=None,
            ),
            patch(
                "claudetube.operations.processor.download_audio",
            ) as mock_dl_audio,
            patch(
                "claudetube.operations.processor.transcribe_audio",
                return_value={"srt": "s", "txt": "t"},
            ),
            patch(
                "claudetube.operations.download.download_video_segment",
            ) as mock_dl_video,
            patch(
                "claudetube.operations.processor.FFmpegTool",
            ) as mock_ffmpeg_cls,
        ):
            mock_dl_audio.side_effect = lambda url, path: path.parent.mkdir(
                parents=True, exist_ok=True
            ) or path.write_bytes(b"audio")

            # Simulate video download creating the file
            def _fake_video_download(**kwargs):
                out = kwargs["output_path"]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"fake-video")
                return out

            mock_dl_video.side_effect = _fake_video_download
            mock_ffmpeg_cls.return_value.extract_frames_interval.return_value = (
                fake_frames
            )

            result = process_video(
                FAKE_URL,
                output_base=cache_base,
                extract_frames=True,
                frame_interval=15,
            )

        assert result.frames == fake_frames
        # State should record frame info
        state = json.loads((result.output_dir / "state.json").read_text())
        assert state["frames_count"] == 2
        assert state["frame_interval"] == 15
