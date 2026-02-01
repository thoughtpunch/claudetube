"""Tests for claudetube MCP server."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from claudetube.core import VideoResult


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory and patch get_cache_dir."""
    with patch("claudetube.mcp_server.get_cache_dir", return_value=tmp_path):
        yield tmp_path


@pytest.fixture
def cached_video(cache_dir):
    """Create a cached video with state.json, audio.txt, and audio.srt."""
    video_dir = cache_dir / "test12345678"
    video_dir.mkdir()

    state = {
        "video_id": "test12345678",
        "title": "Test Video",
        "duration_string": "5:00",
        "transcript_complete": True,
        "transcript_source": "uploaded",
    }
    (video_dir / "state.json").write_text(json.dumps(state))
    (video_dir / "audio.txt").write_text("Hello world, this is a test transcript.")
    (video_dir / "audio.srt").write_text(
        "1\n00:00:00,000 --> 00:00:05,000\nHello world, this is a test transcript.\n"
    )
    return video_dir


class TestProcessVideoTool:
    """Tests for the process_video MCP tool."""

    @pytest.mark.asyncio
    @patch("claudetube.mcp_server.process_video")
    async def test_returns_metadata_and_transcript(self, mock_pv, cache_dir):
        """Successful processing returns JSON with metadata and transcript."""
        from claudetube.mcp_server import process_video_tool

        txt_path = cache_dir / "test123" / "audio.txt"
        txt_path.parent.mkdir(parents=True)
        txt_path.write_text("Hello world")

        mock_pv.return_value = VideoResult(
            success=True,
            video_id="test123",
            output_dir=cache_dir / "test123",
            transcript_txt=txt_path,
            transcript_srt=cache_dir / "test123" / "audio.srt",
            metadata={"title": "Test"},
        )

        result = json.loads(await process_video_tool("https://youtu.be/test123"))

        assert result["video_id"] == "test123"
        assert result["transcript"] == "Hello world"
        assert result["metadata"]["title"] == "Test"

    @pytest.mark.asyncio
    @patch("claudetube.mcp_server.process_video")
    async def test_returns_error_on_failure(self, mock_pv, cache_dir):
        """Failed processing returns JSON with error."""
        from claudetube.mcp_server import process_video_tool

        mock_pv.return_value = VideoResult(
            success=False,
            video_id="fail",
            output_dir=cache_dir / "fail",
            error="Download failed",
        )

        result = json.loads(await process_video_tool("https://youtu.be/fail"))

        assert "error" in result
        assert result["error"] == "Download failed"

    @pytest.mark.asyncio
    @patch("claudetube.mcp_server.process_video")
    async def test_truncates_long_transcript(self, mock_pv, cache_dir):
        """Transcript longer than 50k chars is truncated."""
        from claudetube.mcp_server import process_video_tool

        txt_path = cache_dir / "long" / "audio.txt"
        txt_path.parent.mkdir(parents=True)
        txt_path.write_text("x" * 60_000)

        mock_pv.return_value = VideoResult(
            success=True,
            video_id="long",
            output_dir=cache_dir / "long",
            transcript_txt=txt_path,
            metadata={},
        )

        result = json.loads(await process_video_tool("https://youtu.be/long"))

        assert "truncated" in result["transcript"].lower()
        assert len(result["transcript"]) < 60_000


class TestGetFramesTool:
    """Tests for the get_frames MCP tool."""

    @pytest.mark.asyncio
    @patch("claudetube.mcp_server.get_frames_at")
    async def test_returns_frame_paths(self, mock_gf, cache_dir):
        """Returns JSON list of frame paths."""
        from claudetube.mcp_server import get_frames

        mock_gf.return_value = [
            Path("/fake/frame_00-00.jpg"),
            Path("/fake/frame_00-01.jpg"),
        ]

        result = json.loads(await get_frames("test123", start_time=0.0))

        assert result["frame_count"] == 2
        assert len(result["frame_paths"]) == 2

    @pytest.mark.asyncio
    @patch("claudetube.mcp_server.get_frames_at")
    async def test_returns_empty_on_no_frames(self, mock_gf, cache_dir):
        """Returns empty list when no frames extracted."""
        from claudetube.mcp_server import get_frames

        mock_gf.return_value = []

        result = json.loads(await get_frames("test123", start_time=0.0))

        assert result["frame_count"] == 0
        assert result["frame_paths"] == []


class TestGetHqFramesTool:
    """Tests for the get_hq_frames MCP tool."""

    @pytest.mark.asyncio
    @patch("claudetube.mcp_server.get_hq_frames_at")
    async def test_returns_hq_frame_paths(self, mock_ghf, cache_dir):
        """Returns JSON list of HQ frame paths."""
        from claudetube.mcp_server import get_hq_frames

        mock_ghf.return_value = [Path("/fake/hq_01-00.jpg")]

        result = json.loads(await get_hq_frames("test123", start_time=60.0))

        assert result["frame_count"] == 1
        assert "hq_01-00.jpg" in result["frame_paths"][0]

    @pytest.mark.asyncio
    @patch("claudetube.mcp_server.get_hq_frames_at")
    async def test_returns_empty_on_failure(self, mock_ghf, cache_dir):
        """Returns empty list when HQ extraction fails."""
        from claudetube.mcp_server import get_hq_frames

        mock_ghf.return_value = []

        result = json.loads(await get_hq_frames("test123", start_time=0.0))

        assert result["frame_count"] == 0


class TestListCachedVideosTool:
    """Tests for the list_cached_videos MCP tool."""

    @pytest.mark.asyncio
    async def test_lists_cached_videos(self, cached_video):
        """Returns list of cached videos from state.json files."""
        from claudetube.mcp_server import list_cached_videos

        result = json.loads(await list_cached_videos())

        assert result["count"] == 1
        assert result["videos"][0]["video_id"] == "test12345678"
        assert result["videos"][0]["title"] == "Test Video"
        assert result["videos"][0]["transcript_complete"] is True

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_cache(self, cache_dir):
        """Returns empty list when cache dir has no videos."""
        from claudetube.mcp_server import list_cached_videos

        result = json.loads(await list_cached_videos())

        assert result["count"] == 0
        assert result["videos"] == []

    @pytest.mark.asyncio
    async def test_skips_invalid_state_files(self, cache_dir):
        """Skips state.json files with invalid JSON."""
        from claudetube.mcp_server import list_cached_videos

        bad_dir = cache_dir / "badvideo"
        bad_dir.mkdir()
        (bad_dir / "state.json").write_text("not valid json")

        result = json.loads(await list_cached_videos())

        assert result["count"] == 0


class TestGetTranscriptTool:
    """Tests for the get_transcript MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_txt_transcript(self, cached_video):
        """Returns plain text transcript by default."""
        from claudetube.mcp_server import get_transcript

        result = json.loads(await get_transcript("test12345678"))

        assert result["video_id"] == "test12345678"
        assert result["format"] == "txt"
        assert "Hello world" in result["transcript"]

    @pytest.mark.asyncio
    async def test_returns_srt_transcript(self, cached_video):
        """Returns SRT transcript when format=srt."""
        from claudetube.mcp_server import get_transcript

        result = json.loads(await get_transcript("test12345678", format="srt"))

        assert result["format"] == "srt"
        assert "-->" in result["transcript"]

    @pytest.mark.asyncio
    async def test_returns_error_for_unknown_video(self, cache_dir):
        """Returns error for non-existent video ID."""
        from claudetube.mcp_server import get_transcript

        result = json.loads(await get_transcript("nonexistent1"))

        assert "error" in result

    @pytest.mark.asyncio
    async def test_falls_back_to_other_format(self, cache_dir):
        """Falls back to SRT if TXT doesn't exist."""
        from claudetube.mcp_server import get_transcript

        video_dir = cache_dir / "onlysrt12345"
        video_dir.mkdir()
        (video_dir / "audio.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")

        result = json.loads(await get_transcript("onlysrt12345", format="txt"))

        assert result["format"] == "srt"
        assert "Hi" in result["transcript"]

    @pytest.mark.asyncio
    async def test_returns_error_when_no_transcript_files(self, cache_dir):
        """Returns error when video dir exists but has no transcript files."""
        from claudetube.mcp_server import get_transcript

        video_dir = cache_dir / "notranscript"
        video_dir.mkdir()

        result = json.loads(await get_transcript("notranscript"))

        assert "error" in result
