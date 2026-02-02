"""Tests for Audio Description MCP tools."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory and patch get_cache_dir."""
    with patch("claudetube.mcp_server.get_cache_dir", return_value=tmp_path):
        yield tmp_path


@pytest.fixture
def cached_video(cache_dir):
    """Create a cached video with state.json and transcript."""
    video_dir = cache_dir / "testvideo123"
    video_dir.mkdir()

    state = {
        "video_id": "testvideo123",
        "url": "https://youtu.be/testvideo123",
        "title": "Test Video",
        "duration": 300.0,
        "duration_string": "5:00",
        "transcript_complete": True,
        "transcript_source": "uploaded",
        "ad_complete": False,
        "ad_source": None,
        "ad_track_available": None,
    }
    (video_dir / "state.json").write_text(json.dumps(state))
    (video_dir / "audio.txt").write_text(
        "Hello world, this is a test transcript about code."
    )
    (video_dir / "audio.srt").write_text(
        "1\n00:00:00,000 --> 00:00:05,000\nHello world, this is a test transcript.\n"
    )
    return video_dir


@pytest.fixture
def cached_video_with_ad(cached_video):
    """Create a cached video that also has audio descriptions."""
    vtt_content = (
        "WEBVTT\nKind: descriptions\nLanguage: en\n\n"
        "1\n00:00:00.000 --> 00:00:30.000\n"
        "A person sitting at a desk typing on a laptop.\n\n"
        "2\n00:00:30.000 --> 00:01:00.000\n"
        "Code editor shown on screen with Python code.\n"
    )
    txt_content = (
        "[00:00] A person sitting at a desk typing on a laptop.\n"
        "[00:30] Code editor shown on screen with Python code."
    )
    (cached_video / "audio.ad.vtt").write_text(vtt_content)
    (cached_video / "audio.ad.txt").write_text(txt_content)

    # Update state
    state = json.loads((cached_video / "state.json").read_text())
    state["ad_complete"] = True
    state["ad_source"] = "scene_compilation"
    (cached_video / "state.json").write_text(json.dumps(state))

    return cached_video


# =============================================================================
# get_descriptions tests
# =============================================================================


class TestGetDescriptions:
    """Tests for the get_descriptions MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_error_for_uncached_video(self, cache_dir):
        """Returns error when video is not cached."""
        from claudetube.mcp_server import get_descriptions

        result = json.loads(await get_descriptions("nonexistent"))

        assert "error" in result
        assert "not cached" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_returns_cached_vtt(self, cached_video_with_ad):
        """Returns cached VTT content when available."""
        from claudetube.mcp_server import get_descriptions

        result = json.loads(await get_descriptions("testvideo123", format="vtt"))

        assert "error" not in result
        assert result["source"] == "cache"
        assert "WEBVTT" in result["content"]
        assert "typing on a laptop" in result["content"]

    @pytest.mark.asyncio
    async def test_returns_cached_txt(self, cached_video_with_ad):
        """Returns cached TXT content when format=txt."""
        from claudetube.mcp_server import get_descriptions

        result = json.loads(await get_descriptions("testvideo123", format="txt"))

        assert "error" not in result
        assert result["source"] == "cache"
        assert "[00:00]" in result["content"]
        assert "typing on a laptop" in result["content"]

    @pytest.mark.asyncio
    @patch("claudetube.operations.audio_description.compile_scene_descriptions")
    @patch("claudetube.mcp_server.has_scenes")
    async def test_compiles_from_scenes(
        self, mock_has_scenes, mock_compile, cached_video
    ):
        """Compiles AD from scene data when no cache exists."""
        from claudetube.mcp_server import get_descriptions

        mock_has_scenes.return_value = True

        # Mock compile to create the AD files (simulating real behavior)
        def compile_side_effect(video_id, force=False, output_base=None):
            (cached_video / "audio.ad.vtt").write_text(
                "WEBVTT\nKind: descriptions\nLanguage: en\n\n"
                "1\n00:00:00.000 --> 00:00:30.000\nTest desc.\n"
            )
            (cached_video / "audio.ad.txt").write_text("[00:00] Test desc.")
            return {
                "video_id": "testvideo123",
                "status": "compiled",
                "cue_count": 1,
                "source": "scene_compilation",
            }

        mock_compile.side_effect = compile_side_effect

        result = json.loads(await get_descriptions("testvideo123"))

        assert "error" not in result
        assert result["source"] == "scene_compilation"

    @pytest.mark.asyncio
    async def test_regenerate_bypasses_cache(self, cached_video_with_ad):
        """regenerate=True bypasses cached AD and tries to regenerate."""
        from claudetube.mcp_server import get_descriptions

        # With regenerate=False, returns cache
        result1 = json.loads(await get_descriptions("testvideo123", regenerate=False))
        assert result1["source"] == "cache"

        # With regenerate=True, it won't return the cache directly
        # (it will try other strategies; since no scenes/providers exist, it may error)
        # We just verify it doesn't return "cache" as source
        with (
            patch("claudetube.mcp_server.has_scenes", return_value=False),
            patch(
                "claudetube.operations.audio_description.AudioDescriptionGenerator.generate",
                new_callable=AsyncMock,
                return_value={"error": "No scenes found."},
            ),
        ):
            result2 = json.loads(
                await get_descriptions("testvideo123", regenerate=True)
            )
            # Should not be "cache" source since regenerate bypasses it
            assert result2.get("source") != "cache" or "error" in result2


# =============================================================================
# describe_moment tests
# =============================================================================


class TestDescribeMoment:
    """Tests for the describe_moment MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_error_for_uncached_video(self, cache_dir):
        """Returns error when video is not cached."""
        from claudetube.mcp_server import describe_moment

        result = json.loads(await describe_moment("nonexistent", timestamp=10.0))

        assert "error" in result
        assert "not cached" in result["error"].lower()

    @pytest.mark.asyncio
    @patch("claudetube.mcp_server.get_hq_frames_at")
    async def test_returns_error_when_no_frames(self, mock_frames, cached_video):
        """Returns error when frame extraction fails."""
        from claudetube.mcp_server import describe_moment

        mock_frames.return_value = []

        result = json.loads(await describe_moment("testvideo123", timestamp=10.0))

        assert "error" in result
        assert "frames" in result["error"].lower()

    @pytest.mark.asyncio
    @patch("claudetube.mcp_server.get_factory")
    @patch("claudetube.mcp_server.get_hq_frames_at")
    async def test_returns_frames_without_provider(
        self, mock_frames, mock_factory, cached_video
    ):
        """Returns frame paths when no vision provider available."""
        from claudetube.mcp_server import describe_moment

        mock_frames.return_value = [Path("/fake/frame1.jpg"), Path("/fake/frame2.jpg")]
        mock_factory.side_effect = RuntimeError("No provider")

        result = json.loads(await describe_moment("testvideo123", timestamp=10.0))

        assert result["frame_count"] == 2
        assert result["description"] is None
        assert "No vision provider" in result["note"]

    @pytest.mark.asyncio
    @patch("claudetube.mcp_server.get_factory")
    @patch("claudetube.mcp_server.get_hq_frames_at")
    async def test_returns_description_with_provider(
        self, mock_frames, mock_factory, cached_video
    ):
        """Returns description when vision provider is available."""
        from claudetube.mcp_server import describe_moment

        mock_frames.return_value = [Path("/fake/frame1.jpg")]

        mock_vision = MagicMock()
        mock_vision.info.name = "test-provider"
        mock_vision.analyze_images = AsyncMock(
            return_value="A person typing on a keyboard at a desk."
        )

        mock_factory_inst = MagicMock()
        mock_factory_inst.get_vision_analyzer.return_value = mock_vision
        mock_factory.return_value = mock_factory_inst

        result = json.loads(await describe_moment("testvideo123", timestamp=10.0))

        assert result["frame_count"] == 1
        assert result["description"] == "A person typing on a keyboard at a desk."
        assert result["provider"] == "test-provider"

    @pytest.mark.asyncio
    @patch("claudetube.mcp_server.get_factory")
    @patch("claudetube.mcp_server.get_hq_frames_at")
    async def test_passes_context_to_prompt(
        self, mock_frames, mock_factory, cached_video
    ):
        """Context parameter is included in the vision prompt."""
        from claudetube.mcp_server import describe_moment

        mock_frames.return_value = [Path("/fake/frame1.jpg")]

        mock_vision = MagicMock()
        mock_vision.info.name = "test-provider"
        mock_vision.analyze_images = AsyncMock(return_value="desc")

        mock_factory_inst = MagicMock()
        mock_factory_inst.get_vision_analyzer.return_value = mock_vision
        mock_factory.return_value = mock_factory_inst

        await describe_moment(
            "testvideo123", timestamp=10.0, context="looking at the code editor"
        )

        # Check that context was included in the prompt
        call_args = mock_vision.analyze_images.call_args
        assert "looking at the code editor" in call_args.kwargs["prompt"]


# =============================================================================
# get_accessible_transcript tests
# =============================================================================


class TestGetAccessibleTranscript:
    """Tests for the get_accessible_transcript MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_error_for_uncached_video(self, cache_dir):
        """Returns error when video is not cached."""
        from claudetube.mcp_server import get_accessible_transcript

        result = json.loads(await get_accessible_transcript("nonexistent"))

        assert "error" in result
        assert "not cached" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_returns_error_when_no_transcript(self, cache_dir):
        """Returns error when transcript is missing."""
        from claudetube.mcp_server import get_accessible_transcript

        video_dir = cache_dir / "notranscript"
        video_dir.mkdir()

        result = json.loads(await get_accessible_transcript("notranscript"))

        assert "error" in result
        assert "transcript" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_returns_error_when_no_ad(self, cached_video):
        """Returns error when no audio descriptions exist."""
        from claudetube.mcp_server import get_accessible_transcript

        result = json.loads(await get_accessible_transcript("testvideo123"))

        assert "error" in result
        assert "audio descriptions" in result["error"].lower()
        assert result.get("transcript_available") is True

    @pytest.mark.asyncio
    async def test_returns_merged_transcript(self, cached_video_with_ad):
        """Returns merged transcript with AD entries."""
        from claudetube.mcp_server import get_accessible_transcript

        result = json.loads(await get_accessible_transcript("testvideo123"))

        assert "error" not in result
        assert result["format"] == "accessible_transcript"
        assert result["ad_entry_count"] == 2
        assert "ACCESSIBLE TRANSCRIPT" in result["content"]
        assert "Spoken Transcript" in result["content"]
        assert "Visual Descriptions" in result["content"]
        assert "[AD 00:00]" in result["content"]
        assert "[AD 00:30]" in result["content"]
        assert "typing on a laptop" in result["content"]


# =============================================================================
# has_audio_description tests
# =============================================================================


class TestHasAudioDescription:
    """Tests for the has_audio_description MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_error_for_uncached_video(self, cache_dir):
        """Returns error when video is not cached."""
        from claudetube.mcp_server import has_audio_description

        result = json.loads(await has_audio_description("nonexistent"))

        assert "error" in result
        assert "not cached" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_detects_no_ad(self, cached_video):
        """Returns false when no AD exists."""
        from claudetube.mcp_server import has_audio_description

        # Set ad_track_available to False to skip yt-dlp check
        state = json.loads((cached_video / "state.json").read_text())
        state["ad_track_available"] = False
        (cached_video / "state.json").write_text(json.dumps(state))

        result = json.loads(await has_audio_description("testvideo123"))

        assert result["has_cached_ad"] is False
        assert result["ad_complete"] is False
        assert result["has_source_ad_track"] is False

    @pytest.mark.asyncio
    async def test_detects_cached_ad(self, cached_video_with_ad):
        """Returns true when AD files exist in cache."""
        from claudetube.mcp_server import has_audio_description

        result = json.loads(await has_audio_description("testvideo123"))

        assert result["has_cached_ad"] is True
        assert result["ad_complete"] is True
        assert result["ad_source"] == "scene_compilation"

    @pytest.mark.asyncio
    @patch("claudetube.tools.yt_dlp.YtDlpTool.check_audio_description")
    async def test_checks_source_ad_track(self, mock_check_ad, cached_video):
        """Checks yt-dlp for source AD track when unknown."""
        from claudetube.mcp_server import has_audio_description

        mock_check_ad.return_value = {
            "format_id": "251-ad",
            "format_note": "audio description",
            "language": "en-ad",
        }

        result = json.loads(await has_audio_description("testvideo123"))

        assert result["has_source_ad_track"] is True
        assert result["source_ad_format"]["format_id"] == "251-ad"

        # Verify it persisted the discovery to state
        state = json.loads((cached_video / "state.json").read_text())
        assert state["ad_track_available"] is True

    @pytest.mark.asyncio
    @patch("claudetube.tools.yt_dlp.YtDlpTool.check_audio_description")
    async def test_handles_ytdlp_check_failure(self, mock_check_ad, cached_video):
        """Handles yt-dlp check failure gracefully."""
        from claudetube.mcp_server import has_audio_description

        mock_check_ad.side_effect = Exception("Network error")

        result = json.loads(await has_audio_description("testvideo123"))

        assert result["has_source_ad_track"] is None
        assert "source_check_error" in result

    @pytest.mark.asyncio
    async def test_skips_ytdlp_for_local_video(self, cache_dir):
        """Skips yt-dlp check for local videos (no URL)."""
        from claudetube.mcp_server import has_audio_description

        video_dir = cache_dir / "localvideo123"
        video_dir.mkdir()
        state = {
            "video_id": "localvideo123",
            "url": None,
            "source_type": "local",
            "ad_complete": False,
            "ad_source": None,
            "ad_track_available": None,
        }
        (video_dir / "state.json").write_text(json.dumps(state))

        result = json.loads(await has_audio_description("localvideo123"))

        # Should not attempt yt-dlp (no URL), and not error
        assert "error" not in result
        assert result["has_source_ad_track"] is None
