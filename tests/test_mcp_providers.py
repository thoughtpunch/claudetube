"""Tests for MCP server provider integration.

Verifies:
1. list_providers_tool returns provider info
2. transcribe_video provider override works
3. generate_visual_transcripts provider override works
4. track_people_tool provider override works
5. Factory integration for default provider resolution
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory and patch get_cache_dir."""
    with patch("claudetube.mcp_server.get_cache_dir", return_value=tmp_path):
        yield tmp_path


# =============================================================================
# list_providers_tool
# =============================================================================


class TestListProvidersTool:
    """Tests for the list_providers MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_json(self, cache_dir):
        from claudetube.mcp_server import list_providers_tool

        result = await list_providers_tool()
        data = json.loads(result)

        assert "available_providers" in data
        assert "all_providers" in data
        assert "capabilities" in data

    @pytest.mark.asyncio
    async def test_capabilities_structure(self, cache_dir):
        from claudetube.mcp_server import list_providers_tool

        result = await list_providers_tool()
        data = json.loads(result)

        caps = data["capabilities"]
        assert "transcribe" in caps
        assert "vision" in caps
        assert "embed" in caps
        assert "reason" in caps
        assert "video" in caps

    @pytest.mark.asyncio
    async def test_each_capability_has_providers(self, cache_dir):
        from claudetube.mcp_server import list_providers_tool

        result = await list_providers_tool()
        data = json.loads(result)

        # Each capability should list providers with availability
        for cap_name, cap_data in data["capabilities"].items():
            assert "providers" in cap_data
            for p in cap_data["providers"]:
                assert "name" in p
                assert "available" in p

    @pytest.mark.asyncio
    async def test_voyage_listed_under_embed(self, cache_dir):
        from claudetube.mcp_server import list_providers_tool

        result = await list_providers_tool()
        data = json.loads(result)

        embed_providers = [p["name"] for p in data["capabilities"]["embed"]["providers"]]
        assert "voyage" in embed_providers

    @pytest.mark.asyncio
    async def test_whisper_local_listed_under_transcribe(self, cache_dir):
        from claudetube.mcp_server import list_providers_tool

        result = await list_providers_tool()
        data = json.loads(result)

        transcribe_providers = [
            p["name"] for p in data["capabilities"]["transcribe"]["providers"]
        ]
        assert "whisper-local" in transcribe_providers

    @pytest.mark.asyncio
    async def test_preferences_included(self, cache_dir):
        from claudetube.mcp_server import list_providers_tool

        with patch("claudetube.mcp_server.get_factory") as mock_factory:
            mock_config = MagicMock()
            mock_config.transcription_provider = "whisper-local"
            mock_config.transcription_fallbacks = ["whisper-local"]
            mock_config.vision_provider = "claude-code"
            mock_config.vision_fallbacks = ["claude-code"]
            mock_config.video_provider = None
            mock_config.reasoning_provider = "claude-code"
            mock_config.reasoning_fallbacks = ["claude-code"]
            mock_config.embedding_provider = "voyage"
            mock_factory.return_value.config = mock_config

            result = await list_providers_tool()
            data = json.loads(result)

        assert data["capabilities"]["transcribe"]["preferred"] == "whisper-local"
        assert data["capabilities"]["embed"]["preferred"] == "voyage"


# =============================================================================
# transcribe_video provider override
# =============================================================================


class TestTranscribeVideoProvider:
    """Tests for transcribe_video provider override."""

    @pytest.mark.asyncio
    async def test_default_uses_factory(self, cache_dir):
        from claudetube.mcp_server import transcribe_video

        mock_transcriber = MagicMock()

        with (
            patch("claudetube.mcp_server.get_factory") as mock_factory,
            patch("claudetube.mcp_server._transcribe_video", new_callable=AsyncMock) as mock_tv,
        ):
            mock_factory.return_value.get_transcriber.return_value = mock_transcriber
            mock_tv.return_value = {
                "success": True,
                "video_id": "test",
                "source": "whisper-local",
                "whisper_model": "small",
                "message": "ok",
                "transcript_srt": None,
                "transcript_txt": None,
            }

            await transcribe_video("test", whisper_model="small")

            mock_tv.assert_called_once()
            call_kwargs = mock_tv.call_args[1]
            assert call_kwargs["transcriber"] is mock_transcriber

    @pytest.mark.asyncio
    async def test_provider_override(self, cache_dir):
        from claudetube.mcp_server import transcribe_video

        mock_provider = MagicMock()

        with (
            patch("claudetube.providers.registry.get_provider", return_value=mock_provider),
            patch("claudetube.mcp_server._transcribe_video", new_callable=AsyncMock) as mock_tv,
        ):
            mock_tv.return_value = {
                "success": True,
                "video_id": "test",
                "source": "openai",
                "whisper_model": None,
                "message": "ok",
                "transcript_srt": None,
                "transcript_txt": None,
            }

            await transcribe_video("test", provider="openai")

            call_kwargs = mock_tv.call_args[1]
            assert call_kwargs["transcriber"] is mock_provider


# =============================================================================
# generate_visual_transcripts provider override
# =============================================================================


class TestVisualTranscriptsProvider:
    """Tests for generate_visual_transcripts provider override."""

    @pytest.mark.asyncio
    async def test_provider_override(self, cache_dir):
        from claudetube.mcp_server import generate_visual_transcripts

        mock_provider = MagicMock()

        with (
            patch("claudetube.providers.registry.get_provider", return_value=mock_provider),
            patch(
                "claudetube.operations.visual_transcript.generate_visual_transcript",
                return_value={"status": "ok"},
            ) as mock_gvt,
        ):
            await generate_visual_transcripts("test123", provider="anthropic")

            mock_gvt.assert_called_once()
            call_kwargs = mock_gvt.call_args[1]
            assert call_kwargs["vision_analyzer"] is mock_provider


# =============================================================================
# track_people_tool provider override
# =============================================================================


class TestTrackPeopleProvider:
    """Tests for track_people_tool provider override."""

    @pytest.mark.asyncio
    async def test_provider_override_with_vision(self, cache_dir):
        from claudetube.mcp_server import track_people_tool
        from claudetube.providers.base import VisionAnalyzer

        mock_provider = MagicMock(spec=VisionAnalyzer)

        with (
            patch("claudetube.providers.registry.get_provider", return_value=mock_provider),
            patch(
                "claudetube.operations.person_tracking.track_people",
                return_value={"status": "ok"},
            ) as mock_tp,
        ):
            await track_people_tool("test123", provider="anthropic")

            mock_tp.assert_called_once()
            call_kwargs = mock_tp.call_args[1]
            assert call_kwargs["vision_analyzer"] is mock_provider
