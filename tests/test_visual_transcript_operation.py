"""Tests for VisualTranscriptOperation.

Verifies:
1. Operation instantiation with VisionAnalyzer
2. execute() with structured dict response
3. execute() with string (JSON) response fallback
4. Prompt building with and without transcript context
5. Model name extraction from provider info
6. _should_skip_scene logic
7. _get_default_vision_analyzer fallback
8. generate_visual_transcript backward compatibility
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from claudetube.cache.scenes import SceneBoundary
from claudetube.operations.visual_transcript import (
    VISUAL_PROMPT,
    VisualDescription,
    VisualTranscriptOperation,
    _get_default_vision_analyzer,
    _should_skip_scene,
    generate_visual_transcript,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vision_analyzer():
    """Create a mock VisionAnalyzer with structured output support."""
    analyzer = AsyncMock()
    info_mock = MagicMock()
    info_mock.name = "anthropic"
    type(analyzer).info = PropertyMock(return_value=info_mock)
    analyzer.is_available.return_value = True
    return analyzer


@pytest.fixture
def sample_scene():
    """Create a sample SceneBoundary."""
    return SceneBoundary(
        scene_id=0,
        start_time=10.0,
        end_time=25.0,
        title="Introduction",
        transcript_text="Hello and welcome to this video about Python programming.",
    )


@pytest.fixture
def sample_scene_no_transcript():
    """Create a SceneBoundary with no transcript."""
    return SceneBoundary(
        scene_id=1,
        start_time=30.0,
        end_time=45.0,
    )


@pytest.fixture
def sample_keyframes(tmp_path):
    """Create sample keyframe image files."""
    frames = []
    for i in range(3):
        f = tmp_path / f"kf_{i:02d}.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # JPEG header
        frames.append(f)
    return frames


# ---------------------------------------------------------------------------
# VisualTranscriptOperation Tests
# ---------------------------------------------------------------------------


class TestVisualTranscriptOperationInit:
    """Tests for operation instantiation."""

    def test_accepts_vision_analyzer(self, mock_vision_analyzer):
        op = VisualTranscriptOperation(mock_vision_analyzer)
        assert op.vision is mock_vision_analyzer

    def test_stores_reference(self, mock_vision_analyzer):
        op = VisualTranscriptOperation(mock_vision_analyzer)
        assert op.vision is mock_vision_analyzer


class TestBuildPrompt:
    """Tests for prompt building."""

    def test_prompt_without_transcript(
        self, mock_vision_analyzer, sample_scene_no_transcript
    ):
        op = VisualTranscriptOperation(mock_vision_analyzer)
        prompt = op._build_prompt(sample_scene_no_transcript)
        assert "Describe what is visually happening" in prompt
        assert "Transcript context" not in prompt

    def test_prompt_with_transcript(self, mock_vision_analyzer, sample_scene):
        op = VisualTranscriptOperation(mock_vision_analyzer)
        prompt = op._build_prompt(sample_scene)
        assert "Describe what is visually happening" in prompt
        assert "Transcript context: Hello and welcome" in prompt

    def test_prompt_truncates_long_transcript(self, mock_vision_analyzer):
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=10.0,
            transcript_text="word " * 200,  # 1000 chars
        )
        op = VisualTranscriptOperation(mock_vision_analyzer)
        prompt = op._build_prompt(scene)
        # Transcript context should be truncated to 500 chars
        context_part = prompt.split("Transcript context: ")[1]
        assert len(context_part) <= 500

    def test_uses_module_constant_template(
        self, mock_vision_analyzer, sample_scene_no_transcript
    ):
        op = VisualTranscriptOperation(mock_vision_analyzer)
        prompt = op._build_prompt(sample_scene_no_transcript)
        expected = VISUAL_PROMPT.format(context="")
        assert prompt == expected


class TestExecute:
    """Tests for the execute method."""

    @pytest.mark.asyncio
    async def test_execute_with_dict_response(
        self, mock_vision_analyzer, sample_scene, sample_keyframes
    ):
        """Provider returns dict when schema is provided."""
        mock_vision_analyzer.analyze_images.return_value = {
            "description": "A presenter showing Python code on screen",
            "people": ["presenter"],
            "objects": ["laptop", "projector"],
            "text_on_screen": ["def main():"],
            "actions": ["typing"],
            "setting": "conference room",
        }

        op = VisualTranscriptOperation(mock_vision_analyzer)
        result = await op.execute(
            scene_id=0,
            keyframes=sample_keyframes,
            scene=sample_scene,
        )

        assert isinstance(result, VisualDescription)
        assert result.scene_id == 0
        assert result.description == "A presenter showing Python code on screen"
        assert result.people == ["presenter"]
        assert result.objects == ["laptop", "projector"]
        assert result.text_on_screen == ["def main():"]
        assert result.actions == ["typing"]
        assert result.setting == "conference room"
        assert result.keyframe_count == 3
        assert result.model_used == "anthropic"

    @pytest.mark.asyncio
    async def test_execute_with_string_response(
        self, mock_vision_analyzer, sample_scene, sample_keyframes
    ):
        """Fallback: provider returns JSON string instead of dict."""
        mock_vision_analyzer.analyze_images.return_value = json.dumps(
            {
                "description": "Code on screen",
                "people": [],
                "objects": ["monitor"],
                "text_on_screen": ["import os"],
                "actions": [],
                "setting": "office",
            }
        )

        op = VisualTranscriptOperation(mock_vision_analyzer)
        result = await op.execute(
            scene_id=1,
            keyframes=sample_keyframes,
            scene=sample_scene,
        )

        assert isinstance(result, VisualDescription)
        assert result.description == "Code on screen"
        assert result.objects == ["monitor"]

    @pytest.mark.asyncio
    async def test_execute_passes_schema(
        self, mock_vision_analyzer, sample_scene, sample_keyframes
    ):
        """Verify that the Pydantic schema is passed to analyze_images."""
        mock_vision_analyzer.analyze_images.return_value = {
            "description": "test",
            "people": [],
            "objects": [],
            "text_on_screen": [],
            "actions": [],
            "setting": None,
        }

        op = VisualTranscriptOperation(mock_vision_analyzer)
        await op.execute(scene_id=0, keyframes=sample_keyframes, scene=sample_scene)

        # Check that analyze_images was called with a schema argument
        call_args = mock_vision_analyzer.analyze_images.call_args
        assert call_args is not None
        schema = call_args[1].get("schema") or call_args[0][2]
        assert schema is not None
        # Should be the Pydantic VisualDescription model
        assert hasattr(schema, "model_json_schema")

    @pytest.mark.asyncio
    async def test_execute_passes_keyframes(
        self, mock_vision_analyzer, sample_scene, sample_keyframes
    ):
        """Verify keyframes are passed to analyze_images."""
        mock_vision_analyzer.analyze_images.return_value = {
            "description": "test",
            "people": [],
            "objects": [],
            "text_on_screen": [],
            "actions": [],
            "setting": None,
        }

        op = VisualTranscriptOperation(mock_vision_analyzer)
        await op.execute(scene_id=0, keyframes=sample_keyframes, scene=sample_scene)

        call_args = mock_vision_analyzer.analyze_images.call_args
        images = call_args[0][0]
        assert len(images) == 3
        assert all(isinstance(p, Path) for p in images)

    @pytest.mark.asyncio
    async def test_execute_missing_fields_default(
        self, mock_vision_analyzer, sample_scene, sample_keyframes
    ):
        """Missing fields in response should default to empty values."""
        mock_vision_analyzer.analyze_images.return_value = {
            "description": "Minimal response",
        }

        op = VisualTranscriptOperation(mock_vision_analyzer)
        result = await op.execute(
            scene_id=0,
            keyframes=sample_keyframes,
            scene=sample_scene,
        )

        assert result.description == "Minimal response"
        assert result.people == []
        assert result.objects == []
        assert result.text_on_screen == []
        assert result.actions == []
        assert result.setting is None

    @pytest.mark.asyncio
    async def test_execute_model_name_from_info(self, sample_scene, sample_keyframes):
        """Model name should come from provider.info.name."""
        analyzer = AsyncMock()
        info_mock = MagicMock()
        info_mock.name = "google"
        type(analyzer).info = PropertyMock(return_value=info_mock)
        analyzer.analyze_images.return_value = {
            "description": "test",
            "people": [],
            "objects": [],
            "text_on_screen": [],
            "actions": [],
            "setting": None,
        }

        op = VisualTranscriptOperation(analyzer)
        result = await op.execute(
            scene_id=0, keyframes=sample_keyframes, scene=sample_scene
        )

        assert result.model_used == "google"

    @pytest.mark.asyncio
    async def test_execute_no_info_attribute(self, sample_scene, sample_keyframes):
        """Graceful handling when provider has no info attribute."""
        analyzer = AsyncMock(spec=[])
        analyzer.analyze_images = AsyncMock(
            return_value={
                "description": "test",
                "people": [],
                "objects": [],
                "text_on_screen": [],
                "actions": [],
                "setting": None,
            }
        )

        op = VisualTranscriptOperation(analyzer)
        result = await op.execute(
            scene_id=0, keyframes=sample_keyframes, scene=sample_scene
        )

        assert result.model_used is None


# ---------------------------------------------------------------------------
# VisualDescription Tests
# ---------------------------------------------------------------------------


class TestVisualDescription:
    """Tests for the VisualDescription dataclass."""

    def test_to_dict(self):
        desc = VisualDescription(
            scene_id=0,
            description="Test",
            people=["person"],
            objects=["laptop"],
            text_on_screen=["hello"],
            actions=["typing"],
            setting="office",
            keyframe_count=3,
            model_used="anthropic",
        )
        d = desc.to_dict()
        assert d["scene_id"] == 0
        assert d["description"] == "Test"
        assert d["model_used"] == "anthropic"

    def test_from_dict(self):
        data = {
            "scene_id": 1,
            "description": "A scene",
            "people": ["man"],
            "objects": [],
            "text_on_screen": [],
            "actions": ["walking"],
            "setting": "outdoors",
            "keyframe_count": 2,
            "model_used": "google",
        }
        desc = VisualDescription.from_dict(data)
        assert desc.scene_id == 1
        assert desc.description == "A scene"
        assert desc.model_used == "google"

    def test_from_dict_defaults(self):
        desc = VisualDescription.from_dict({})
        assert desc.scene_id == 0
        assert desc.description == ""
        assert desc.people == []
        assert desc.setting is None

    def test_roundtrip(self):
        original = VisualDescription(
            scene_id=5,
            description="Roundtrip test",
            people=["a", "b"],
            objects=["x"],
            text_on_screen=["y"],
            actions=["z"],
            setting="lab",
            keyframe_count=1,
            model_used="test",
        )
        restored = VisualDescription.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()


# ---------------------------------------------------------------------------
# Skip Logic Tests
# ---------------------------------------------------------------------------


class TestShouldSkipScene:
    """Tests for _should_skip_scene."""

    def test_skip_high_density_short_scene(self):
        """High word density + short scene = skip."""
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=10.0,
            transcript_text="word " * 30,  # 3 words/sec
        )
        assert _should_skip_scene(scene) is True

    def test_no_skip_low_density(self):
        """Low word density = don't skip."""
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=10.0,
            transcript_text="few words",
        )
        assert _should_skip_scene(scene) is False

    def test_no_skip_long_scene(self):
        """Long scene (>60s) with high density = don't skip."""
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=120.0,
            transcript_text="word " * 300,  # 2.5 words/sec but long
        )
        assert _should_skip_scene(scene) is False

    def test_no_skip_empty_transcript(self):
        """No transcript = don't skip."""
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=10.0,
        )
        assert _should_skip_scene(scene) is False

    def test_no_skip_zero_duration(self):
        """Zero duration scene = don't skip."""
        scene = SceneBoundary(
            scene_id=0,
            start_time=5.0,
            end_time=5.0,
        )
        assert _should_skip_scene(scene) is False

    def test_uses_transcript_segment_fallback(self):
        """Falls back to transcript_segment if transcript_text is empty."""
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=10.0,
            transcript_segment="word " * 30,
        )
        assert _should_skip_scene(scene) is True


# ---------------------------------------------------------------------------
# Default Vision Analyzer Tests
# ---------------------------------------------------------------------------


class TestGetDefaultVisionAnalyzer:
    """Tests for _get_default_vision_analyzer."""

    def _make_vision_mock(self):
        """Create a mock that satisfies the VisionAnalyzer runtime_checkable protocol."""
        mock = MagicMock()
        mock.is_available.return_value = True
        # Add analyze_images so isinstance(mock, VisionAnalyzer) is True
        mock.analyze_images = AsyncMock()
        return mock

    def test_returns_provider_from_router(self):
        """Should return VisionAnalyzer from ProviderRouter."""
        mock_provider = self._make_vision_mock()

        mock_router = MagicMock()
        mock_router.get_vision_analyzer_for_structured_output.return_value = mock_provider

        with patch(
            "claudetube.providers.router.ProviderRouter",
            return_value=mock_router,
        ):
            result = _get_default_vision_analyzer()
            assert result is mock_provider

    def test_raises_when_no_provider(self):
        """Should raise RuntimeError when router raises NoProviderError."""
        from claudetube.providers.capabilities import Capability
        from claudetube.providers.router import NoProviderError

        mock_router = MagicMock()
        mock_router.get_vision_analyzer_for_structured_output.side_effect = NoProviderError(
            Capability.VISION
        )

        with (
            patch(
                "claudetube.providers.router.ProviderRouter",
                return_value=mock_router,
            ),
            pytest.raises(RuntimeError, match="capability VISION"),
        ):
            _get_default_vision_analyzer()

    def test_raises_on_router_exception(self):
        """Should raise RuntimeError on unexpected router failures."""
        mock_router = MagicMock()
        mock_router.get_vision_analyzer_for_structured_output.side_effect = Exception(
            "config error"
        )

        with (
            patch(
                "claudetube.providers.router.ProviderRouter",
                return_value=mock_router,
            ),
            pytest.raises(RuntimeError, match="No vision provider with structured output"),
        ):
            _get_default_vision_analyzer()


# ---------------------------------------------------------------------------
# Integration: generate_visual_transcript with VisualTranscriptOperation
# ---------------------------------------------------------------------------


class TestGenerateVisualTranscriptIntegration:
    """Tests that generate_visual_transcript uses VisualTranscriptOperation."""

    def test_cached_result_returned_without_analyzer(self, tmp_path):
        """Cached visual.json should be returned without needing a provider."""
        video_id = "test123"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir()

        # Create scenes data
        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir()
        scenes_json = scenes_dir / "scenes.json"
        scenes_json.write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "method": "transcript",
                    "scene_count": 1,
                    "scenes": [
                        {
                            "scene_id": 0,
                            "start_time": 0.0,
                            "end_time": 10.0,
                        }
                    ],
                }
            )
        )

        # Create cached visual.json
        scene_dir = scenes_dir / "scene_000"
        scene_dir.mkdir()
        visual_json = scene_dir / "visual.json"
        visual_json.write_text(
            json.dumps(
                {
                    "scene_id": 0,
                    "description": "Cached description",
                    "people": [],
                    "objects": [],
                    "text_on_screen": [],
                    "actions": [],
                    "setting": "office",
                    "keyframe_count": 3,
                    "model_used": "anthropic",
                }
            )
        )

        result = generate_visual_transcript(
            video_id=video_id,
            output_base=tmp_path,
        )

        assert result["generated"] == 1
        assert result["results"][0]["description"] == "Cached description"
        assert result["errors"] == []

    def test_skipped_scene_without_analyzer(self, tmp_path):
        """Skippable scene should not need a vision provider."""
        video_id = "test456"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir()

        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir()
        scenes_json = scenes_dir / "scenes.json"
        scenes_json.write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "method": "transcript",
                    "scene_count": 1,
                    "scenes": [
                        {
                            "scene_id": 0,
                            "start_time": 0.0,
                            "end_time": 10.0,
                            "transcript_text": "word " * 30,  # High density -> skip
                        }
                    ],
                }
            )
        )

        result = generate_visual_transcript(
            video_id=video_id,
            output_base=tmp_path,
        )

        assert result["skipped"] == 1
        assert result["generated"] == 0
        assert 0 in result["skipped_scene_ids"]

    def test_not_cached_error(self, tmp_path):
        """Missing video cache returns error."""
        result = generate_visual_transcript(
            video_id="nonexistent",
            output_base=tmp_path,
        )
        assert "error" in result

    def test_no_scenes_error(self, tmp_path):
        """Missing scenes data returns error."""
        video_id = "noscenesid"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir()

        result = generate_visual_transcript(
            video_id=video_id,
            output_base=tmp_path,
        )
        assert "error" in result

    def test_accepts_vision_analyzer_parameter(self, tmp_path):
        """generate_visual_transcript accepts vision_analyzer kwarg."""
        video_id = "test789"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir()

        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir()
        scenes_json = scenes_dir / "scenes.json"
        scenes_json.write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "method": "transcript",
                    "scene_count": 1,
                    "scenes": [
                        {
                            "scene_id": 0,
                            "start_time": 0.0,
                            "end_time": 10.0,
                        }
                    ],
                }
            )
        )

        # Create state.json (needed for keyframe extraction)
        state_file = cache_dir / "state.json"
        state_file.write_text(json.dumps({"url": None}))

        # Mock vision_analyzer - won't actually be called because no keyframes available
        mock_analyzer = AsyncMock()

        result = generate_visual_transcript(
            video_id=video_id,
            output_base=tmp_path,
            vision_analyzer=mock_analyzer,
        )

        # No keyframes available, so error expected for the scene
        assert result["errors"][0]["error"] == "No keyframes available"


# ---------------------------------------------------------------------------
# VISUAL_PROMPT constant
# ---------------------------------------------------------------------------


class TestVisualPrompt:
    """Tests for the VISUAL_PROMPT constant."""

    def test_prompt_has_context_placeholder(self):
        assert "{context}" in VISUAL_PROMPT

    def test_prompt_mentions_key_aspects(self):
        assert "visual actions" in VISUAL_PROMPT
        assert "people" in VISUAL_PROMPT
        assert "objects" in VISUAL_PROMPT
        assert "text on screen" in VISUAL_PROMPT
        assert "setting" in VISUAL_PROMPT

    def test_prompt_format_no_context(self):
        result = VISUAL_PROMPT.format(context="")
        assert "{context}" not in result
        assert "Transcript context" not in result

    def test_prompt_format_with_context(self):
        result = VISUAL_PROMPT.format(context="\n\nTranscript context: hello")
        assert "Transcript context: hello" in result
