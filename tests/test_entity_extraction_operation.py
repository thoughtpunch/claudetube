"""Tests for EntityExtractionOperation.

Verifies:
1. Operation instantiation with VisionAnalyzer and Reasoner
2. execute() with structured dict response for visual entities
3. execute() with structured dict response for semantic concepts
4. Prompt building with and without transcript context
5. Model name extraction from provider info
6. _should_skip_entity_extraction logic
7. _get_default_providers fallback
8. extract_entities_for_video backward compatibility
9. EntityExtractionSceneResult.to_visual_json() derived output
10. get_extracted_entities cache reading
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from claudetube.cache.scenes import SceneBoundary
from claudetube.operations.entity_extraction import (
    SEMANTIC_CONCEPT_PROMPT,
    VIDEO_ENTITY_PROMPT,
    VISUAL_ENTITY_PROMPT,
    EntityExtractionOperation,
    EntityExtractionSceneResult,
    _get_default_providers,
    _should_skip_entity_extraction,
    extract_entities_for_video,
    get_extracted_entities,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_video_analyzer():
    """Create a mock VideoAnalyzer with structured output support."""
    analyzer = AsyncMock()
    info_mock = MagicMock()
    info_mock.name = "google"
    type(analyzer).info = PropertyMock(return_value=info_mock)
    analyzer.is_available.return_value = True
    return analyzer


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
def mock_reasoner():
    """Create a mock Reasoner with structured output support."""
    reasoner = AsyncMock()
    info_mock = MagicMock()
    info_mock.name = "anthropic"
    type(reasoner).info = PropertyMock(return_value=info_mock)
    reasoner.is_available.return_value = True
    return reasoner


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


@pytest.fixture
def sample_video_path(tmp_path):
    """Create a sample video file."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"\x00" * 100)  # Dummy video file
    return video


# ---------------------------------------------------------------------------
# EntityExtractionOperation Init Tests
# ---------------------------------------------------------------------------


class TestEntityExtractionOperationInit:
    """Tests for operation instantiation."""

    def test_accepts_video_analyzer(self, mock_video_analyzer):
        op = EntityExtractionOperation(video_analyzer=mock_video_analyzer)
        assert op.video_analyzer is mock_video_analyzer
        assert op.vision is None
        assert op.reasoner is None

    def test_accepts_vision_analyzer(self, mock_vision_analyzer):
        op = EntityExtractionOperation(vision_analyzer=mock_vision_analyzer)
        assert op.vision is mock_vision_analyzer
        assert op.video_analyzer is None
        assert op.reasoner is None

    def test_accepts_reasoner(self, mock_reasoner):
        op = EntityExtractionOperation(reasoner=mock_reasoner)
        assert op.reasoner is mock_reasoner
        assert op.video_analyzer is None
        assert op.vision is None

    def test_accepts_all(
        self, mock_video_analyzer, mock_vision_analyzer, mock_reasoner
    ):
        op = EntityExtractionOperation(
            video_analyzer=mock_video_analyzer,
            vision_analyzer=mock_vision_analyzer,
            reasoner=mock_reasoner,
        )
        assert op.video_analyzer is mock_video_analyzer
        assert op.vision is mock_vision_analyzer
        assert op.reasoner is mock_reasoner


# ---------------------------------------------------------------------------
# Prompt Building Tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Tests for prompt building."""

    def test_visual_prompt_without_transcript(
        self, mock_vision_analyzer, sample_scene_no_transcript
    ):
        op = EntityExtractionOperation(vision_analyzer=mock_vision_analyzer)
        prompt = op._build_visual_prompt(sample_scene_no_transcript)
        assert "extract all entities" in prompt
        assert "Transcript context" not in prompt
        assert "30.0" in prompt  # start_time

    def test_visual_prompt_with_transcript(self, mock_vision_analyzer, sample_scene):
        op = EntityExtractionOperation(vision_analyzer=mock_vision_analyzer)
        prompt = op._build_visual_prompt(sample_scene)
        assert "extract all entities" in prompt
        assert "Transcript context: Hello and welcome" in prompt
        assert "10.0" in prompt  # start_time

    def test_concept_prompt_with_transcript(self, mock_reasoner, sample_scene):
        op = EntityExtractionOperation(reasoner=mock_reasoner)
        prompt = op._build_concept_prompt(sample_scene)
        assert "key concepts" in prompt
        assert "Hello and welcome" in prompt
        assert "10.0" in prompt  # start_time

    def test_concept_prompt_empty_transcript(
        self, mock_reasoner, sample_scene_no_transcript
    ):
        op = EntityExtractionOperation(reasoner=mock_reasoner)
        prompt = op._build_concept_prompt(sample_scene_no_transcript)
        assert "key concepts" in prompt

    def test_visual_prompt_truncates_long_transcript(self, mock_vision_analyzer):
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=10.0,
            transcript_text="word " * 200,  # 1000 chars
        )
        op = EntityExtractionOperation(vision_analyzer=mock_vision_analyzer)
        prompt = op._build_visual_prompt(scene)
        context_part = prompt.split("Transcript context: ")[1]
        assert len(context_part) <= 500


# ---------------------------------------------------------------------------
# Execute Tests
# ---------------------------------------------------------------------------


class TestExecute:
    """Tests for the execute method."""

    @pytest.mark.asyncio
    async def test_execute_with_vision_only(
        self, mock_vision_analyzer, sample_scene, sample_keyframes
    ):
        """Vision-only extraction returns visual entities."""
        mock_vision_analyzer.analyze_images.return_value = {
            "objects": [
                {
                    "name": "laptop",
                    "category": "object",
                    "first_seen_sec": 10.0,
                    "confidence": 0.9,
                }
            ],
            "people": [
                {
                    "name": "presenter",
                    "category": "person",
                    "first_seen_sec": 10.0,
                    "confidence": 0.95,
                }
            ],
            "text_on_screen": [
                {
                    "name": "def main():",
                    "category": "code",
                    "first_seen_sec": 12.0,
                    "confidence": 0.8,
                }
            ],
            "concepts": [],
            "code_snippets": [],
        }

        op = EntityExtractionOperation(vision_analyzer=mock_vision_analyzer)
        result = await op.execute(
            scene_id=0,
            keyframes=sample_keyframes,
            scene=sample_scene,
        )

        assert isinstance(result, EntityExtractionSceneResult)
        assert result.scene_id == 0
        assert len(result.objects) == 1
        assert result.objects[0]["name"] == "laptop"
        assert len(result.people) == 1
        assert result.people[0]["name"] == "presenter"
        assert len(result.text_on_screen) == 1
        assert result.model_used == "anthropic"

    @pytest.mark.asyncio
    async def test_execute_with_reasoner_only(
        self, mock_reasoner, sample_scene, sample_keyframes
    ):
        """Reasoner-only extraction returns semantic concepts."""
        mock_reasoner.reason.return_value = {
            "concepts": [
                {
                    "term": "Python",
                    "definition": "A programming language",
                    "importance": "primary",
                    "first_mention_sec": 10.0,
                    "related_terms": ["programming", "scripting"],
                }
            ]
        }

        op = EntityExtractionOperation(reasoner=mock_reasoner)
        result = await op.execute(
            scene_id=0,
            keyframes=sample_keyframes,
            scene=sample_scene,
        )

        assert isinstance(result, EntityExtractionSceneResult)
        assert len(result.concepts) == 1
        assert result.concepts[0]["term"] == "Python"
        assert result.objects == []
        assert result.model_used == "anthropic"

    @pytest.mark.asyncio
    async def test_execute_with_both_providers(
        self, mock_vision_analyzer, mock_reasoner, sample_scene, sample_keyframes
    ):
        """Both providers run concurrently."""
        mock_vision_analyzer.analyze_images.return_value = {
            "objects": [
                {"name": "laptop", "category": "object", "first_seen_sec": 10.0}
            ],
            "people": [],
            "text_on_screen": [],
            "concepts": [],
            "code_snippets": [],
        }
        mock_reasoner.reason.return_value = {
            "concepts": [
                {
                    "term": "Python",
                    "definition": "A language",
                    "importance": "primary",
                    "first_mention_sec": 10.0,
                }
            ]
        }

        op = EntityExtractionOperation(
            vision_analyzer=mock_vision_analyzer,
            reasoner=mock_reasoner,
        )
        result = await op.execute(
            scene_id=0,
            keyframes=sample_keyframes,
            scene=sample_scene,
        )

        assert len(result.objects) == 1
        assert len(result.concepts) == 1
        # Vision analyzer called
        mock_vision_analyzer.analyze_images.assert_called_once()
        # Reasoner called
        mock_reasoner.reason.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_passes_schema_to_vision(
        self, mock_vision_analyzer, sample_scene, sample_keyframes
    ):
        """Pydantic schema is passed to analyze_images."""
        mock_vision_analyzer.analyze_images.return_value = {
            "objects": [],
            "people": [],
            "text_on_screen": [],
            "concepts": [],
            "code_snippets": [],
        }

        op = EntityExtractionOperation(vision_analyzer=mock_vision_analyzer)
        await op.execute(scene_id=0, keyframes=sample_keyframes, scene=sample_scene)

        call_args = mock_vision_analyzer.analyze_images.call_args
        assert call_args is not None
        schema = call_args[1].get("schema") or call_args[0][2]
        assert schema is not None
        assert hasattr(schema, "model_json_schema")

    @pytest.mark.asyncio
    async def test_execute_handles_string_response(
        self, mock_vision_analyzer, sample_scene, sample_keyframes
    ):
        """Handles provider returning JSON string instead of dict."""
        mock_vision_analyzer.analyze_images.return_value = json.dumps(
            {
                "objects": [
                    {"name": "monitor", "category": "object", "first_seen_sec": 10.0}
                ],
                "people": [],
                "text_on_screen": [],
                "concepts": [],
                "code_snippets": [],
            }
        )

        op = EntityExtractionOperation(vision_analyzer=mock_vision_analyzer)
        result = await op.execute(
            scene_id=0,
            keyframes=sample_keyframes,
            scene=sample_scene,
        )

        assert len(result.objects) == 1
        assert result.objects[0]["name"] == "monitor"

    @pytest.mark.asyncio
    async def test_execute_no_providers(self, sample_scene, sample_keyframes):
        """No providers returns empty result."""
        op = EntityExtractionOperation()
        result = await op.execute(
            scene_id=0,
            keyframes=sample_keyframes,
            scene=sample_scene,
        )

        assert result.objects == []
        assert result.people == []
        assert result.concepts == []

    @pytest.mark.asyncio
    async def test_execute_no_keyframes_with_vision(
        self, mock_vision_analyzer, sample_scene
    ):
        """Vision analyzer is not called when no keyframes available."""
        op = EntityExtractionOperation(vision_analyzer=mock_vision_analyzer)
        result = await op.execute(
            scene_id=0,
            keyframes=[],
            scene=sample_scene,
        )

        mock_vision_analyzer.analyze_images.assert_not_called()
        assert result.objects == []

    @pytest.mark.asyncio
    async def test_execute_model_name_from_vision_info(
        self, sample_scene, sample_keyframes
    ):
        """Model name comes from vision provider info."""
        analyzer = AsyncMock()
        info_mock = MagicMock()
        info_mock.name = "google"
        type(analyzer).info = PropertyMock(return_value=info_mock)
        analyzer.analyze_images.return_value = {
            "objects": [],
            "people": [],
            "text_on_screen": [],
            "concepts": [],
            "code_snippets": [],
        }

        op = EntityExtractionOperation(vision_analyzer=analyzer)
        result = await op.execute(
            scene_id=0, keyframes=sample_keyframes, scene=sample_scene
        )

        assert result.model_used == "google"

    @pytest.mark.asyncio
    async def test_execute_model_name_from_reasoner_info(self, sample_scene):
        """Model name comes from reasoner info when no vision provider."""
        reasoner = AsyncMock()
        info_mock = MagicMock()
        info_mock.name = "openai"
        type(reasoner).info = PropertyMock(return_value=info_mock)
        reasoner.reason.return_value = {"concepts": []}

        op = EntityExtractionOperation(reasoner=reasoner)
        result = await op.execute(scene_id=0, keyframes=[], scene=sample_scene)

        assert result.model_used == "openai"

    @pytest.mark.asyncio
    async def test_execute_vision_error_handled(
        self, mock_vision_analyzer, mock_reasoner, sample_scene, sample_keyframes
    ):
        """Vision failure doesn't prevent concept extraction."""
        mock_vision_analyzer.analyze_images.side_effect = RuntimeError(
            "Vision API error"
        )
        mock_reasoner.reason.return_value = {
            "concepts": [
                {
                    "term": "Test",
                    "definition": "A test",
                    "importance": "primary",
                    "first_mention_sec": 0.0,
                }
            ]
        }

        op = EntityExtractionOperation(
            vision_analyzer=mock_vision_analyzer,
            reasoner=mock_reasoner,
        )
        result = await op.execute(
            scene_id=0,
            keyframes=sample_keyframes,
            scene=sample_scene,
        )

        # Visual entities empty due to error, but concepts still extracted
        assert result.objects == []
        assert len(result.concepts) == 1


# ---------------------------------------------------------------------------
# VideoAnalyzer Execute Tests
# ---------------------------------------------------------------------------


class TestExecuteWithVideoAnalyzer:
    """Tests for VideoAnalyzer support in execute."""

    @pytest.mark.asyncio
    async def test_video_analyzer_used_when_available(
        self,
        mock_video_analyzer,
        mock_vision_analyzer,
        sample_scene,
        sample_keyframes,
        sample_video_path,
    ):
        """VideoAnalyzer is preferred over VisionAnalyzer."""
        mock_video_analyzer.analyze_video.return_value = {
            "objects": [
                {
                    "name": "laptop",
                    "category": "object",
                    "first_seen_sec": 10.0,
                    "confidence": 0.9,
                }
            ],
            "people": [],
            "text_on_screen": [],
            "concepts": [],
            "code_snippets": [],
        }

        op = EntityExtractionOperation(
            video_analyzer=mock_video_analyzer,
            vision_analyzer=mock_vision_analyzer,
        )
        result = await op.execute(
            scene_id=0,
            keyframes=sample_keyframes,
            scene=sample_scene,
            video_path=sample_video_path,
        )

        assert len(result.objects) == 1
        assert result.objects[0]["name"] == "laptop"
        assert result.model_used == "google"
        # VideoAnalyzer called, VisionAnalyzer NOT called
        mock_video_analyzer.analyze_video.assert_called_once()
        mock_vision_analyzer.analyze_images.assert_not_called()

    @pytest.mark.asyncio
    async def test_video_analyzer_passes_time_range(
        self,
        mock_video_analyzer,
        sample_scene,
        sample_video_path,
    ):
        """VideoAnalyzer receives start_time and end_time."""
        mock_video_analyzer.analyze_video.return_value = {
            "objects": [],
            "people": [],
            "text_on_screen": [],
            "concepts": [],
            "code_snippets": [],
        }

        op = EntityExtractionOperation(video_analyzer=mock_video_analyzer)
        await op.execute(
            scene_id=0,
            keyframes=[],
            scene=sample_scene,
            video_path=sample_video_path,
        )

        call_kwargs = mock_video_analyzer.analyze_video.call_args[1]
        assert call_kwargs["start_time"] == 10.0
        assert call_kwargs["end_time"] == 25.0

    @pytest.mark.asyncio
    async def test_falls_back_to_vision_on_video_error(
        self,
        mock_video_analyzer,
        mock_vision_analyzer,
        sample_scene,
        sample_keyframes,
        sample_video_path,
    ):
        """When VideoAnalyzer fails, falls back to VisionAnalyzer."""
        mock_video_analyzer.analyze_video.side_effect = RuntimeError("Gemini API error")
        mock_vision_analyzer.analyze_images.return_value = {
            "objects": [
                {"name": "monitor", "category": "object", "first_seen_sec": 10.0}
            ],
            "people": [],
            "text_on_screen": [],
            "concepts": [],
            "code_snippets": [],
        }

        op = EntityExtractionOperation(
            video_analyzer=mock_video_analyzer,
            vision_analyzer=mock_vision_analyzer,
        )
        result = await op.execute(
            scene_id=0,
            keyframes=sample_keyframes,
            scene=sample_scene,
            video_path=sample_video_path,
        )

        assert len(result.objects) == 1
        assert result.objects[0]["name"] == "monitor"
        mock_video_analyzer.analyze_video.assert_called_once()
        mock_vision_analyzer.analyze_images.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_video_path_uses_vision(
        self,
        mock_video_analyzer,
        mock_vision_analyzer,
        sample_scene,
        sample_keyframes,
    ):
        """Without video_path, VisionAnalyzer is used even when VideoAnalyzer is set."""
        mock_vision_analyzer.analyze_images.return_value = {
            "objects": [
                {"name": "keyboard", "category": "object", "first_seen_sec": 10.0}
            ],
            "people": [],
            "text_on_screen": [],
            "concepts": [],
            "code_snippets": [],
        }

        op = EntityExtractionOperation(
            video_analyzer=mock_video_analyzer,
            vision_analyzer=mock_vision_analyzer,
        )
        result = await op.execute(
            scene_id=0,
            keyframes=sample_keyframes,
            scene=sample_scene,
            video_path=None,
        )

        assert len(result.objects) == 1
        mock_video_analyzer.analyze_video.assert_not_called()
        mock_vision_analyzer.analyze_images.assert_called_once()

    @pytest.mark.asyncio
    async def test_video_analyzer_with_reasoner(
        self,
        mock_video_analyzer,
        mock_reasoner,
        sample_scene,
        sample_video_path,
    ):
        """VideoAnalyzer and Reasoner run concurrently."""
        mock_video_analyzer.analyze_video.return_value = {
            "objects": [
                {"name": "laptop", "category": "object", "first_seen_sec": 10.0}
            ],
            "people": [],
            "text_on_screen": [],
            "concepts": [],
            "code_snippets": [],
        }
        mock_reasoner.reason.return_value = {
            "concepts": [
                {
                    "term": "Python",
                    "definition": "A language",
                    "importance": "primary",
                    "first_mention_sec": 10.0,
                }
            ]
        }

        op = EntityExtractionOperation(
            video_analyzer=mock_video_analyzer,
            reasoner=mock_reasoner,
        )
        result = await op.execute(
            scene_id=0,
            keyframes=[],
            scene=sample_scene,
            video_path=sample_video_path,
        )

        assert len(result.objects) == 1
        assert len(result.concepts) == 1
        mock_video_analyzer.analyze_video.assert_called_once()
        mock_reasoner.reason.assert_called_once()

    @pytest.mark.asyncio
    async def test_video_analyzer_handles_string_response(
        self,
        mock_video_analyzer,
        sample_scene,
        sample_video_path,
    ):
        """VideoAnalyzer returning JSON string is parsed correctly."""
        mock_video_analyzer.analyze_video.return_value = json.dumps(
            {
                "objects": [
                    {"name": "mouse", "category": "object", "first_seen_sec": 10.0}
                ],
                "people": [],
                "text_on_screen": [],
                "concepts": [],
                "code_snippets": [],
            }
        )

        op = EntityExtractionOperation(video_analyzer=mock_video_analyzer)
        result = await op.execute(
            scene_id=0,
            keyframes=[],
            scene=sample_scene,
            video_path=sample_video_path,
        )

        assert len(result.objects) == 1
        assert result.objects[0]["name"] == "mouse"


# ---------------------------------------------------------------------------
# Video Prompt Tests
# ---------------------------------------------------------------------------


class TestBuildVideoPrompt:
    """Tests for video prompt building."""

    def test_video_prompt_has_time_range(self, mock_video_analyzer, sample_scene):
        op = EntityExtractionOperation(video_analyzer=mock_video_analyzer)
        prompt = op._build_video_prompt(sample_scene)
        assert "10.0" in prompt  # start_time
        assert "25.0" in prompt  # end_time

    def test_video_prompt_with_transcript(self, mock_video_analyzer, sample_scene):
        op = EntityExtractionOperation(video_analyzer=mock_video_analyzer)
        prompt = op._build_video_prompt(sample_scene)
        assert "Transcript context: Hello and welcome" in prompt

    def test_video_prompt_without_transcript(
        self, mock_video_analyzer, sample_scene_no_transcript
    ):
        op = EntityExtractionOperation(video_analyzer=mock_video_analyzer)
        prompt = op._build_video_prompt(sample_scene_no_transcript)
        assert "Transcript context" not in prompt

    def test_video_prompt_constant_has_placeholders(self):
        assert "{start_time" in VIDEO_ENTITY_PROMPT
        assert "{end_time" in VIDEO_ENTITY_PROMPT
        assert "{context}" in VIDEO_ENTITY_PROMPT


# ---------------------------------------------------------------------------
# EntityExtractionSceneResult Tests
# ---------------------------------------------------------------------------


class TestEntityExtractionSceneResult:
    """Tests for the EntityExtractionSceneResult dataclass."""

    def test_to_dict(self):
        result = EntityExtractionSceneResult(
            scene_id=0,
            objects=[{"name": "laptop", "category": "object"}],
            people=[{"name": "presenter", "category": "person"}],
            text_on_screen=[],
            concepts=[{"term": "Python", "importance": "primary"}],
            code_snippets=[],
            model_used="anthropic",
        )
        d = result.to_dict()
        assert d["scene_id"] == 0
        assert len(d["objects"]) == 1
        assert d["model_used"] == "anthropic"

    def test_from_dict(self):
        data = {
            "scene_id": 1,
            "objects": [{"name": "monitor"}],
            "people": [],
            "text_on_screen": [],
            "concepts": [{"term": "ML"}],
            "code_snippets": [],
            "model_used": "google",
        }
        result = EntityExtractionSceneResult.from_dict(data)
        assert result.scene_id == 1
        assert len(result.objects) == 1
        assert result.model_used == "google"

    def test_from_dict_defaults(self):
        result = EntityExtractionSceneResult.from_dict({})
        assert result.scene_id == 0
        assert result.objects == []
        assert result.model_used is None

    def test_roundtrip(self):
        original = EntityExtractionSceneResult(
            scene_id=5,
            objects=[{"name": "book"}],
            people=[{"name": "author"}],
            text_on_screen=[{"name": "Chapter 1"}],
            concepts=[{"term": "AI"}],
            code_snippets=[{"language": "python", "code": "print('hi')"}],
            model_used="test",
        )
        restored = EntityExtractionSceneResult.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_to_visual_json(self):
        result = EntityExtractionSceneResult(
            scene_id=0,
            objects=[{"name": "laptop"}, {"name": "monitor"}],
            people=[{"name": "presenter"}],
            text_on_screen=[{"name": "def main():"}],
            concepts=[],
            model_used="anthropic",
        )
        visual = result.to_visual_json()
        assert visual["scene_id"] == 0
        assert "presenter" in visual["people"]
        assert "laptop" in visual["objects"]
        assert "monitor" in visual["objects"]
        assert "def main():" in visual["text_on_screen"]
        assert visual["model_used"] == "anthropic"
        assert "People: presenter" in visual["description"]

    def test_to_visual_json_empty(self):
        result = EntityExtractionSceneResult(scene_id=0)
        visual = result.to_visual_json()
        assert visual["description"] == ""
        assert visual["people"] == []
        assert visual["objects"] == []


# ---------------------------------------------------------------------------
# Skip Logic Tests
# ---------------------------------------------------------------------------


class TestShouldSkipEntityExtraction:
    """Tests for _should_skip_entity_extraction."""

    def test_skip_very_short_scene_no_transcript(self):
        """Very short scene with no transcript = skip."""
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=1.0,
        )
        assert _should_skip_entity_extraction(scene) is True

    def test_no_skip_short_scene_with_transcript(self):
        """Short scene WITH transcript = don't skip."""
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=1.0,
            transcript_text="Some content here",
        )
        assert _should_skip_entity_extraction(scene) is False

    def test_no_skip_normal_scene(self):
        """Normal duration scene = don't skip."""
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=10.0,
        )
        assert _should_skip_entity_extraction(scene) is False

    def test_no_skip_scene_with_transcript(self):
        """Scene with transcript = don't skip."""
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=30.0,
            transcript_text="Python programming is great",
        )
        assert _should_skip_entity_extraction(scene) is False

    def test_skip_boundary_at_2_seconds(self):
        """Exactly 2s scene with no transcript = skip."""
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=1.99,
        )
        assert _should_skip_entity_extraction(scene) is True

    def test_no_skip_at_2_seconds(self):
        """Exactly 2.0s scene = don't skip."""
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=2.0,
        )
        assert _should_skip_entity_extraction(scene) is False


# ---------------------------------------------------------------------------
# Default Provider Tests
# ---------------------------------------------------------------------------


class TestGetDefaultProviders:
    """Tests for _get_default_providers."""

    def _make_provider_mock(
        self, available=True, has_video=False, has_vision=True, has_reasoner=True
    ):
        """Create a mock provider with specified capabilities.

        Uses a real class (not MagicMock) so that isinstance checks with
        @runtime_checkable protocols work correctly across Python 3.10-3.12.
        """
        instance = type("MockProvider", (), {})()
        instance.is_available = MagicMock(return_value=available)
        instance.info = {"name": "MockProvider"}
        if has_video:
            instance.analyze_video = AsyncMock()
        if has_vision:
            instance.analyze_images = AsyncMock()
        if has_reasoner:
            instance.reason = AsyncMock()
        return instance

    def test_returns_all_from_single_provider(self):
        """Single provider with all capabilities satisfies all roles."""
        mock_video = self._make_provider_mock(has_video=True)
        mock_vision = self._make_provider_mock(has_vision=True)
        mock_reasoner = self._make_provider_mock(has_reasoner=True)

        mock_router = MagicMock()
        mock_router.get_video_analyzer.return_value = mock_video
        mock_router.get_vision_analyzer_for_structured_output.return_value = mock_vision
        mock_router.get_reasoner_for_structured_output.return_value = mock_reasoner

        with patch(
            "claudetube.providers.router.ProviderRouter",
            return_value=mock_router,
        ):
            video, vision, reasoner = _get_default_providers()
            assert video is mock_video
            assert vision is mock_vision
            assert reasoner is mock_reasoner

    def test_returns_vision_and_reasoner_without_video(self):
        """Provider without VideoAnalyzer returns None for video."""
        mock_vision = self._make_provider_mock(has_vision=True)
        mock_reasoner = self._make_provider_mock(has_reasoner=True)

        mock_router = MagicMock()
        # get_video_analyzer returns None when no video provider available
        mock_router.get_video_analyzer.return_value = None
        mock_router.get_vision_analyzer_for_structured_output.return_value = mock_vision
        mock_router.get_reasoner_for_structured_output.return_value = mock_reasoner

        with patch(
            "claudetube.providers.router.ProviderRouter",
            return_value=mock_router,
        ):
            video, vision, reasoner = _get_default_providers()
            assert video is None
            assert vision is mock_vision
            assert reasoner is mock_reasoner

    def test_returns_none_when_unavailable(self):
        """Unavailable providers return None for all."""
        from claudetube.providers.capabilities import Capability
        from claudetube.providers.router import NoProviderError

        mock_router = MagicMock()
        # get_video_analyzer returns None (doesn't raise)
        mock_router.get_video_analyzer.return_value = None
        mock_router.get_vision_analyzer_for_structured_output.side_effect = NoProviderError(
            Capability.VISION
        )
        mock_router.get_reasoner_for_structured_output.side_effect = NoProviderError(
            Capability.REASON
        )

        with patch(
            "claudetube.providers.router.ProviderRouter",
            return_value=mock_router,
        ):
            video, vision, reasoner = _get_default_providers()
            assert video is None
            assert vision is None
            assert reasoner is None

    def test_handles_import_error(self):
        """Import error returns None for all."""
        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=ImportError("No module"),
        ):
            video, vision, reasoner = _get_default_providers()
            assert video is None
            assert vision is None
            assert reasoner is None


# ---------------------------------------------------------------------------
# Integration: extract_entities_for_video
# ---------------------------------------------------------------------------


class TestExtractEntitiesForVideoIntegration:
    """Tests that extract_entities_for_video uses EntityExtractionOperation."""

    def test_cached_result_returned_without_provider(self, tmp_path):
        """Cached entities.json should be returned without needing a provider."""
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

        # Create cached entities.json
        scene_dir = scenes_dir / "scene_000"
        scene_dir.mkdir()
        entities_json = scene_dir / "entities.json"
        entities_json.write_text(
            json.dumps(
                {
                    "scene_id": 0,
                    "objects": [{"name": "laptop"}],
                    "people": [],
                    "text_on_screen": [],
                    "concepts": [],
                    "code_snippets": [],
                    "model_used": "anthropic",
                }
            )
        )

        result = extract_entities_for_video(
            video_id=video_id,
            output_base=tmp_path,
        )

        assert result["extracted"] == 1
        assert result["results"][0]["objects"][0]["name"] == "laptop"
        assert result["errors"] == []

    def test_skipped_scene_without_provider(self, tmp_path):
        """Very short scene with no transcript should be skipped."""
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
                            "end_time": 1.0,
                            # No transcript_text = skip
                        }
                    ],
                }
            )
        )

        result = extract_entities_for_video(
            video_id=video_id,
            output_base=tmp_path,
        )

        assert result["skipped"] == 1
        assert result["extracted"] == 0
        assert 0 in result["skipped_scene_ids"]

    def test_not_cached_error(self, tmp_path):
        """Missing video cache returns error."""
        result = extract_entities_for_video(
            video_id="nonexistent",
            output_base=tmp_path,
        )
        assert "error" in result

    def test_no_scenes_error(self, tmp_path):
        """Missing scenes data returns error."""
        video_id = "noscenesid"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir()

        result = extract_entities_for_video(
            video_id=video_id,
            output_base=tmp_path,
        )
        assert "error" in result

    def test_generates_visual_json_from_entities(self, tmp_path):
        """When generate_visual=True, visual.json should be derived from entities."""
        video_id = "testvisual"
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
                            "end_time": 30.0,
                            "transcript_text": "Hello world",
                        }
                    ],
                }
            )
        )

        # Create a mock reasoner that returns concepts
        mock_reasoner = AsyncMock()
        info_mock = MagicMock()
        info_mock.name = "test-reasoner"
        type(mock_reasoner).info = PropertyMock(return_value=info_mock)
        mock_reasoner.reason.return_value = {
            "concepts": [
                {
                    "term": "greeting",
                    "definition": "A hello",
                    "importance": "primary",
                    "first_mention_sec": 0.0,
                }
            ]
        }

        result = extract_entities_for_video(
            video_id=video_id,
            output_base=tmp_path,
            reasoner=mock_reasoner,
            generate_visual=True,
        )

        assert result["extracted"] == 1

        # Check that visual.json was generated
        visual_path = scenes_dir / "scene_000" / "visual.json"
        assert visual_path.exists()
        visual_data = json.loads(visual_path.read_text())
        assert visual_data["scene_id"] == 0
        assert visual_data["model_used"] == "test-reasoner"


# ---------------------------------------------------------------------------
# get_extracted_entities Tests
# ---------------------------------------------------------------------------


class TestGetExtractedEntities:
    """Tests for get_extracted_entities cache reader."""

    def test_returns_cached_entities(self, tmp_path):
        video_id = "testcache"
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
                    "scenes": [{"scene_id": 0, "start_time": 0.0, "end_time": 10.0}],
                }
            )
        )

        scene_dir = scenes_dir / "scene_000"
        scene_dir.mkdir()
        entities_json = scene_dir / "entities.json"
        entities_json.write_text(
            json.dumps(
                {
                    "scene_id": 0,
                    "objects": [{"name": "laptop"}],
                    "people": [],
                    "text_on_screen": [],
                    "concepts": [],
                    "code_snippets": [],
                }
            )
        )

        result = get_extracted_entities(video_id, output_base=tmp_path)

        assert result["count"] == 1
        assert result["missing"] == []
        assert result["results"][0]["objects"][0]["name"] == "laptop"

    def test_reports_missing_scenes(self, tmp_path):
        video_id = "testmissing"
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
                    "scene_count": 2,
                    "scenes": [
                        {"scene_id": 0, "start_time": 0.0, "end_time": 10.0},
                        {"scene_id": 1, "start_time": 10.0, "end_time": 20.0},
                    ],
                }
            )
        )

        # Only scene 0 has entities
        scene_dir = scenes_dir / "scene_000"
        scene_dir.mkdir()
        entities_json = scene_dir / "entities.json"
        entities_json.write_text(
            json.dumps(
                {
                    "scene_id": 0,
                    "objects": [],
                    "people": [],
                    "text_on_screen": [],
                    "concepts": [],
                    "code_snippets": [],
                }
            )
        )

        result = get_extracted_entities(video_id, output_base=tmp_path)

        assert result["count"] == 1
        assert result["missing"] == [1]

    def test_error_when_not_cached(self, tmp_path):
        result = get_extracted_entities("nonexistent", output_base=tmp_path)
        assert "error" in result

    def test_error_when_no_scenes(self, tmp_path):
        video_id = "noscenes"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir()

        result = get_extracted_entities(video_id, output_base=tmp_path)
        assert "error" in result


# ---------------------------------------------------------------------------
# Prompt Constants Tests
# ---------------------------------------------------------------------------


class TestPromptConstants:
    """Tests for prompt template constants."""

    def test_visual_prompt_has_placeholders(self):
        assert "{start_time" in VISUAL_ENTITY_PROMPT
        assert "{context}" in VISUAL_ENTITY_PROMPT

    def test_visual_prompt_mentions_categories(self):
        assert "object" in VISUAL_ENTITY_PROMPT
        assert "person" in VISUAL_ENTITY_PROMPT
        assert "text" in VISUAL_ENTITY_PROMPT
        assert "code" in VISUAL_ENTITY_PROMPT

    def test_concept_prompt_has_placeholders(self):
        assert "{start_time" in SEMANTIC_CONCEPT_PROMPT
        assert "{transcript}" in SEMANTIC_CONCEPT_PROMPT

    def test_concept_prompt_mentions_importance_levels(self):
        assert "primary" in SEMANTIC_CONCEPT_PROMPT
        assert "secondary" in SEMANTIC_CONCEPT_PROMPT
        assert "mentioned" in SEMANTIC_CONCEPT_PROMPT
