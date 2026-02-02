"""Tests for PersonTrackingOperation.

Verifies:
1. Operation instantiation with VideoAnalyzer and VisionAnalyzer
2. execute() with VideoAnalyzer (whole-video tracking)
3. execute() with VisionAnalyzer fallback (frame-by-frame)
4. Prompt building with and without transcript context
5. Model name extraction from provider info
6. _get_default_providers fallback chain
7. track_people backward compatibility (cache, visual, AI, face_recognition)
8. get_people_tracking cache reading
9. Data class serialization roundtrips
10. _parse_tracking_result from AI response
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from claudetube.cache.scenes import SceneBoundary
from claudetube.operations.person_tracking import (
    FRAME_PERSON_TRACKING_PROMPT,
    VIDEO_PERSON_TRACKING_PROMPT,
    PeopleTrackingData,
    PersonAppearance,
    PersonTrack,
    PersonTrackingOperation,
    _get_default_providers,
    _track_from_visual_transcripts,
    get_people_json_path,
    get_people_tracking,
    track_people,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_video_analyzer():
    """Create a mock VideoAnalyzer."""
    analyzer = AsyncMock()
    info_mock = MagicMock()
    info_mock.name = "google"
    type(analyzer).info = PropertyMock(return_value=info_mock)
    analyzer.is_available.return_value = True
    return analyzer


@pytest.fixture
def mock_vision_analyzer():
    """Create a mock VisionAnalyzer."""
    analyzer = AsyncMock()
    info_mock = MagicMock()
    info_mock.name = "anthropic"
    type(analyzer).info = PropertyMock(return_value=info_mock)
    analyzer.is_available.return_value = True
    return analyzer


@pytest.fixture
def sample_scenes():
    """Create sample scene boundaries."""
    return [
        SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=15.0,
            title="Introduction",
            transcript_text="Hello and welcome to this video about Python.",
        ),
        SceneBoundary(
            scene_id=1,
            start_time=15.0,
            end_time=30.0,
            title="Main Content",
            transcript_text="Let me show you how to use Python decorators.",
        ),
    ]


@pytest.fixture
def sample_scene_no_transcript():
    """Create a SceneBoundary with no transcript."""
    return SceneBoundary(
        scene_id=2,
        start_time=30.0,
        end_time=45.0,
    )


@pytest.fixture
def sample_keyframes(tmp_path):
    """Create sample keyframe image files in scene directories."""
    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()

    for scene_id in range(2):
        scene_dir = scenes_dir / f"scene_{scene_id:03d}" / "keyframes"
        scene_dir.mkdir(parents=True)
        for i in range(2):
            f = scene_dir / f"kf_{i:02d}.jpg"
            f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    return tmp_path


@pytest.fixture
def video_tracking_response():
    """Sample AI response for video-level tracking."""
    return {
        "people": [
            {
                "person_id": "person_0",
                "description": "man in blue shirt",
                "appearances": [
                    {
                        "scene_id": 0,
                        "timestamp": 7.5,
                        "action": "presenting",
                        "confidence": 0.95,
                    },
                    {
                        "scene_id": 1,
                        "timestamp": 22.5,
                        "action": "coding",
                        "confidence": 0.90,
                    },
                ],
            },
            {
                "person_id": "person_1",
                "description": "woman with glasses",
                "appearances": [
                    {
                        "scene_id": 1,
                        "timestamp": 20.0,
                        "action": "listening",
                        "confidence": 0.85,
                    },
                ],
            },
        ]
    }


@pytest.fixture
def frame_tracking_response():
    """Sample AI response for frame-level tracking."""
    return {
        "people": [
            {
                "person_id": "person_0",
                "description": "man in blue shirt",
                "appearances": [
                    {
                        "scene_id": 0,
                        "timestamp": 7.5,
                        "action": "presenting",
                        "confidence": 0.9,
                    },
                ],
            },
        ]
    }


# ---------------------------------------------------------------------------
# PersonTrackingOperation Init Tests
# ---------------------------------------------------------------------------


class TestPersonTrackingOperationInit:
    """Tests for operation instantiation."""

    def test_accepts_video_analyzer(self, mock_video_analyzer):
        op = PersonTrackingOperation(video_analyzer=mock_video_analyzer)
        assert op.video_analyzer is mock_video_analyzer
        assert op.vision is None

    def test_accepts_vision_analyzer(self, mock_vision_analyzer):
        op = PersonTrackingOperation(vision_analyzer=mock_vision_analyzer)
        assert op.vision is mock_vision_analyzer
        assert op.video_analyzer is None

    def test_accepts_both(self, mock_video_analyzer, mock_vision_analyzer):
        op = PersonTrackingOperation(
            video_analyzer=mock_video_analyzer,
            vision_analyzer=mock_vision_analyzer,
        )
        assert op.video_analyzer is mock_video_analyzer
        assert op.vision is mock_vision_analyzer


# ---------------------------------------------------------------------------
# Prompt Building Tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Tests for prompt building."""

    def test_video_prompt_includes_scene_boundaries(
        self, mock_video_analyzer, sample_scenes
    ):
        op = PersonTrackingOperation(video_analyzer=mock_video_analyzer)
        prompt = op._build_video_prompt(sample_scenes)
        assert "scene 0: 0.0s-15.0s" in prompt
        assert "scene 1: 15.0s-30.0s" in prompt
        assert "Track people consistently" in prompt

    def test_frame_prompt_with_transcript(self, mock_vision_analyzer, sample_scenes):
        op = PersonTrackingOperation(vision_analyzer=mock_vision_analyzer)
        prompt = op._build_frame_prompt(sample_scenes[0])
        assert "scene 0" in prompt
        assert "0.0s - 15.0s" in prompt
        assert "Transcript context: Hello and welcome" in prompt

    def test_frame_prompt_without_transcript(
        self, mock_vision_analyzer, sample_scene_no_transcript
    ):
        op = PersonTrackingOperation(vision_analyzer=mock_vision_analyzer)
        prompt = op._build_frame_prompt(sample_scene_no_transcript)
        assert "scene 2" in prompt
        assert "Transcript context" not in prompt

    def test_frame_prompt_truncates_long_transcript(self, mock_vision_analyzer):
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=10.0,
            transcript_text="word " * 200,
        )
        op = PersonTrackingOperation(vision_analyzer=mock_vision_analyzer)
        prompt = op._build_frame_prompt(scene)
        context_part = prompt.split("Transcript context: ")[1].split("\n")[0]
        assert len(context_part) <= 500


# ---------------------------------------------------------------------------
# Execute Tests
# ---------------------------------------------------------------------------


class TestExecute:
    """Tests for the execute method."""

    @pytest.mark.asyncio
    async def test_execute_with_video_analyzer(
        self, mock_video_analyzer, sample_scenes, tmp_path, video_tracking_response
    ):
        """VideoAnalyzer tracks people across entire video."""
        mock_video_analyzer.analyze_video.return_value = video_tracking_response

        # Create a fake video file
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"\x00" * 100)

        op = PersonTrackingOperation(video_analyzer=mock_video_analyzer)
        result = await op.execute(
            scenes=sample_scenes,
            cache_dir=tmp_path,
            video_path=video_path,
        )

        assert isinstance(result, PeopleTrackingData)
        assert result.method == "video_analyzer"
        assert len(result.people) == 2
        assert result.people[0].description == "man in blue shirt"
        assert len(result.people[0].appearances) == 2
        assert result.people[1].description == "woman with glasses"
        mock_video_analyzer.analyze_video.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_vision_fallback(
        self,
        mock_video_analyzer,
        mock_vision_analyzer,
        sample_scenes,
        sample_keyframes,
        frame_tracking_response,
    ):
        """Falls back to VisionAnalyzer when VideoAnalyzer fails."""
        mock_video_analyzer.analyze_video.side_effect = RuntimeError("API error")
        mock_vision_analyzer.analyze_images.return_value = frame_tracking_response

        video_path = sample_keyframes / "video.mp4"
        video_path.write_bytes(b"\x00" * 100)

        op = PersonTrackingOperation(
            video_analyzer=mock_video_analyzer,
            vision_analyzer=mock_vision_analyzer,
        )
        result = await op.execute(
            scenes=sample_scenes,
            cache_dir=sample_keyframes,
            video_path=video_path,
        )

        assert result.method == "vision_analyzer"
        assert len(result.people) >= 1
        mock_vision_analyzer.analyze_images.assert_called()

    @pytest.mark.asyncio
    async def test_execute_vision_only(
        self,
        mock_vision_analyzer,
        sample_scenes,
        sample_keyframes,
        frame_tracking_response,
    ):
        """VisionAnalyzer works alone without VideoAnalyzer."""
        mock_vision_analyzer.analyze_images.return_value = frame_tracking_response

        op = PersonTrackingOperation(vision_analyzer=mock_vision_analyzer)
        result = await op.execute(
            scenes=sample_scenes,
            cache_dir=sample_keyframes,
        )

        assert result.method == "vision_analyzer"
        mock_vision_analyzer.analyze_images.assert_called()

    @pytest.mark.asyncio
    async def test_execute_no_providers(self, sample_scenes, tmp_path):
        """No providers returns empty result."""
        op = PersonTrackingOperation()
        result = await op.execute(
            scenes=sample_scenes,
            cache_dir=tmp_path,
        )

        assert result.method == "none"
        assert result.people == []

    @pytest.mark.asyncio
    async def test_execute_video_analyzer_no_video_path(
        self,
        mock_video_analyzer,
        mock_vision_analyzer,
        sample_scenes,
        sample_keyframes,
        frame_tracking_response,
    ):
        """Without video_path, skips VideoAnalyzer and uses VisionAnalyzer."""
        mock_vision_analyzer.analyze_images.return_value = frame_tracking_response

        op = PersonTrackingOperation(
            video_analyzer=mock_video_analyzer,
            vision_analyzer=mock_vision_analyzer,
        )
        result = await op.execute(
            scenes=sample_scenes,
            cache_dir=sample_keyframes,
            video_path=None,
        )

        mock_video_analyzer.analyze_video.assert_not_called()
        assert result.method == "vision_analyzer"

    @pytest.mark.asyncio
    async def test_execute_passes_schema_to_video_analyzer(
        self, mock_video_analyzer, sample_scenes, tmp_path, video_tracking_response
    ):
        """Pydantic schema is passed to analyze_video."""
        mock_video_analyzer.analyze_video.return_value = video_tracking_response

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"\x00" * 100)

        op = PersonTrackingOperation(video_analyzer=mock_video_analyzer)
        await op.execute(
            scenes=sample_scenes, cache_dir=tmp_path, video_path=video_path
        )

        call_args = mock_video_analyzer.analyze_video.call_args
        assert call_args is not None
        schema = call_args[1].get("schema") or call_args[0][2]
        assert schema is not None
        assert hasattr(schema, "model_json_schema")

    @pytest.mark.asyncio
    async def test_execute_handles_string_response(
        self, mock_video_analyzer, sample_scenes, tmp_path, video_tracking_response
    ):
        """Handles provider returning JSON string instead of dict."""
        mock_video_analyzer.analyze_video.return_value = json.dumps(
            video_tracking_response
        )

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"\x00" * 100)

        op = PersonTrackingOperation(video_analyzer=mock_video_analyzer)
        result = await op.execute(
            scenes=sample_scenes, cache_dir=tmp_path, video_path=video_path
        )

        assert len(result.people) == 2
        assert result.people[0].description == "man in blue shirt"

    @pytest.mark.asyncio
    async def test_execute_vision_merges_same_person(
        self,
        mock_vision_analyzer,
        sample_scenes,
        sample_keyframes,
    ):
        """VisionAnalyzer merges same person across scenes by description."""
        # Both scenes return the same person
        mock_vision_analyzer.analyze_images.return_value = {
            "people": [
                {
                    "person_id": "person_0",
                    "description": "man in blue shirt",
                    "appearances": [
                        {
                            "scene_id": 0,
                            "timestamp": 7.5,
                            "action": "speaking",
                            "confidence": 0.9,
                        },
                    ],
                },
            ]
        }

        op = PersonTrackingOperation(vision_analyzer=mock_vision_analyzer)
        result = await op.execute(
            scenes=sample_scenes,
            cache_dir=sample_keyframes,
        )

        # Should be merged into one person with 2 appearances
        matching = [p for p in result.people if p.description == "man in blue shirt"]
        assert len(matching) == 1
        assert len(matching[0].appearances) == 2

    @pytest.mark.asyncio
    async def test_execute_vision_handles_scene_error(
        self,
        mock_vision_analyzer,
        sample_scenes,
        sample_keyframes,
    ):
        """Vision error on one scene doesn't prevent other scenes."""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API error")
            return {
                "people": [
                    {
                        "person_id": "person_0",
                        "description": "presenter",
                        "appearances": [
                            {
                                "scene_id": 1,
                                "timestamp": 22.5,
                                "action": "coding",
                                "confidence": 0.9,
                            },
                        ],
                    },
                ]
            }

        mock_vision_analyzer.analyze_images.side_effect = side_effect

        op = PersonTrackingOperation(vision_analyzer=mock_vision_analyzer)
        result = await op.execute(
            scenes=sample_scenes,
            cache_dir=sample_keyframes,
        )

        assert len(result.people) == 1
        assert result.people[0].description == "presenter"


# ---------------------------------------------------------------------------
# Parse Tracking Result Tests
# ---------------------------------------------------------------------------


class TestParseTrackingResult:
    """Tests for _parse_tracking_result."""

    def test_parses_dict_response(self, video_tracking_response):
        op = PersonTrackingOperation()
        result = op._parse_tracking_result(video_tracking_response, method="test")

        assert result.method == "test"
        assert len(result.people) == 2
        assert result.people[0].person_id == "person_0"
        assert result.people[0].description == "man in blue shirt"
        assert len(result.people[0].appearances) == 2
        assert result.people[0].appearances[0].action == "presenting"

    def test_parses_empty_response(self):
        op = PersonTrackingOperation()
        result = op._parse_tracking_result({"people": []}, method="test")

        assert result.people == []

    def test_parses_missing_fields(self):
        op = PersonTrackingOperation()
        result = op._parse_tracking_result(
            {"people": [{"description": "someone", "appearances": [{"scene_id": 0}]}]},
            method="test",
        )

        assert len(result.people) == 1
        assert result.people[0].appearances[0].timestamp == 0.0
        assert result.people[0].appearances[0].confidence == 1.0


# ---------------------------------------------------------------------------
# Data Class Tests
# ---------------------------------------------------------------------------


class TestDataClasses:
    """Tests for dataclass serialization."""

    def test_person_appearance_roundtrip(self):
        original = PersonAppearance(
            scene_id=1, timestamp=15.0, action="speaking", confidence=0.9
        )
        restored = PersonAppearance.from_dict(original.to_dict())
        assert restored.scene_id == original.scene_id
        assert restored.timestamp == original.timestamp
        assert restored.action == original.action
        assert restored.confidence == original.confidence

    def test_person_track_roundtrip(self):
        original = PersonTrack(
            person_id="person_0",
            description="man in blue shirt",
            appearances=[
                PersonAppearance(scene_id=0, timestamp=7.5, action="speaking"),
                PersonAppearance(scene_id=1, timestamp=22.5, action="coding"),
            ],
        )
        restored = PersonTrack.from_dict(original.to_dict())
        assert restored.person_id == original.person_id
        assert restored.description == original.description
        assert len(restored.appearances) == 2
        assert restored.scene_count == 2
        assert restored.total_screen_time == 2.0

    def test_people_tracking_data_roundtrip(self):
        original = PeopleTrackingData(
            video_id="test123",
            method="video_analyzer",
            people=[
                PersonTrack(
                    person_id="person_0",
                    description="presenter",
                    appearances=[PersonAppearance(scene_id=0, timestamp=5.0)],
                ),
            ],
        )
        d = original.to_dict()
        assert d["video_id"] == "test123"
        assert d["method"] == "video_analyzer"
        assert d["people_count"] == 1

        restored = PeopleTrackingData.from_dict(d)
        assert restored.video_id == original.video_id
        assert restored.method == original.method
        assert len(restored.people) == 1

    def test_person_track_encoding_not_serialized(self):
        """Face encoding should not appear in serialized output."""
        track = PersonTrack(
            person_id="person_0",
            description="test",
            encoding=[0.1, 0.2, 0.3],
        )
        d = track.to_dict()
        assert "encoding" not in d


# ---------------------------------------------------------------------------
# Visual Transcript Tracking Tests
# ---------------------------------------------------------------------------


class TestTrackFromVisualTranscripts:
    """Tests for _track_from_visual_transcripts."""

    def test_extracts_people_from_visual_data(self, tmp_path):
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        scene_dir = scenes_dir / "scene_000"
        scene_dir.mkdir()
        visual_path = scene_dir / "visual.json"
        visual_path.write_text(
            json.dumps(
                {
                    "people": ["man in blue shirt", "woman with glasses"],
                    "actions": ["presenting"],
                    "description": "Two people in a conference room",
                }
            )
        )

        scenes = [SceneBoundary(scene_id=0, start_time=0.0, end_time=10.0)]
        result = _track_from_visual_transcripts(scenes, tmp_path)

        assert len(result.people) == 2
        assert result.method == "visual_transcript"
        assert result.people[0].appearances[0].confidence == 0.8

    def test_merges_same_person_across_scenes(self, tmp_path):
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()

        for scene_id in range(2):
            scene_dir = scenes_dir / f"scene_{scene_id:03d}"
            scene_dir.mkdir()
            visual_path = scene_dir / "visual.json"
            visual_path.write_text(
                json.dumps(
                    {
                        "people": ["man in blue shirt"],
                        "actions": ["speaking"],
                    }
                )
            )

        scenes = [
            SceneBoundary(scene_id=0, start_time=0.0, end_time=10.0),
            SceneBoundary(scene_id=1, start_time=10.0, end_time=20.0),
        ]
        result = _track_from_visual_transcripts(scenes, tmp_path)

        assert len(result.people) == 1
        assert len(result.people[0].appearances) == 2

    def test_handles_missing_visual_json(self, tmp_path):
        scenes = [SceneBoundary(scene_id=0, start_time=0.0, end_time=10.0)]
        result = _track_from_visual_transcripts(scenes, tmp_path)

        assert result.people == []

    def test_handles_invalid_json(self, tmp_path):
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        scene_dir = scenes_dir / "scene_000"
        scene_dir.mkdir()
        visual_path = scene_dir / "visual.json"
        visual_path.write_text("not json")

        scenes = [SceneBoundary(scene_id=0, start_time=0.0, end_time=10.0)]
        result = _track_from_visual_transcripts(scenes, tmp_path)

        assert result.people == []


# ---------------------------------------------------------------------------
# Default Provider Tests
# ---------------------------------------------------------------------------


class TestGetDefaultProviders:
    """Tests for _get_default_providers."""

    def _make_provider_mock(self, available=True, has_video=False, has_vision=True):
        mock = MagicMock()
        mock.is_available.return_value = available
        if has_video:
            mock.analyze_video = AsyncMock()
        if has_vision:
            mock.analyze_images = AsyncMock()
        return mock

    def test_returns_google_for_video_and_vision(self):
        mock_google = self._make_provider_mock(has_video=True, has_vision=True)

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_google,
        ):
            video, vision = _get_default_providers()
            assert video is mock_google
            assert vision is mock_google

    def test_returns_none_when_unavailable(self):
        mock_provider = self._make_provider_mock(available=False)

        with patch(
            "claudetube.providers.registry.get_provider",
            return_value=mock_provider,
        ):
            video, vision = _get_default_providers()
            assert video is None
            assert vision is None

    def test_handles_import_error(self):
        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=ImportError("No module"),
        ):
            video, vision = _get_default_providers()
            assert video is None
            assert vision is None

    def test_google_first_for_video_analyzer(self):
        """Google is tried first for VideoAnalyzer capability."""
        call_order = []

        def mock_get_provider(name, **kwargs):
            call_order.append(name)
            mock = self._make_provider_mock(
                has_video=(name == "google"),
                has_vision=True,
            )
            return mock

        with patch(
            "claudetube.providers.registry.get_provider",
            side_effect=mock_get_provider,
        ):
            video, vision = _get_default_providers()
            assert call_order[0] == "google"
            assert video is not None


# ---------------------------------------------------------------------------
# Integration: track_people
# ---------------------------------------------------------------------------


class TestTrackPeopleIntegration:
    """Tests that track_people uses PersonTrackingOperation."""

    def test_cached_result_returned_without_provider(self, tmp_path):
        """Cached people.json should be returned without needing a provider."""
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
                    "scenes": [{"scene_id": 0, "start_time": 0.0, "end_time": 10.0}],
                }
            )
        )

        # Create cached people.json
        entities_dir = cache_dir / "entities"
        entities_dir.mkdir()
        people_json = entities_dir / "people.json"
        people_json.write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "method": "visual_transcript",
                    "people_count": 1,
                    "people": {
                        "person_0": {
                            "person_id": "person_0",
                            "description": "presenter",
                            "appearances": [{"scene_id": 0, "timestamp": 5.0}],
                        }
                    },
                }
            )
        )

        result = track_people(video_id=video_id, output_base=tmp_path)

        assert result["people_count"] == 1
        assert "person_0" in result["people"]

    def test_visual_transcript_fallback(self, tmp_path):
        """Falls back to visual transcripts when no cache exists."""
        video_id = "testvt"
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

        # Create visual.json with people
        scene_dir = scenes_dir / "scene_000"
        scene_dir.mkdir()
        visual_path = scene_dir / "visual.json"
        visual_path.write_text(
            json.dumps(
                {
                    "people": ["presenter"],
                    "actions": ["speaking"],
                }
            )
        )

        result = track_people(video_id=video_id, output_base=tmp_path)

        assert result["people_count"] == 1
        assert result["method"] == "visual_transcript"

    def test_not_cached_error(self, tmp_path):
        result = track_people(video_id="nonexistent", output_base=tmp_path)
        assert "error" in result

    def test_no_scenes_error(self, tmp_path):
        video_id = "noscenes"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir()

        result = track_people(video_id=video_id, output_base=tmp_path)
        assert "error" in result

    def test_force_regenerates(self, tmp_path):
        """force=True should regenerate even when cached."""
        video_id = "testforce"
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

        # Create cached people.json with 5 people
        entities_dir = cache_dir / "entities"
        entities_dir.mkdir()
        people_json = entities_dir / "people.json"
        people_json.write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "method": "visual_transcript",
                    "people_count": 5,
                    "people": {},
                }
            )
        )

        # Force should regenerate (no visual.json → 0 people from visual transcripts)
        result = track_people(video_id=video_id, output_base=tmp_path, force=True)

        # Should have regenerated with 0 people (no visual.json data)
        assert result["people_count"] == 0

    def test_saves_to_cache(self, tmp_path):
        """Results should be saved to entities/people.json."""
        video_id = "testsave"
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

        track_people(video_id=video_id, output_base=tmp_path)

        people_path = get_people_json_path(cache_dir)
        assert people_path.exists()
        data = json.loads(people_path.read_text())
        assert data["video_id"] == video_id

    def test_updates_state_json(self, tmp_path):
        """Should update state.json with people_tracking_complete."""
        video_id = "teststate"
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

        state_file = cache_dir / "state.json"
        state_file.write_text(json.dumps({"video_id": video_id}))

        track_people(video_id=video_id, output_base=tmp_path)

        state = json.loads(state_file.read_text())
        assert state["people_tracking_complete"] is True


# ---------------------------------------------------------------------------
# get_people_tracking Tests
# ---------------------------------------------------------------------------


class TestGetPeopleTracking:
    """Tests for get_people_tracking cache reader."""

    def test_returns_cached_data(self, tmp_path):
        video_id = "testcache"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir()

        entities_dir = cache_dir / "entities"
        entities_dir.mkdir()
        people_json = entities_dir / "people.json"
        people_json.write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "method": "visual_transcript",
                    "people_count": 1,
                    "people": {
                        "person_0": {"person_id": "person_0", "description": "test"}
                    },
                }
            )
        )

        result = get_people_tracking(video_id, output_base=tmp_path)
        assert result["people_count"] == 1

    def test_error_when_not_cached(self, tmp_path):
        result = get_people_tracking("nonexistent", output_base=tmp_path)
        assert "error" in result

    def test_error_when_no_people_json(self, tmp_path):
        video_id = "nopeople"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir()

        result = get_people_tracking(video_id, output_base=tmp_path)
        assert "error" in result
        assert "Run track_people" in result["error"]

    def test_error_on_invalid_json(self, tmp_path):
        video_id = "badjson"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir()

        entities_dir = cache_dir / "entities"
        entities_dir.mkdir()
        people_json = entities_dir / "people.json"
        people_json.write_text("not json")

        result = get_people_tracking(video_id, output_base=tmp_path)
        assert "error" in result


# ---------------------------------------------------------------------------
# Prompt Constants Tests
# ---------------------------------------------------------------------------


class TestPromptConstants:
    """Tests for prompt template constants."""

    def test_video_prompt_has_placeholder(self):
        assert "{scene_boundaries}" in VIDEO_PERSON_TRACKING_PROMPT

    def test_video_prompt_mentions_tracking(self):
        assert "Track people consistently" in VIDEO_PERSON_TRACKING_PROMPT

    def test_frame_prompt_has_placeholders(self):
        assert "{scene_id}" in FRAME_PERSON_TRACKING_PROMPT
        assert "{start_time" in FRAME_PERSON_TRACKING_PROMPT
        assert "{end_time" in FRAME_PERSON_TRACKING_PROMPT
        assert "{context}" in FRAME_PERSON_TRACKING_PROMPT

    def test_frame_prompt_mentions_person_id(self):
        assert "person_id" in FRAME_PERSON_TRACKING_PROMPT
