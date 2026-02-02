"""Tests for AudioDescriptionGenerator provider integration."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from claudetube.cache.manager import CacheManager
from claudetube.cache.scenes import SceneBoundary, ScenesData
from claudetube.models.state import VideoState
from claudetube.operations.audio_description import (
    AudioDescriptionGenerator,
    _find_provider_for_capability,
)
from claudetube.operations.visual_transcript import VisualDescription
from claudetube.providers.base import (
    Provider,
    Transcriber,
    VideoAnalyzer,
    VisionAnalyzer,
)
from claudetube.providers.capabilities import Capability, ProviderInfo
from claudetube.providers.types import TranscriptionResult, TranscriptionSegment

# ── Fake Providers ────────────────────────────────────────────────────


class FakeVideoProvider(Provider):
    """Fake provider implementing VideoAnalyzer for testing."""

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="fake-video",
            capabilities=frozenset({Capability.VIDEO, Capability.VISION}),
        )

    def is_available(self) -> bool:
        return True

    async def analyze_video(self, video, prompt, schema=None, start_time=None, end_time=None, **kwargs):
        return json.dumps({
            "description": "A person demonstrates code on screen.",
            "people": ["presenter"],
            "objects": ["laptop", "projector"],
            "text_on_screen": ["def hello():"],
            "actions": ["typing"],
            "setting": "conference room",
        })

    async def analyze_images(self, images, prompt, schema=None, **kwargs):
        return json.dumps({
            "description": "Frame showing a code editor.",
            "people": [],
            "objects": ["code editor"],
            "text_on_screen": ["import os"],
            "actions": [],
            "setting": "desktop",
        })


# Register the protocol implementations
VideoAnalyzer.register(FakeVideoProvider)
VisionAnalyzer.register(FakeVideoProvider)


class FakeVisionProvider(Provider):
    """Fake provider implementing only VisionAnalyzer."""

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="fake-vision",
            capabilities=frozenset({Capability.VISION}),
        )

    def is_available(self) -> bool:
        return True

    async def analyze_images(self, images, prompt, schema=None, **kwargs):
        return json.dumps({
            "description": "A diagram is shown on a whiteboard.",
            "people": ["instructor"],
            "objects": ["whiteboard", "marker"],
            "text_on_screen": ["Step 1", "Step 2"],
            "actions": ["pointing at diagram"],
            "setting": "classroom",
        })


VisionAnalyzer.register(FakeVisionProvider)


class FakeTranscriptionProvider(Provider):
    """Fake provider implementing Transcriber."""

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="fake-transcriber",
            capabilities=frozenset({Capability.TRANSCRIBE}),
        )

    def is_available(self) -> bool:
        return True

    async def transcribe(self, audio, language=None, **kwargs):
        return TranscriptionResult(
            text="A person walks into frame and sits down at the desk.",
            segments=[
                TranscriptionSegment(start=0.0, end=3.0, text="A person walks into frame."),
                TranscriptionSegment(start=3.0, end=6.0, text="They sit down at the desk."),
            ],
            language="en",
            duration=6.0,
            provider="fake-transcriber",
        )


Transcriber.register(FakeTranscriptionProvider)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def video_id():
    return "test_ad_gen_123"


@pytest.fixture
def cache_with_scenes(tmp_path, video_id):
    """Set up a video cache with scenes and visual data."""
    cache = CacheManager(tmp_path)
    cache_dir = cache.ensure_cache_dir(video_id)

    # State
    state = VideoState(video_id=video_id, scenes_processed=True, scene_count=2)
    cache.save_state(video_id, state)

    # Scenes
    scenes_data = ScenesData(
        video_id=video_id,
        method="transcript",
        scenes=[
            SceneBoundary(
                scene_id=0, start_time=0.0, end_time=30.0,
                transcript_text="Welcome to the tutorial.",
            ),
            SceneBoundary(
                scene_id=1, start_time=30.0, end_time=60.0,
                transcript_text="Now let's look at the code.",
            ),
        ],
    )
    scenes_dir = cache_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "scenes.json").write_text(json.dumps(scenes_data.to_dict(), indent=2))

    return cache, cache_dir


@pytest.fixture
def cache_with_scenes_and_visuals(cache_with_scenes, video_id):
    """Cache with scenes AND pre-existing visual.json for each scene."""
    cache, cache_dir = cache_with_scenes

    for i in range(2):
        scene_dir = cache_dir / "scenes" / f"scene_{i:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        visual = VisualDescription(
            scene_id=i, description=f"Description for scene {i}.",
            people=[], objects=[], text_on_screen=[], actions=[],
        )
        (scene_dir / "visual.json").write_text(json.dumps(visual.to_dict(), indent=2))

    return cache, cache_dir


# ── Tests: AudioDescriptionGenerator ──────────────────────────────────


class TestAudioDescriptionGeneratorInit:
    """Tests for AudioDescriptionGenerator construction."""

    def test_default_construction(self):
        gen = AudioDescriptionGenerator()
        assert gen._video_provider is None
        assert gen._vision_provider is None
        assert gen._transcription_provider is None

    def test_explicit_providers(self):
        video = FakeVideoProvider()
        vision = FakeVisionProvider()
        trans = FakeTranscriptionProvider()
        gen = AudioDescriptionGenerator(
            video_provider=video,
            vision_provider=vision,
            transcription_provider=trans,
        )
        assert gen._get_video_provider() is video
        assert gen._get_vision_provider() is vision
        assert gen._get_transcription_provider() is trans


class TestGenerateCache:
    """Tests for cache behavior in generate()."""

    @pytest.mark.asyncio
    async def test_returns_cached_when_exists(self, tmp_path, video_id, cache_with_scenes):
        cache, cache_dir = cache_with_scenes

        # Write AD files directly
        vtt_path, txt_path = cache.get_ad_paths(video_id)
        vtt_path.write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:30.000\nCached description")
        txt_path.write_text("[00:00] Cached description")

        gen = AudioDescriptionGenerator()
        result = await gen.generate(video_id, output_base=tmp_path)

        assert result["status"] == "cached"
        assert "vtt_path" in result

    @pytest.mark.asyncio
    async def test_force_bypasses_cache(self, tmp_path, video_id, cache_with_scenes_and_visuals):
        cache, cache_dir = cache_with_scenes_and_visuals

        # Write AD files directly
        vtt_path, txt_path = cache.get_ad_paths(video_id)
        vtt_path.write_text("WEBVTT\n\nOld content")
        txt_path.write_text("[00:00] Old content")

        gen = AudioDescriptionGenerator(vision_provider=FakeVisionProvider())

        # Mock _get_scene_keyframes to return fake keyframe paths
        fake_kf = tmp_path / "fake_kf.jpg"
        fake_kf.write_bytes(b"\xff\xd8\xff\xe0")  # minimal JPEG header

        with patch.object(AudioDescriptionGenerator, "_get_scene_keyframes", return_value=[fake_kf]):
            result = await gen.generate(video_id, force=True, output_base=tmp_path)

        assert result["status"] == "generated"


class TestGenerateNativeVideo:
    """Tests for native video generation path."""

    @pytest.mark.asyncio
    async def test_uses_video_provider_when_available(self, tmp_path, video_id, cache_with_scenes):
        cache, cache_dir = cache_with_scenes

        # Create a fake video source file
        state = cache.get_state(video_id)
        state.cached_file = "source.mp4"
        cache.save_state(video_id, state)
        (cache_dir / "source.mp4").write_bytes(b"fake video content")

        provider = FakeVideoProvider()
        gen = AudioDescriptionGenerator(video_provider=provider)

        result = await gen.generate(video_id, output_base=tmp_path)

        assert result["status"] == "generated"
        assert result["source"] == "native_video"
        assert result["provider"] == "fake-video"
        assert result["cue_count"] == 2

    @pytest.mark.asyncio
    async def test_falls_back_to_vision_when_no_video_source(self, tmp_path, video_id, cache_with_scenes):
        """If video file not on disk, fall back to frame-by-frame."""
        cache, cache_dir = cache_with_scenes

        video_prov = FakeVideoProvider()
        vision_prov = FakeVisionProvider()

        gen = AudioDescriptionGenerator(video_provider=video_prov, vision_provider=vision_prov)

        # Mock keyframes to exist
        fake_kf = tmp_path / "fake_kf.jpg"
        fake_kf.write_bytes(b"\xff\xd8\xff\xe0")

        with patch.object(AudioDescriptionGenerator, "_get_scene_keyframes", return_value=[fake_kf]):
            result = await gen.generate(video_id, output_base=tmp_path)

        assert result["status"] == "generated"
        assert result["source"] == "frame_vision"
        assert result["provider"] == "fake-vision"


class TestGenerateFromFrames:
    """Tests for frame-by-frame vision generation path."""

    @pytest.mark.asyncio
    async def test_uses_vision_provider(self, tmp_path, video_id, cache_with_scenes):
        cache, cache_dir = cache_with_scenes

        provider = FakeVisionProvider()
        gen = AudioDescriptionGenerator(vision_provider=provider)

        fake_kf = tmp_path / "fake_kf.jpg"
        fake_kf.write_bytes(b"\xff\xd8\xff\xe0")

        with patch.object(AudioDescriptionGenerator, "_get_scene_keyframes", return_value=[fake_kf]):
            result = await gen.generate(video_id, output_base=tmp_path)

        assert result["status"] == "generated"
        assert result["source"] == "frame_vision"
        assert result["provider"] == "fake-vision"
        assert result["generated_count"] == 2

    @pytest.mark.asyncio
    async def test_uses_cached_visual_json(self, tmp_path, video_id, cache_with_scenes_and_visuals):
        """Should use existing visual.json without calling provider."""
        cache, cache_dir = cache_with_scenes_and_visuals

        provider = FakeVisionProvider()
        # Wrap analyze_images to track calls
        original = provider.analyze_images
        call_count = 0

        async def tracking_analyze(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return await original(*args, **kwargs)

        provider.analyze_images = tracking_analyze

        gen = AudioDescriptionGenerator(vision_provider=provider)
        result = await gen.generate(video_id, output_base=tmp_path)

        assert result["status"] == "generated"
        # Provider should NOT have been called — both scenes have cached visuals
        assert call_count == 0

    @pytest.mark.asyncio
    async def test_handles_no_keyframes(self, tmp_path, video_id, cache_with_scenes):
        cache, cache_dir = cache_with_scenes

        provider = FakeVisionProvider()
        gen = AudioDescriptionGenerator(vision_provider=provider)

        with patch.object(AudioDescriptionGenerator, "_get_scene_keyframes", return_value=[]):
            result = await gen.generate(video_id, output_base=tmp_path)

        # All scenes should have errors about missing keyframes
        assert len(result.get("errors", [])) == 2


class TestGenerateCompileOnlyFallback:
    """Tests for compile-only fallback when no AI providers available."""

    @pytest.mark.asyncio
    async def test_falls_back_to_compile(self, tmp_path, video_id, cache_with_scenes_and_visuals):
        """No providers at all should compile from existing data."""
        gen = AudioDescriptionGenerator()

        # Patch out auto-discovery to return None
        with patch.object(gen, "_get_video_provider", return_value=None), \
             patch.object(gen, "_get_vision_provider", return_value=None):
            result = await gen.generate(video_id, output_base=tmp_path)

        assert result["status"] == "compiled"
        assert result["source"] == "scene_compilation"


class TestTranscribeAdTrack:
    """Tests for AD track transcription."""

    @pytest.mark.asyncio
    async def test_transcribes_ad_track(self, tmp_path, video_id):
        cache = CacheManager(tmp_path)
        cache_dir = cache.ensure_cache_dir(video_id)
        state = VideoState(video_id=video_id)
        cache.save_state(video_id, state)

        # Create fake AD audio
        ad_audio = cache_dir / "ad_audio.mp3"
        ad_audio.write_bytes(b"fake audio data")

        provider = FakeTranscriptionProvider()
        gen = AudioDescriptionGenerator(transcription_provider=provider)

        result = await gen.transcribe_ad_track(video_id, ad_audio, output_base=tmp_path)

        assert result["status"] == "transcribed"
        assert result["segment_count"] == 2
        assert result["source"] == "source_track"
        assert result["provider"] == "fake-transcriber"

        # Verify output files
        vtt_path, txt_path = cache.get_ad_paths(video_id)
        assert vtt_path.exists()
        assert txt_path.exists()

        vtt_content = vtt_path.read_text()
        assert "WEBVTT" in vtt_content
        assert "Kind: descriptions" in vtt_content
        assert "walks into frame" in vtt_content

        # Verify state updated
        state = cache.get_state(video_id)
        assert state.ad_complete is True
        assert state.ad_source == "source_track"

    @pytest.mark.asyncio
    async def test_error_when_no_provider(self, tmp_path, video_id):
        cache = CacheManager(tmp_path)
        cache.ensure_cache_dir(video_id)
        state = VideoState(video_id=video_id)
        cache.save_state(video_id, state)

        gen = AudioDescriptionGenerator()

        with patch.object(gen, "_get_transcription_provider", return_value=None):
            result = await gen.transcribe_ad_track(
                video_id, Path("/fake/audio.mp3"), output_base=tmp_path,
            )

        assert "error" in result
        assert "No transcription provider" in result["error"]

    @pytest.mark.asyncio
    async def test_error_when_file_missing(self, tmp_path, video_id):
        cache = CacheManager(tmp_path)
        cache.ensure_cache_dir(video_id)
        state = VideoState(video_id=video_id)
        cache.save_state(video_id, state)

        gen = AudioDescriptionGenerator(transcription_provider=FakeTranscriptionProvider())
        result = await gen.transcribe_ad_track(
            video_id, Path("/nonexistent/audio.mp3"), output_base=tmp_path,
        )

        assert "error" in result
        assert "not found" in result["error"]


class TestParseDescriptionResponse:
    """Tests for _parse_description_response static method."""

    def test_parses_json_string(self):
        response = json.dumps({
            "description": "A scene.",
            "people": ["person"],
            "objects": [],
            "text_on_screen": [],
            "actions": [],
            "setting": "office",
        })
        scene = SceneBoundary(scene_id=0, start_time=0, end_time=10)
        result = AudioDescriptionGenerator._parse_description_response(response, scene)

        assert result is not None
        assert result.description == "A scene."
        assert result.setting == "office"
        assert result.scene_id == 0

    def test_parses_dict_response(self):
        response = {
            "description": "A dict scene.",
            "people": [],
            "objects": [],
            "text_on_screen": [],
            "actions": [],
        }
        scene = SceneBoundary(scene_id=5, start_time=100, end_time=200)
        result = AudioDescriptionGenerator._parse_description_response(response, scene)

        assert result is not None
        assert result.description == "A dict scene."
        assert result.scene_id == 5

    def test_handles_markdown_code_blocks(self):
        response = '```json\n{"description": "Markdown.", "people": [], "objects": [], "text_on_screen": [], "actions": []}\n```'
        scene = SceneBoundary(scene_id=0, start_time=0, end_time=10)
        result = AudioDescriptionGenerator._parse_description_response(response, scene)

        assert result is not None
        assert result.description == "Markdown."

    def test_returns_none_on_invalid_json(self):
        scene = SceneBoundary(scene_id=0, start_time=0, end_time=10)
        result = AudioDescriptionGenerator._parse_description_response("not json", scene)
        assert result is None

    def test_returns_none_on_none_response(self):
        scene = SceneBoundary(scene_id=0, start_time=0, end_time=10)
        result = AudioDescriptionGenerator._parse_description_response(None, scene)
        assert result is None


class TestFindProviderForCapability:
    """Tests for _find_provider_for_capability helper."""

    def test_returns_none_when_no_providers(self):
        with patch("claudetube.providers.registry.list_available", return_value=[]):
            result = _find_provider_for_capability("VIDEO")
        assert result is None

    def test_finds_matching_provider(self):
        fake_provider = FakeVideoProvider()
        with patch("claudetube.providers.registry.list_available", return_value=["fake-video"]), \
             patch("claudetube.providers.registry.get_provider", return_value=fake_provider):
            result = _find_provider_for_capability("VIDEO")
        assert result is fake_provider

    def test_skips_provider_without_capability(self):
        fake_vision = FakeVisionProvider()
        with patch("claudetube.providers.registry.list_available", return_value=["fake-vision"]), \
             patch("claudetube.providers.registry.get_provider", return_value=fake_vision):
            result = _find_provider_for_capability("VIDEO")
        assert result is None


class TestGenerateErrorCases:
    """Tests for error handling in generate()."""

    @pytest.mark.asyncio
    async def test_error_when_video_not_cached(self, tmp_path):
        gen = AudioDescriptionGenerator()
        result = await gen.generate("nonexistent", output_base=tmp_path)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_error_when_no_scenes(self, tmp_path, video_id):
        cache = CacheManager(tmp_path)
        cache.ensure_cache_dir(video_id)
        state = VideoState(video_id=video_id)
        cache.save_state(video_id, state)

        gen = AudioDescriptionGenerator()
        result = await gen.generate(video_id, output_base=tmp_path)
        assert "error" in result
        assert "scenes" in result["error"].lower()


class TestModuleExports:
    """Tests for module exports."""

    def test_generator_exported_from_operations(self):
        from claudetube.operations import AudioDescriptionGenerator
        assert AudioDescriptionGenerator is not None

    def test_find_provider_importable(self):
        from claudetube.operations.audio_description import (
            _find_provider_for_capability,
        )
        assert callable(_find_provider_for_capability)
