"""Tests for audio description scene compiler."""

import json

from claudetube.cache.manager import CacheManager
from claudetube.cache.scenes import SceneBoundary, ScenesData
from claudetube.models.state import VideoState
from claudetube.operations.audio_description import (
    SUPPORTED_AD_LANGUAGES,
    _compile_vtt,
    _format_description,
    _format_vtt_timestamp,
    _resolve_ad_language,
    compile_scene_descriptions,
    get_scene_descriptions,
)
from claudetube.operations.visual_transcript import VisualDescription


class TestFormatVttTimestamp:
    """Tests for VTT timestamp formatting."""

    def test_zero_seconds(self):
        assert _format_vtt_timestamp(0.0) == "00:00:00.000"

    def test_simple_seconds(self):
        assert _format_vtt_timestamp(5.0) == "00:00:05.000"

    def test_minutes_and_seconds(self):
        assert _format_vtt_timestamp(90.0) == "00:01:30.000"

    def test_hours(self):
        assert _format_vtt_timestamp(3661.5) == "01:01:01.500"

    def test_milliseconds(self):
        assert _format_vtt_timestamp(1.234) == "00:00:01.234"

    def test_large_value(self):
        assert _format_vtt_timestamp(7200.0) == "02:00:00.000"


class TestFormatDescription:
    """Tests for description formatting."""

    def test_with_visual_description(self):
        visual = VisualDescription(
            scene_id=0,
            description="A developer types at a desk with two monitors.",
            people=["developer"],
            objects=["monitors", "keyboard"],
            text_on_screen=[],
            actions=["typing"],
            setting="home office",
        )
        result = _format_description(visual, "some transcript", None)
        assert "A developer types at a desk" in result
        assert "home office" in result

    def test_with_title(self):
        visual = VisualDescription(
            scene_id=0,
            description="Speaker introduces topic.",
            people=[],
            objects=[],
            text_on_screen=[],
            actions=[],
        )
        result = _format_description(visual, "", "Introduction")
        assert result.startswith("Introduction.")

    def test_with_text_on_screen(self):
        visual = VisualDescription(
            scene_id=0,
            description="Code editor is shown.",
            people=[],
            objects=[],
            text_on_screen=["def main():", "import os"],
            actions=[],
        )
        result = _format_description(visual, "", None)
        assert "On screen:" in result
        assert "def main()" in result

    def test_without_visual_falls_back_to_transcript(self):
        result = _format_description(
            None, "Hello everyone welcome to the tutorial.", None
        )
        assert "Speaker:" in result
        assert "Hello everyone" in result

    def test_without_visual_and_short_transcript(self):
        result = _format_description(None, "Hi.", None)
        assert result == ""

    def test_without_visual_and_empty_transcript(self):
        result = _format_description(None, "", None)
        assert result == ""

    def test_setting_not_duplicated_in_description(self):
        visual = VisualDescription(
            scene_id=0,
            description="A classroom with students.",
            people=["students"],
            objects=[],
            text_on_screen=[],
            actions=[],
            setting="classroom",
        )
        result = _format_description(visual, "", None)
        # Setting should not appear separately since "classroom" is in description
        assert result.count("classroom") == 1

    def test_actions_used_when_no_description(self):
        visual = VisualDescription(
            scene_id=0,
            description="",
            people=[],
            objects=[],
            text_on_screen=[],
            actions=["scrolling through code", "clicking run button"],
        )
        result = _format_description(visual, "", None)
        assert "Scrolling through code" in result

    def test_text_on_screen_limited_to_5(self):
        visual = VisualDescription(
            scene_id=0,
            description="Code shown.",
            people=[],
            objects=[],
            text_on_screen=[
                "line1",
                "line2",
                "line3",
                "line4",
                "line5",
                "line6",
                "line7",
            ],
            actions=[],
        )
        result = _format_description(visual, "", None)
        assert "line5" in result
        assert "line6" not in result


class TestCompileVtt:
    """Tests for VTT compilation."""

    def _make_scenes_data(self, scenes):
        return ScenesData(video_id="test123", method="transcript", scenes=scenes)

    def _write_visual_json(self, cache_dir, scene_id, visual):
        scene_dir = cache_dir / "scenes" / f"scene_{scene_id:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        path = scene_dir / "visual.json"
        path.write_text(json.dumps(visual.to_dict(), indent=2))

    def test_basic_compilation(self, tmp_path):
        scenes = [
            SceneBoundary(
                scene_id=0, start_time=0.0, end_time=30.0, transcript_text="Hello world"
            ),
            SceneBoundary(
                scene_id=1,
                start_time=30.0,
                end_time=60.0,
                transcript_text="Next section",
            ),
        ]
        scenes_data = self._make_scenes_data(scenes)

        # Write visual.json for scene 0
        visual = VisualDescription(
            scene_id=0,
            description="Speaker at desk.",
            people=[],
            objects=[],
            text_on_screen=[],
            actions=[],
        )
        self._write_visual_json(tmp_path, 0, visual)

        visual2 = VisualDescription(
            scene_id=1,
            description="Code editor shown.",
            people=[],
            objects=[],
            text_on_screen=["def hello():"],
            actions=[],
        )
        self._write_visual_json(tmp_path, 1, visual2)

        vtt_lines, txt_lines = _compile_vtt(scenes_data, tmp_path)

        # VTT header
        assert vtt_lines[0] == "WEBVTT"
        assert vtt_lines[1] == "Kind: descriptions"
        assert vtt_lines[2] == "Language: en"

        # Should have 2 cues
        assert len(txt_lines) == 2
        assert txt_lines[0].startswith("[00:00]")
        assert txt_lines[1].startswith("[00:30]")

    def test_scene_without_visual_json(self, tmp_path):
        """Scenes without visual.json should fall back to transcript."""
        scenes = [
            SceneBoundary(
                scene_id=0,
                start_time=0.0,
                end_time=30.0,
                transcript_text="Hello everyone, welcome to the tutorial on Python basics.",
            ),
        ]
        scenes_data = self._make_scenes_data(scenes)

        vtt_lines, txt_lines = _compile_vtt(scenes_data, tmp_path)

        assert len(txt_lines) == 1
        assert "Speaker:" in txt_lines[0]

    def test_empty_scene_produces_no_cue(self, tmp_path):
        """Scene with no visual data and short transcript should be skipped."""
        scenes = [
            SceneBoundary(
                scene_id=0, start_time=0.0, end_time=5.0, transcript_text="Hi."
            ),
        ]
        scenes_data = self._make_scenes_data(scenes)

        vtt_lines, txt_lines = _compile_vtt(scenes_data, tmp_path)

        assert len(txt_lines) == 0

    def test_vtt_timing_format(self, tmp_path):
        scenes = [
            SceneBoundary(
                scene_id=0, start_time=65.5, end_time=130.0, transcript_text=""
            ),
        ]
        scenes_data = self._make_scenes_data(scenes)

        visual = VisualDescription(
            scene_id=0,
            description="A diagram is shown.",
            people=[],
            objects=[],
            text_on_screen=[],
            actions=[],
        )
        self._write_visual_json(tmp_path, 0, visual)

        vtt_lines, _ = _compile_vtt(scenes_data, tmp_path)

        # Find the timing line
        timing_line = [line for line in vtt_lines if "-->" in line][0]
        assert timing_line == "00:01:05.500 --> 00:02:10.000"

    def test_invalid_visual_json_handled(self, tmp_path):
        """Invalid visual.json should not crash compilation."""
        scenes = [
            SceneBoundary(
                scene_id=0,
                start_time=0.0,
                end_time=30.0,
                transcript_text="Welcome to the introduction to machine learning concepts.",
            ),
        ]
        scenes_data = self._make_scenes_data(scenes)

        # Write invalid JSON
        scene_dir = tmp_path / "scenes" / "scene_000"
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "visual.json").write_text("not valid json")

        vtt_lines, txt_lines = _compile_vtt(scenes_data, tmp_path)

        # Should still produce output from transcript fallback
        assert len(txt_lines) == 1

    def test_language_parameter_in_header(self, tmp_path):
        """Language parameter should appear in VTT header."""
        scenes = [
            SceneBoundary(
                scene_id=0,
                start_time=0.0,
                end_time=30.0,
                transcript_text="Hola mundo",
            ),
        ]
        scenes_data = self._make_scenes_data(scenes)

        visual = VisualDescription(
            scene_id=0,
            description="Presentador en escritorio.",
            people=[],
            objects=[],
            text_on_screen=[],
            actions=[],
        )
        self._write_visual_json(tmp_path, 0, visual)

        vtt_lines, _ = _compile_vtt(scenes_data, tmp_path, language="es")

        assert vtt_lines[0] == "WEBVTT"
        assert vtt_lines[1] == "Kind: descriptions"
        assert vtt_lines[2] == "Language: es"

    def test_default_language_is_english(self, tmp_path):
        """Default language should be 'en' when not specified."""
        scenes = [
            SceneBoundary(
                scene_id=0,
                start_time=0.0,
                end_time=30.0,
                transcript_text="Hello world",
            ),
        ]
        scenes_data = self._make_scenes_data(scenes)

        visual = VisualDescription(
            scene_id=0,
            description="Speaker at desk.",
            people=[],
            objects=[],
            text_on_screen=[],
            actions=[],
        )
        self._write_visual_json(tmp_path, 0, visual)

        vtt_lines, _ = _compile_vtt(scenes_data, tmp_path)
        assert vtt_lines[2] == "Language: en"

    def test_chapter_titles_included(self, tmp_path):
        scenes = [
            SceneBoundary(
                scene_id=0,
                start_time=0.0,
                end_time=60.0,
                title="Getting Started",
                transcript_text="",
            ),
        ]
        scenes_data = self._make_scenes_data(scenes)

        visual = VisualDescription(
            scene_id=0,
            description="Title card shown.",
            people=[],
            objects=[],
            text_on_screen=[],
            actions=[],
        )
        self._write_visual_json(tmp_path, 0, visual)

        _, txt_lines = _compile_vtt(scenes_data, tmp_path)

        assert "Getting Started." in txt_lines[0]


class TestCompileSceneDescriptions:
    """Tests for the main compile_scene_descriptions function."""

    def _setup_video(self, tmp_path, video_id="test123"):
        """Set up a cached video with scenes and visual data."""
        cache = CacheManager(tmp_path)
        cache_dir = cache.ensure_cache_dir(video_id)

        # Create state
        state = VideoState(video_id=video_id, scenes_processed=True, scene_count=2)
        cache.save_state(video_id, state)

        # Create scenes data
        scenes_data = ScenesData(
            video_id=video_id,
            method="transcript",
            scenes=[
                SceneBoundary(
                    scene_id=0,
                    start_time=0.0,
                    end_time=30.0,
                    transcript_text="First scene",
                ),
                SceneBoundary(
                    scene_id=1,
                    start_time=30.0,
                    end_time=60.0,
                    transcript_text="Second scene",
                ),
            ],
        )
        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        (scenes_dir / "scenes.json").write_text(
            json.dumps(scenes_data.to_dict(), indent=2)
        )

        # Create visual.json for both scenes
        for i in range(2):
            scene_dir = scenes_dir / f"scene_{i:03d}"
            scene_dir.mkdir(parents=True, exist_ok=True)
            visual = VisualDescription(
                scene_id=i,
                description=f"Visual description for scene {i}.",
                people=[],
                objects=[],
                text_on_screen=[],
                actions=[],
            )
            (scene_dir / "visual.json").write_text(
                json.dumps(visual.to_dict(), indent=2)
            )

        return cache, cache_dir

    def test_compiles_successfully(self, tmp_path):
        cache, cache_dir = self._setup_video(tmp_path)
        result = compile_scene_descriptions("test123", output_base=tmp_path)

        assert result["status"] == "compiled"
        assert result["cue_count"] == 2
        assert result["source"] == "scene_compilation"

        # Verify files exist
        vtt_path = cache_dir / "audio.ad.vtt"
        txt_path = cache_dir / "audio.ad.txt"
        assert vtt_path.exists()
        assert txt_path.exists()

        # Verify VTT content
        vtt_content = vtt_path.read_text()
        assert vtt_content.startswith("WEBVTT")
        assert "Kind: descriptions" in vtt_content

    def test_updates_state(self, tmp_path):
        self._setup_video(tmp_path)
        compile_scene_descriptions("test123", output_base=tmp_path)

        cache = CacheManager(tmp_path)
        state = cache.get_state("test123")
        assert state.ad_complete is True
        assert state.ad_source == "scene_compilation"

    def test_returns_cached_when_exists(self, tmp_path):
        self._setup_video(tmp_path)

        # Compile once
        result1 = compile_scene_descriptions("test123", output_base=tmp_path)
        assert result1["status"] == "compiled"

        # Second call should return cached
        result2 = compile_scene_descriptions("test123", output_base=tmp_path)
        assert result2["status"] == "cached"

    def test_force_recompiles(self, tmp_path):
        self._setup_video(tmp_path)

        # Compile once
        compile_scene_descriptions("test123", output_base=tmp_path)

        # Force recompile
        result = compile_scene_descriptions("test123", force=True, output_base=tmp_path)
        assert result["status"] == "compiled"

    def test_error_when_video_not_cached(self, tmp_path):
        result = compile_scene_descriptions("nonexistent", output_base=tmp_path)
        assert "error" in result

    def test_error_when_no_scenes(self, tmp_path):
        cache = CacheManager(tmp_path)
        cache.ensure_cache_dir("test123")
        state = VideoState(video_id="test123")
        cache.save_state("test123", state)

        result = compile_scene_descriptions("test123", output_base=tmp_path)
        assert "error" in result
        assert "scenes" in result["error"].lower()

    def test_error_when_empty_scenes(self, tmp_path):
        cache = CacheManager(tmp_path)
        cache_dir = cache.ensure_cache_dir("test123")
        state = VideoState(video_id="test123")
        cache.save_state("test123", state)

        # Create scenes.json with empty scenes list
        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        scenes_data = ScenesData(video_id="test123", method="transcript", scenes=[])
        (scenes_dir / "scenes.json").write_text(
            json.dumps(scenes_data.to_dict(), indent=2)
        )

        result = compile_scene_descriptions("test123", output_base=tmp_path)
        assert "error" in result


class TestGetSceneDescriptions:
    """Tests for get_scene_descriptions read-only accessor."""

    def test_returns_error_when_no_ad(self, tmp_path):
        cache = CacheManager(tmp_path)
        cache.ensure_cache_dir("test123")

        result = get_scene_descriptions("test123", output_base=tmp_path)
        assert "error" in result

    def test_returns_cached_content(self, tmp_path):
        cache = CacheManager(tmp_path)
        cache.ensure_cache_dir("test123")

        # Write AD files directly
        vtt_path, txt_path = cache.get_ad_paths("test123")
        vtt_path.write_text(
            "WEBVTT\n\n1\n00:00:00.000 --> 00:00:30.000\nTest description"
        )
        txt_path.write_text("[00:00] Test description")

        result = get_scene_descriptions("test123", output_base=tmp_path)
        assert "vtt" in result
        assert "txt" in result
        assert "WEBVTT" in result["vtt"]
        assert "[00:00]" in result["txt"]


class TestResolveAdLanguage:
    """Tests for _resolve_ad_language helper."""

    def test_none_returns_english(self):
        assert _resolve_ad_language(None) == "en"

    def test_empty_string_returns_english(self):
        assert _resolve_ad_language("") == "en"

    def test_english(self):
        assert _resolve_ad_language("en") == "en"

    def test_spanish(self):
        assert _resolve_ad_language("es") == "es"

    def test_french(self):
        assert _resolve_ad_language("fr") == "fr"

    def test_german(self):
        assert _resolve_ad_language("de") == "de"

    def test_japanese(self):
        assert _resolve_ad_language("ja") == "ja"

    def test_chinese(self):
        assert _resolve_ad_language("zh") == "zh"

    def test_portuguese(self):
        assert _resolve_ad_language("pt") == "pt"

    def test_regional_variant_normalized(self):
        """pt-BR should resolve to pt."""
        assert _resolve_ad_language("pt-BR") == "pt"

    def test_chinese_variant_normalized(self):
        """zh-Hans should resolve to zh."""
        assert _resolve_ad_language("zh-Hans") == "zh"

    def test_uppercase_normalized(self):
        assert _resolve_ad_language("ES") == "es"

    def test_unsupported_language_falls_back(self):
        assert _resolve_ad_language("ko") == "en"

    def test_supported_languages_set(self):
        assert {"en", "es", "fr", "de", "ja", "zh", "pt"} == SUPPORTED_AD_LANGUAGES


class TestCompileSceneDescriptionsLanguage:
    """Tests for language-aware compilation in compile_scene_descriptions."""

    def _setup_video_with_language(self, tmp_path, video_id="test123", language=None):
        """Set up a cached video with language metadata."""
        cache = CacheManager(tmp_path)
        cache_dir = cache.ensure_cache_dir(video_id)

        state = VideoState(
            video_id=video_id,
            scenes_processed=True,
            scene_count=1,
            language=language,
        )
        cache.save_state(video_id, state)

        scenes_data = ScenesData(
            video_id=video_id,
            method="transcript",
            scenes=[
                SceneBoundary(
                    scene_id=0,
                    start_time=0.0,
                    end_time=30.0,
                    transcript_text="Content",
                ),
            ],
        )
        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        (scenes_dir / "scenes.json").write_text(
            json.dumps(scenes_data.to_dict(), indent=2)
        )

        scene_dir = scenes_dir / "scene_000"
        scene_dir.mkdir(parents=True, exist_ok=True)
        visual = VisualDescription(
            scene_id=0,
            description="Visual content.",
            people=[],
            objects=[],
            text_on_screen=[],
            actions=[],
        )
        (scene_dir / "visual.json").write_text(json.dumps(visual.to_dict(), indent=2))

        return cache, cache_dir

    def test_spanish_language_in_vtt(self, tmp_path):
        cache, cache_dir = self._setup_video_with_language(tmp_path, language="es")
        result = compile_scene_descriptions("test123", output_base=tmp_path)

        assert result["status"] == "compiled"
        vtt_content = (cache_dir / "audio.ad.vtt").read_text()
        assert "Language: es" in vtt_content

    def test_none_language_defaults_to_english(self, tmp_path):
        cache, cache_dir = self._setup_video_with_language(tmp_path, language=None)
        result = compile_scene_descriptions("test123", output_base=tmp_path)

        assert result["status"] == "compiled"
        vtt_content = (cache_dir / "audio.ad.vtt").read_text()
        assert "Language: en" in vtt_content

    def test_unsupported_language_defaults_to_english(self, tmp_path):
        cache, cache_dir = self._setup_video_with_language(tmp_path, language="ko")
        result = compile_scene_descriptions("test123", output_base=tmp_path)

        assert result["status"] == "compiled"
        vtt_content = (cache_dir / "audio.ad.vtt").read_text()
        assert "Language: en" in vtt_content

    def test_regional_variant_normalized_in_vtt(self, tmp_path):
        cache, cache_dir = self._setup_video_with_language(tmp_path, language="pt-BR")
        result = compile_scene_descriptions("test123", output_base=tmp_path)

        assert result["status"] == "compiled"
        vtt_content = (cache_dir / "audio.ad.vtt").read_text()
        assert "Language: pt" in vtt_content


class TestModuleExports:
    """Tests for module exports from operations package."""

    def test_compile_exported(self):
        from claudetube.operations import compile_scene_descriptions

        assert callable(compile_scene_descriptions)

    def test_get_exported(self):
        from claudetube.operations import get_scene_descriptions

        assert callable(get_scene_descriptions)
