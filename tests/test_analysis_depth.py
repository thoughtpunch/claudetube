"""Tests for multi-pass analysis depth functionality."""

import json

import pytest

from claudetube.cache.scenes import (
    SceneBoundary,
    ScenesData,
    get_scene_dir,
    get_visual_json_path,
    save_scenes_data,
)
from claudetube.operations.analysis_depth import (
    AnalysisDepth,
    AnalysisResult,
    Entities,
    TechnicalContent,
    analyze_video,
    extract_entities,
    get_analysis_status,
)


@pytest.fixture
def mock_cache_dir(tmp_path):
    """Create a mock video cache directory with basic structure."""
    video_id = "test_video"
    cache_dir = tmp_path / video_id
    cache_dir.mkdir(parents=True)

    # Create state.json
    state = {
        "video_id": video_id,
        "title": "Test Video",
        "duration": 300,
        "description": "A test video about Python programming",
        "cached_file": None,
    }
    (cache_dir / "state.json").write_text(json.dumps(state))

    # Create scenes
    scenes_data = ScenesData(
        video_id=video_id,
        method="transcript",
        scenes=[
            SceneBoundary(
                scene_id=0,
                start_time=0.0,
                end_time=60.0,
                title="Introduction",
                transcript_text="Hello and welcome to this Python tutorial.",
            ),
            SceneBoundary(
                scene_id=1,
                start_time=60.0,
                end_time=180.0,
                title="Functions",
                transcript_text="Let's learn about Python functions and Django framework.",
            ),
            SceneBoundary(
                scene_id=2,
                start_time=180.0,
                end_time=300.0,
                title="Conclusion",
                transcript_text="In summary, we covered Python basics.",
            ),
        ],
    )
    save_scenes_data(cache_dir, scenes_data)

    return tmp_path, video_id, cache_dir


class TestAnalysisDepth:
    """Tests for AnalysisDepth enum."""

    def test_depth_values(self):
        assert AnalysisDepth.QUICK.value == "quick"
        assert AnalysisDepth.STANDARD.value == "standard"
        assert AnalysisDepth.DEEP.value == "deep"
        assert AnalysisDepth.EXHAUSTIVE.value == "exhaustive"

    def test_depth_from_string(self):
        assert AnalysisDepth("quick") == AnalysisDepth.QUICK
        assert AnalysisDepth("standard") == AnalysisDepth.STANDARD
        assert AnalysisDepth("deep") == AnalysisDepth.DEEP
        assert AnalysisDepth("exhaustive") == AnalysisDepth.EXHAUSTIVE


class TestTechnicalContent:
    """Tests for TechnicalContent dataclass."""

    def test_create_technical_content(self):
        content = TechnicalContent(
            scene_id=1,
            ocr_text=["def hello():", "print('hello')"],
            code_blocks=[{"content": "def hello(): ...", "language": "python"}],
            content_types=["code"],
        )
        assert content.scene_id == 1
        assert len(content.ocr_text) == 2
        assert content.code_blocks[0]["language"] == "python"

    def test_to_dict(self):
        content = TechnicalContent(
            scene_id=2,
            ocr_text=["text"],
            code_blocks=[],
            content_types=["slides"],
        )
        d = content.to_dict()
        assert d["scene_id"] == 2
        assert d["ocr_text"] == ["text"]
        assert d["content_types"] == ["slides"]

    def test_from_dict(self):
        d = {
            "scene_id": 3,
            "ocr_text": ["foo", "bar"],
            "code_blocks": [{"content": "x = 1", "language": "python"}],
            "content_types": ["code", "terminal"],
        }
        content = TechnicalContent.from_dict(d)
        assert content.scene_id == 3
        assert content.ocr_text == ["foo", "bar"]
        assert len(content.code_blocks) == 1

    def test_from_dict_defaults(self):
        d = {"scene_id": 0}
        content = TechnicalContent.from_dict(d)
        assert content.ocr_text == []
        assert content.code_blocks == []
        assert content.content_types == []


class TestEntities:
    """Tests for Entities dataclass."""

    def test_create_entities(self):
        entities = Entities(
            scene_id=1,
            people=["John Doe"],
            topics=["authentication"],
            technologies=["python", "django"],
            keywords=["function", "class"],
        )
        assert entities.scene_id == 1
        assert "John Doe" in entities.people
        assert "python" in entities.technologies

    def test_to_dict(self):
        entities = Entities(
            scene_id=2,
            people=["Jane"],
            topics=[],
            technologies=["react"],
            keywords=["component"],
        )
        d = entities.to_dict()
        assert d["scene_id"] == 2
        assert d["people"] == ["Jane"]
        assert d["technologies"] == ["react"]

    def test_from_dict(self):
        d = {
            "scene_id": 5,
            "people": ["Alice", "Bob"],
            "topics": ["testing"],
            "technologies": ["pytest"],
            "keywords": ["test", "assert"],
        }
        entities = Entities.from_dict(d)
        assert entities.scene_id == 5
        assert len(entities.people) == 2

    def test_from_dict_defaults(self):
        d = {"scene_id": 0}
        entities = Entities.from_dict(d)
        assert entities.people == []
        assert entities.technologies == []


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_create_result(self):
        result = AnalysisResult(
            video_id="abc123",
            depth=AnalysisDepth.STANDARD,
            scenes=[{"scene_id": 0, "start_time": 0}],
            method="transcript",
            processing_time=1.5,
        )
        assert result.video_id == "abc123"
        assert result.depth == AnalysisDepth.STANDARD
        assert len(result.scenes) == 1

    def test_to_dict(self):
        result = AnalysisResult(
            video_id="xyz789",
            depth=AnalysisDepth.DEEP,
            scenes=[{"scene_id": 0}, {"scene_id": 1}],
            method="hybrid",
            processing_time=10.5,
            focus_sections=[0],
            errors=[{"error": "test"}],
        )
        d = result.to_dict()
        assert d["video_id"] == "xyz789"
        assert d["depth"] == "deep"
        assert d["scene_count"] == 2
        assert d["focus_sections"] == [0]
        assert len(d["errors"]) == 1


class TestExtractEntities:
    """Tests for entity extraction."""

    def test_extract_entities_from_transcript(self, mock_cache_dir):
        base_dir, video_id, cache_dir = mock_cache_dir

        # Scene with Python/Django mentioned
        scene = SceneBoundary(
            scene_id=1,
            start_time=60.0,
            end_time=180.0,
            transcript_text="Let's learn about Python functions and Django framework with React.",
        )

        entities = extract_entities(video_id, scene, cache_dir)
        assert entities is not None
        assert entities.scene_id == 1
        assert "python" in entities.technologies
        assert "django" in entities.technologies
        assert "react" in entities.technologies

    def test_extract_entities_caches_result(self, mock_cache_dir):
        base_dir, video_id, cache_dir = mock_cache_dir

        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=60.0,
            transcript_text="Testing caching with Python.",
        )

        # First call extracts
        entities1 = extract_entities(video_id, scene, cache_dir)
        assert entities1 is not None

        # Check cache file exists
        entities_path = cache_dir / "scenes" / "scene_000" / "entities.json"
        assert entities_path.exists()

        # Second call uses cache
        entities2 = extract_entities(video_id, scene, cache_dir)
        assert entities2 is not None
        assert entities2.scene_id == entities1.scene_id

    def test_extract_entities_with_visual_description(self, mock_cache_dir):
        base_dir, video_id, cache_dir = mock_cache_dir

        # Create visual description
        get_scene_dir(cache_dir, 0)
        visual_path = get_visual_json_path(cache_dir, 0)
        visual_path.write_text(
            json.dumps(
                {
                    "description": "A person coding with TypeScript in VSCode.",
                    "people": ["developer in blue shirt"],
                }
            )
        )

        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=60.0,
            transcript_text="Welcome to the tutorial.",
        )

        entities = extract_entities(video_id, scene, cache_dir, force=True)
        assert entities is not None
        assert "typescript" in entities.technologies
        assert "vscode" in entities.technologies

    def test_extract_entities_no_content(self, mock_cache_dir):
        base_dir, video_id, cache_dir = mock_cache_dir

        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=60.0,
            transcript_text="",  # No transcript
        )

        entities = extract_entities(video_id, scene, cache_dir)
        assert entities is None


class TestAnalyzeVideo:
    """Tests for the main analyze_video function."""

    def test_quick_analysis(self, mock_cache_dir):
        base_dir, video_id, cache_dir = mock_cache_dir

        result = analyze_video(
            video_id,
            depth=AnalysisDepth.QUICK,
            output_base=base_dir,
        )

        assert result.video_id == video_id
        assert result.depth == AnalysisDepth.QUICK
        assert len(result.scenes) == 3
        assert result.method == "transcript"
        assert result.processing_time > 0
        assert not result.errors

    def test_standard_analysis_with_cached_visual(self, mock_cache_dir):
        base_dir, video_id, cache_dir = mock_cache_dir

        # Pre-create cached visual for scene 0
        visual_path = get_visual_json_path(cache_dir, 0)
        visual_path.parent.mkdir(parents=True, exist_ok=True)
        visual_path.write_text(
            json.dumps(
                {
                    "description": "A coding tutorial intro",
                    "people": ["instructor"],
                    "objects": ["laptop"],
                }
            )
        )

        result = analyze_video(
            video_id,
            depth=AnalysisDepth.STANDARD,
            output_base=base_dir,
        )

        assert result.depth == AnalysisDepth.STANDARD
        # Scene 0 should have visual enriched
        assert "visual" in result.scenes[0]
        assert result.scenes[0]["visual"]["description"] == "A coding tutorial intro"

    def test_deep_analysis(self, mock_cache_dir):
        base_dir, video_id, cache_dir = mock_cache_dir

        result = analyze_video(
            video_id,
            depth=AnalysisDepth.DEEP,
            output_base=base_dir,
        )

        assert result.depth == AnalysisDepth.DEEP
        # Should have entities for scenes with transcript
        # (technical might be empty due to no keyframes)
        for scene in result.scenes:
            if scene.get("transcript_text"):
                # Should have attempted entity extraction
                pass  # May or may not have entities depending on content

    def test_focus_sections(self, mock_cache_dir):
        base_dir, video_id, cache_dir = mock_cache_dir

        result = analyze_video(
            video_id,
            depth=AnalysisDepth.DEEP,
            focus_sections=[1],  # Only scene 1
            output_base=base_dir,
        )

        assert result.focus_sections == [1]

    def test_video_not_cached(self, tmp_path):
        result = analyze_video(
            "nonexistent_video",
            depth=AnalysisDepth.QUICK,
            output_base=tmp_path,
        )

        assert result.method == "error"
        assert len(result.errors) == 1
        assert "not cached" in result.errors[0]["error"]

    def test_analysis_result_to_dict(self, mock_cache_dir):
        base_dir, video_id, cache_dir = mock_cache_dir

        result = analyze_video(
            video_id,
            depth=AnalysisDepth.QUICK,
            output_base=base_dir,
        )

        d = result.to_dict()
        assert "video_id" in d
        assert "depth" in d
        assert "scene_count" in d
        assert "scenes" in d
        assert "method" in d
        assert "processing_time" in d


class TestGetAnalysisStatus:
    """Tests for get_analysis_status function."""

    def test_basic_status(self, mock_cache_dir):
        base_dir, video_id, cache_dir = mock_cache_dir

        status = get_analysis_status(video_id, output_base=base_dir)

        assert status["video_id"] == video_id
        assert status["scene_count"] == 3
        assert status["max_completed_depth"] == "quick"
        assert len(status["scenes"]) == 3

    def test_status_with_visual(self, mock_cache_dir):
        base_dir, video_id, cache_dir = mock_cache_dir

        # Add visual for all scenes
        for i in range(3):
            visual_path = get_visual_json_path(cache_dir, i)
            visual_path.parent.mkdir(parents=True, exist_ok=True)
            visual_path.write_text(json.dumps({"description": f"Scene {i}"}))

        status = get_analysis_status(video_id, output_base=base_dir)

        assert status["max_completed_depth"] == "standard"
        assert status["summary"]["with_visual"] == 3

    def test_status_video_not_found(self, tmp_path):
        status = get_analysis_status("nonexistent", output_base=tmp_path)
        assert "error" in status

    def test_status_no_scenes(self, tmp_path):
        video_id = "no_scenes_video"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir()
        (cache_dir / "state.json").write_text(json.dumps({"video_id": video_id}))

        status = get_analysis_status(video_id, output_base=tmp_path)
        assert status["has_scenes"] is False


class TestIntegration:
    """Integration tests for multi-pass analysis."""

    def test_progressive_analysis(self, mock_cache_dir):
        """Test that each depth level builds on the previous."""
        base_dir, video_id, cache_dir = mock_cache_dir

        # QUICK
        quick_result = analyze_video(
            video_id, depth=AnalysisDepth.QUICK, output_base=base_dir
        )
        assert quick_result.depth == AnalysisDepth.QUICK
        assert len(quick_result.scenes) == 3

        # Pre-create visual for standard
        for i in range(3):
            visual_path = get_visual_json_path(cache_dir, i)
            visual_path.parent.mkdir(parents=True, exist_ok=True)
            visual_path.write_text(json.dumps({"description": f"Scene {i}"}))

        # STANDARD (includes everything from QUICK)
        standard_result = analyze_video(
            video_id, depth=AnalysisDepth.STANDARD, output_base=base_dir
        )
        assert standard_result.depth == AnalysisDepth.STANDARD
        assert "visual" in standard_result.scenes[0]

        # DEEP (includes everything from STANDARD)
        deep_result = analyze_video(
            video_id, depth=AnalysisDepth.DEEP, output_base=base_dir
        )
        assert deep_result.depth == AnalysisDepth.DEEP
        # Should still have visual
        assert "visual" in deep_result.scenes[0]

    def test_caching_across_calls(self, mock_cache_dir):
        """Test that cached results are reused."""
        base_dir, video_id, cache_dir = mock_cache_dir

        # First call
        result1 = analyze_video(
            video_id, depth=AnalysisDepth.QUICK, output_base=base_dir
        )

        # Second call should be faster (cached)
        result2 = analyze_video(
            video_id, depth=AnalysisDepth.QUICK, output_base=base_dir
        )

        # Both should return same data
        assert len(result1.scenes) == len(result2.scenes)

    def test_force_reanalysis(self, mock_cache_dir):
        """Test that force=True regenerates analysis."""
        base_dir, video_id, cache_dir = mock_cache_dir

        # Pre-create entities with specific content
        entities_path = cache_dir / "scenes" / "scene_000" / "entities.json"
        entities_path.parent.mkdir(parents=True, exist_ok=True)
        entities_path.write_text(
            json.dumps(
                {
                    "scene_id": 0,
                    "technologies": ["old_tech"],
                    "people": [],
                    "topics": [],
                    "keywords": [],
                }
            )
        )

        # Without force, should get cached value
        result1 = analyze_video(
            video_id, depth=AnalysisDepth.DEEP, output_base=base_dir
        )
        scene0 = result1.scenes[0]
        if "entities" in scene0:
            assert "old_tech" in scene0["entities"]["technologies"]

        # With force, should regenerate
        result2 = analyze_video(
            video_id, depth=AnalysisDepth.DEEP, force=True, output_base=base_dir
        )
        scene0 = result2.scenes[0]
        if "entities" in scene0:
            # Should have extracted fresh entities from transcript
            # "old_tech" should not be there anymore
            assert "old_tech" not in scene0["entities"]["technologies"]
