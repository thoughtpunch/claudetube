"""Tests for scene cache management."""

import json
from unittest.mock import patch

from claudetube.cache import (
    CacheManager,
    SceneBoundary,
    ScenesData,
    SceneStatus,
    get_all_scene_statuses,
    get_keyframes_dir,
    get_scene_dir,
    get_scene_status,
    get_scenes_dir,
    get_scenes_json_path,
    get_technical_json_path,
    get_visual_json_path,
    has_scenes,
    list_scene_keyframes,
    load_scenes_data,
    save_scenes_data,
)
from claudetube.models.state import VideoState


class TestSceneBoundary:
    """Tests for SceneBoundary dataclass."""

    def test_create_scene_boundary(self):
        scene = SceneBoundary(
            scene_id=0,
            start_time=0.0,
            end_time=30.5,
            title="Introduction",
            transcript_segment="Hello and welcome...",
        )
        assert scene.scene_id == 0
        assert scene.start_time == 0.0
        assert scene.end_time == 30.5
        assert scene.title == "Introduction"
        assert scene.transcript_segment == "Hello and welcome..."

    def test_duration_calculation(self):
        scene = SceneBoundary(scene_id=1, start_time=10.0, end_time=45.0)
        assert scene.duration() == 35.0

    def test_to_dict(self):
        scene = SceneBoundary(
            scene_id=2, start_time=60.0, end_time=120.0, title="Main Content"
        )
        d = scene.to_dict()
        assert d["scene_id"] == 2
        assert d["start_time"] == 60.0
        assert d["end_time"] == 120.0
        assert d["title"] == "Main Content"
        # transcript_segment is omitted when not set (sparse output)
        assert "transcript_segment" not in d

    def test_from_dict(self):
        d = {
            "scene_id": 3,
            "start_time": 90.0,
            "end_time": 150.0,
            "title": "Conclusion",
            "transcript_segment": "Thank you for watching",
        }
        scene = SceneBoundary.from_dict(d)
        assert scene.scene_id == 3
        assert scene.start_time == 90.0
        assert scene.end_time == 150.0
        assert scene.title == "Conclusion"
        assert scene.transcript_segment == "Thank you for watching"

    def test_from_dict_minimal(self):
        d = {"scene_id": 0, "start_time": 0.0, "end_time": 10.0}
        scene = SceneBoundary.from_dict(d)
        assert scene.scene_id == 0
        assert scene.title is None
        assert scene.transcript_segment is None


class TestScenesData:
    """Tests for ScenesData dataclass."""

    def test_create_scenes_data(self):
        scenes = [
            SceneBoundary(scene_id=0, start_time=0, end_time=30),
            SceneBoundary(scene_id=1, start_time=30, end_time=60),
        ]
        data = ScenesData(video_id="test123", method="transcript", scenes=scenes)
        assert data.video_id == "test123"
        assert data.method == "transcript"
        assert len(data.scenes) == 2

    def test_to_dict(self):
        scenes = [
            SceneBoundary(scene_id=0, start_time=0, end_time=30),
            SceneBoundary(scene_id=1, start_time=30, end_time=60),
        ]
        data = ScenesData(video_id="test123", method="visual", scenes=scenes)
        d = data.to_dict()
        assert d["video_id"] == "test123"
        assert d["method"] == "visual"
        assert d["scene_count"] == 2
        assert len(d["scenes"]) == 2

    def test_from_dict(self):
        d = {
            "video_id": "abc456",
            "method": "hybrid",
            "scenes": [
                {"scene_id": 0, "start_time": 0, "end_time": 15},
                {"scene_id": 1, "start_time": 15, "end_time": 30},
            ],
        }
        data = ScenesData.from_dict(d)
        assert data.video_id == "abc456"
        assert data.method == "hybrid"
        assert len(data.scenes) == 2
        assert data.scenes[0].scene_id == 0


class TestSceneStatus:
    """Tests for SceneStatus dataclass."""

    def test_default_status(self):
        status = SceneStatus()
        assert status.keyframes is False
        assert status.visual is False
        assert status.technical is False
        assert status.is_complete() is False

    def test_partial_status(self):
        status = SceneStatus(keyframes=True)
        assert status.keyframes is True
        assert status.visual is False
        assert status.is_complete() is False

    def test_complete_status(self):
        status = SceneStatus(keyframes=True, visual=True, technical=True)
        assert status.is_complete() is True

    def test_to_dict(self):
        status = SceneStatus(keyframes=True, visual=False, technical=True)
        d = status.to_dict()
        assert d == {"keyframes": True, "visual": False, "technical": True}


class TestSceneCacheHelpers:
    """Tests for scene cache helper functions."""

    def test_get_scenes_dir_creates_directory(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        scenes_dir = get_scenes_dir(cache_dir)
        assert scenes_dir.exists()
        assert scenes_dir.name == "scenes"
        assert scenes_dir.parent == cache_dir

    def test_get_scene_dir_creates_directory(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        scene_dir = get_scene_dir(cache_dir, 0)
        assert scene_dir.exists()
        assert scene_dir.name == "scene_000"

        scene_dir_5 = get_scene_dir(cache_dir, 5)
        assert scene_dir_5.name == "scene_005"

        scene_dir_42 = get_scene_dir(cache_dir, 42)
        assert scene_dir_42.name == "scene_042"

    def test_get_keyframes_dir_creates_directory(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        kf_dir = get_keyframes_dir(cache_dir, 3)
        assert kf_dir.exists()
        assert kf_dir.name == "keyframes"
        assert kf_dir.parent.name == "scene_003"

    def test_get_scenes_json_path(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        path = get_scenes_json_path(cache_dir)
        assert path == cache_dir / "scenes" / "scenes.json"
        # Note: Function creates scenes dir as side effect
        assert (cache_dir / "scenes").exists()

    def test_get_visual_json_path(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        path = get_visual_json_path(cache_dir, 2)
        assert path == cache_dir / "scenes" / "scene_002" / "visual.json"

    def test_get_technical_json_path(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        path = get_technical_json_path(cache_dir, 7)
        assert path == cache_dir / "scenes" / "scene_007" / "technical.json"

    def test_has_scenes_false_when_no_file(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()
        assert has_scenes(cache_dir) is False

    def test_has_scenes_true_when_file_exists(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()
        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir()
        (scenes_dir / "scenes.json").write_text("{}")
        assert has_scenes(cache_dir) is True


class TestSceneStatusDetection:
    """Tests for get_scene_status function."""

    def test_status_all_false_when_empty(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        status = get_scene_status(cache_dir, 0)
        assert status.keyframes is False
        assert status.visual is False
        assert status.technical is False

    def test_status_keyframes_true_when_has_files(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()
        kf_dir = cache_dir / "scenes" / "scene_000" / "keyframes"
        kf_dir.mkdir(parents=True)
        (kf_dir / "kf_000.jpg").write_bytes(b"fake")

        status = get_scene_status(cache_dir, 0)
        assert status.keyframes is True
        assert status.visual is False
        assert status.technical is False

    def test_status_visual_true_when_exists(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()
        scene_dir = cache_dir / "scenes" / "scene_001"
        scene_dir.mkdir(parents=True)
        (scene_dir / "visual.json").write_text("{}")

        status = get_scene_status(cache_dir, 1)
        assert status.keyframes is False
        assert status.visual is True
        assert status.technical is False

    def test_status_technical_true_when_exists(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()
        scene_dir = cache_dir / "scenes" / "scene_002"
        scene_dir.mkdir(parents=True)
        (scene_dir / "technical.json").write_text("{}")

        status = get_scene_status(cache_dir, 2)
        assert status.keyframes is False
        assert status.visual is False
        assert status.technical is True


class TestScenesDataPersistence:
    """Tests for saving and loading scenes data."""

    def test_save_and_load_scenes_data(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        scenes = [
            SceneBoundary(scene_id=0, start_time=0, end_time=30, title="Intro"),
            SceneBoundary(scene_id=1, start_time=30, end_time=90, title="Main"),
        ]
        data = ScenesData(video_id="video123", method="transcript", scenes=scenes)

        save_scenes_data(cache_dir, data)
        loaded = load_scenes_data(cache_dir)

        assert loaded is not None
        assert loaded.video_id == "video123"
        assert loaded.method == "transcript"
        assert len(loaded.scenes) == 2
        assert loaded.scenes[0].title == "Intro"

    def test_load_scenes_data_returns_none_when_missing(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        assert load_scenes_data(cache_dir) is None

    def test_load_scenes_data_returns_none_on_invalid_json(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()
        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir()
        (scenes_dir / "scenes.json").write_text("not valid json")

        assert load_scenes_data(cache_dir) is None


class TestListSceneKeyframes:
    """Tests for list_scene_keyframes function."""

    def test_returns_empty_when_no_dir(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        frames = list_scene_keyframes(cache_dir, 0)
        assert frames == []

    def test_returns_sorted_keyframes(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()
        kf_dir = cache_dir / "scenes" / "scene_000" / "keyframes"
        kf_dir.mkdir(parents=True)

        (kf_dir / "kf_002.jpg").write_bytes(b"frame2")
        (kf_dir / "kf_000.jpg").write_bytes(b"frame0")
        (kf_dir / "kf_001.jpg").write_bytes(b"frame1")

        frames = list_scene_keyframes(cache_dir, 0)
        assert len(frames) == 3
        assert frames[0].name == "kf_000.jpg"
        assert frames[1].name == "kf_001.jpg"
        assert frames[2].name == "kf_002.jpg"

    def test_only_matches_jpg_pattern(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()
        kf_dir = cache_dir / "scenes" / "scene_000" / "keyframes"
        kf_dir.mkdir(parents=True)

        (kf_dir / "kf_000.jpg").write_bytes(b"frame0")
        (kf_dir / "other.txt").write_text("not a frame")

        frames = list_scene_keyframes(cache_dir, 0)
        assert len(frames) == 1


class TestGetAllSceneStatuses:
    """Tests for get_all_scene_statuses function."""

    def test_returns_empty_dict_when_no_scenes(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        statuses = get_all_scene_statuses(cache_dir)
        assert statuses == {}

    def test_returns_statuses_for_all_scenes(self, tmp_path):
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        # Create scenes.json with 2 scenes
        scenes = [
            SceneBoundary(scene_id=0, start_time=0, end_time=30),
            SceneBoundary(scene_id=1, start_time=30, end_time=60),
        ]
        data = ScenesData(video_id="video123", method="transcript", scenes=scenes)
        save_scenes_data(cache_dir, data)

        # Create keyframes for scene 0 only
        kf_dir = cache_dir / "scenes" / "scene_000" / "keyframes"
        kf_dir.mkdir(parents=True)
        (kf_dir / "kf_000.jpg").write_bytes(b"frame")

        statuses = get_all_scene_statuses(cache_dir)
        assert len(statuses) == 2
        assert statuses[0].keyframes is True
        assert statuses[1].keyframes is False


class TestVideoStateSceneFields:
    """Tests for scene-related fields in VideoState."""

    def test_default_scene_fields(self):
        state = VideoState(video_id="test123")
        assert state.scenes_processed is False
        assert state.scenes_method is None
        assert state.scene_count is None
        assert state.visual_transcripts_complete is False
        assert state.technical_extraction_complete is False

    def test_scene_fields_in_to_dict(self):
        state = VideoState(
            video_id="test123",
            scenes_processed=True,
            scenes_method="transcript",
            scene_count=5,
            visual_transcripts_complete=True,
            technical_extraction_complete=False,
        )
        d = state.to_dict()
        assert d["scenes_processed"] is True
        assert d["scenes_method"] == "transcript"
        assert d["scene_count"] == 5
        assert d["visual_transcripts_complete"] is True
        assert d["technical_extraction_complete"] is False

    def test_scene_fields_from_dict(self):
        d = {
            "video_id": "test123",
            "scenes_processed": True,
            "scenes_method": "hybrid",
            "scene_count": 10,
            "visual_transcripts_complete": False,
            "technical_extraction_complete": True,
        }
        state = VideoState.from_dict(d)
        assert state.scenes_processed is True
        assert state.scenes_method == "hybrid"
        assert state.scene_count == 10
        assert state.visual_transcripts_complete is False
        assert state.technical_extraction_complete is True

    def test_scene_fields_backward_compatible(self):
        """Old state.json without scene fields should load with defaults."""
        d = {
            "video_id": "test123",
            "title": "Old Video",
            "transcript_complete": True,
            # No scene fields
        }
        state = VideoState.from_dict(d)
        assert state.scenes_processed is False
        assert state.scenes_method is None
        assert state.scene_count is None


class TestCacheManagerSceneMethods:
    """Tests for scene methods on CacheManager."""

    def test_get_scenes_dir(self, tmp_path):
        manager = CacheManager(cache_base=tmp_path)
        scenes_dir = manager.get_scenes_dir("video123")
        assert scenes_dir.exists()
        assert scenes_dir == tmp_path / "video123" / "scenes"

    def test_get_scene_dir(self, tmp_path):
        manager = CacheManager(cache_base=tmp_path)
        scene_dir = manager.get_scene_dir("video123", 5)
        assert scene_dir.exists()
        assert scene_dir.name == "scene_005"

    def test_get_keyframes_dir(self, tmp_path):
        manager = CacheManager(cache_base=tmp_path)
        kf_dir = manager.get_keyframes_dir("video123", 3)
        assert kf_dir.exists()
        assert kf_dir.name == "keyframes"

    def test_has_scenes(self, tmp_path):
        manager = CacheManager(cache_base=tmp_path)
        assert manager.has_scenes("video123") is False

        # Create scenes.json
        scenes_dir = tmp_path / "video123" / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        (scenes_dir / "scenes.json").write_text("{}")

        assert manager.has_scenes("video123") is True

    def test_save_and_load_scenes_data(self, tmp_path):
        manager = CacheManager(cache_base=tmp_path)
        (tmp_path / "video123").mkdir()

        scenes = [SceneBoundary(scene_id=0, start_time=0, end_time=30)]
        data = ScenesData(video_id="video123", method="transcript", scenes=scenes)

        manager.save_scenes_data("video123", data)
        loaded = manager.load_scenes_data("video123")

        assert loaded is not None
        assert loaded.video_id == "video123"

    def test_get_scene_status(self, tmp_path):
        manager = CacheManager(cache_base=tmp_path)
        status = manager.get_scene_status("video123", 0)
        assert status.keyframes is False

    def test_list_scene_keyframes(self, tmp_path):
        manager = CacheManager(cache_base=tmp_path)
        kf_dir = tmp_path / "video123" / "scenes" / "scene_000" / "keyframes"
        kf_dir.mkdir(parents=True)
        (kf_dir / "kf_000.jpg").write_bytes(b"frame")

        frames = manager.list_scene_keyframes("video123", 0)
        assert len(frames) == 1

    def test_get_all_scene_statuses(self, tmp_path):
        manager = CacheManager(cache_base=tmp_path)
        (tmp_path / "video123").mkdir()

        scenes = [
            SceneBoundary(scene_id=0, start_time=0, end_time=30),
            SceneBoundary(scene_id=1, start_time=30, end_time=60),
        ]
        data = ScenesData(video_id="video123", method="transcript", scenes=scenes)
        manager.save_scenes_data("video123", data)

        statuses = manager.get_all_scene_statuses("video123")
        assert len(statuses) == 2


class TestGetScenesSyncEnrich:
    """Tests for _get_scenes_sync enrich parameter."""

    def _setup_cached_video(self, tmp_path):
        """Create a minimal cached video with scenes."""
        cache_dir = tmp_path / "video123"
        cache_dir.mkdir()

        # Create state.json
        state = {"video_id": "video123", "duration": 60, "description": "Test video"}
        (cache_dir / "state.json").write_text(json.dumps(state))

        # Create scenes
        scenes = [
            SceneBoundary(scene_id=0, start_time=0, end_time=30, title="Intro"),
            SceneBoundary(scene_id=1, start_time=30, end_time=60, title="Main"),
        ]
        data = ScenesData(video_id="video123", method="transcript", scenes=scenes)
        save_scenes_data(cache_dir, data)

        return cache_dir

    @patch("claudetube.mcp_server.get_cache_dir")
    def test_enrich_false_does_not_call_visual_transcript(
        self, mock_cache_dir, tmp_path
    ):
        """When enrich=False (default), visual transcript generation is not triggered."""
        self._setup_cached_video(tmp_path)
        mock_cache_dir.return_value = tmp_path

        from claudetube.mcp_server import _get_scenes_sync

        with patch(
            "claudetube.operations.visual_transcript.generate_visual_transcript"
        ) as mock_gen:
            result = _get_scenes_sync("video123", force=False, enrich=False)

        mock_gen.assert_not_called()
        assert "error" not in result
        assert len(result["scenes"]) == 2

    @patch("claudetube.mcp_server.get_cache_dir")
    def test_enrich_true_calls_visual_transcript(self, mock_cache_dir, tmp_path):
        """When enrich=True, generate_visual_transcript is called."""
        self._setup_cached_video(tmp_path)
        mock_cache_dir.return_value = tmp_path

        from claudetube.mcp_server import _get_scenes_sync

        with patch(
            "claudetube.operations.visual_transcript.generate_visual_transcript",
            return_value={"results": [], "errors": [], "generated": 0, "skipped": 2},
        ) as mock_gen:
            result = _get_scenes_sync("video123", force=False, enrich=True)

        mock_gen.assert_called_once_with(video_id="video123", output_base=tmp_path)
        assert "error" not in result
        assert len(result["scenes"]) == 2

    @patch("claudetube.mcp_server.get_cache_dir")
    def test_enrich_true_includes_visual_data_in_result(self, mock_cache_dir, tmp_path):
        """When enrich=True, visual data is included in scene dicts."""
        cache_dir = self._setup_cached_video(tmp_path)
        mock_cache_dir.return_value = tmp_path

        # Write visual.json for scene 0
        scene_dir = cache_dir / "scenes" / "scene_000"
        scene_dir.mkdir(parents=True, exist_ok=True)
        visual_data = {
            "scene_id": 0,
            "description": "A person talking",
            "people": ["host"],
        }
        (scene_dir / "visual.json").write_text(json.dumps(visual_data))

        from claudetube.mcp_server import _get_scenes_sync

        with patch(
            "claudetube.operations.visual_transcript.generate_visual_transcript",
            return_value={"results": [], "errors": [], "generated": 0, "skipped": 2},
        ):
            result = _get_scenes_sync("video123", force=False, enrich=True)

        # Scene 0 should have visual data attached
        scene_0 = result["scenes"][0]
        assert "visual" in scene_0
        assert scene_0["visual"]["description"] == "A person talking"

        # Scene 1 should not have visual data
        scene_1 = result["scenes"][1]
        assert "visual" not in scene_1

    @patch("claudetube.mcp_server.get_cache_dir")
    def test_enrich_default_is_false(self, mock_cache_dir, tmp_path):
        """The enrich parameter defaults to False."""
        self._setup_cached_video(tmp_path)
        mock_cache_dir.return_value = tmp_path

        from claudetube.mcp_server import _get_scenes_sync

        with patch(
            "claudetube.operations.visual_transcript.generate_visual_transcript"
        ) as mock_gen:
            result = _get_scenes_sync("video123")

        mock_gen.assert_not_called()
        assert "error" not in result
