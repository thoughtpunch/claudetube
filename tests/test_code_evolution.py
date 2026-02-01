"""
Tests for code evolution tracking.
"""

import json

import pytest

from claudetube.operations.code_evolution import (
    CodeEvolutionData,
    CodeSnapshot,
    CodeUnit,
    detect_change_type,
    get_code_evolution_path,
    identify_code_unit,
    query_code_evolution,
    track_code_evolution,
)


class TestIdentifyCodeUnit:
    """Tests for code unit identification."""

    def test_python_function(self):
        """Should identify Python function."""
        code = """def validate_token(token):
    if not token:
        return False
    return True"""
        unit_id, unit_type, name = identify_code_unit(code)
        assert unit_id == "function:validate_token"
        assert unit_type == "function"
        assert name == "validate_token"

    def test_javascript_function(self):
        """Should identify JavaScript function."""
        code = """function fetchData(url) {
    return fetch(url).then(r => r.json());
}"""
        unit_id, unit_type, name = identify_code_unit(code)
        assert unit_id == "function:fetchData"
        assert unit_type == "function"
        assert name == "fetchData"

    def test_javascript_arrow_function(self):
        """Should identify JS arrow function."""
        code = """const handleClick = async (event) => {
    event.preventDefault();
};"""
        unit_id, unit_type, name = identify_code_unit(code)
        assert unit_id == "function:handleClick"
        assert unit_type == "function"
        assert name == "handleClick"

    def test_python_class(self):
        """Should identify Python class."""
        code = """class UserAuthenticator:
    def __init__(self):
        pass"""
        unit_id, unit_type, name = identify_code_unit(code)
        assert unit_id == "class:UserAuthenticator"
        assert unit_type == "class"
        assert name == "UserAuthenticator"

    def test_rust_function(self):
        """Should identify Rust function."""
        code = """fn process_request(req: Request) -> Response {
    // handle request
}"""
        unit_id, unit_type, name = identify_code_unit(code)
        assert unit_id == "function:process_request"
        assert unit_type == "function"
        assert name == "process_request"

    def test_go_function(self):
        """Should identify Go function."""
        code = """func HandleRequest(w http.ResponseWriter, r *http.Request) {
    w.Write([]byte("Hello"))
}"""
        unit_id, unit_type, name = identify_code_unit(code)
        assert unit_id == "function:HandleRequest"
        assert unit_type == "function"
        assert name == "HandleRequest"

    def test_rust_struct(self):
        """Should identify Rust struct."""
        code = """struct Config {
    timeout: u64,
    retries: u32,
}"""
        unit_id, unit_type, name = identify_code_unit(code)
        assert unit_id == "struct:Config"
        assert unit_type == "struct"
        assert name == "Config"

    def test_typescript_interface(self):
        """Should identify TypeScript interface."""
        code = """interface UserData {
    id: string;
    name: string;
}"""
        unit_id, unit_type, name = identify_code_unit(code)
        assert unit_id == "interface:UserData"
        assert unit_type == "interface"
        assert name == "UserData"

    def test_snippet_fallback(self):
        """Should create snippet ID for unidentifiable code."""
        code = """x = 1
y = 2
print(x + y)"""
        unit_id, unit_type, name = identify_code_unit(code)
        assert unit_type == "snippet"
        assert unit_id.startswith("snippet:")


class TestDetectChangeType:
    """Tests for change detection."""

    def test_unchanged(self):
        """Should detect unchanged code."""
        old = "def foo():\n    pass"
        new = "def foo():\n    pass"
        change_type, summary = detect_change_type(old, new)
        assert change_type == "unchanged"
        assert summary is None

    def test_added_lines(self):
        """Should detect added lines."""
        old = "def foo():\n    pass"
        new = "def foo():\n    x = 1\n    pass"
        change_type, summary = detect_change_type(old, new)
        assert change_type == "added_lines"
        assert "+1 lines" in summary

    def test_removed_lines(self):
        """Should detect removed lines."""
        old = "def foo():\n    x = 1\n    y = 2\n    pass"
        new = "def foo():\n    pass"
        change_type, summary = detect_change_type(old, new)
        assert change_type == "removed_lines"
        assert "-2 lines" in summary

    def test_modified(self):
        """Should detect modifications (both adds and removes)."""
        old = "def foo():\n    x = 1\n    pass"
        new = "def foo():\n    y = 2\n    return"
        change_type, summary = detect_change_type(old, new)
        assert change_type == "modified"
        assert "+" in summary and "-" in summary


class TestCodeSnapshot:
    """Tests for CodeSnapshot dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        snapshot = CodeSnapshot(
            scene_id=1,
            timestamp=120.5,
            content="def foo(): pass",
            language="python",
            change_type="shown",
            diff_summary=None,
        )
        data = snapshot.to_dict()
        assert data["scene_id"] == 1
        assert data["timestamp"] == 120.5
        assert data["content"] == "def foo(): pass"
        assert data["language"] == "python"
        assert data["change_type"] == "shown"

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "scene_id": 2,
            "timestamp": 245.0,
            "content": "def bar(): return 1",
            "language": "python",
            "change_type": "modified",
            "diff_summary": "+1 lines",
        }
        snapshot = CodeSnapshot.from_dict(data)
        assert snapshot.scene_id == 2
        assert snapshot.timestamp == 245.0
        assert snapshot.change_type == "modified"


class TestCodeUnit:
    """Tests for CodeUnit dataclass."""

    def test_properties(self):
        """Should calculate derived properties correctly."""
        unit = CodeUnit(
            unit_id="function:test",
            unit_type="function",
            name="test",
            snapshots=[
                CodeSnapshot(1, 10.0, "v1", "python", "shown", None),
                CodeSnapshot(2, 20.0, "v2", "python", "modified", "+1"),
                CodeSnapshot(3, 30.0, "v2", "python", "unchanged", None),
                CodeSnapshot(4, 40.0, "v3", "python", "added_lines", "+2"),
            ],
        )
        assert unit.first_seen == 10.0
        assert unit.last_seen == 40.0
        assert unit.change_count == 2  # modified + added_lines

    def test_roundtrip(self):
        """Should serialize and deserialize correctly."""
        unit = CodeUnit(
            unit_id="class:User",
            unit_type="class",
            name="User",
            snapshots=[
                CodeSnapshot(0, 5.0, "class User: pass", "python", "shown", None),
            ],
        )
        data = unit.to_dict()
        restored = CodeUnit.from_dict(data)
        assert restored.unit_id == unit.unit_id
        assert restored.unit_type == unit.unit_type
        assert len(restored.snapshots) == 1


class TestCodeEvolutionData:
    """Tests for CodeEvolutionData container."""

    def test_summary_generation(self):
        """Should generate accurate summary."""
        data = CodeEvolutionData(
            video_id="test123",
            method="technical_json",
            code_units=[
                CodeUnit(
                    "function:foo", "function", "foo",
                    [
                        CodeSnapshot(0, 0, "v1", "python", "shown", None),
                        CodeSnapshot(1, 10, "v2", "python", "modified", "+1"),
                    ],
                ),
                CodeUnit(
                    "class:Bar", "class", "Bar",
                    [CodeSnapshot(0, 0, "v1", "python", "shown", None)],
                ),
                CodeUnit(
                    "function:baz", "function", "baz",
                    [
                        CodeSnapshot(0, 0, "v1", "python", "shown", None),
                        CodeSnapshot(1, 10, "v2", "python", "modified", "+1"),
                        CodeSnapshot(2, 20, "v3", "python", "modified", "+2"),
                    ],
                ),
            ],
        )
        result = data.to_dict()

        assert result["unit_count"] == 3
        assert result["summary"]["total_units"] == 3
        assert result["summary"]["by_type"]["function"] == 2
        assert result["summary"]["by_type"]["class"] == 1

        # Most modified should be baz (2 changes), then foo (1 change)
        most_modified = result["summary"]["most_modified"]
        assert len(most_modified) == 2
        assert most_modified[0]["name"] == "baz"
        assert most_modified[0]["change_count"] == 2

    def test_roundtrip(self):
        """Should serialize and deserialize correctly."""
        original = CodeEvolutionData(
            video_id="vid123",
            method="technical_json",
            code_units=[
                CodeUnit(
                    "function:test", "function", "test",
                    [CodeSnapshot(0, 0, "code", "python", "shown", None)],
                ),
            ],
        )
        data = original.to_dict()
        restored = CodeEvolutionData.from_dict(data)

        assert restored.video_id == original.video_id
        assert restored.method == original.method
        assert len(restored.code_units) == 1
        assert restored.code_units[0].unit_id == "function:test"


class TestGetCodeEvolutionPath:
    """Tests for cache path generation."""

    def test_creates_entities_dir(self, tmp_path):
        """Should create entities directory if needed."""
        path = get_code_evolution_path(tmp_path)
        assert path.parent.name == "entities"
        assert path.parent.exists()
        assert path.name == "code_evolution.json"


class TestTrackCodeEvolution:
    """Integration tests for track_code_evolution."""

    def test_returns_error_for_uncached_video(self, tmp_path):
        """Should return error if video not cached."""
        result = track_code_evolution("nonexistent", output_base=tmp_path)
        assert "error" in result
        assert "not cached" in result["error"]

    def test_caches_results(self, tmp_path):
        """Should cache results and return from cache on second call."""
        # Set up minimal cache structure
        video_id = "test_video"
        cache_dir = tmp_path / video_id
        cache_dir.mkdir(parents=True)

        # Create state.json
        (cache_dir / "state.json").write_text(json.dumps({"video_id": video_id}))

        # Create scenes data
        scenes_dir = cache_dir / "scenes"
        scenes_dir.mkdir()
        (scenes_dir / "scenes.json").write_text(json.dumps({
            "video_id": video_id,
            "method": "transcript",
            "scenes": [
                {"scene_id": 0, "start_time": 0, "end_time": 30, "title": "Intro"},
            ],
        }))

        # Create scene directory with technical.json
        scene_dir = scenes_dir / "scene_000"
        scene_dir.mkdir()
        (scene_dir / "technical.json").write_text(json.dumps({
            "version": 1,
            "frames": [
                {
                    "timestamp": 10.0,
                    "code_blocks": [
                        {"content": "def hello():\n    print('hi')", "language": "python"},
                    ],
                },
            ],
        }))

        # First call - generates data
        result1 = track_code_evolution(video_id, output_base=tmp_path)
        assert "error" not in result1
        assert result1["unit_count"] == 1

        # Verify cached
        evolution_path = get_code_evolution_path(cache_dir)
        assert evolution_path.exists()

        # Second call - from cache
        result2 = track_code_evolution(video_id, output_base=tmp_path)
        assert result2 == result1


class TestQueryCodeEvolution:
    """Tests for querying code evolution."""

    def test_query_returns_matching_units(self, tmp_path):
        """Should return units matching query."""
        # Set up cached evolution data
        video_id = "query_test"
        cache_dir = tmp_path / video_id
        entities_dir = cache_dir / "entities"
        entities_dir.mkdir(parents=True)

        evolution_data = {
            "video_id": video_id,
            "method": "technical_json",
            "code_units": {
                "function:validate_token": {
                    "unit_id": "function:validate_token",
                    "unit_type": "function",
                    "name": "validate_token",
                    "snapshots": [],
                },
                "function:refresh_token": {
                    "unit_id": "function:refresh_token",
                    "unit_type": "function",
                    "name": "refresh_token",
                    "snapshots": [],
                },
                "class:User": {
                    "unit_id": "class:User",
                    "unit_type": "class",
                    "name": "User",
                    "snapshots": [],
                },
            },
        }
        (entities_dir / "code_evolution.json").write_text(json.dumps(evolution_data))

        # Query for "token"
        result = query_code_evolution(video_id, "token", output_base=tmp_path)
        assert "error" not in result
        assert result["match_count"] == 2  # validate_token and refresh_token

        # Query for "User"
        result = query_code_evolution(video_id, "User", output_base=tmp_path)
        assert result["match_count"] == 1

    def test_query_returns_error_for_missing_data(self, tmp_path):
        """Should return error if no evolution data."""
        result = query_code_evolution("nonexistent", "foo", output_base=tmp_path)
        assert "error" in result
