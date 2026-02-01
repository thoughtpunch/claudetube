"""
Tests for project-level .claudetube/config.yaml support.
"""

from pathlib import Path


class TestProjectConfig:
    """Tests for project-level configuration."""

    def test_find_project_config_in_current_dir(self, tmp_path, monkeypatch):
        """Should find config in current directory."""
        from claudetube.config.loader import _find_project_config

        # Create .claudetube/config.yaml in tmp_path
        config_dir = tmp_path / ".claudetube"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("cache_dir: ./cache")

        monkeypatch.chdir(tmp_path)
        result = _find_project_config()
        assert result == config_file

    def test_find_project_config_in_parent_dir(self, tmp_path, monkeypatch):
        """Should find config in parent directory."""
        from claudetube.config.loader import _find_project_config

        # Create .claudetube/config.yaml in tmp_path (parent)
        config_dir = tmp_path / ".claudetube"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("cache_dir: ./cache")

        # Create a subdirectory to be the "cwd"
        subdir = tmp_path / "project" / "src"
        subdir.mkdir(parents=True)

        monkeypatch.chdir(subdir)
        result = _find_project_config()
        assert result == config_file

    def test_find_project_config_not_found(self, tmp_path, monkeypatch):
        """Should return None when no config exists."""
        from claudetube.config.loader import _find_project_config

        monkeypatch.chdir(tmp_path)
        result = _find_project_config()
        assert result is None

    def test_relative_path_resolves_to_config_location(self, tmp_path):
        """Relative cache_dir should resolve relative to config file."""
        from claudetube.config.loader import _get_cache_dir_from_yaml

        config_path = tmp_path / ".claudetube" / "config.yaml"
        config = {"cache_dir": "./my-cache"}

        result = _get_cache_dir_from_yaml(config, config_path)

        # Should resolve to tmp_path/.claudetube/my-cache
        expected = (tmp_path / ".claudetube" / "my-cache").resolve()
        assert result == expected

    def test_absolute_path_unchanged(self, tmp_path):
        """Absolute cache_dir should not be modified."""
        from claudetube.config.loader import _get_cache_dir_from_yaml

        config_path = tmp_path / ".claudetube" / "config.yaml"
        config = {"cache_dir": "/absolute/path/to/cache"}

        result = _get_cache_dir_from_yaml(config, config_path)

        assert result == Path("/absolute/path/to/cache")

    def test_tilde_expansion(self, tmp_path):
        """Should expand ~ in paths."""
        from claudetube.config.loader import _get_cache_dir_from_yaml

        config_path = tmp_path / ".claudetube" / "config.yaml"
        config = {"cache_dir": "~/my-cache"}

        result = _get_cache_dir_from_yaml(config, config_path)

        expected = Path.home() / "my-cache"
        assert result == expected

    def test_project_config_priority(self, tmp_path, monkeypatch):
        """Project config should override user config but not env."""
        from claudetube.config.loader import (
            ConfigSource,
            _resolve_config,
            clear_config_cache,
        )

        # Create project config
        config_dir = tmp_path / ".claudetube"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("cache_dir: ./project-cache")

        # Create the expected cache directory
        expected_cache = (config_dir / "project-cache").resolve()

        # Clear any cached config
        clear_config_cache()

        # Change to tmp_path and ensure no env var
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CLAUDETUBE_CACHE_DIR", raising=False)

        result = _resolve_config()

        assert result.source == ConfigSource.PROJECT
        assert result.cache_dir == expected_cache

        # Clear cache after test
        clear_config_cache()

    def test_empty_config_falls_through(self, tmp_path, monkeypatch):
        """Empty config file should fall through to next priority."""
        from claudetube.config.loader import (
            ConfigSource,
            _resolve_config,
            clear_config_cache,
        )

        # Create empty project config
        config_dir = tmp_path / ".claudetube"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("")

        clear_config_cache()

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CLAUDETUBE_CACHE_DIR", raising=False)

        result = _resolve_config()

        # Should fall through to default (no user config in test)
        assert result.source == ConfigSource.DEFAULT

        clear_config_cache()

    def test_missing_cache_dir_key_falls_through(self, tmp_path, monkeypatch):
        """Config without cache_dir should fall through."""
        from claudetube.config.loader import (
            ConfigSource,
            _resolve_config,
            clear_config_cache,
        )

        # Create config with other keys but no cache_dir
        config_dir = tmp_path / ".claudetube"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("other_setting: value")

        clear_config_cache()

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CLAUDETUBE_CACHE_DIR", raising=False)

        result = _resolve_config()

        assert result.source == ConfigSource.DEFAULT

        clear_config_cache()
