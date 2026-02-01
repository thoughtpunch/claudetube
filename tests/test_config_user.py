"""
Tests for user-level ~/.config/claudetube/config.yaml support.
"""

from pathlib import Path


class TestUserConfig:
    """Tests for user-level configuration."""

    def test_user_config_path_unix(self, monkeypatch):
        """Should use ~/.config/claudetube on Unix systems."""
        from claudetube.config.loader import _get_user_config_path

        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

        result = _get_user_config_path()

        assert result == Path.home() / ".config" / "claudetube" / "config.yaml"

    def test_user_config_path_xdg(self, tmp_path, monkeypatch):
        """Should respect XDG_CONFIG_HOME on Unix systems."""
        from claudetube.config.loader import _get_user_config_path

        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

        result = _get_user_config_path()

        assert result == tmp_path / "xdg" / "claudetube" / "config.yaml"

    def test_user_config_path_windows(self, tmp_path, monkeypatch):
        """Should use APPDATA on Windows."""
        from claudetube.config.loader import _get_user_config_path

        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))

        result = _get_user_config_path()

        expected = tmp_path / "AppData" / "Roaming" / "claudetube" / "config.yaml"
        assert result == expected

    def test_user_config_path_windows_fallback(self, monkeypatch):
        """Should fallback to home/AppData/Roaming on Windows without APPDATA."""
        from claudetube.config.loader import _get_user_config_path

        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.delenv("APPDATA", raising=False)

        result = _get_user_config_path()

        expected = Path.home() / "AppData" / "Roaming" / "claudetube" / "config.yaml"
        assert result == expected

    def test_user_config_loaded(self, tmp_path, monkeypatch):
        """User config should be loaded when present."""
        from claudetube.config.loader import (
            ConfigSource,
            _resolve_config,
            clear_config_cache,
        )

        # Create user config at a known location
        user_config_dir = tmp_path / ".config" / "claudetube"
        user_config_dir.mkdir(parents=True)
        user_config_file = user_config_dir / "config.yaml"
        user_config_file.write_text("cache_dir: ~/user-cache")

        # Mock the user config path function
        monkeypatch.setattr(
            "claudetube.config.loader._get_user_config_path",
            lambda: user_config_file,
        )

        # Ensure no project config or env var
        monkeypatch.setattr(
            "claudetube.config.loader._find_project_config",
            lambda: None,
        )
        monkeypatch.delenv("CLAUDETUBE_CACHE_DIR", raising=False)

        clear_config_cache()

        result = _resolve_config()

        assert result.source == ConfigSource.USER
        assert result.cache_dir == Path.home() / "user-cache"

        clear_config_cache()

    def test_user_config_lower_priority_than_project(self, tmp_path, monkeypatch):
        """Project config should override user config."""
        from claudetube.config.loader import (
            ConfigSource,
            _resolve_config,
            clear_config_cache,
        )

        # Create project config
        project_config_dir = tmp_path / ".claudetube"
        project_config_dir.mkdir()
        project_config_file = project_config_dir / "config.yaml"
        project_config_file.write_text("cache_dir: ./project-cache")

        # Create user config
        user_config_dir = tmp_path / ".config" / "claudetube"
        user_config_dir.mkdir(parents=True)
        user_config_file = user_config_dir / "config.yaml"
        user_config_file.write_text("cache_dir: ~/user-cache")

        # Mock functions
        monkeypatch.setattr(
            "claudetube.config.loader._find_project_config",
            lambda: project_config_file,
        )
        monkeypatch.setattr(
            "claudetube.config.loader._get_user_config_path",
            lambda: user_config_file,
        )
        monkeypatch.delenv("CLAUDETUBE_CACHE_DIR", raising=False)

        clear_config_cache()

        result = _resolve_config()

        # Project config should win
        assert result.source == ConfigSource.PROJECT
        expected = (project_config_dir / "project-cache").resolve()
        assert result.cache_dir == expected

        clear_config_cache()

    def test_user_config_lower_priority_than_env(self, tmp_path, monkeypatch):
        """Environment variable should override user config."""
        from claudetube.config.loader import (
            ConfigSource,
            _resolve_config,
            clear_config_cache,
        )

        # Create user config
        user_config_dir = tmp_path / ".config" / "claudetube"
        user_config_dir.mkdir(parents=True)
        user_config_file = user_config_dir / "config.yaml"
        user_config_file.write_text("cache_dir: ~/user-cache")

        # Set env var
        env_cache = tmp_path / "env-cache"
        monkeypatch.setenv("CLAUDETUBE_CACHE_DIR", str(env_cache))

        # Mock user config path
        monkeypatch.setattr(
            "claudetube.config.loader._get_user_config_path",
            lambda: user_config_file,
        )
        monkeypatch.setattr(
            "claudetube.config.loader._find_project_config",
            lambda: None,
        )

        clear_config_cache()

        result = _resolve_config()

        # Env should win
        assert result.source == ConfigSource.ENV
        assert result.cache_dir == env_cache.resolve()

        clear_config_cache()
