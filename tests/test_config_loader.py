"""Tests for the unified config loader."""

from pathlib import Path
from unittest.mock import patch

import pytest

from claudetube.config.loader import (
    ClaudetubeConfig,
    ConfigSource,
    _find_project_config,
    _get_cache_dir_from_yaml,
    _get_default_cache_dir,
    _get_user_config_path,
    _load_yaml_config,
    _resolve_config,
    clear_config_cache,
    get_cache_dir,
    get_config,
)


class TestConfigSource:
    """Tests for ConfigSource enum."""

    def test_config_source_values(self):
        """Test ConfigSource has expected values."""
        assert ConfigSource.ENV.value == "env"
        assert ConfigSource.PROJECT.value == "project"
        assert ConfigSource.USER.value == "user"
        assert ConfigSource.DEFAULT.value == "default"


class TestClaudetubeConfig:
    """Tests for ClaudetubeConfig dataclass."""

    def test_config_creation(self):
        """Test creating a config instance."""
        config = ClaudetubeConfig(cache_dir=Path("/tmp/cache"), source=ConfigSource.ENV)
        assert config.cache_dir == Path("/tmp/cache")
        assert config.source == ConfigSource.ENV

    def test_config_repr(self):
        """Test config string representation."""
        config = ClaudetubeConfig(
            cache_dir=Path("/tmp/cache"), source=ConfigSource.USER
        )
        repr_str = repr(config)
        assert "cache_dir=" in repr_str
        assert "source='user'" in repr_str

    def test_config_is_frozen(self):
        """Test config is immutable."""
        config = ClaudetubeConfig(
            cache_dir=Path("/tmp/cache"), source=ConfigSource.DEFAULT
        )
        with pytest.raises(AttributeError):
            config.cache_dir = Path("/other")  # type: ignore


class TestLoadYamlConfig:
    """Tests for _load_yaml_config."""

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading a file that doesn't exist returns None."""
        result = _load_yaml_config(tmp_path / "nonexistent.yaml")
        assert result is None

    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("cache_dir: /custom/cache\nother_key: value\n")
        result = _load_yaml_config(config_file)
        assert result == {"cache_dir": "/custom/cache", "other_key": "value"}

    def test_load_empty_yaml(self, tmp_path):
        """Test loading an empty YAML file returns empty dict."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        result = _load_yaml_config(config_file)
        assert result == {}

    def test_load_invalid_yaml_type(self, tmp_path):
        """Test loading YAML that's not a dict returns None."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- item1\n- item2\n")
        result = _load_yaml_config(config_file)
        assert result is None


class TestGetCacheDirFromYaml:
    """Tests for _get_cache_dir_from_yaml."""

    def test_none_config(self):
        """Test None config returns None."""
        assert _get_cache_dir_from_yaml(None) is None

    def test_missing_cache_dir(self):
        """Test config without cache_dir returns None."""
        assert _get_cache_dir_from_yaml({"other_key": "value"}) is None

    def test_with_cache_dir(self):
        """Test config with cache_dir returns expanded Path."""
        result = _get_cache_dir_from_yaml({"cache_dir": "/custom/cache"})
        assert result == Path("/custom/cache")

    def test_expands_tilde(self):
        """Test cache_dir with ~ is expanded."""
        result = _get_cache_dir_from_yaml({"cache_dir": "~/my_cache"})
        assert result == Path.home() / "my_cache"


class TestFindProjectConfig:
    """Tests for _find_project_config."""

    def test_finds_config_in_cwd(self, tmp_path, monkeypatch):
        """Test finding config in current directory."""
        config_dir = tmp_path / ".claudetube"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("cache_dir: /project/cache\n")

        monkeypatch.chdir(tmp_path)
        result = _find_project_config()
        assert result == config_file

    def test_finds_config_in_parent(self, tmp_path, monkeypatch):
        """Test finding config in parent directory."""
        config_dir = tmp_path / ".claudetube"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("cache_dir: /project/cache\n")

        subdir = tmp_path / "src" / "module"
        subdir.mkdir(parents=True)

        monkeypatch.chdir(subdir)
        result = _find_project_config()
        assert result == config_file

    def test_no_config_found(self, tmp_path, monkeypatch):
        """Test no config found returns None."""
        subdir = tmp_path / "project"
        subdir.mkdir()
        monkeypatch.chdir(subdir)
        result = _find_project_config()
        assert result is None


class TestGetUserConfigPath:
    """Tests for _get_user_config_path."""

    def test_returns_expected_path(self):
        """Test user config path is in ~/.config/claudetube/."""
        result = _get_user_config_path()
        assert result == Path.home() / ".config" / "claudetube" / "config.yaml"


class TestGetDefaultCacheDir:
    """Tests for _get_default_cache_dir."""

    def test_returns_expected_path(self):
        """Test default cache dir is ~/.claude/video_cache."""
        result = _get_default_cache_dir()
        assert result == Path.home() / ".claude" / "video_cache"


class TestResolveConfig:
    """Tests for _resolve_config."""

    def test_env_takes_priority(self, tmp_path, monkeypatch):
        """Test environment variable takes highest priority."""
        env_cache = str(tmp_path / "env_cache")
        monkeypatch.setenv("CLAUDETUBE_CACHE_DIR", env_cache)

        # Also set up project config to ensure env wins
        config_dir = tmp_path / ".claudetube"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text("cache_dir: /project/cache\n")
        monkeypatch.chdir(tmp_path)

        result = _resolve_config()
        assert result.cache_dir == Path(env_cache)
        assert result.source == ConfigSource.ENV

    def test_project_config_second_priority(self, tmp_path, monkeypatch):
        """Test project config is used when env not set."""
        monkeypatch.delenv("CLAUDETUBE_CACHE_DIR", raising=False)

        config_dir = tmp_path / ".claudetube"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text("cache_dir: /project/cache\n")
        monkeypatch.chdir(tmp_path)

        # Mock user config to not exist
        with patch("claudetube.config.loader._get_user_config_path") as mock_user:
            mock_user.return_value = tmp_path / "nonexistent" / "config.yaml"
            result = _resolve_config()

        assert result.cache_dir == Path("/project/cache")
        assert result.source == ConfigSource.PROJECT

    def test_user_config_third_priority(self, tmp_path, monkeypatch):
        """Test user config is used when env and project not set."""
        monkeypatch.delenv("CLAUDETUBE_CACHE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)  # No project config here

        user_config = tmp_path / "user_config.yaml"
        user_config.write_text("cache_dir: /user/cache\n")

        with patch("claudetube.config.loader._get_user_config_path") as mock_user:
            mock_user.return_value = user_config
            result = _resolve_config()

        assert result.cache_dir == Path("/user/cache")
        assert result.source == ConfigSource.USER

    def test_default_fallback(self, tmp_path, monkeypatch):
        """Test default is used when no other config is set."""
        monkeypatch.delenv("CLAUDETUBE_CACHE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)

        with patch("claudetube.config.loader._get_user_config_path") as mock_user:
            mock_user.return_value = tmp_path / "nonexistent" / "config.yaml"
            result = _resolve_config()

        assert result.cache_dir == Path.home() / ".claude" / "video_cache"
        assert result.source == ConfigSource.DEFAULT


class TestGetConfig:
    """Tests for get_config."""

    def test_returns_config(self, tmp_path, monkeypatch):
        """Test get_config returns a valid config."""
        monkeypatch.delenv("CLAUDETUBE_CACHE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        clear_config_cache()

        with patch("claudetube.config.loader._get_user_config_path") as mock_user:
            mock_user.return_value = tmp_path / "nonexistent" / "config.yaml"
            config = get_config()

        assert isinstance(config, ClaudetubeConfig)
        assert isinstance(config.cache_dir, Path)
        assert isinstance(config.source, ConfigSource)

    def test_caches_result(self, tmp_path, monkeypatch):
        """Test get_config caches the result."""
        monkeypatch.delenv("CLAUDETUBE_CACHE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        clear_config_cache()

        with patch("claudetube.config.loader._resolve_config") as mock_resolve:
            mock_resolve.return_value = ClaudetubeConfig(
                cache_dir=Path("/cached"), source=ConfigSource.DEFAULT
            )
            config1 = get_config()
            config2 = get_config()

        assert config1 is config2
        mock_resolve.assert_called_once()


class TestGetCacheDir:
    """Tests for get_cache_dir."""

    def test_returns_path(self, tmp_path, monkeypatch):
        """Test get_cache_dir returns just the cache_dir Path."""
        env_cache = str(tmp_path / "env_cache")
        monkeypatch.setenv("CLAUDETUBE_CACHE_DIR", env_cache)
        clear_config_cache()

        result = get_cache_dir()
        assert result == Path(env_cache)


class TestClearConfigCache:
    """Tests for clear_config_cache."""

    def test_clears_cache(self, tmp_path, monkeypatch):
        """Test clearing cache forces re-resolution."""
        clear_config_cache()

        # First call with one env value
        env_cache1 = str(tmp_path / "cache1")
        monkeypatch.setenv("CLAUDETUBE_CACHE_DIR", env_cache1)
        config1 = get_config()
        assert config1.cache_dir == Path(env_cache1)

        # Change env and clear cache
        env_cache2 = str(tmp_path / "cache2")
        monkeypatch.setenv("CLAUDETUBE_CACHE_DIR", env_cache2)
        clear_config_cache()

        config2 = get_config()
        assert config2.cache_dir == Path(env_cache2)
