"""Tests for the validate-config CLI command."""

from pathlib import Path
from unittest.mock import patch

import pytest


class TestValidateConfigCommand:
    """Tests for claudetube validate-config."""

    def test_no_config_file_exits_zero(self, capsys):
        """Test exits 0 when no config file is found."""
        from claudetube.cli import main

        with (
            patch("sys.argv", ["claudetube", "validate-config", "--skip-availability"]),
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=None,
            ),
            patch(
                "claudetube.config.loader._get_user_config_path",
                return_value=Path("/nonexistent/config.yaml"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "No config file found" in captured.out

    def test_valid_config_exits_zero(self, tmp_path, capsys):
        """Test exits 0 with a valid config."""
        from claudetube.cli import main

        config_file = tmp_path / "config.yaml"
        valid_yaml = {
            "providers": {
                "openai": {"api_key": "sk-test"},
                "preferences": {"vision": "claude-code"},
            }
        }

        with (
            patch("sys.argv", ["claudetube", "validate-config", "--skip-availability"]),
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=config_file,
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=valid_yaml,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Config is valid" in captured.out

    def test_invalid_config_exits_one(self, tmp_path, capsys):
        """Test exits 1 with an invalid config."""
        from claudetube.cli import main

        config_file = tmp_path / "config.yaml"
        invalid_yaml = {
            "providers": {
                "openai": "not-a-dict",
                "local": {"whisper_model": "mega"},
            }
        }

        with (
            patch("sys.argv", ["claudetube", "validate-config", "--skip-availability"]),
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=config_file,
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=invalid_yaml,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid" in captured.out.lower()
        assert "error" in captured.out.lower()

    def test_shows_errors_and_warnings(self, tmp_path, capsys):
        """Test that errors and warnings are printed."""
        from claudetube.cli import main

        config_file = tmp_path / "config.yaml"
        yaml_with_issues = {
            "providers": {
                "openai": "not-a-dict",
                "preferences": {"vision": "whisper-local"},
            }
        }

        with (
            patch("sys.argv", ["claudetube", "validate-config", "--skip-availability"]),
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=config_file,
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=yaml_with_issues,
            ),
            pytest.raises(SystemExit),
        ):
            main()

        captured = capsys.readouterr()
        # Should show errors (openai not a dict)
        assert "Errors" in captured.out
        # Should show warnings (whisper-local can't do vision)
        assert "Warnings" in captured.out

    def test_warnings_only_still_valid(self, tmp_path, capsys):
        """Test config with only warnings exits 0."""
        from claudetube.cli import main

        config_file = tmp_path / "config.yaml"
        yaml_with_warnings = {
            "providers": {
                "preferences": {"vision": "whisper-local"},
            }
        }

        with (
            patch("sys.argv", ["claudetube", "validate-config", "--skip-availability"]),
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=config_file,
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=yaml_with_warnings,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "warning" in captured.out.lower()

    def test_config_file_path_printed(self, tmp_path, capsys):
        """Test the config file path is shown in output."""
        from claudetube.cli import main

        config_file = tmp_path / "config.yaml"

        with (
            patch("sys.argv", ["claudetube", "validate-config", "--skip-availability"]),
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=config_file,
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value={"providers": {}},
            ),
            pytest.raises(SystemExit),
        ):
            main()

        captured = capsys.readouterr()
        assert str(config_file) in captured.out

    def test_skip_availability_flag(self, tmp_path, capsys):
        """Test --skip-availability skips provider checks."""
        from claudetube.cli import main

        config_file = tmp_path / "config.yaml"

        with (
            patch("sys.argv", ["claudetube", "validate-config", "--skip-availability"]),
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=config_file,
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value={"providers": {}},
            ),
            pytest.raises(SystemExit),
        ):
            main()

        captured = capsys.readouterr()
        assert "Provider availability" not in captured.out

    def test_provider_availability_shown(self, tmp_path, capsys):
        """Test provider availability is shown without --skip-availability."""
        from claudetube.cli import main

        config_file = tmp_path / "config.yaml"

        with (
            patch("sys.argv", ["claudetube", "validate-config"]),
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=config_file,
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value={"providers": {}},
            ),
            patch(
                "claudetube.providers.registry.list_all",
                return_value=["claude-code", "openai"],
            ),
            patch(
                "claudetube.providers.registry.list_available",
                return_value=["claude-code"],
            ),
            pytest.raises(SystemExit),
        ):
            main()

        captured = capsys.readouterr()
        assert "Provider availability" in captured.out
        assert "claude-code: available" in captured.out
        assert "openai: not available" in captured.out

    def test_failed_parse_exits_one(self, tmp_path, capsys):
        """Test exits 1 when config file fails to parse."""
        from claudetube.cli import main

        user_path = tmp_path / "config.yaml"
        user_path.touch()  # Make it exist

        with (
            patch("sys.argv", ["claudetube", "validate-config", "--skip-availability"]),
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=None,
            ),
            patch(
                "claudetube.config.loader._get_user_config_path",
                return_value=user_path,
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=None,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Failed to parse" in captured.out

    def test_user_config_fallback(self, tmp_path, capsys):
        """Test falls back to user config when no project config."""
        from claudetube.cli import main

        user_path = tmp_path / "config.yaml"
        user_path.touch()

        with (
            patch("sys.argv", ["claudetube", "validate-config", "--skip-availability"]),
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=None,
            ),
            patch(
                "claudetube.config.loader._get_user_config_path",
                return_value=user_path,
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value={"providers": {}},
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert str(user_path) in captured.out
