"""Tests for tools/yt_dlp.py — YouTube config args and POT provider detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from claudetube.tools.yt_dlp import YtDlpTool


@pytest.fixture()
def tool():
    return YtDlpTool()


# ---------------------------------------------------------------------------
# _youtube_config_args
# ---------------------------------------------------------------------------


class TestYoutubeConfigArgs:
    """Tests for _youtube_config_args() method."""

    def test_returns_empty_when_no_config(self, tool):
        """No config files → empty args."""
        with patch(
            "claudetube.config.loader._find_project_config", return_value=None
        ), patch(
            "claudetube.config.loader._load_yaml_config", return_value=None
        ):
            assert tool._youtube_config_args() == []

    def test_returns_empty_when_no_youtube_section(self, tool):
        """Config exists but no 'youtube' key → empty args."""
        fake_config = {"providers": {"openai": {"api_key": "sk-test"}}}
        with patch(
            "claudetube.config.loader._find_project_config",
            return_value=Path("/fake/config.yaml"),
        ), patch(
            "claudetube.config.loader._load_yaml_config",
            return_value=fake_config,
        ):
            assert tool._youtube_config_args() == []

    def test_po_token(self, tool):
        """po_token maps to --extractor-args youtube:po_token=..."""
        fake_config = {"youtube": {"po_token": "mweb.gvs+TOKEN123"}}
        with patch(
            "claudetube.config.loader._find_project_config",
            return_value=Path("/fake/config.yaml"),
        ), patch(
            "claudetube.config.loader._load_yaml_config",
            return_value=fake_config,
        ):
            args = tool._youtube_config_args()
            assert "--extractor-args" in args
            assert "youtube:po_token=mweb.gvs+TOKEN123" in args

    def test_cookies_file_exists(self, tool, tmp_path):
        """cookies_file maps to --cookies when file exists."""
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("# Netscape HTTP Cookie File\n")

        fake_config = {"youtube": {"cookies_file": str(cookie_file)}}
        with patch(
            "claudetube.config.loader._find_project_config",
            return_value=Path("/fake/config.yaml"),
        ), patch(
            "claudetube.config.loader._load_yaml_config",
            return_value=fake_config,
        ):
            args = tool._youtube_config_args()
            assert "--cookies" in args
            assert str(cookie_file) in args

    def test_cookies_file_missing(self, tool):
        """cookies_file that doesn't exist → warning, no args."""
        fake_config = {"youtube": {"cookies_file": "/nonexistent/cookies.txt"}}
        with patch(
            "claudetube.config.loader._find_project_config",
            return_value=Path("/fake/config.yaml"),
        ), patch(
            "claudetube.config.loader._load_yaml_config",
            return_value=fake_config,
        ):
            args = tool._youtube_config_args()
            assert "--cookies" not in args

    def test_pot_server_url(self, tool):
        """pot_server_url maps to --extractor-args youtubepot-bgutilhttp:base_url=..."""
        fake_config = {
            "youtube": {"pot_server_url": "http://192.168.1.100:4416"}
        }
        with patch(
            "claudetube.config.loader._find_project_config",
            return_value=Path("/fake/config.yaml"),
        ), patch(
            "claudetube.config.loader._load_yaml_config",
            return_value=fake_config,
        ):
            args = tool._youtube_config_args()
            assert "--extractor-args" in args
            assert (
                "youtubepot-bgutilhttp:base_url=http://192.168.1.100:4416"
                in args
            )

    def test_pot_script_path_exists(self, tool, tmp_path):
        """pot_script_path maps to --extractor-args when script exists."""
        script = tmp_path / "generate_once.js"
        script.write_text("// bgutil script\n")

        fake_config = {"youtube": {"pot_script_path": str(script)}}
        with patch(
            "claudetube.config.loader._find_project_config",
            return_value=Path("/fake/config.yaml"),
        ), patch(
            "claudetube.config.loader._load_yaml_config",
            return_value=fake_config,
        ):
            args = tool._youtube_config_args()
            assert "--extractor-args" in args
            # There may be multiple --extractor-args; find the bgutilscript one
            bgutil_args = [
                a for a in args if "youtubepot-bgutilscript:" in a
            ]
            assert len(bgutil_args) == 1
            assert f"script_path={script}" in bgutil_args[0]

    def test_pot_script_path_missing(self, tool):
        """pot_script_path that doesn't exist → warning, no args."""
        fake_config = {
            "youtube": {"pot_script_path": "/nonexistent/generate_once.js"}
        }
        with patch(
            "claudetube.config.loader._find_project_config",
            return_value=Path("/fake/config.yaml"),
        ), patch(
            "claudetube.config.loader._load_yaml_config",
            return_value=fake_config,
        ):
            args = tool._youtube_config_args()
            bgutil_args = [
                a for a in args if "youtubepot-bgutilscript:" in a
            ]
            assert len(bgutil_args) == 0

    def test_all_options_combined(self, tool, tmp_path):
        """All options produce the expected args together."""
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("# cookies\n")
        script = tmp_path / "generate_once.js"
        script.write_text("// script\n")

        fake_config = {
            "youtube": {
                "po_token": "mweb.gvs+COMBINED",
                "cookies_file": str(cookie_file),
                "pot_server_url": "http://localhost:4416",
                "pot_script_path": str(script),
            }
        }
        with patch(
            "claudetube.config.loader._find_project_config",
            return_value=Path("/fake/config.yaml"),
        ), patch(
            "claudetube.config.loader._load_yaml_config",
            return_value=fake_config,
        ):
            args = tool._youtube_config_args()
            assert "youtube:po_token=mweb.gvs+COMBINED" in args
            assert "--cookies" in args
            assert "youtubepot-bgutilhttp:base_url=http://localhost:4416" in args
            assert any("youtubepot-bgutilscript:" in a for a in args)

    def test_config_load_exception_returns_empty(self, tool):
        """Any exception during config load → empty args, no crash."""
        with patch(
            "claudetube.config.loader._find_project_config",
            side_effect=RuntimeError("boom"),
        ):
            args = tool._youtube_config_args()
            assert args == []

    def test_youtube_section_not_dict(self, tool):
        """Non-dict youtube value → empty args."""
        fake_config = {"youtube": "invalid"}
        with patch(
            "claudetube.config.loader._find_project_config",
            return_value=Path("/fake/config.yaml"),
        ), patch(
            "claudetube.config.loader._load_yaml_config",
            return_value=fake_config,
        ):
            args = tool._youtube_config_args()
            assert args == []


# ---------------------------------------------------------------------------
# check_pot_providers
# ---------------------------------------------------------------------------


class TestCheckPotProviders:
    """Tests for check_pot_providers() method."""

    def test_detects_bgutil_providers(self, tool):
        """Parses provider list from verbose output."""
        fake_stderr = (
            "[debug] [youtube] [pot] PO Token Providers: "
            "bgutil:http-1.2.2 (external), bgutil:script-1.2.2 (external)\n"
        )
        with patch.object(
            tool,
            "_run",
            return_value=type(
                "R", (), {"stderr": fake_stderr, "stdout": "", "success": True}
            )(),
        ):
            providers = tool.check_pot_providers()
            assert len(providers) == 2
            assert "bgutil:http-1.2.2 (external)" in providers
            assert "bgutil:script-1.2.2 (external)" in providers

    def test_returns_empty_when_none(self, tool):
        """'none' provider list → empty list."""
        fake_stderr = "[pot] PO Token Providers: none\n"
        with patch.object(
            tool,
            "_run",
            return_value=type(
                "R", (), {"stderr": fake_stderr, "stdout": "", "success": True}
            )(),
        ):
            providers = tool.check_pot_providers()
            assert providers == []

    def test_returns_empty_when_no_match(self, tool):
        """No provider line in output → empty list."""
        fake_stderr = "[debug] some other output\n"
        with patch.object(
            tool,
            "_run",
            return_value=type(
                "R", (), {"stderr": fake_stderr, "stdout": "", "success": True}
            )(),
        ):
            providers = tool.check_pot_providers()
            assert providers == []


# ---------------------------------------------------------------------------
# _is_youtube_url
# ---------------------------------------------------------------------------


class TestIsYoutubeUrl:
    """Tests for _is_youtube_url() static method."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=abc123",
            "https://youtube.com/watch?v=abc123",
            "https://m.youtube.com/watch?v=abc123",
            "https://youtu.be/abc123",
            "http://www.youtube.com/watch?v=abc123",
        ],
    )
    def test_youtube_urls(self, url):
        assert YtDlpTool._is_youtube_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://vimeo.com/123456",
            "https://example.com/video",
            "https://dailymotion.com/video/x123",
        ],
    )
    def test_non_youtube_urls(self, url):
        assert YtDlpTool._is_youtube_url(url) is False
