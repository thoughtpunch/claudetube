"""Tests for tools/yt_dlp.py — YouTube config args, POT provider detection, and auth health checks."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        with (
            patch("claudetube.config.loader._find_project_config", return_value=None),
            patch("claudetube.config.loader._load_yaml_config", return_value=None),
        ):
            assert tool._youtube_config_args() == []

    def test_returns_empty_when_no_youtube_section(self, tool):
        """Config exists but no 'youtube' key → empty args."""
        fake_config = {"providers": {"openai": {"api_key": "sk-test"}}}
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            assert tool._youtube_config_args() == []

    def test_po_token(self, tool):
        """po_token maps to --extractor-args youtube:po_token=..."""
        fake_config = {"youtube": {"po_token": "mweb.gvs+TOKEN123"}}
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            args = tool._youtube_config_args()
            assert "--extractor-args" in args
            assert "youtube:po_token=mweb.gvs+TOKEN123" in args

    def test_cookies_file_exists(self, tool, tmp_path):
        """cookies_file maps to --cookies when file exists."""
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("# Netscape HTTP Cookie File\n")

        fake_config = {"youtube": {"cookies_file": str(cookie_file)}}
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            args = tool._youtube_config_args()
            assert "--cookies" in args
            assert str(cookie_file) in args

    def test_cookies_file_missing(self, tool):
        """cookies_file that doesn't exist → warning, no args."""
        fake_config = {"youtube": {"cookies_file": "/nonexistent/cookies.txt"}}
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            args = tool._youtube_config_args()
            assert "--cookies" not in args

    def test_pot_server_url(self, tool):
        """pot_server_url maps to --extractor-args youtubepot-bgutilhttp:base_url=..."""
        fake_config = {"youtube": {"pot_server_url": "http://192.168.1.100:4416"}}
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            args = tool._youtube_config_args()
            assert "--extractor-args" in args
            assert "youtubepot-bgutilhttp:base_url=http://192.168.1.100:4416" in args

    def test_pot_script_path_exists(self, tool, tmp_path):
        """pot_script_path maps to --extractor-args when script exists."""
        script = tmp_path / "generate_once.js"
        script.write_text("// bgutil script\n")

        fake_config = {"youtube": {"pot_script_path": str(script)}}
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            args = tool._youtube_config_args()
            assert "--extractor-args" in args
            # There may be multiple --extractor-args; find the bgutilscript one
            bgutil_args = [a for a in args if "youtubepot-bgutilscript:" in a]
            assert len(bgutil_args) == 1
            assert f"script_path={script}" in bgutil_args[0]

    def test_pot_script_path_missing(self, tool):
        """pot_script_path that doesn't exist → warning, no args."""
        fake_config = {"youtube": {"pot_script_path": "/nonexistent/generate_once.js"}}
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            args = tool._youtube_config_args()
            bgutil_args = [a for a in args if "youtubepot-bgutilscript:" in a]
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
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
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
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            args = tool._youtube_config_args()
            assert args == []

    def test_cookies_from_browser_valid(self, tool):
        """cookies_from_browser with supported browser → --cookies-from-browser."""
        fake_config = {"youtube": {"cookies_from_browser": "firefox"}}
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            args = tool._youtube_config_args()
            assert "--cookies-from-browser" in args
            assert "firefox" in args

    def test_cookies_from_browser_unsupported(self, tool):
        """cookies_from_browser with unsupported browser → warning, no args."""
        fake_config = {"youtube": {"cookies_from_browser": "netscape"}}
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            args = tool._youtube_config_args()
            assert "--cookies-from-browser" not in args
            assert "--cookies" not in args

    def test_cookies_from_browser_takes_priority_over_cookies_file(
        self, tool, tmp_path
    ):
        """cookies_from_browser wins over cookies_file when both set."""
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("# Netscape HTTP Cookie File\n")

        fake_config = {
            "youtube": {
                "cookies_from_browser": "chrome",
                "cookies_file": str(cookie_file),
            }
        }
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            args = tool._youtube_config_args()
            assert "--cookies-from-browser" in args
            assert "chrome" in args
            # cookies_file should NOT be used when cookies_from_browser is set
            assert "--cookies" not in args

    def test_cookies_from_browser_case_insensitive(self, tool):
        """Browser name is normalized to lowercase."""
        fake_config = {"youtube": {"cookies_from_browser": "Firefox"}}
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            args = tool._youtube_config_args()
            assert "--cookies-from-browser" in args
            assert "firefox" in args

    def test_cookies_from_browser_fallback_to_cookies_file(self, tool, tmp_path):
        """Unsupported browser falls through to cookies_file."""
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("# cookies\n")

        fake_config = {
            "youtube": {
                "cookies_from_browser": "netscape",  # unsupported
                "cookies_file": str(cookie_file),
            }
        }
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            args = tool._youtube_config_args()
            assert "--cookies-from-browser" not in args
            assert "--cookies" in args
            assert str(cookie_file) in args

    @pytest.mark.parametrize(
        "browser",
        [
            "brave",
            "chrome",
            "chromium",
            "edge",
            "firefox",
            "opera",
            "safari",
            "vivaldi",
            "whale",
        ],
    )
    def test_all_supported_browsers(self, tool, browser):
        """All documented browsers are accepted."""
        fake_config = {"youtube": {"cookies_from_browser": browser}}
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            args = tool._youtube_config_args()
            assert "--cookies-from-browser" in args
            assert browser in args


# ---------------------------------------------------------------------------
# YouTube config args applied to all methods
# ---------------------------------------------------------------------------


class TestYoutubeConfigArgsApplied:
    """Verify _youtube_config_args() is called for YouTube URLs in all methods."""

    def _make_success_result(self, stdout="", stderr=""):
        """Create a mock ToolResult-like object."""
        return type(
            "R",
            (),
            {"success": True, "stdout": stdout, "stderr": stderr, "returncode": 0},
        )()

    def test_get_metadata_uses_youtube_config(self, tool):
        """get_metadata passes YouTube config args for YouTube URLs."""
        with (
            patch.object(
                tool, "_youtube_config_args", return_value=["--cookies", "/f"]
            ) as mock_cfg,
            patch.object(
                tool,
                "_run",
                return_value=self._make_success_result(stdout='{"id":"x"}'),
            ),
        ):
            tool.get_metadata("https://www.youtube.com/watch?v=abc123")
            mock_cfg.assert_called_once()
            run_args = tool._run.call_args[0][0]
            assert "--cookies" in run_args

    def test_get_metadata_skips_config_for_non_youtube(self, tool):
        """get_metadata does NOT call _youtube_config_args for non-YouTube URLs."""
        with (
            patch.object(tool, "_youtube_config_args") as mock_cfg,
            patch.object(
                tool,
                "_run",
                return_value=self._make_success_result(stdout='{"id":"x"}'),
            ),
        ):
            tool.get_metadata("https://vimeo.com/123456")
            mock_cfg.assert_not_called()

    def test_download_thumbnail_uses_youtube_config(self, tool, tmp_path):
        """download_thumbnail passes YouTube config args for YouTube URLs."""
        (tmp_path / "thumbnail.jpg").write_bytes(b"\xff\xd8")
        with (
            patch.object(
                tool, "_youtube_config_args", return_value=["--cookies", "/f"]
            ) as mock_cfg,
            patch.object(tool, "_run", return_value=self._make_success_result()),
        ):
            tool.download_thumbnail("https://www.youtube.com/watch?v=abc123", tmp_path)
            mock_cfg.assert_called_once()

    def test_fetch_subtitles_uses_youtube_config(self, tool, tmp_path):
        """fetch_subtitles passes YouTube config args for YouTube URLs."""
        with (
            patch.object(
                tool, "_youtube_config_args", return_value=["--cookies", "/f"]
            ) as mock_cfg,
            patch.object(tool, "_run", return_value=self._make_success_result()),
        ):
            tool.fetch_subtitles("https://www.youtube.com/watch?v=abc123", tmp_path)
            mock_cfg.assert_called_once()

    def test_get_formats_uses_youtube_config(self, tool):
        """get_formats passes YouTube config args for YouTube URLs."""
        with (
            patch.object(
                tool, "_youtube_config_args", return_value=["--cookies", "/f"]
            ) as mock_cfg,
            patch.object(
                tool,
                "_run",
                return_value=self._make_success_result(stdout='{"formats":[]}'),
            ),
        ):
            tool.get_formats("https://www.youtube.com/watch?v=abc123")
            mock_cfg.assert_called_once()

    def test_download_video_segment_uses_youtube_config(self, tool, tmp_path):
        """download_video_segment passes YouTube config args for YouTube URLs."""
        out = tmp_path / "segment.mp4"
        with (
            patch.object(
                tool, "_youtube_config_args", return_value=["--cookies", "/f"]
            ) as mock_cfg,
            patch.object(tool, "_run", return_value=self._make_success_result()),
        ):
            tool.download_video_segment(
                "https://www.youtube.com/watch?v=abc123", out, 0, 10
            )
            mock_cfg.assert_called_once()

    def test_download_audio_description_uses_youtube_config(self, tool, tmp_path):
        """download_audio_description passes YouTube config args for YouTube URLs."""
        out = tmp_path / "ad.mp3"
        out.write_bytes(b"\x00")
        with (
            patch.object(
                tool, "_youtube_config_args", return_value=["--cookies", "/f"]
            ) as mock_cfg,
            patch.object(tool, "_run", return_value=self._make_success_result()),
        ):
            tool.download_audio_description(
                "https://www.youtube.com/watch?v=abc123",
                out,
                format_id="251",
            )
            mock_cfg.assert_called_once()


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


# ---------------------------------------------------------------------------
# check_youtube_auth_status
# ---------------------------------------------------------------------------


class TestCheckYoutubeAuthStatus:
    """Tests for check_youtube_auth_status() method."""

    def test_level_0_nothing_configured(self, tool):
        """No deno, no config → auth level 0."""
        with (
            patch.object(tool, "_subprocess_env", return_value={"PATH": "/usr/bin"}),
            patch("shutil.which", return_value=None),
            patch.object(tool, "check_pot_providers", return_value=[]),
            patch.object(tool, "_load_youtube_config", return_value={}),
        ):
            status = tool.check_youtube_auth_status()
            assert status["auth_level"] == 0
            assert status["deno_available"] is False
            assert status["cookies_configured"] is False
            assert status["po_token_configured"] is False
            assert status["pot_plugin_loaded"] is False
            assert len(status["recommendations"]) > 0

    def test_level_1_deno_only(self, tool):
        """Deno available, nothing else → auth level 1."""
        fake_proc = MagicMock()
        fake_proc.returncode = 0
        fake_proc.stdout = "deno 2.1.4\n"

        with (
            patch.object(tool, "_subprocess_env", return_value={"PATH": "/usr/bin"}),
            patch("shutil.which", return_value="/usr/local/bin/deno"),
            patch("subprocess.run", return_value=fake_proc),
            patch.object(tool, "check_pot_providers", return_value=[]),
            patch.object(tool, "_load_youtube_config", return_value={}),
        ):
            status = tool.check_youtube_auth_status()
            assert status["auth_level"] == 1
            assert status["deno_available"] is True
            assert status["deno_version"] == "2.1.4"

    def test_level_2_cookies_and_deno(self, tool, tmp_path):
        """Deno + cookies → auth level 2."""
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("# cookies\n")

        fake_proc = MagicMock()
        fake_proc.returncode = 0
        fake_proc.stdout = "deno 2.1.4\n"

        yt_cfg = {"cookies_file": str(cookie_file)}

        with (
            patch.object(tool, "_subprocess_env", return_value={"PATH": "/usr/bin"}),
            patch("shutil.which", return_value="/usr/local/bin/deno"),
            patch("subprocess.run", return_value=fake_proc),
            patch.object(tool, "check_pot_providers", return_value=[]),
            patch.object(tool, "_load_youtube_config", return_value=yt_cfg),
        ):
            status = tool.check_youtube_auth_status()
            assert status["auth_level"] == 2
            assert status["cookies_configured"] is True
            assert status["cookies_source"] == f"file:{cookie_file}"

    def test_level_3_po_token_cookies_deno(self, tool, tmp_path):
        """Deno + cookies + manual PO token → auth level 3."""
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("# cookies\n")

        fake_proc = MagicMock()
        fake_proc.returncode = 0
        fake_proc.stdout = "deno 2.1.4\n"

        yt_cfg = {
            "cookies_file": str(cookie_file),
            "po_token": "mweb.gvs+TOKENVALUE",
        }

        with (
            patch.object(tool, "_subprocess_env", return_value={"PATH": "/usr/bin"}),
            patch("shutil.which", return_value="/usr/local/bin/deno"),
            patch("subprocess.run", return_value=fake_proc),
            patch.object(tool, "check_pot_providers", return_value=[]),
            patch.object(tool, "_load_youtube_config", return_value=yt_cfg),
        ):
            status = tool.check_youtube_auth_status()
            assert status["auth_level"] == 3
            assert status["po_token_configured"] is True
            assert status["po_token_type"] == "mweb.gvs"

    def test_level_4_plugin_cookies_deno(self, tool, tmp_path):
        """Deno + cookies + POT plugin → auth level 4."""
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("# cookies\n")

        fake_proc = MagicMock()
        fake_proc.returncode = 0
        fake_proc.stdout = "deno 2.1.4\n"

        yt_cfg = {"cookies_file": str(cookie_file)}

        with (
            patch.object(tool, "_subprocess_env", return_value={"PATH": "/usr/bin"}),
            patch("shutil.which", return_value="/usr/local/bin/deno"),
            patch("subprocess.run", return_value=fake_proc),
            patch.object(
                tool,
                "check_pot_providers",
                return_value=["bgutil:http-1.2.2 (external)"],
            ),
            patch.object(tool, "_load_youtube_config", return_value=yt_cfg),
        ):
            status = tool.check_youtube_auth_status()
            assert status["auth_level"] == 4
            assert status["pot_plugin_loaded"] is True
            assert status["pot_plugin_version"] == "1.2.2"

    def test_cookies_from_browser_detected(self, tool):
        """cookies_from_browser in config is correctly detected."""
        fake_proc = MagicMock()
        fake_proc.returncode = 0
        fake_proc.stdout = "deno 2.1.4\n"

        yt_cfg = {"cookies_from_browser": "firefox"}

        with (
            patch.object(tool, "_subprocess_env", return_value={"PATH": "/usr/bin"}),
            patch("shutil.which", return_value="/usr/local/bin/deno"),
            patch("subprocess.run", return_value=fake_proc),
            patch.object(tool, "check_pot_providers", return_value=[]),
            patch.object(tool, "_load_youtube_config", return_value=yt_cfg),
        ):
            status = tool.check_youtube_auth_status()
            assert status["cookies_configured"] is True
            assert status["cookies_source"] == "browser:firefox"

    def test_pot_server_reachable_check(self, tool):
        """pot_server_url triggers reachability check."""
        yt_cfg = {"pot_server_url": "http://localhost:4416"}

        with (
            patch.object(tool, "_subprocess_env", return_value={"PATH": "/usr/bin"}),
            patch("shutil.which", return_value=None),
            patch.object(tool, "check_pot_providers", return_value=[]),
            patch.object(tool, "_load_youtube_config", return_value=yt_cfg),
            patch("urllib.request.urlopen", side_effect=ConnectionRefusedError),
        ):
            status = tool.check_youtube_auth_status()
            assert status["pot_server_reachable"] is False

    def test_pot_server_not_configured(self, tool):
        """No pot_server_url → pot_server_reachable is None."""
        with (
            patch.object(tool, "_subprocess_env", return_value={"PATH": "/usr/bin"}),
            patch("shutil.which", return_value=None),
            patch.object(tool, "check_pot_providers", return_value=[]),
            patch.object(tool, "_load_youtube_config", return_value={}),
        ):
            status = tool.check_youtube_auth_status()
            assert status["pot_server_reachable"] is None

    def test_recommendations_when_nothing_configured(self, tool):
        """No config → recommendations include deno, cookies, and PO token."""
        with (
            patch.object(tool, "_subprocess_env", return_value={"PATH": "/usr/bin"}),
            patch("shutil.which", return_value=None),
            patch.object(tool, "check_pot_providers", return_value=[]),
            patch.object(tool, "_load_youtube_config", return_value={}),
        ):
            status = tool.check_youtube_auth_status()
            recs = status["recommendations"]
            assert any("deno" in r.lower() for r in recs)
            assert any("cookies" in r.lower() for r in recs)
            assert any("bgutil" in r.lower() for r in recs)

    def test_no_recommendations_when_fully_configured(self, tool, tmp_path):
        """Level 4 → no recommendations."""
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("# cookies\n")

        fake_proc = MagicMock()
        fake_proc.returncode = 0
        fake_proc.stdout = "deno 2.1.4\n"

        yt_cfg = {"cookies_file": str(cookie_file)}

        with (
            patch.object(tool, "_subprocess_env", return_value={"PATH": "/usr/bin"}),
            patch("shutil.which", return_value="/usr/local/bin/deno"),
            patch("subprocess.run", return_value=fake_proc),
            patch.object(
                tool,
                "check_pot_providers",
                return_value=["bgutil:http-1.2.2 (external)"],
            ),
            patch.object(tool, "_load_youtube_config", return_value=yt_cfg),
        ):
            status = tool.check_youtube_auth_status()
            assert status["recommendations"] == []


# ---------------------------------------------------------------------------
# format_auth_error_message
# ---------------------------------------------------------------------------


class TestFormatAuthErrorMessage:
    """Tests for format_auth_error_message() method."""

    def test_includes_auth_level(self, tool):
        """Message includes auth level."""
        status = {
            "auth_level": 0,
            "deno_available": False,
            "deno_version": None,
            "cookies_configured": False,
            "cookies_source": None,
            "pot_plugin_loaded": False,
            "pot_plugin_version": None,
            "po_token_configured": False,
            "po_token_type": None,
            "pot_server_reachable": None,
            "recommendations": ["Install deno"],
        }
        msg = tool.format_auth_error_message(status)
        assert "auth level: 0/4" in msg
        assert "403" in msg
        assert "youtube-auth.md" in msg

    def test_includes_po_token_expiry_warning(self, tool):
        """When PO token configured but 403, mentions expiry."""
        status = {
            "auth_level": 3,
            "deno_available": True,
            "deno_version": "2.1.4",
            "cookies_configured": True,
            "cookies_source": "browser:firefox",
            "pot_plugin_loaded": False,
            "pot_plugin_version": None,
            "po_token_configured": True,
            "po_token_type": "mweb.gvs",
            "pot_server_reachable": None,
            "recommendations": [],
        }
        msg = tool.format_auth_error_message(status)
        assert "expire" in msg.lower()
        assert "12 hours" in msg

    def test_uses_computed_status_when_none(self, tool):
        """When auth_status is None, computes it."""
        with patch.object(
            tool,
            "check_youtube_auth_status",
            return_value={
                "auth_level": 0,
                "deno_available": False,
                "deno_version": None,
                "cookies_configured": False,
                "cookies_source": None,
                "pot_plugin_loaded": False,
                "pot_plugin_version": None,
                "po_token_configured": False,
                "po_token_type": None,
                "pot_server_reachable": None,
                "recommendations": [],
            },
        ) as mock_check:
            msg = tool.format_auth_error_message()
            mock_check.assert_called_once()
            assert "403" in msg


# ---------------------------------------------------------------------------
# _run 403 actionable error
# ---------------------------------------------------------------------------


class TestRunActionableError:
    """Tests for _run() emitting actionable error on 403 failures."""

    def test_403_retry_includes_auth_guidance(self, tool):
        """When 403 retry fails, stderr includes auth guidance."""
        # First call returns 403 error, second call also fails
        first_result = subprocess.CompletedProcess(
            args=["yt-dlp"],
            returncode=1,
            stdout="",
            stderr="ERROR: HTTP Error 403: Forbidden",
        )
        retry_result = subprocess.CompletedProcess(
            args=["yt-dlp"],
            returncode=1,
            stdout="",
            stderr="ERROR: HTTP Error 403: Forbidden (retry)",
        )

        with (
            patch("subprocess.run", side_effect=[first_result, retry_result]),
            patch.object(tool, "_subprocess_env", return_value={"PATH": "/usr/bin"}),
            patch.object(
                tool, "format_auth_error_message", return_value="\nAuth guidance here"
            ),
        ):
            result = tool._run(
                ["--dump-json", "https://youtube.com/watch?v=x"], retry_clients=True
            )
            assert not result.success
            assert "Auth guidance here" in result.stderr

    def test_non_403_error_no_auth_guidance(self, tool):
        """Non-403 errors don't include auth guidance."""
        error_result = subprocess.CompletedProcess(
            args=["yt-dlp"],
            returncode=1,
            stdout="",
            stderr="ERROR: Video unavailable",
        )

        with (
            patch("subprocess.run", return_value=error_result),
            patch.object(tool, "_subprocess_env", return_value={"PATH": "/usr/bin"}),
        ):
            result = tool._run(
                ["--dump-json", "https://example.com/video"], retry_clients=True
            )
            assert not result.success
            assert "auth" not in result.stderr.lower()


# ---------------------------------------------------------------------------
# _load_youtube_config
# ---------------------------------------------------------------------------


class TestLoadYoutubeConfig:
    """Tests for _load_youtube_config() helper."""

    def test_returns_youtube_section(self, tool):
        """Returns the youtube config dict."""
        fake_config = {"youtube": {"po_token": "mweb.gvs+TOKEN"}}
        with (
            patch(
                "claudetube.config.loader._find_project_config",
                return_value=Path("/fake/config.yaml"),
            ),
            patch(
                "claudetube.config.loader._load_yaml_config",
                return_value=fake_config,
            ),
        ):
            cfg = tool._load_youtube_config()
            assert cfg == {"po_token": "mweb.gvs+TOKEN"}

    def test_returns_empty_when_no_config(self, tool):
        """No config → empty dict."""
        with (
            patch("claudetube.config.loader._find_project_config", return_value=None),
            patch("claudetube.config.loader._load_yaml_config", return_value=None),
        ):
            cfg = tool._load_youtube_config()
            assert cfg == {}

    def test_returns_empty_on_exception(self, tool):
        """Exception → empty dict, no crash."""
        with patch(
            "claudetube.config.loader._find_project_config",
            side_effect=RuntimeError("boom"),
        ):
            cfg = tool._load_youtube_config()
            assert cfg == {}
