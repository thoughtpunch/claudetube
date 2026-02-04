"""Tests for slash command markdown files."""

from pathlib import Path

import pytest

COMMANDS_DIR = Path(__file__).parent.parent / "commands"


class TestYtCommand:
    """Tests for the main /yt command."""

    def test_yt_md_exists(self):
        """yt.md should exist."""
        assert (COMMANDS_DIR / "yt.md").exists()

    def test_yt_md_has_cache_check_first(self):
        """yt.md should check cache before invoking Python."""
        content = (COMMANDS_DIR / "yt.md").read_text()
        assert "Check Cache First" in content
        assert "Fast Path" in content

    def test_yt_md_has_video_id_patterns(self):
        """yt.md should document video ID extraction patterns."""
        content = (COMMANDS_DIR / "yt.md").read_text()
        assert "youtube.com/watch?v=" in content
        assert "youtu.be/" in content
        assert "youtube.com/embed/" in content

    def test_yt_md_has_cache_location(self):
        """yt.md should document the cache location."""
        content = (COMMANDS_DIR / "yt.md").read_text()
        assert "~/.claudetube/cache/" in content
        assert "state.json" in content

    def test_yt_md_has_skip_python_instruction(self):
        """yt.md should instruct to skip Python when cached."""
        content = (COMMANDS_DIR / "yt.md").read_text()
        assert "Skip Python entirely" in content
        assert "Read the cached files directly" in content

    def test_yt_md_has_fallback_to_python(self):
        """yt.md should fall back to Python when not cached."""
        content = (COMMANDS_DIR / "yt.md").read_text()
        assert "If NOT cached" in content
        assert "process_video" in content


class TestYtTranscriptCommand:
    """Tests for /yt:transcript command."""

    def test_transcript_md_exists(self):
        """transcript.md should exist."""
        assert (COMMANDS_DIR / "yt" / "transcript.md").exists()

    def test_transcript_md_is_bash_only(self):
        """transcript.md should not invoke Python for reading cached transcripts."""
        content = (COMMANDS_DIR / "yt" / "transcript.md").read_text()
        # Should use bash to check cache
        assert "CACHE_DIR=" in content
        # Should not have process_video call
        assert "process_video" not in content

    def test_transcript_md_suggests_yt_first(self):
        """transcript.md should tell user to run /yt first if not cached."""
        content = (COMMANDS_DIR / "yt" / "transcript.md").read_text()
        assert "/yt" in content
        assert "not cached" in content.lower()


class TestYtListCommand:
    """Tests for /yt:list command."""

    def test_list_md_exists(self):
        """list.md should exist."""
        assert (COMMANDS_DIR / "yt" / "list.md").exists()

    def test_list_md_is_bash_only(self):
        """list.md should use pure bash, no Python."""
        content = (COMMANDS_DIR / "yt" / "list.md").read_text()
        assert "CACHE_DIR=" in content
        # Should not invoke Python
        assert "python3" not in content.lower()


class TestYtSeeCommand:
    """Tests for /yt:see command."""

    def test_see_md_exists(self):
        """see.md should exist."""
        assert (COMMANDS_DIR / "yt" / "see.md").exists()

    def test_see_md_has_cache_check_first(self):
        """see.md should check cache before invoking Python."""
        content = (COMMANDS_DIR / "yt" / "see.md").read_text()
        assert "Check Cache First" in content
        assert "state.json" in content

    def test_see_md_suggests_yt_first(self):
        """see.md should tell user to run /yt first if not cached."""
        content = (COMMANDS_DIR / "yt" / "see.md").read_text()
        assert "/yt" in content
        assert "NOT cached" in content

    def test_see_md_has_quality_escalation(self):
        """see.md should document quality escalation."""
        content = (COMMANDS_DIR / "yt" / "see.md").read_text()
        assert "lowest" in content
        assert "highest" in content
        assert "Auto-Escalation" in content or "escalat" in content.lower()


class TestYtHqCommand:
    """Tests for /yt:hq command."""

    def test_hq_md_exists(self):
        """hq.md should exist."""
        assert (COMMANDS_DIR / "yt" / "hq.md").exists()

    def test_hq_md_has_cache_check_first(self):
        """hq.md should check cache before invoking Python."""
        content = (COMMANDS_DIR / "yt" / "hq.md").read_text()
        assert "Check Cache First" in content
        assert "state.json" in content

    def test_hq_md_suggests_yt_first(self):
        """hq.md should tell user to run /yt first if not cached."""
        content = (COMMANDS_DIR / "yt" / "hq.md").read_text()
        assert "/yt" in content
        assert "NOT cached" in content

    def test_hq_md_mentions_code_and_text(self):
        """hq.md should mention its use case for code/text."""
        content = (COMMANDS_DIR / "yt" / "hq.md").read_text()
        assert "code" in content.lower()
        assert "text" in content.lower()


class TestAllCommandsHaveFrontmatter:
    """Tests that all command files have proper frontmatter."""

    @pytest.mark.parametrize(
        "command_path",
        [
            "yt.md",
            "yt/transcript.md",
            "yt/list.md",
            "yt/see.md",
            "yt/hq.md",
        ],
    )
    def test_has_frontmatter(self, command_path):
        """Command files should have YAML frontmatter."""
        content = (COMMANDS_DIR / command_path).read_text()
        assert content.startswith("---")
        assert "description:" in content
        assert "allowed-tools:" in content

    @pytest.mark.parametrize(
        "command_path",
        [
            "yt.md",
            "yt/transcript.md",
            "yt/list.md",
            "yt/see.md",
            "yt/hq.md",
        ],
    )
    def test_has_description(self, command_path):
        """Command files should have a description."""
        content = (COMMANDS_DIR / command_path).read_text()
        # Extract frontmatter
        parts = content.split("---", 2)
        assert len(parts) >= 3
        frontmatter = parts[1]
        assert "description:" in frontmatter
