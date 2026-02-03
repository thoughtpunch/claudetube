"""Tests for yt-dlp output templates."""

from pathlib import Path

import pytest

from claudetube.config.output_templates import (
    NO_CHANNEL,
    NO_PLAYLIST,
    TEMPLATES,
    OutputTemplates,
    build_outtmpl_dict,
    get_output_path,
)


class TestOutputTemplatesConstants:
    """Tests for template constants."""

    def test_sentinel_values(self):
        """Test sentinel values match VideoPath conventions."""
        assert NO_CHANNEL == "no_channel"
        assert NO_PLAYLIST == "no_playlist"


class TestOutputTemplates:
    """Tests for OutputTemplates dataclass."""

    def test_default_templates_exist(self):
        """Test default templates are created."""
        templates = OutputTemplates.default()
        assert templates.audio is not None
        assert templates.video is not None
        assert templates.thumbnail is not None
        assert templates.subtitle is not None
        assert templates.infojson is not None

    def test_templates_contain_extractor(self):
        """Test all templates start with extractor placeholder."""
        templates = OutputTemplates.default()
        for name in ["audio", "video", "thumbnail", "subtitle", "infojson"]:
            template = getattr(templates, name)
            assert template.startswith("%(extractor)s/"), f"{name} should start with extractor"

    def test_templates_contain_channel_fallback(self):
        """Test templates have channel_id with fallback."""
        templates = OutputTemplates.default()
        # Check audio template as representative
        assert f"%(channel_id|uploader_id|{NO_CHANNEL})s" in templates.audio

    def test_templates_contain_playlist_fallback(self):
        """Test templates have playlist_id with fallback."""
        templates = OutputTemplates.default()
        assert f"%(playlist_id|{NO_PLAYLIST})s" in templates.audio

    def test_templates_contain_video_id(self):
        """Test templates have video id placeholder."""
        templates = OutputTemplates.default()
        assert "%(id)s" in templates.audio

    def test_audio_template_filename(self):
        """Test audio template ends with audio.%(ext)s."""
        templates = OutputTemplates.default()
        assert templates.audio.endswith("/audio.%(ext)s")

    def test_video_template_filename(self):
        """Test video template ends with video.%(ext)s."""
        templates = OutputTemplates.default()
        assert templates.video.endswith("/video.%(ext)s")

    def test_thumbnail_template_filename(self):
        """Test thumbnail template ends with thumbnail.%(ext)s."""
        templates = OutputTemplates.default()
        assert templates.thumbnail.endswith("/thumbnail.%(ext)s")

    def test_subtitle_template_filename(self):
        """Test subtitle template uses video id in filename."""
        templates = OutputTemplates.default()
        assert templates.subtitle.endswith("/%(id)s.%(ext)s")

    def test_infojson_template_filename(self):
        """Test infojson template ends with info.json."""
        templates = OutputTemplates.default()
        assert templates.infojson.endswith("/info.json")

    def test_templates_immutable(self):
        """Test OutputTemplates is frozen (immutable)."""
        templates = OutputTemplates.default()
        with pytest.raises(AttributeError):
            templates.audio = "new_value"


class TestGlobalTemplates:
    """Tests for global TEMPLATES instance."""

    def test_templates_exists(self):
        """Test TEMPLATES global is available."""
        assert TEMPLATES is not None
        assert isinstance(TEMPLATES, OutputTemplates)

    def test_templates_equals_default(self):
        """Test TEMPLATES matches default templates."""
        default = OutputTemplates.default()
        assert TEMPLATES.audio == default.audio
        assert TEMPLATES.video == default.video


class TestGetOutputPath:
    """Tests for get_output_path function."""

    def test_audio_path(self):
        """Test audio output path construction."""
        cache_base = Path("/cache")
        result = get_output_path("audio", cache_base)
        assert result.startswith("/cache/")
        assert "audio.%(ext)s" in result

    def test_video_path(self):
        """Test video output path construction."""
        cache_base = Path("/cache")
        result = get_output_path("video", cache_base)
        assert result.startswith("/cache/")
        assert "video.%(ext)s" in result

    def test_thumbnail_path(self):
        """Test thumbnail output path construction."""
        cache_base = Path("/cache")
        result = get_output_path("thumbnail", cache_base)
        assert result.startswith("/cache/")
        assert "thumbnail.%(ext)s" in result

    def test_subtitle_path(self):
        """Test subtitle output path construction."""
        cache_base = Path("/cache")
        result = get_output_path("subtitle", cache_base)
        assert result.startswith("/cache/")
        assert "%(id)s.%(ext)s" in result

    def test_infojson_path(self):
        """Test infojson output path construction."""
        cache_base = Path("/cache")
        result = get_output_path("infojson", cache_base)
        assert result.startswith("/cache/")
        assert "info.json" in result

    def test_invalid_template_type(self):
        """Test ValueError for invalid template type."""
        with pytest.raises(ValueError) as exc_info:
            get_output_path("invalid", Path("/cache"))
        assert "Unknown template type" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_path_is_absolute(self):
        """Test output path is absolute when cache_base is absolute."""
        cache_base = Path("/home/user/.cache")
        result = get_output_path("audio", cache_base)
        assert result.startswith("/home/user/.cache/")


class TestBuildOuttmplDict:
    """Tests for build_outtmpl_dict function."""

    def test_returns_dict(self):
        """Test function returns a dictionary."""
        result = build_outtmpl_dict(Path("/cache"))
        assert isinstance(result, dict)

    def test_contains_default_key(self):
        """Test dict contains 'default' key for video."""
        result = build_outtmpl_dict(Path("/cache"))
        assert "default" in result
        assert "video.%(ext)s" in result["default"]

    def test_contains_thumbnail_key(self):
        """Test dict contains 'thumbnail' key."""
        result = build_outtmpl_dict(Path("/cache"))
        assert "thumbnail" in result
        assert "thumbnail.%(ext)s" in result["thumbnail"]

    def test_contains_subtitle_key(self):
        """Test dict contains 'subtitle' key."""
        result = build_outtmpl_dict(Path("/cache"))
        assert "subtitle" in result
        assert "%(id)s.%(ext)s" in result["subtitle"]

    def test_contains_infojson_key(self):
        """Test dict contains 'infojson' key."""
        result = build_outtmpl_dict(Path("/cache"))
        assert "infojson" in result
        assert "info.json" in result["infojson"]

    def test_all_paths_use_cache_base(self):
        """Test all paths start with the cache base."""
        cache_base = Path("/custom/cache/path")
        result = build_outtmpl_dict(cache_base)
        for key, path in result.items():
            assert path.startswith("/custom/cache/path/"), f"{key} should use cache_base"


class TestTemplateHierarchy:
    """Tests for correct hierarchical path structure."""

    def test_full_hierarchy_structure(self):
        """Test templates follow domain/channel/playlist/video_id structure."""
        templates = OutputTemplates.default()
        # Audio template as representative
        parts = templates.audio.split("/")

        # Should be: extractor / channel / playlist / video_id / filename
        assert parts[0] == "%(extractor)s"
        assert NO_CHANNEL in parts[1]  # channel with fallback
        assert NO_PLAYLIST in parts[2]  # playlist with fallback
        assert parts[3] == "%(id)s"
        assert parts[4] == "audio.%(ext)s"

    def test_channel_fallback_order(self):
        """Test channel fallback order: channel_id -> uploader_id -> no_channel."""
        templates = OutputTemplates.default()
        channel_part = templates.audio.split("/")[1]
        # Should try channel_id first, then uploader_id, then fallback
        assert channel_part == f"%(channel_id|uploader_id|{NO_CHANNEL})s"
