"""Tests for yt-dlp output templates."""

from pathlib import Path

import pytest

from claudetube.config.output_templates import (
    NO_CHANNEL,
    NO_PLAYLIST,
    TEMPLATES,
    OutputTemplates,
    build_audio_download_args,
    build_cli_args,
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


class TestBuildCliArgs:
    """Tests for build_cli_args function."""

    def test_returns_list(self):
        """Test function returns a list."""
        result = build_cli_args(Path("/cache"))
        assert isinstance(result, list)

    def test_contains_home_path(self):
        """Test args contain -P home:CACHE_BASE."""
        result = build_cli_args(Path("/cache"))
        assert "-P" in result
        # Find the home path argument
        p_indices = [i for i, x in enumerate(result) if x == "-P"]
        home_args = [result[i + 1] for i in p_indices if i + 1 < len(result)]
        assert any(arg.startswith("home:") for arg in home_args)

    def test_contains_thumbnail_path(self):
        """Test args contain -P thumbnail:CACHE_BASE."""
        result = build_cli_args(Path("/cache"))
        p_indices = [i for i, x in enumerate(result) if x == "-P"]
        path_args = [result[i + 1] for i in p_indices if i + 1 < len(result)]
        assert any(arg.startswith("thumbnail:") for arg in path_args)

    def test_contains_subtitle_path(self):
        """Test args contain -P subtitle:CACHE_BASE."""
        result = build_cli_args(Path("/cache"))
        p_indices = [i for i, x in enumerate(result) if x == "-P"]
        path_args = [result[i + 1] for i in p_indices if i + 1 < len(result)]
        assert any(arg.startswith("subtitle:") for arg in path_args)

    def test_contains_infojson_path(self):
        """Test args contain -P infojson:CACHE_BASE."""
        result = build_cli_args(Path("/cache"))
        p_indices = [i for i, x in enumerate(result) if x == "-P"]
        path_args = [result[i + 1] for i in p_indices if i + 1 < len(result)]
        assert any(arg.startswith("infojson:") for arg in path_args)

    def test_contains_output_templates(self):
        """Test args contain -o TYPE:TEMPLATE for each type."""
        result = build_cli_args(Path("/cache"))
        o_indices = [i for i, x in enumerate(result) if x == "-o"]
        output_args = [result[i + 1] for i in o_indices if i + 1 < len(result)]

        # Should have thumbnail, subtitle, and infojson type prefixes
        assert any(arg.startswith("thumbnail:") for arg in output_args)
        assert any(arg.startswith("subtitle:") for arg in output_args)
        assert any(arg.startswith("infojson:") for arg in output_args)

    def test_contains_write_flags(self):
        """Test args contain write flags for each asset type."""
        result = build_cli_args(Path("/cache"))
        assert "--write-thumbnail" in result
        assert "--write-subs" in result
        assert "--write-info-json" in result

    def test_exclude_thumbnail(self):
        """Test thumbnail can be excluded."""
        result = build_cli_args(Path("/cache"), include_thumbnail=False)
        assert "--write-thumbnail" not in result
        # No thumbnail path prefix
        p_indices = [i for i, x in enumerate(result) if x == "-P"]
        path_args = [result[i + 1] for i in p_indices if i + 1 < len(result)]
        assert not any(arg.startswith("thumbnail:") for arg in path_args)

    def test_exclude_subtitles(self):
        """Test subtitles can be excluded."""
        result = build_cli_args(Path("/cache"), include_subtitles=False)
        assert "--write-subs" not in result
        # No subtitle path prefix
        p_indices = [i for i, x in enumerate(result) if x == "-P"]
        path_args = [result[i + 1] for i in p_indices if i + 1 < len(result)]
        assert not any(arg.startswith("subtitle:") for arg in path_args)

    def test_exclude_infojson(self):
        """Test infojson can be excluded."""
        result = build_cli_args(Path("/cache"), include_infojson=False)
        assert "--write-info-json" not in result

    def test_subtitle_options(self):
        """Test subtitle-specific options are included."""
        result = build_cli_args(Path("/cache"))
        assert "--write-auto-subs" in result
        assert "--sub-langs" in result
        assert "--convert-subs" in result

    def test_thumbnail_conversion(self):
        """Test thumbnail conversion to jpg is included."""
        result = build_cli_args(Path("/cache"))
        assert "--convert-thumbnails" in result
        idx = result.index("--convert-thumbnails")
        assert result[idx + 1] == "jpg"


class TestBuildAudioDownloadArgs:
    """Tests for build_audio_download_args function."""

    def test_returns_list(self):
        """Test function returns a list."""
        result = build_audio_download_args(Path("/cache"), "https://example.com/video")
        assert isinstance(result, list)

    def test_url_at_end(self):
        """Test URL is the last argument."""
        url = "https://example.com/video"
        result = build_audio_download_args(Path("/cache"), url)
        assert result[-1] == url

    def test_contains_audio_extraction(self):
        """Test args contain audio extraction flags."""
        result = build_audio_download_args(Path("/cache"), "https://example.com/video")
        assert "-x" in result
        assert "--audio-format" in result
        assert "mp3" in result

    def test_contains_quality_setting(self):
        """Test args contain audio quality setting."""
        result = build_audio_download_args(
            Path("/cache"), "https://example.com/video", quality="128K"
        )
        assert "--audio-quality" in result
        idx = result.index("--audio-quality")
        assert result[idx + 1] == "128K"

    def test_default_quality(self):
        """Test default quality is 64K."""
        result = build_audio_download_args(Path("/cache"), "https://example.com/video")
        idx = result.index("--audio-quality")
        assert result[idx + 1] == "64K"

    def test_no_playlist_flag(self):
        """Test --no-playlist flag is included."""
        result = build_audio_download_args(Path("/cache"), "https://example.com/video")
        assert "--no-playlist" in result

    def test_thumbnail_included_by_default(self):
        """Test thumbnail is included by default."""
        result = build_audio_download_args(Path("/cache"), "https://example.com/video")
        assert "--write-thumbnail" in result

    def test_subtitles_included_by_default(self):
        """Test subtitles are included by default."""
        result = build_audio_download_args(Path("/cache"), "https://example.com/video")
        assert "--write-subs" in result

    def test_infojson_excluded_by_default(self):
        """Test infojson is excluded by default."""
        result = build_audio_download_args(Path("/cache"), "https://example.com/video")
        assert "--write-info-json" not in result

    def test_include_infojson(self):
        """Test infojson can be included."""
        result = build_audio_download_args(
            Path("/cache"), "https://example.com/video", include_infojson=True
        )
        assert "--write-info-json" in result

    def test_exclude_thumbnail(self):
        """Test thumbnail can be excluded."""
        result = build_audio_download_args(
            Path("/cache"), "https://example.com/video", include_thumbnail=False
        )
        assert "--write-thumbnail" not in result

    def test_exclude_subtitles(self):
        """Test subtitles can be excluded."""
        result = build_audio_download_args(
            Path("/cache"), "https://example.com/video", include_subtitles=False
        )
        assert "--write-subs" not in result

    def test_contains_format_selection(self):
        """Test args contain best audio format selection."""
        result = build_audio_download_args(Path("/cache"), "https://example.com/video")
        assert "-f" in result
        idx = result.index("-f")
        assert result[idx + 1] == "ba"

    def test_contains_multi_path_args(self):
        """Test args contain multi-path output configuration."""
        result = build_audio_download_args(Path("/cache"), "https://example.com/video")
        # Should have -P for home path at minimum
        assert "-P" in result
        p_indices = [i for i, x in enumerate(result) if x == "-P"]
        path_args = [result[i + 1] for i in p_indices if i + 1 < len(result)]
        assert any(arg.startswith("home:") for arg in path_args)
