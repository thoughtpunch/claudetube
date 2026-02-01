"""Tests for URL parsing and video ID extraction."""

import pytest

from claudetube.urls import (
    LocalFile,
    LocalFileError,
    VideoURL,
    extract_playlist_id,
    extract_url_context,
    extract_video_id,
    get_provider_count,
    get_provider_for_url,
    is_local_file,
    is_url,
    list_supported_providers,
    parse_input,
)


class TestVideoURL:
    """Tests for VideoURL Pydantic model."""

    # YouTube tests
    def test_youtube_standard_url(self):
        v = VideoURL.parse("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert v.video_id == "dQw4w9WgXcQ"
        assert v.provider == "YouTube"

    def test_youtube_short_url(self):
        v = VideoURL.parse("https://youtu.be/dQw4w9WgXcQ")
        assert v.video_id == "dQw4w9WgXcQ"
        assert v.provider == "YouTube"

    def test_youtube_embed_url(self):
        v = VideoURL.parse("https://youtube.com/embed/dQw4w9WgXcQ")
        assert v.video_id == "dQw4w9WgXcQ"
        assert v.provider == "YouTube"

    def test_youtube_shorts_url(self):
        v = VideoURL.parse("https://youtube.com/shorts/dQw4w9WgXcQ")
        assert v.video_id == "dQw4w9WgXcQ"
        assert v.provider == "YouTube"

    def test_youtube_with_extra_params(self):
        v = VideoURL.parse("https://youtube.com/watch?v=dQw4w9WgXcQ&t=120&list=PLxyz")
        assert v.video_id == "dQw4w9WgXcQ"

    # TikTok tests
    def test_tiktok_video_url(self):
        v = VideoURL.parse(
            "https://www.tiktok.com/@celiatoks/video/7454538602173697313"
        )
        assert v.video_id == "7454538602173697313"
        assert v.provider == "TikTok"

    # Rumble tests
    def test_rumble_url(self):
        v = VideoURL.parse("https://rumble.com/v4p5yd5-title-here.html")
        assert v.video_id == "v4p5yd5"
        assert v.provider == "Rumble"

    # Vimeo tests
    def test_vimeo_url(self):
        v = VideoURL.parse("https://vimeo.com/347119375")
        assert v.video_id == "347119375"
        assert v.provider == "Vimeo"

    # Dailymotion tests
    def test_dailymotion_url(self):
        v = VideoURL.parse("https://www.dailymotion.com/video/x9yl448")
        assert v.video_id == "x9yl448"
        assert v.provider == "Dailymotion"

    def test_dailymotion_short_url(self):
        v = VideoURL.parse("https://dai.ly/x9yl448")
        assert v.video_id == "x9yl448"
        assert v.provider == "Dailymotion"

    # Twitter/X tests
    def test_twitter_url(self):
        v = VideoURL.parse("https://twitter.com/user/status/1855025943389229210")
        assert v.video_id == "1855025943389229210"
        assert v.provider == "Twitter/X"

    def test_x_url(self):
        v = VideoURL.parse("https://x.com/user/status/1855025943389229210")
        assert v.video_id == "1855025943389229210"
        assert v.provider == "Twitter/X"

    # Twitch tests
    def test_twitch_vod_url(self):
        v = VideoURL.parse("https://www.twitch.tv/videos/1234567890")
        assert v.video_id == "1234567890"
        assert v.provider == "Twitch"

    def test_twitch_clip_url(self):
        v = VideoURL.parse("https://clips.twitch.tv/AmusedStrongSnood-g5jFT33XOQBOJIRV")
        assert v.video_id == "AmusedStrongSnood-g5jFT33XOQBOJIRV"
        assert v.provider == "Twitch"

    # Reddit tests
    def test_reddit_url(self):
        v = VideoURL.parse("https://www.reddit.com/r/funny/comments/1bnkjoh/title/")
        assert v.video_id == "1bnkjoh"
        assert v.provider == "Reddit"

    # Odysee tests
    def test_odysee_url(self):
        v = VideoURL.parse("https://odysee.com/@channel:a/video-title:c")
        assert v.video_id == "video-title"
        assert v.provider == "Odysee"
        assert v.provider_data.get("channel") == "channel"

    # BitChute tests
    def test_bitchute_url(self):
        v = VideoURL.parse("https://www.bitchute.com/video/9ctGQyScZWK7/")
        assert v.video_id == "9ctGQyScZWK7"
        assert v.provider == "BitChute"

    # Archive.org tests
    def test_archive_org_url(self):
        v = VideoURL.parse("https://archive.org/details/SampleVideo_908")
        assert v.video_id == "SampleVideo_908"
        assert v.provider == "Archive.org"

    # Bilibili tests
    def test_bilibili_bv_url(self):
        v = VideoURL.parse("https://www.bilibili.com/video/BV1GJ411x7h7")
        assert v.video_id == "BV1GJ411x7h7"
        assert v.provider == "Bilibili"

    # Instagram tests
    def test_instagram_reel_url(self):
        v = VideoURL.parse("https://www.instagram.com/reel/ABC123xyz/")
        assert v.video_id == "ABC123xyz"
        assert v.provider == "Instagram"

    # Facebook tests
    def test_facebook_watch_url(self):
        v = VideoURL.parse("https://www.facebook.com/watch/?v=1234567890")
        assert v.video_id == "1234567890"
        assert v.provider == "Facebook"

    # Generic/fallback tests
    def test_unknown_site_fallback(self):
        v = VideoURL.parse("https://unknown-site.com/video/abc123")
        assert v.video_id == "abc123"
        assert v.provider is None
        assert v.is_known_provider is False

    def test_url_without_scheme(self):
        v = VideoURL.parse("youtube.com/watch?v=dQw4w9WgXcQ")
        assert v.video_id == "dQw4w9WgXcQ"
        assert v.url.startswith("https://")

    def test_empty_url_raises(self):
        with pytest.raises(ValueError, match="empty"):
            VideoURL.parse("")

    def test_whitespace_url_raises(self):
        with pytest.raises(ValueError, match="empty"):
            VideoURL.parse("   ")

    def test_try_parse_returns_none_on_error(self):
        result = VideoURL.try_parse("")
        assert result is None

    def test_cache_key_short_id(self):
        v = VideoURL.parse("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert v.cache_key == "dQw4w9WgXcQ"

    def test_str_representation(self):
        v = VideoURL.parse("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert str(v) == "YouTube:dQw4w9WgXcQ"

    def test_str_representation_unknown(self):
        v = VideoURL.parse("https://unknown.com/video/abc123")
        assert str(v) == "abc123"


class TestExtractVideoId:
    """Tests for backwards-compatible extract_video_id function."""

    def test_youtube_url(self):
        assert (
            extract_video_id("https://youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        )

    def test_short_url(self):
        assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_rumble_url(self):
        assert extract_video_id("https://rumble.com/v4p5yd5-title.html") == "v4p5yd5"

    def test_invalid_url_returns_sanitized(self):
        result = extract_video_id("not a url at all")
        assert "/" not in result
        assert ":" not in result


class TestExtractPlaylistId:
    """Tests for playlist ID extraction."""

    def test_extracts_playlist_id(self):
        url = "https://youtube.com/watch?v=abc&list=PLtest123"
        assert extract_playlist_id(url) == "PLtest123"

    def test_returns_none_without_playlist(self):
        url = "https://youtube.com/watch?v=abc"
        assert extract_playlist_id(url) is None

    def test_returns_none_for_empty(self):
        assert extract_playlist_id("") is None


class TestExtractUrlContext:
    """Tests for URL context extraction."""

    def test_extracts_full_context(self):
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ&list=PLtest"
        ctx = extract_url_context(url)
        assert ctx["video_id"] == "dQw4w9WgXcQ"
        assert ctx["playlist_id"] == "PLtest"
        assert ctx["provider"] == "YouTube"
        assert "list=" not in ctx["clean_url"]

    def test_includes_provider_data(self):
        url = "https://odysee.com/@channel:a/video:c"
        ctx = extract_url_context(url)
        assert ctx["provider"] == "Odysee"
        assert "channel" in ctx["provider_data"]


class TestGetProviderForUrl:
    """Tests for provider detection."""

    def test_youtube(self):
        assert (
            get_provider_for_url("https://youtube.com/watch?v=dQw4w9WgXcQ") == "YouTube"
        )

    def test_tiktok(self):
        assert get_provider_for_url("https://tiktok.com/@user/video/123") == "TikTok"

    def test_unknown(self):
        assert get_provider_for_url("https://unknown.com/video") is None


class TestListSupportedProviders:
    """Tests for provider listing."""

    def test_returns_list(self):
        providers = list_supported_providers()
        assert isinstance(providers, list)
        assert len(providers) == 70  # We have exactly 70 providers

    def test_provider_count(self):
        assert get_provider_count() == 70

    def test_includes_major_providers(self):
        providers = list_supported_providers()
        assert "YouTube" in providers
        assert "TikTok" in providers
        assert "Vimeo" in providers
        assert "Twitter/X" in providers
        assert "Facebook" in providers
        assert "Instagram" in providers
        assert "Twitch" in providers
        assert "Reddit" in providers


class TestMultiSiteUrlParsing:
    """Integration tests across multiple sites."""

    @pytest.mark.parametrize(
        "url,expected_id,expected_provider",
        [
            # Top platforms
            ("https://www.youtube.com/watch?v=XqZsoesa55w", "XqZsoesa55w", "YouTube"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ", "YouTube"),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ", "YouTube"),
            ("https://www.facebook.com/watch/?v=1234567890", "1234567890", "Facebook"),
            ("https://www.instagram.com/reel/ABC123xyz/", "ABC123xyz", "Instagram"),
            (
                "https://www.tiktok.com/@user/video/7454538602173697313",
                "7454538602173697313",
                "TikTok",
            ),
            ("https://twitter.com/user/status/123456789", "123456789", "Twitter/X"),
            ("https://x.com/user/status/123456789", "123456789", "Twitter/X"),
            # Video platforms
            ("https://vimeo.com/347119375", "347119375", "Vimeo"),
            ("https://player.vimeo.com/video/347119375", "347119375", "Vimeo"),
            ("https://www.dailymotion.com/video/x9yl448", "x9yl448", "Dailymotion"),
            ("https://dai.ly/x9yl448", "x9yl448", "Dailymotion"),
            ("https://rumble.com/v4p5yd5-title.html", "v4p5yd5", "Rumble"),
            ("https://rumble.com/embed/v4p5yd5", "v4p5yd5", "Rumble"),
            ("https://streamable.com/abc123", "abc123", "Streamable"),
            # Streaming platforms
            ("https://www.twitch.tv/videos/1234567890", "1234567890", "Twitch"),
            (
                "https://clips.twitch.tv/AmusedStrongSnood-ABC123",
                "AmusedStrongSnood-ABC123",
                "Twitch",
            ),
            ("https://kick.com/username/clips/abc123", "abc123", "Kick"),
            # Alternative platforms
            ("https://odysee.com/@channel:a/video-title:c", "video-title", "Odysee"),
            (
                "https://www.bitchute.com/video/9ctGQyScZWK7/",
                "9ctGQyScZWK7",
                "BitChute",
            ),
            ("https://nebula.tv/videos/video-slug", "video-slug", "Nebula"),
            ("https://floatplane.com/post/abc123", "abc123", "Floatplane"),
            # Educational/business
            (
                "https://www.ted.com/talks/speaker_title_here",
                "speaker_title_here",
                "Ted",
            ),
            (
                "https://loom.com/share/abc123def456789012345678901234ab",
                "abc123def456789012345678901234ab",
                "Loom",
            ),
            ("https://wistia.com/medias/abc123", "abc123", "Wistia"),
            ("https://vidyard.com/watch/ABC123", "ABC123", "Vidyard"),
            # Regional platforms
            ("https://www.bilibili.com/video/BV1GJ411x7h7", "BV1GJ411x7h7", "Bilibili"),
            ("https://www.nicovideo.jp/watch/sm12345678", "sm12345678", "Niconico"),
            # Archives
            (
                "https://archive.org/details/SampleVideo_908",
                "SampleVideo_908",
                "Archive.org",
            ),
            # Social/Reddit
            ("https://www.reddit.com/r/sub/comments/abc123/title/", "abc123", "Reddit"),
            ("https://v.redd.it/abc123xyz", "abc123xyz", "Reddit"),
            # Media
            ("https://imgur.com/gallery/abc123", "abc123", "Imgur"),
            ("https://gfycat.com/FastHappyCat", "FastHappyCat", "Gfycat"),
            ("https://giphy.com/gifs/funny-abc123XYZ", "abc123XYZ", "Giphy"),
            # Audio
            ("https://soundcloud.com/artist/track-name", "track-name", "Soundcloud"),
            ("https://mixcloud.com/dj/mix-name", "mix-name", "Mixcloud"),
            # News/Sports
            ("https://espn.com/video/clip/_/id/12345678", "12345678", "ESPN"),
            ("https://www.9gag.com/gag/abc123XY", "abc123XY", "9GAG"),
        ],
    )
    def test_url_parsing(self, url, expected_id, expected_provider):
        v = VideoURL.parse(url)
        assert v.video_id == expected_id
        assert v.provider == expected_provider


# =============================================================================
# Local File Detection Tests
# =============================================================================


class TestIsLocalFile:
    """Tests for is_local_file() function."""

    def test_http_url_not_local(self):
        assert is_local_file("https://youtube.com/watch?v=abc") is False
        assert is_local_file("http://example.com/video.mp4") is False

    def test_nonexistent_file_not_local(self):
        assert is_local_file("/nonexistent/path/video.mp4") is False

    def test_empty_string_not_local(self):
        assert is_local_file("") is False
        assert is_local_file("   ") is False

    def test_none_not_local(self):
        assert is_local_file(None) is False

    def test_absolute_path_is_local(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.touch()
        assert is_local_file(str(video)) is True

    def test_relative_path_is_local(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "test.mp4"
        video.touch()
        assert is_local_file("./test.mp4") is True

    def test_home_relative_path(self, tmp_path, monkeypatch):
        # Create a test file in home directory (use tmp as fake home)
        monkeypatch.setenv("HOME", str(tmp_path))
        video = tmp_path / "test_video.mp4"
        video.touch()
        # This test uses expanduser which reads HOME env var
        result = is_local_file("~/test_video.mp4")
        assert result is True

    def test_file_uri_is_local(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.touch()
        assert is_local_file(f"file://{video}") is True
        assert is_local_file(f"file:///{video}") is True

    def test_domain_like_string_is_url(self):
        # Should be treated as URL, not local file
        assert is_local_file("youtube.com/watch?v=abc") is False
        assert is_local_file("example.com/video.mp4") is False


class TestIsUrl:
    """Tests for is_url() function."""

    def test_http_urls(self):
        assert is_url("https://youtube.com/watch?v=abc") is True
        assert is_url("http://example.com/video.mp4") is True

    def test_file_uri_not_url(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.touch()
        assert is_url(f"file://{video}") is False

    def test_local_file_not_url(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.touch()
        assert is_url(str(video)) is False

    def test_empty_string_not_url(self):
        assert is_url("") is False
        assert is_url("   ") is False


class TestLocalFile:
    """Tests for LocalFile model."""

    def test_parse_absolute_path(self, tmp_path):
        video = tmp_path / "test_video.mp4"
        video.touch()

        lf = LocalFile.parse(str(video))
        assert lf.path == video
        assert lf.extension == ".mp4"
        assert lf.is_video is True
        assert lf.filename == "test_video.mp4"
        assert lf.stem == "test_video"

    def test_parse_file_uri(self, tmp_path):
        video = tmp_path / "test.mkv"
        video.touch()

        lf = LocalFile.parse(f"file://{video}")
        assert lf.path == video
        assert lf.extension == ".mkv"

    def test_parse_file_uri_with_spaces(self, tmp_path):
        video = tmp_path / "my video file.mp4"
        video.touch()

        # URL-encoded spaces
        lf = LocalFile.parse(f"file://{str(video).replace(' ', '%20')}")
        assert lf.path == video

    def test_parse_audio_file(self, tmp_path):
        audio = tmp_path / "podcast.mp3"
        audio.touch()

        lf = LocalFile.parse(str(audio))
        assert lf.extension == ".mp3"
        assert lf.is_video is False

    def test_parse_nonexistent_raises(self):
        with pytest.raises(LocalFileError, match="File not found"):
            LocalFile.parse("/nonexistent/video.mp4")

    def test_parse_unsupported_format_raises(self, tmp_path):
        txt = tmp_path / "document.txt"
        txt.touch()

        with pytest.raises(LocalFileError, match="Unsupported file format"):
            LocalFile.parse(str(txt))

    def test_parse_directory_raises(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        with pytest.raises(LocalFileError, match="Not a file"):
            LocalFile.parse(str(subdir))

    def test_try_parse_returns_none_on_error(self):
        assert LocalFile.try_parse("/nonexistent/video.mp4") is None
        assert LocalFile.try_parse("") is None

    def test_all_video_extensions(self, tmp_path):
        """Test that all documented video extensions are supported."""
        from claudetube.urls import SUPPORTED_VIDEO_EXTENSIONS

        for ext in SUPPORTED_VIDEO_EXTENSIONS:
            video = tmp_path / f"test{ext}"
            video.touch()
            lf = LocalFile.parse(str(video))
            assert lf.is_video is True
            video.unlink()

    def test_all_audio_extensions(self, tmp_path):
        """Test that all documented audio extensions are supported."""
        from claudetube.urls import SUPPORTED_AUDIO_EXTENSIONS

        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            audio = tmp_path / f"test{ext}"
            audio.touch()
            lf = LocalFile.parse(str(audio))
            assert lf.is_video is False
            audio.unlink()


class TestParseInput:
    """Tests for parse_input() unified input parser."""

    def test_parse_url(self):
        result = parse_input("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert result["type"] == "url"
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["provider"] == "YouTube"

    def test_parse_local_file(self, tmp_path):
        video = tmp_path / "my_recording.mp4"
        video.touch()

        result = parse_input(str(video))
        assert result["type"] == "local"
        assert result["path"] == str(video)
        assert result["filename"] == "my_recording.mp4"
        assert result["stem"] == "my_recording"
        assert result["extension"] == ".mp4"
        assert result["is_video"] is True

    def test_parse_local_audio(self, tmp_path):
        audio = tmp_path / "podcast.mp3"
        audio.touch()

        result = parse_input(str(audio))
        assert result["type"] == "local"
        assert result["is_video"] is False

    def test_parse_file_uri(self, tmp_path):
        video = tmp_path / "test.webm"
        video.touch()

        result = parse_input(f"file://{video}")
        assert result["type"] == "local"
        assert result["extension"] == ".webm"


class TestLocalFileEdgeCases:
    """Edge case tests for local file detection."""

    def test_relative_path_with_dots(self, tmp_path, monkeypatch):
        """Test ../path/to/video.mp4 style paths."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        video = tmp_path / "video.mp4"
        video.touch()

        monkeypatch.chdir(subdir)
        assert is_local_file("../video.mp4") is True

    def test_path_with_special_chars(self, tmp_path):
        """Test paths with spaces and special characters."""
        video = tmp_path / "my video (2024) - final.mp4"
        video.touch()

        lf = LocalFile.parse(str(video))
        assert lf.path == video

    def test_uppercase_extension(self, tmp_path):
        """Test that uppercase extensions work."""
        video = tmp_path / "VIDEO.MP4"
        video.touch()

        lf = LocalFile.parse(str(video))
        assert lf.extension == ".mp4"  # Should be normalized to lowercase
        assert lf.is_video is True

    def test_url_without_scheme_not_confused_with_file(self):
        """Ensure URLs without http:// aren't confused with files."""
        # These should be treated as URLs, not local files
        assert is_local_file("youtube.com/watch?v=abc") is False
        assert is_local_file("vimeo.com/123456") is False

    def test_bare_filename_in_cwd(self, tmp_path, monkeypatch):
        """Test bare filename (no path) works if file exists in cwd."""
        monkeypatch.chdir(tmp_path)
        video = tmp_path / "video.mp4"
        video.touch()

        assert is_local_file("video.mp4") is True
