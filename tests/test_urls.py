"""Tests for URL parsing and video ID extraction."""

import pytest

from claudetube.urls import (
    VideoURL,
    extract_playlist_id,
    extract_url_context,
    extract_video_id,
    get_provider_count,
    get_provider_for_url,
    list_supported_providers,
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
