"""Tests for VideoPath model and sanitize_domain function."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from claudetube.models.video_path import VideoPath, sanitize_domain


class TestSanitizeDomain:
    """Tests for the sanitize_domain() function."""

    def test_simple_domain(self):
        assert sanitize_domain("youtube.com") == "youtube"

    def test_strips_www(self):
        assert sanitize_domain("www.youtube.com") == "youtube"

    def test_strips_m_prefix(self):
        assert sanitize_domain("m.facebook.com") == "facebook"

    def test_strips_mobile_prefix(self):
        assert sanitize_domain("mobile.twitter.com") == "twitter"

    def test_strips_clips_prefix(self):
        assert sanitize_domain("clips.twitch.tv") == "twitch"

    def test_strips_player_prefix(self):
        assert sanitize_domain("player.vimeo.com") == "vimeo"

    def test_strips_music_prefix(self):
        assert sanitize_domain("music.youtube.com") == "youtube"

    def test_country_code_tld(self):
        assert sanitize_domain("bbc.co.uk") == "bbc"

    def test_short_domain(self):
        assert sanitize_domain("youtu.be") == "youtu"

    def test_already_clean(self):
        assert sanitize_domain("vimeo.com") == "vimeo"

    def test_mixed_case(self):
        assert sanitize_domain("YouTube.com") == "youtube"

    def test_strips_whitespace(self):
        assert sanitize_domain("  youtube.com  ") == "youtube"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Cannot extract domain"):
            sanitize_domain("")

    def test_only_dots_raises(self):
        with pytest.raises(ValueError, match="Cannot extract domain"):
            sanitize_domain("...")

    def test_only_prefix_and_tld(self):
        # www.com -> strip www. -> "com" -> take first part -> "com"
        assert sanitize_domain("www.com") == "com"

    def test_dailymotion(self):
        assert sanitize_domain("dailymotion.com") == "dailymotion"

    def test_dai_ly(self):
        assert sanitize_domain("dai.ly") == "dai"

    def test_reddit(self):
        assert sanitize_domain("www.reddit.com") == "reddit"

    def test_old_reddit(self):
        assert sanitize_domain("old.reddit.com") == "old"

    def test_b23_tv(self):
        assert sanitize_domain("b23.tv") == "b23"

    def test_single_part_hostname(self):
        assert sanitize_domain("localhost") == "localhost"


class TestVideoPathConstruction:
    """Tests for basic VideoPath construction and validation."""

    def test_basic_construction(self):
        vp = VideoPath(domain="youtube", video_id="abc123")
        assert vp.domain == "youtube"
        assert vp.video_id == "abc123"
        assert vp.channel is None
        assert vp.playlist is None

    def test_full_construction(self):
        vp = VideoPath(
            domain="youtube",
            channel="UCxxx",
            playlist="PLxxx",
            video_id="dQw4w9WgXcQ",
        )
        assert vp.domain == "youtube"
        assert vp.channel == "UCxxx"
        assert vp.playlist == "PLxxx"
        assert vp.video_id == "dQw4w9WgXcQ"

    def test_none_channel_accepted(self):
        vp = VideoPath(domain="youtube", channel=None, video_id="abc")
        assert vp.channel is None

    def test_none_playlist_accepted(self):
        vp = VideoPath(domain="youtube", playlist=None, video_id="abc")
        assert vp.playlist is None

    def test_empty_domain_raises(self):
        with pytest.raises(ValidationError):
            VideoPath(domain="", video_id="abc")

    def test_empty_video_id_raises(self):
        with pytest.raises(ValidationError):
            VideoPath(domain="youtube", video_id="")

    def test_whitespace_video_id_raises(self):
        with pytest.raises(ValidationError):
            VideoPath(domain="youtube", video_id="   ")

    def test_empty_channel_raises(self):
        with pytest.raises(ValidationError):
            VideoPath(domain="youtube", channel="", video_id="abc")

    def test_whitespace_channel_raises(self):
        with pytest.raises(ValidationError):
            VideoPath(domain="youtube", channel="   ", video_id="abc")

    def test_empty_playlist_raises(self):
        with pytest.raises(ValidationError):
            VideoPath(domain="youtube", playlist="", video_id="abc")

    def test_whitespace_playlist_raises(self):
        with pytest.raises(ValidationError):
            VideoPath(domain="youtube", playlist="   ", video_id="abc")

    def test_uppercase_domain_raises(self):
        with pytest.raises(ValidationError):
            VideoPath(domain="YouTube", video_id="abc")

    def test_domain_with_special_chars_raises(self):
        with pytest.raises(ValidationError):
            VideoPath(domain="you-tube", video_id="abc")

    def test_domain_starting_with_number_raises(self):
        with pytest.raises(ValidationError):
            VideoPath(domain="9gag", video_id="abc")

    def test_domain_with_dots_raises(self):
        with pytest.raises(ValidationError):
            VideoPath(domain="youtube.com", video_id="abc")

    def test_frozen(self):
        vp = VideoPath(domain="youtube", video_id="abc")
        with pytest.raises(ValidationError):
            vp.domain = "vimeo"  # type: ignore[misc]

    def test_local_domain(self):
        vp = VideoPath(domain="local", video_id="my_video_abc123")
        assert vp.domain == "local"


class TestVideoPathRelativePath:
    """Tests for relative_path() method."""

    def test_all_components(self):
        vp = VideoPath(
            domain="youtube",
            channel="UCxxx",
            playlist="PLxxx",
            video_id="abc",
        )
        assert vp.relative_path() == Path("youtube/UCxxx/PLxxx/abc")

    def test_no_channel_no_playlist(self):
        vp = VideoPath(domain="youtube", video_id="abc")
        assert vp.relative_path() == Path("youtube/no_channel/no_playlist/abc")

    def test_channel_no_playlist(self):
        vp = VideoPath(domain="youtube", channel="UCxxx", video_id="abc")
        assert vp.relative_path() == Path("youtube/UCxxx/no_playlist/abc")

    def test_no_channel_with_playlist(self):
        vp = VideoPath(domain="youtube", playlist="PLxxx", video_id="abc")
        assert vp.relative_path() == Path("youtube/no_channel/PLxxx/abc")

    def test_local_path(self):
        vp = VideoPath(domain="local", video_id="screen_rec_a3f2dd1e")
        assert vp.relative_path() == Path(
            "local/no_channel/no_playlist/screen_rec_a3f2dd1e"
        )


class TestVideoPathFromCachePath:
    """Tests for from_cache_path() class method."""

    def test_full_path(self):
        vp = VideoPath.from_cache_path("youtube/UCxxx/PLxxx/abc")
        assert vp.domain == "youtube"
        assert vp.channel == "UCxxx"
        assert vp.playlist == "PLxxx"
        assert vp.video_id == "abc"

    def test_no_channel_sentinel(self):
        vp = VideoPath.from_cache_path("youtube/no_channel/PLxxx/abc")
        assert vp.channel is None
        assert vp.playlist == "PLxxx"

    def test_no_playlist_sentinel(self):
        vp = VideoPath.from_cache_path("youtube/UCxxx/no_playlist/abc")
        assert vp.channel == "UCxxx"
        assert vp.playlist is None

    def test_both_sentinels(self):
        vp = VideoPath.from_cache_path("youtube/no_channel/no_playlist/abc")
        assert vp.channel is None
        assert vp.playlist is None

    def test_too_few_components_raises(self):
        with pytest.raises(ValueError, match="at least 4 components"):
            VideoPath.from_cache_path("youtube/abc")

    def test_extra_components_ignored(self):
        vp = VideoPath.from_cache_path("youtube/UCxxx/PLxxx/abc/extra/parts")
        assert vp.domain == "youtube"
        assert vp.video_id == "abc"

    def test_local_path(self):
        vp = VideoPath.from_cache_path(
            "local/no_channel/no_playlist/screen_rec_a3f2dd1e"
        )
        assert vp.domain == "local"
        assert vp.channel is None
        assert vp.playlist is None
        assert vp.video_id == "screen_rec_a3f2dd1e"


class TestVideoPathRoundTrip:
    """Tests that from_cache_path(str(vp.relative_path())) == vp."""

    @pytest.mark.parametrize(
        "domain,channel,playlist,video_id",
        [
            ("youtube", None, None, "dQw4w9WgXcQ"),
            ("youtube", "UCxxx", None, "abc123"),
            ("youtube", None, "PLxxx", "abc123"),
            ("youtube", "UCxxx", "PLxxx", "abc123"),
            ("twitter", "elikiowa", None, "1879432010"),
            ("vimeo", None, None, "912345678"),
            ("local", None, None, "screen_rec_a3f2dd1e"),
            ("twitch", "ninja", None, "12345"),
            ("dailymotion", "cnn", None, "x8fgh12"),
        ],
    )
    def test_round_trip(self, domain, channel, playlist, video_id):
        original = VideoPath(
            domain=domain,
            channel=channel,
            playlist=playlist,
            video_id=video_id,
        )
        reconstructed = VideoPath.from_cache_path(str(original.relative_path()))
        assert reconstructed == original


class TestVideoPathCacheDir:
    """Tests for cache_dir() method."""

    def test_cache_dir(self):
        vp = VideoPath(domain="youtube", channel="UCxxx", video_id="abc")
        base = Path("/home/user/.claude/video_cache")
        expected = base / "youtube" / "UCxxx" / "no_playlist" / "abc"
        assert vp.cache_dir(base) == expected

    def test_cache_dir_with_all_components(self):
        vp = VideoPath(
            domain="youtube",
            channel="UCxxx",
            playlist="PLxxx",
            video_id="abc",
        )
        base = Path("/cache")
        assert vp.cache_dir(base) == Path("/cache/youtube/UCxxx/PLxxx/abc")


class TestVideoPathFromLocal:
    """Tests for from_local() class method."""

    def test_from_local(self):
        mock_local = MagicMock()
        mock_local.video_id = "screen_rec_a3f2dd1e"
        vp = VideoPath.from_local(mock_local)
        assert vp.domain == "local"
        assert vp.channel is None
        assert vp.playlist is None
        assert vp.video_id == "screen_rec_a3f2dd1e"


class TestVideoPathFromUrl:
    """Tests for from_url() class method."""

    def test_youtube_url(self):
        vp = VideoPath.from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert vp.domain == "youtube"
        assert vp.video_id == "dQw4w9WgXcQ"

    def test_youtube_short_url(self):
        vp = VideoPath.from_url("https://youtu.be/dQw4w9WgXcQ")
        assert vp.domain == "youtu"
        assert vp.video_id == "dQw4w9WgXcQ"

    def test_vimeo_url(self):
        vp = VideoPath.from_url("https://vimeo.com/912345678")
        assert vp.domain == "vimeo"
        assert vp.video_id == "912345678"

    def test_twitter_url(self):
        vp = VideoPath.from_url("https://x.com/user/status/1879432010")
        assert vp.domain == "x"
        assert vp.video_id == "1879432010"

    def test_with_metadata_channel(self):
        vp = VideoPath.from_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            metadata={"channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw"},
        )
        assert vp.domain == "youtube"
        assert vp.channel == "UCuAXFkgsw1L7xaCfnd5JJOw"
        assert vp.video_id == "dQw4w9WgXcQ"

    def test_with_metadata_playlist(self):
        vp = VideoPath.from_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            metadata={"playlist_id": "PLRqwX-V7Uu6ZiZxtDDRCi6uhfTH4FilpH"},
        )
        assert vp.playlist == "PLRqwX-V7Uu6ZiZxtDDRCi6uhfTH4FilpH"

    def test_with_metadata_uploader_id_fallback(self):
        vp = VideoPath.from_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            metadata={"uploader_id": "RickAstleyVEVO"},
        )
        assert vp.channel == "RickAstleyVEVO"

    def test_with_metadata_channel_name_fallback(self):
        vp = VideoPath.from_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            metadata={"channel": "Rick Astley"},
        )
        assert vp.channel == "Rick_Astley"

    def test_with_metadata_playlist_title_fallback(self):
        vp = VideoPath.from_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            metadata={"playlist_title": "My Favorite Videos!"},
        )
        assert vp.playlist == "My_Favorite_Videos_"

    def test_no_metadata(self):
        vp = VideoPath.from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert vp.channel is None
        assert vp.playlist is None

    def test_twitch_url_with_channel(self):
        vp = VideoPath.from_url(
            "https://www.twitch.tv/ninja/clip/AwkwardHelplessSalamanderSwiftRage"
        )
        assert vp.domain == "twitch"
        assert vp.channel == "ninja"

    def test_odysee_url_with_channel(self):
        vp = VideoPath.from_url("https://odysee.com/@DistroTube:2/my-video:a")
        assert vp.domain == "odysee"
        assert vp.channel == "DistroTube"

    def test_metadata_channel_id_takes_priority(self):
        """channel_id from metadata takes priority over uploader_id."""
        vp = VideoPath.from_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            metadata={
                "channel_id": "UC_actual_id",
                "uploader_id": "uploader_fallback",
                "channel": "Display Name",
            },
        )
        assert vp.channel == "UC_actual_id"

    def test_long_channel_name_truncated(self):
        vp = VideoPath.from_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            metadata={"channel": "A" * 100},
        )
        assert len(vp.channel) <= 60

    def test_empty_metadata_values_ignored(self):
        vp = VideoPath.from_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            metadata={"channel_id": "", "playlist_id": ""},
        )
        assert vp.channel is None
        assert vp.playlist is None


class TestVideoPathEquality:
    """Tests for equality and hashing."""

    def test_equal_instances(self):
        a = VideoPath(domain="youtube", video_id="abc")
        b = VideoPath(domain="youtube", video_id="abc")
        assert a == b

    def test_unequal_instances(self):
        a = VideoPath(domain="youtube", video_id="abc")
        b = VideoPath(domain="youtube", video_id="xyz")
        assert a != b

    def test_hashable(self):
        vp = VideoPath(domain="youtube", video_id="abc")
        s = {vp}
        assert vp in s
