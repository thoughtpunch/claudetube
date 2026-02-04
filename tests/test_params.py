"""Tests for URL query parameter extraction and timestamp parsing."""

from __future__ import annotations

from claudetube.parsing.params import extract_query_params, parse_timestamp


class TestExtractQueryParams:
    """Tests for extract_query_params()."""

    def test_youtube_playlist_and_timestamp(self):
        url = "https://youtube.com/watch?v=abc&list=PLxxx&t=120"
        result = extract_query_params(url, "youtube")
        assert result == {"playlist": "PLxxx", "start_time": "120"}

    def test_youtube_index_and_channel(self):
        url = "https://youtube.com/watch?v=abc&index=3&ab_channel=Test"
        result = extract_query_params(url, "youtube")
        assert result == {"playlist_position": "3", "channel_hint": "Test"}

    def test_youtube_start_param(self):
        url = "https://youtube.com/watch?v=abc&start=300"
        result = extract_query_params(url, "youtube")
        assert result == {"start_time": "300"}

    def test_youtube_feature_param(self):
        url = "https://youtube.com/watch?v=abc&feature=shared"
        result = extract_query_params(url, "youtube")
        assert result == {"referral_source": "shared"}

    def test_youtube_t_takes_precedence_over_start(self):
        """When both t= and start= are present, t= wins (listed first in mapping)."""
        url = "https://youtube.com/watch?v=abc&t=60&start=120"
        result = extract_query_params(url, "youtube")
        # Both map to start_time; t= is processed first so it wins
        assert result["start_time"] == "60"

    def test_vimeo_private_hash(self):
        url = "https://vimeo.com/123?h=abc123"
        result = extract_query_params(url, "vimeo")
        assert result == {"private_hash": "abc123"}

    def test_vimeo_time(self):
        url = "https://vimeo.com/123?time=1m30s"
        result = extract_query_params(url, "vimeo")
        assert result == {"start_time": "1m30s"}

    def test_twitch_timestamp(self):
        url = "https://twitch.tv/videos/123?t=1h30m45s"
        result = extract_query_params(url, "twitch")
        assert result == {"start_time": "1h30m45s"}

    def test_generic_utm_params(self):
        url = "https://example.com/video?utm_source=twitter&utm_medium=social"
        result = extract_query_params(url, "unknown_provider")
        assert result == {"referral_source": "twitter", "referral_medium": "social"}

    def test_utm_params_with_known_provider(self):
        url = "https://youtube.com/watch?v=abc&utm_source=twitter"
        result = extract_query_params(url, "youtube")
        assert result == {"referral_source": "twitter"}

    def test_provider_specific_overrides_utm(self):
        """Provider-specific referral_source (feature=) takes priority over utm_source."""
        url = "https://youtube.com/watch?v=abc&feature=shared&utm_source=twitter"
        result = extract_query_params(url, "youtube")
        # feature= maps to referral_source and is provider-specific, so it wins
        assert result["referral_source"] == "shared"

    def test_no_query_params(self):
        url = "https://youtube.com/watch?v=abc"
        result = extract_query_params(url, "youtube")
        assert result == {}

    def test_unknown_provider_no_params(self):
        url = "https://example.com/video"
        result = extract_query_params(url, "unknown")
        assert result == {}

    def test_case_insensitive_provider(self):
        url = "https://youtube.com/watch?v=abc&t=60"
        result = extract_query_params(url, "YouTube")
        assert result == {"start_time": "60"}

    def test_empty_query_string(self):
        url = "https://youtube.com/watch"
        result = extract_query_params(url, "youtube")
        assert result == {}

    def test_all_youtube_params(self):
        url = (
            "https://youtube.com/watch?v=abc&list=PLxxx&t=120"
            "&index=5&ab_channel=TestChannel&feature=shared"
        )
        result = extract_query_params(url, "youtube")
        assert result == {
            "playlist": "PLxxx",
            "start_time": "120",
            "playlist_position": "5",
            "channel_hint": "TestChannel",
            "referral_source": "shared",
        }


class TestParseTimestamp:
    """Tests for parse_timestamp()."""

    # Pure seconds
    def test_pure_seconds_integer(self):
        assert parse_timestamp("120") == 120.0

    def test_pure_seconds_string(self):
        assert parse_timestamp("300") == 300.0

    def test_zero(self):
        assert parse_timestamp("0") == 0.0

    def test_float_seconds(self):
        assert parse_timestamp("90.5") == 90.5

    # Compact format (XhYmZs)
    def test_minutes_and_seconds(self):
        assert parse_timestamp("2m30s") == 150.0

    def test_hours_minutes_seconds(self):
        assert parse_timestamp("1h2m3s") == 3723.0

    def test_hours_minutes_seconds_large(self):
        assert parse_timestamp("1h30m45s") == 5445.0

    def test_hours_only(self):
        assert parse_timestamp("2h") == 7200.0

    def test_minutes_only(self):
        assert parse_timestamp("5m") == 300.0

    def test_seconds_only_compact(self):
        assert parse_timestamp("45s") == 45.0

    def test_hours_and_seconds_no_minutes(self):
        assert parse_timestamp("1h30s") == 3630.0

    def test_compact_case_insensitive(self):
        assert parse_timestamp("1H2M3S") == 3723.0

    # Colon format
    def test_colon_hms(self):
        assert parse_timestamp("1:30:45") == 5445.0

    def test_colon_ms(self):
        assert parse_timestamp("2:30") == 150.0

    def test_colon_zero(self):
        assert parse_timestamp("0:00") == 0.0

    def test_colon_large_hours(self):
        assert parse_timestamp("10:00:00") == 36000.0

    # Invalid inputs
    def test_invalid_string(self):
        assert parse_timestamp("invalid") is None

    def test_empty_string(self):
        assert parse_timestamp("") is None

    def test_whitespace_only(self):
        assert parse_timestamp("   ") is None

    def test_negative_seconds(self):
        assert parse_timestamp("-10") is None

    def test_random_letters(self):
        assert parse_timestamp("abc") is None

    def test_partial_format(self):
        # "h" alone without digits doesn't match
        assert parse_timestamp("h") is None

    # Edge cases
    def test_whitespace_padding(self):
        assert parse_timestamp("  120  ") == 120.0

    def test_zero_compact(self):
        assert parse_timestamp("0h0m0s") == 0.0
