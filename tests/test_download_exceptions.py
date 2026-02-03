"""Tests for typed download exception hierarchy."""

import pytest

from claudetube.exceptions import (
    AgeRestrictedError,
    DownloadError,
    ExtractorError,
    FormatNotAvailableError,
    GeoRestrictedError,
    NetworkError,
    RateLimitError,
    VideoUnavailableError,
    YouTubeAuthError,
)
from claudetube.tools.yt_dlp import (
    YtDlpError,
    parse_yt_dlp_error,
    yt_dlp_error_to_exception,
)


class TestDownloadErrorHierarchy:
    """Test that all download exceptions inherit from DownloadError."""

    def test_youtube_auth_error_inherits_from_download_error(self):
        error = YouTubeAuthError("HTTP Error 403: Forbidden")
        assert isinstance(error, DownloadError)

    def test_geo_restricted_error_inherits_from_download_error(self):
        error = GeoRestrictedError("Video not available in your country")
        assert isinstance(error, DownloadError)

    def test_format_not_available_error_inherits_from_download_error(self):
        error = FormatNotAvailableError("Requested format is not available")
        assert isinstance(error, DownloadError)

    def test_rate_limit_error_inherits_from_download_error(self):
        error = RateLimitError("HTTP Error 429: Too Many Requests")
        assert isinstance(error, DownloadError)

    def test_extractor_error_inherits_from_download_error(self):
        error = ExtractorError("Unsupported URL")
        assert isinstance(error, DownloadError)

    def test_network_error_inherits_from_download_error(self):
        error = NetworkError("Connection refused")
        assert isinstance(error, DownloadError)

    def test_video_unavailable_error_inherits_from_download_error(self):
        error = VideoUnavailableError("Video has been removed")
        assert isinstance(error, DownloadError)

    def test_age_restricted_error_inherits_from_download_error(self):
        error = AgeRestrictedError("Sign in to confirm your age")
        assert isinstance(error, DownloadError)


class TestDownloadErrorAttributes:
    """Test that exceptions have the correct attributes."""

    def test_youtube_auth_error_attributes(self):
        error = YouTubeAuthError(
            "HTTP Error 403",
            stderr="ERROR: HTTP Error 403: Forbidden",
            auth_level=2,
            clients_tried=["default", "mweb"],
        )
        assert error.message == "HTTP Error 403"
        assert error.category == "auth"
        assert error.auth_level == 2
        assert error.clients_tried == ["default", "mweb"]
        assert "auth_level" in error.details
        assert "clients_tried" in error.details

    def test_geo_restricted_error_attributes(self):
        error = GeoRestrictedError(
            "Not available in your country",
            stderr="ERROR: Video not available",
        )
        assert error.message == "Not available in your country"
        assert error.category == "geo_restricted"
        assert "VPN" in error.suggestion

    def test_format_not_available_error_attributes(self):
        error = FormatNotAvailableError(
            "No audio formats",
            available_formats=["mp4", "webm"],
        )
        assert error.message == "No audio formats"
        assert error.category == "format_unavailable"
        assert error.available_formats == ["mp4", "webm"]
        assert "available_formats" in error.details

    def test_network_error_http_code(self):
        error = NetworkError("HTTP Error 500", http_code=500)
        assert error.http_code == 500
        assert error.details["http_code"] == 500

    def test_video_unavailable_error_reason(self):
        error = VideoUnavailableError("Video removed", reason="copyright")
        assert error.reason == "copyright"
        assert error.details["reason"] == "copyright"


class TestDownloadErrorToDict:
    """Test the to_dict method for MCP error responses."""

    def test_youtube_auth_error_to_dict(self):
        error = YouTubeAuthError(
            "HTTP Error 403",
            auth_level=2,
            clients_tried=["default", "mweb"],
        )
        result = error.to_dict()

        assert result["type"] == "YouTubeAuthError"
        assert result["message"] == "HTTP Error 403"
        assert result["category"] == "auth"
        assert result["details"]["auth_level"] == 2
        assert result["details"]["clients_tried"] == ["default", "mweb"]
        assert "suggestion" in result

    def test_base_download_error_to_dict(self):
        error = DownloadError(
            "Unknown error",
            category="unknown",
            details={"extra": "info"},
        )
        result = error.to_dict()

        assert result["type"] == "DownloadError"
        assert result["message"] == "Unknown error"
        assert result["category"] == "unknown"
        assert result["details"]["extra"] == "info"


class TestParseYtDlpError:
    """Test parsing yt-dlp stderr into YtDlpError."""

    def test_parse_403_error(self):
        stderr = "ERROR: HTTP Error 403: Forbidden"
        error = parse_yt_dlp_error(stderr)

        assert error.category == "auth"
        assert "403" in error.message

    def test_parse_geo_restricted(self):
        stderr = "ERROR: Video not available in your country"
        error = parse_yt_dlp_error(stderr)

        assert error.category == "geo_restricted"

    def test_parse_format_unavailable(self):
        stderr = "ERROR: Requested format is not available"
        error = parse_yt_dlp_error(stderr)

        assert error.category == "format_unavailable"

    def test_parse_rate_limited(self):
        stderr = "ERROR: HTTP Error 429: Too Many Requests"
        error = parse_yt_dlp_error(stderr)

        assert error.category == "rate_limited"

    def test_parse_age_restricted(self):
        stderr = "ERROR: Sign in to confirm your age"
        error = parse_yt_dlp_error(stderr)

        assert error.category == "age_restricted"

    def test_parse_private_video(self):
        stderr = "ERROR: This video is private"
        error = parse_yt_dlp_error(stderr)

        assert error.category == "private"

    def test_parse_copyright(self):
        stderr = "ERROR: Video blocked due to copyright"
        error = parse_yt_dlp_error(stderr)

        assert error.category == "copyright"

    def test_parse_network_error(self):
        stderr = "ERROR: Connection refused"
        error = parse_yt_dlp_error(stderr)

        assert error.category == "network"

    def test_parse_http_error_with_code(self):
        stderr = "ERROR: HTTP Error 500: Internal Server Error"
        error = parse_yt_dlp_error(stderr)

        assert error.category == "http_error"
        assert error.details.get("http_code") == 500

    def test_parse_unknown_error(self):
        stderr = "ERROR: Something unexpected happened"
        error = parse_yt_dlp_error(stderr)

        assert error.category == "unknown"


class TestYtDlpErrorToException:
    """Test converting YtDlpError to typed exceptions."""

    def test_auth_error_to_youtube_auth_error(self):
        yt_error = YtDlpError(
            category="auth",
            message="HTTP Error 403",
            stderr="ERROR: HTTP Error 403",
            details={},
        )
        exc = yt_dlp_error_to_exception(yt_error, is_youtube=True)

        assert isinstance(exc, YouTubeAuthError)

    def test_geo_restricted_to_geo_restricted_error(self):
        yt_error = YtDlpError(
            category="geo_restricted",
            message="Not available in your country",
            stderr="ERROR: Not available",
            details={},
        )
        exc = yt_dlp_error_to_exception(yt_error)

        assert isinstance(exc, GeoRestrictedError)

    def test_format_unavailable_to_format_not_available_error(self):
        yt_error = YtDlpError(
            category="format_unavailable",
            message="Format not available",
            stderr="ERROR: Format not available",
            details={},
        )
        exc = yt_dlp_error_to_exception(yt_error)

        assert isinstance(exc, FormatNotAvailableError)

    def test_rate_limited_to_rate_limit_error(self):
        yt_error = YtDlpError(
            category="rate_limited",
            message="Too many requests",
            stderr="ERROR: 429",
            details={},
        )
        exc = yt_dlp_error_to_exception(yt_error)

        assert isinstance(exc, RateLimitError)

    def test_network_to_network_error(self):
        yt_error = YtDlpError(
            category="network",
            message="Connection refused",
            stderr="ERROR: Connection refused",
            details={},
        )
        exc = yt_dlp_error_to_exception(yt_error)

        assert isinstance(exc, NetworkError)

    def test_unavailable_to_video_unavailable_error(self):
        yt_error = YtDlpError(
            category="unavailable",
            message="Video unavailable",
            stderr="ERROR: Video unavailable",
            details={},
        )
        exc = yt_dlp_error_to_exception(yt_error)

        assert isinstance(exc, VideoUnavailableError)

    def test_age_restricted_to_age_restricted_error(self):
        yt_error = YtDlpError(
            category="age_restricted",
            message="Age verification required",
            stderr="ERROR: Sign in to confirm your age",
            details={},
        )
        exc = yt_dlp_error_to_exception(yt_error)

        assert isinstance(exc, AgeRestrictedError)

    def test_http_403_youtube_to_youtube_auth_error(self):
        """HTTP 403 on YouTube should become YouTubeAuthError."""
        yt_error = YtDlpError(
            category="http_error",
            message="HTTP Error 403",
            stderr="ERROR: HTTP Error 403",
            details={"http_code": 403},
        )
        exc = yt_dlp_error_to_exception(yt_error, is_youtube=True)

        assert isinstance(exc, YouTubeAuthError)

    def test_http_500_to_network_error(self):
        """Other HTTP errors should become NetworkError."""
        yt_error = YtDlpError(
            category="http_error",
            message="HTTP Error 500",
            stderr="ERROR: HTTP Error 500",
            details={"http_code": 500},
        )
        exc = yt_dlp_error_to_exception(yt_error, is_youtube=True)

        assert isinstance(exc, NetworkError)
        assert exc.http_code == 500

    def test_unknown_to_download_error(self):
        """Unknown errors should become base DownloadError."""
        yt_error = YtDlpError(
            category="unknown",
            message="Something failed",
            stderr="ERROR: Something failed",
            details={},
        )
        exc = yt_dlp_error_to_exception(yt_error)

        assert type(exc) is DownloadError


class TestBackwardsCompatibility:
    """Test that code catching DownloadError still works."""

    def test_catch_youtube_auth_error_as_download_error(self):
        """YouTubeAuthError should be catchable as DownloadError."""
        with pytest.raises(DownloadError):
            raise YouTubeAuthError("Auth failed")

    def test_catch_all_subclasses_as_download_error(self):
        """All typed exceptions should be catchable as DownloadError."""
        exceptions = [
            YouTubeAuthError("test"),
            GeoRestrictedError("test"),
            FormatNotAvailableError("test"),
            RateLimitError("test"),
            ExtractorError("test"),
            NetworkError("test"),
            VideoUnavailableError("test"),
            AgeRestrictedError("test"),
        ]

        for exc in exceptions:
            with pytest.raises(DownloadError):
                raise exc
