"""Tests for claudetube."""

import pytest
from pathlib import Path

from claudetube.fast import extract_video_id, _format_srt_time


class TestExtractVideoId:
    """Tests for video ID extraction."""

    def test_standard_url(self):
        assert extract_video_id("https://youtube.com/watch?v=dYP2V_nK8o0") == "dYP2V_nK8o0"

    def test_short_url(self):
        assert extract_video_id("https://youtu.be/dYP2V_nK8o0") == "dYP2V_nK8o0"

    def test_embed_url(self):
        assert extract_video_id("https://youtube.com/embed/dYP2V_nK8o0") == "dYP2V_nK8o0"

    def test_just_id(self):
        assert extract_video_id("dYP2V_nK8o0") == "dYP2V_nK8o0"

    def test_with_extra_params(self):
        assert extract_video_id("https://youtube.com/watch?v=dYP2V_nK8o0&t=120") == "dYP2V_nK8o0"


class TestFormatSrtTime:
    """Tests for SRT timestamp formatting."""

    def test_seconds_only(self):
        assert _format_srt_time(45.5) == "00:00:45,500"

    def test_minutes_and_seconds(self):
        assert _format_srt_time(90.123) == "00:01:30,123"

    def test_hours(self):
        assert _format_srt_time(3661.5) == "01:01:01,500"

    def test_zero(self):
        assert _format_srt_time(0) == "00:00:00,000"
