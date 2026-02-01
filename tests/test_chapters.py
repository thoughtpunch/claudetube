"""Tests for chapter extraction."""

import pytest

from claudetube.models import Chapter
from claudetube.operations.chapters import extract_youtube_chapters, parse_timestamp


class TestParseTimestamp:
    """Tests for timestamp parsing."""

    def test_minutes_seconds(self):
        assert parse_timestamp("1:23") == 83.0

    def test_hours_minutes_seconds(self):
        assert parse_timestamp("1:23:45") == 5025.0

    def test_zero(self):
        assert parse_timestamp("0:00") == 0.0

    def test_single_digit_minutes(self):
        assert parse_timestamp("5:30") == 330.0

    def test_double_digit_minutes(self):
        assert parse_timestamp("12:45") == 765.0

    def test_large_hours(self):
        assert parse_timestamp("10:00:00") == 36000.0


class TestChapterModel:
    """Tests for Chapter dataclass."""

    def test_create_chapter(self):
        ch = Chapter(
            title="Introduction",
            start=0.0,
            end=60.0,
            source="youtube_chapters",
            confidence=0.95,
        )
        assert ch.title == "Introduction"
        assert ch.start == 0.0
        assert ch.end == 60.0
        assert ch.source == "youtube_chapters"
        assert ch.confidence == 0.95

    def test_default_values(self):
        ch = Chapter(title="Test", start=10.0)
        assert ch.end is None
        assert ch.source == "unknown"
        assert ch.confidence == 0.5

    def test_to_dict(self):
        ch = Chapter(
            title="Test",
            start=10.0,
            end=20.0,
            source="test",
            confidence=0.8,
        )
        d = ch.to_dict()
        assert d["title"] == "Test"
        assert d["start"] == 10.0
        assert d["end"] == 20.0
        assert d["source"] == "test"
        assert d["confidence"] == 0.8

    def test_from_dict(self):
        d = {
            "title": "From Dict",
            "start": 5.0,
            "end": 15.0,
            "source": "dict_source",
            "confidence": 0.7,
        }
        ch = Chapter.from_dict(d)
        assert ch.title == "From Dict"
        assert ch.start == 5.0
        assert ch.end == 15.0
        assert ch.source == "dict_source"
        assert ch.confidence == 0.7

    def test_from_dict_defaults(self):
        ch = Chapter.from_dict({})
        assert ch.title == ""
        assert ch.start == 0.0
        assert ch.end is None
        assert ch.source == "unknown"
        assert ch.confidence == 0.5


class TestExtractYoutubeChapters:
    """Tests for extract_youtube_chapters function."""

    def test_native_chapters(self):
        """Extract chapters from yt-dlp native chapters field."""
        video_info = {
            "chapters": [
                {"title": "Intro", "start_time": 0.0, "end_time": 60.0},
                {"title": "Main Content", "start_time": 60.0, "end_time": 300.0},
                {"title": "Conclusion", "start_time": 300.0, "end_time": 360.0},
            ]
        }
        chapters = extract_youtube_chapters(video_info)

        assert len(chapters) == 3
        assert chapters[0].title == "Intro"
        assert chapters[0].start == 0.0
        assert chapters[0].end == 60.0
        assert chapters[0].source == "youtube_chapters"
        assert chapters[0].confidence == 0.95

        assert chapters[1].title == "Main Content"
        assert chapters[2].title == "Conclusion"

    def test_description_parsing_simple(self):
        """Parse chapters from video description."""
        video_info = {
            "description": """Check out this video!

0:00 Introduction
1:30 First Topic
5:00 Second Topic
10:00 Conclusion

Thanks for watching!""",
            "duration": 720,
        }
        chapters = extract_youtube_chapters(video_info)

        assert len(chapters) == 4
        assert chapters[0].title == "Introduction"
        assert chapters[0].start == 0.0
        assert chapters[0].end == 90.0  # 1:30
        assert chapters[0].source == "description_parsed"
        assert chapters[0].confidence == 0.9

        assert chapters[1].title == "First Topic"
        assert chapters[1].start == 90.0
        assert chapters[1].end == 300.0

        assert chapters[3].title == "Conclusion"
        assert chapters[3].end == 720.0  # video duration

    def test_description_with_dashes(self):
        """Parse chapters with dash separators."""
        video_info = {
            "description": """0:00 - Intro
2:30 - Main Part
5:00 - Outro""",
            "duration": 400,
        }
        chapters = extract_youtube_chapters(video_info)

        assert len(chapters) == 3
        assert chapters[0].title == "Intro"
        assert chapters[1].title == "Main Part"
        assert chapters[2].title == "Outro"

    def test_description_with_hours(self):
        """Parse chapters with hour timestamps."""
        video_info = {
            "description": """0:00:00 Start
1:30:00 Middle Section
2:00:30 End Part""",
            "duration": 7500,
        }
        chapters = extract_youtube_chapters(video_info)

        assert len(chapters) == 3
        assert chapters[0].start == 0.0
        assert chapters[1].start == 5400.0  # 1:30:00
        assert chapters[2].start == 7230.0  # 2:00:30

    def test_native_chapters_preferred(self):
        """Native chapters take precedence over description parsing."""
        video_info = {
            "chapters": [
                {"title": "Native Chapter", "start_time": 0.0, "end_time": 60.0},
            ],
            "description": "0:00 Description Chapter",
        }
        chapters = extract_youtube_chapters(video_info)

        assert len(chapters) == 1
        assert chapters[0].title == "Native Chapter"
        assert chapters[0].source == "youtube_chapters"

    def test_empty_video_info(self):
        """Empty video info returns empty list."""
        chapters = extract_youtube_chapters({})
        assert chapters == []

    def test_no_chapters_no_description(self):
        """No chapters and no description returns empty list."""
        video_info = {"title": "Test Video"}
        chapters = extract_youtube_chapters(video_info)
        assert chapters == []

    def test_description_without_timestamps(self):
        """Description without timestamps returns empty list."""
        video_info = {
            "description": "This is a great video about Python programming."
        }
        chapters = extract_youtube_chapters(video_info)
        assert chapters == []

    def test_sorted_by_start_time(self):
        """Chapters are sorted by start time."""
        video_info = {
            "description": """5:00 Middle
0:00 Start
10:00 End""",
            "duration": 720,
        }
        chapters = extract_youtube_chapters(video_info)

        assert len(chapters) == 3
        assert chapters[0].title == "Start"
        assert chapters[1].title == "Middle"
        assert chapters[2].title == "End"

    def test_end_times_filled_in(self):
        """End times are filled from next chapter's start."""
        video_info = {
            "description": """0:00 First
1:00 Second
2:00 Third""",
            "duration": 180,
        }
        chapters = extract_youtube_chapters(video_info)

        assert chapters[0].end == 60.0  # start of Second
        assert chapters[1].end == 120.0  # start of Third
        assert chapters[2].end == 180.0  # video duration

    def test_none_description(self):
        """None description is handled gracefully."""
        video_info = {"description": None}
        chapters = extract_youtube_chapters(video_info)
        assert chapters == []

    def test_empty_native_chapters(self):
        """Empty chapters list falls through to description parsing."""
        video_info = {
            "chapters": [],
            "description": "0:00 From Description",
            "duration": 60,
        }
        chapters = extract_youtube_chapters(video_info)

        assert len(chapters) == 1
        assert chapters[0].title == "From Description"
        assert chapters[0].source == "description_parsed"
