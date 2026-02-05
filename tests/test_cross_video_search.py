"""Tests for cross-video search functionality."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claudetube.navigation.cross_video import (
    ChapterMatch,
    PlaylistChapterIndex,
    PlaylistSearchResult,
    _create_preview,
    _format_timestamp,
    build_chapter_index,
    find_chapters_by_topic,
    load_chapter_index,
    save_chapter_index,
    search_playlist_transcripts,
)


class TestDataclasses:
    """Tests for dataclass serialization."""

    def test_playlist_search_result_to_dict(self):
        """Test PlaylistSearchResult serialization."""
        result = PlaylistSearchResult(
            video_id="abc123",
            video_title="Test Video",
            video_position=0,
            scene_id=1,
            start_time=60.0,
            end_time=120.0,
            relevance=0.8,
            preview="This is a preview...",
            timestamp_str="1:00",
            match_type="fts",
            chapter_title="Introduction",
        )

        d = result.to_dict()
        assert d["video_id"] == "abc123"
        assert d["video_title"] == "Test Video"
        assert d["video_position"] == 0
        assert d["scene_id"] == 1
        assert d["start_time"] == 60.0
        assert d["relevance"] == 0.8
        assert d["match_type"] == "fts"
        assert d["chapter_title"] == "Introduction"

    def test_chapter_match_to_dict(self):
        """Test ChapterMatch serialization."""
        match = ChapterMatch(
            video_id="xyz789",
            video_title="Tutorial",
            video_position=2,
            chapter_title="Setup",
            chapter_index=0,
            start_time=0.0,
            end_time=300.0,
            relevance=0.9,
            timestamp_str="0:00",
        )

        d = match.to_dict()
        assert d["video_id"] == "xyz789"
        assert d["chapter_title"] == "Setup"
        assert d["chapter_index"] == 0
        assert d["relevance"] == 0.9

    def test_playlist_chapter_index_serialization(self):
        """Test PlaylistChapterIndex to_dict and from_dict."""
        chapters = [
            {
                "video_id": "v1",
                "video_title": "Video 1",
                "video_position": 0,
                "chapter_title": "Intro",
                "chapter_index": 0,
                "start_time": 0.0,
                "end_time": 60.0,
                "timestamp_str": "0:00",
            }
        ]

        index = PlaylistChapterIndex(playlist_id="playlist123", chapters=chapters)
        d = index.to_dict()

        assert d["playlist_id"] == "playlist123"
        assert d["chapter_count"] == 1
        assert len(d["chapters"]) == 1

        # Round-trip
        reconstructed = PlaylistChapterIndex.from_dict(d)
        assert reconstructed.playlist_id == "playlist123"
        assert len(reconstructed.chapters) == 1


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_format_timestamp_minutes_seconds(self):
        """Test MM:SS format for < 1 hour."""
        assert _format_timestamp(0) == "0:00"
        assert _format_timestamp(65) == "1:05"
        assert _format_timestamp(3599) == "59:59"

    def test_format_timestamp_hours(self):
        """Test HH:MM:SS format for >= 1 hour."""
        assert _format_timestamp(3600) == "1:00:00"
        assert _format_timestamp(3665) == "1:01:05"
        assert _format_timestamp(7200) == "2:00:00"

    def test_create_preview_short_text(self):
        """Test preview for text shorter than max_len."""
        text = "Short text"
        assert _create_preview(text, "short", max_len=100) == text

    def test_create_preview_centered_on_match(self):
        """Test preview centers on query match."""
        text = "A" * 100 + " MATCH " + "B" * 100
        preview = _create_preview(text, "MATCH", max_len=50)
        assert "MATCH" in preview
        assert preview.startswith("...")
        assert preview.endswith("...")

    def test_create_preview_no_match(self):
        """Test preview when no match found returns start of text."""
        text = "A" * 200
        preview = _create_preview(text, "NOMATCH", max_len=50)
        assert preview.startswith("A")
        assert preview.endswith("...")


class TestChapterIndexing:
    """Tests for chapter indexing functionality."""

    def test_build_chapter_index_empty_playlist(self, tmp_path):
        """Test building index for playlist without videos."""
        # Create empty playlist
        playlist_dir = tmp_path / "playlists" / "empty_playlist"
        playlist_dir.mkdir(parents=True)
        playlist_file = playlist_dir / "playlist.json"
        playlist_file.write_text(
            json.dumps(
                {
                    "playlist_id": "empty_playlist",
                    "title": "Empty",
                    "videos": [],
                }
            )
        )

        index = build_chapter_index("empty_playlist", cache_base=tmp_path)
        assert index.playlist_id == "empty_playlist"
        assert len(index.chapters) == 0

    def test_save_and_load_chapter_index(self, tmp_path):
        """Test saving and loading chapter index."""
        # Create playlist directory
        playlist_dir = tmp_path / "playlists" / "test_playlist"
        playlist_dir.mkdir(parents=True)

        index = PlaylistChapterIndex(
            playlist_id="test_playlist",
            chapters=[
                {
                    "video_id": "v1",
                    "video_title": "Video 1",
                    "video_position": 0,
                    "chapter_title": "Getting Started",
                    "chapter_index": 0,
                    "start_time": 0.0,
                    "end_time": 120.0,
                    "timestamp_str": "0:00",
                }
            ],
        )

        # Save
        path = save_chapter_index(index, cache_base=tmp_path)
        assert path.exists()
        assert "chapter_index.json" in str(path)

        # Load
        loaded = load_chapter_index("test_playlist", cache_base=tmp_path)
        assert loaded is not None
        assert loaded.playlist_id == "test_playlist"
        assert len(loaded.chapters) == 1

    def test_load_chapter_index_not_found(self, tmp_path):
        """Test loading non-existent chapter index returns None."""
        result = load_chapter_index("nonexistent", cache_base=tmp_path)
        assert result is None


class TestChapterSearch:
    """Tests for chapter-based topic search."""

    @pytest.fixture
    def chapter_index(self, tmp_path):
        """Create a chapter index for testing."""
        playlist_dir = tmp_path / "playlists" / "course_playlist"
        playlist_dir.mkdir(parents=True)

        # Create playlist metadata
        playlist_file = playlist_dir / "playlist.json"
        playlist_file.write_text(
            json.dumps(
                {
                    "playlist_id": "course_playlist",
                    "title": "Python Course",
                    "videos": [
                        {"video_id": "v1", "title": "Introduction", "position": 0},
                        {"video_id": "v2", "title": "Setup", "position": 1},
                    ],
                }
            )
        )

        # Create chapter index
        index = PlaylistChapterIndex(
            playlist_id="course_playlist",
            chapters=[
                {
                    "video_id": "v1",
                    "video_title": "Introduction",
                    "video_position": 0,
                    "chapter_title": "Welcome and Overview",
                    "chapter_index": 0,
                    "start_time": 0.0,
                    "end_time": 60.0,
                    "timestamp_str": "0:00",
                },
                {
                    "video_id": "v1",
                    "video_title": "Introduction",
                    "video_position": 0,
                    "chapter_title": "Setting Up Your Environment",
                    "chapter_index": 1,
                    "start_time": 60.0,
                    "end_time": 180.0,
                    "timestamp_str": "1:00",
                },
                {
                    "video_id": "v2",
                    "video_title": "Setup",
                    "video_position": 1,
                    "chapter_title": "Installing Python",
                    "chapter_index": 0,
                    "start_time": 0.0,
                    "end_time": 120.0,
                    "timestamp_str": "0:00",
                },
                {
                    "video_id": "v2",
                    "video_title": "Setup",
                    "video_position": 1,
                    "chapter_title": "Installing Dependencies",
                    "chapter_index": 1,
                    "start_time": 120.0,
                    "end_time": 240.0,
                    "timestamp_str": "2:00",
                },
            ],
        )
        save_chapter_index(index, cache_base=tmp_path)
        return tmp_path

    def test_find_chapters_exact_match(self, chapter_index):
        """Test finding chapters with exact match."""
        results = find_chapters_by_topic(
            "course_playlist",
            "Installing Python",
            cache_base=chapter_index,
        )
        assert len(results) > 0
        assert results[0].chapter_title == "Installing Python"
        assert results[0].relevance > 0.5

    def test_find_chapters_partial_match(self, chapter_index):
        """Test finding chapters with partial word match."""
        results = find_chapters_by_topic(
            "course_playlist",
            "install",
            cache_base=chapter_index,
        )
        assert len(results) >= 2
        # Should match both "Installing Python" and "Installing Dependencies"
        titles = [r.chapter_title for r in results]
        assert "Installing Python" in titles
        assert "Installing Dependencies" in titles

    def test_find_chapters_no_match(self, chapter_index):
        """Test finding chapters with no match."""
        results = find_chapters_by_topic(
            "course_playlist",
            "kubernetes",  # Not in any chapter
            cache_base=chapter_index,
        )
        assert len(results) == 0

    def test_find_chapters_sorted_by_relevance(self, chapter_index):
        """Test that results are sorted by relevance."""
        results = find_chapters_by_topic(
            "course_playlist",
            "Setup",
            top_k=10,
            cache_base=chapter_index,
        )
        # Results should be in descending relevance order
        for i in range(len(results) - 1):
            assert results[i].relevance >= results[i + 1].relevance


class TestPlaylistTranscriptSearch:
    """Tests for playlist-wide transcript search."""

    def test_search_playlist_no_playlist(self, tmp_path):
        """Test search on non-existent playlist returns empty."""
        results = search_playlist_transcripts(
            "nonexistent",
            "test query",
            cache_base=tmp_path,
        )
        assert results == []

    def test_search_playlist_empty_videos(self, tmp_path):
        """Test search on playlist with no videos."""
        playlist_dir = tmp_path / "playlists" / "empty"
        playlist_dir.mkdir(parents=True)
        (playlist_dir / "playlist.json").write_text(
            json.dumps(
                {
                    "playlist_id": "empty",
                    "title": "Empty Playlist",
                    "videos": [],
                }
            )
        )

        results = search_playlist_transcripts(
            "empty",
            "test",
            cache_base=tmp_path,
        )
        assert results == []

    @pytest.fixture
    def playlist_with_scenes(self, tmp_path):
        """Create a playlist with videos that have scenes."""
        # Create playlist
        playlist_dir = tmp_path / "playlists" / "test_playlist"
        playlist_dir.mkdir(parents=True)
        (playlist_dir / "playlist.json").write_text(
            json.dumps(
                {
                    "playlist_id": "test_playlist",
                    "title": "Test Playlist",
                    "videos": [
                        {"video_id": "vid1", "title": "Video One", "position": 0},
                        {"video_id": "vid2", "title": "Video Two", "position": 1},
                    ],
                }
            )
        )

        # Create video with scenes
        vid1_dir = tmp_path / "vid1" / "scenes"
        vid1_dir.mkdir(parents=True)
        (vid1_dir / "scenes.json").write_text(
            json.dumps(
                {
                    "video_id": "vid1",
                    "method": "transcript",
                    "scenes": [
                        {
                            "scene_id": 0,
                            "start_time": 0.0,
                            "end_time": 60.0,
                            "title": "Intro",
                            "transcript_text": "Welcome to this tutorial about Python authentication.",
                        },
                        {
                            "scene_id": 1,
                            "start_time": 60.0,
                            "end_time": 120.0,
                            "title": "Setup",
                            "transcript_text": "First we need to install the dependencies.",
                        },
                    ],
                }
            )
        )

        vid2_dir = tmp_path / "vid2" / "scenes"
        vid2_dir.mkdir(parents=True)
        (vid2_dir / "scenes.json").write_text(
            json.dumps(
                {
                    "video_id": "vid2",
                    "method": "transcript",
                    "scenes": [
                        {
                            "scene_id": 0,
                            "start_time": 0.0,
                            "end_time": 90.0,
                            "title": None,
                            "transcript_text": "Now let's implement user authentication with JWT tokens.",
                        },
                    ],
                }
            )
        )

        return tmp_path

    def test_search_playlist_memory_fallback(self, playlist_with_scenes):
        """Test in-memory search when FTS unavailable."""
        results = search_playlist_transcripts(
            "test_playlist",
            "authentication",
            cache_base=playlist_with_scenes,
        )

        assert len(results) >= 1
        # Both videos mention authentication
        video_ids = {r.video_id for r in results}
        assert "vid1" in video_ids or "vid2" in video_ids

    def test_search_playlist_relevance_ranking(self, playlist_with_scenes):
        """Test that results are ranked by relevance."""
        results = search_playlist_transcripts(
            "test_playlist",
            "install dependencies",
            cache_base=playlist_with_scenes,
        )

        # Results should be sorted by relevance
        for i in range(len(results) - 1):
            assert results[i].relevance >= results[i + 1].relevance

    def test_search_playlist_top_k_limit(self, playlist_with_scenes):
        """Test that top_k limits results."""
        results = search_playlist_transcripts(
            "test_playlist",
            "to",  # Should match multiple scenes
            top_k=1,
            cache_base=playlist_with_scenes,
        )

        assert len(results) <= 1
