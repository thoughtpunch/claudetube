"""Tests for playlist metadata extraction."""


class TestClassifyPlaylistType:
    """Test playlist type classification."""

    def test_course_detection_title(self):
        from claudetube.operations.playlist import classify_playlist_type

        playlist_info = {"title": "Python Course for Beginners", "description": ""}
        videos = [{"title": "Lesson 1"}, {"title": "Lesson 2"}]
        assert classify_playlist_type(playlist_info, videos) == "course"

    def test_course_detection_description(self):
        from claudetube.operations.playlist import classify_playlist_type

        playlist_info = {
            "title": "Programming Videos",
            "description": "A complete tutorial series",
        }
        videos = []
        assert classify_playlist_type(playlist_info, videos) == "course"

    def test_series_detection_numbered(self):
        from claudetube.operations.playlist import classify_playlist_type

        playlist_info = {"title": "My Series", "description": ""}
        videos = [
            {"title": "Part 1 - Introduction"},
            {"title": "Part 2 - Setup"},
            {"title": "Part 3 - Building"},
            {"title": "Part 4 - Testing"},
            {"title": "Part 5 - Deployment"},
        ]
        assert classify_playlist_type(playlist_info, videos) == "series"

    def test_series_detection_episodes(self):
        from claudetube.operations.playlist import classify_playlist_type

        playlist_info = {"title": "Documentary", "description": ""}
        videos = [
            {"title": "Episode 1: Origins"},
            {"title": "Episode 2: Growth"},
            {"title": "Episode 3: Crisis"},
        ]
        assert classify_playlist_type(playlist_info, videos) == "series"

    def test_conference_detection(self):
        from claudetube.operations.playlist import classify_playlist_type

        playlist_info = {"title": "PyCon 2024 Talks", "description": ""}
        videos = [{"title": "Keynote"}, {"title": "Talk 1"}]
        assert classify_playlist_type(playlist_info, videos) == "conference"

    def test_collection_default(self):
        from claudetube.operations.playlist import classify_playlist_type

        playlist_info = {"title": "My Favorites", "description": "Random videos"}
        videos = [{"title": "Video A"}, {"title": "Video B"}]
        assert classify_playlist_type(playlist_info, videos) == "collection"


class TestExtractPlaylistId:
    """Test playlist ID extraction from URLs."""

    def test_youtube_playlist_url(self):
        from claudetube.operations.playlist import _extract_playlist_id

        url = "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        assert _extract_playlist_id(url) == "PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"

    def test_youtube_playlist_in_video_url(self):
        from claudetube.operations.playlist import _extract_playlist_id

        url = "https://www.youtube.com/watch?v=abc123&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        assert _extract_playlist_id(url) == "PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"

    def test_fallback_hash(self):
        from claudetube.operations.playlist import _extract_playlist_id

        url = "https://example.com/some/playlist"
        result = _extract_playlist_id(url)
        assert len(result) == 12  # SHA256 hash prefix
