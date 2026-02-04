"""Tests for cache storage with slim VideoState.

Tests that VideoState no longer contains queryable metadata fields
(description, categories, tags, view_count, like_count) and that
these are properly synced to SQLite.
"""

from claudetube.models.state import VideoState


class TestVideoStateSlim:
    """Tests for slim VideoState without queryable metadata."""

    def test_no_description_field(self):
        """VideoState should not have description field."""
        state = VideoState(video_id="test")
        assert not hasattr(state, "description")

    def test_no_categories_field(self):
        """VideoState should not have categories field."""
        state = VideoState(video_id="test")
        assert not hasattr(state, "categories")

    def test_no_tags_field(self):
        """VideoState should not have tags field."""
        state = VideoState(video_id="test")
        assert not hasattr(state, "tags")

    def test_no_view_count_field(self):
        """VideoState should not have view_count field."""
        state = VideoState(video_id="test")
        assert not hasattr(state, "view_count")

    def test_no_like_count_field(self):
        """VideoState should not have like_count field."""
        state = VideoState(video_id="test")
        assert not hasattr(state, "like_count")

    def test_to_dict_excludes_queryable_fields(self):
        """to_dict should not include queryable metadata fields."""
        state = VideoState(video_id="test", title="Test Video")
        d = state.to_dict()

        # These fields should NOT be in the dict
        assert "description" not in d
        assert "categories" not in d
        assert "tags" not in d
        assert "view_count" not in d
        assert "like_count" not in d

        # These fields should still be present
        assert "video_id" in d
        assert "title" in d
        assert "duration" in d
        assert "chapters" in d

    def test_from_dict_ignores_deprecated_fields(self):
        """from_dict should silently ignore deprecated fields from old state.json."""
        data = {
            "video_id": "test",
            "title": "Test Video",
            # Deprecated fields that may exist in old state.json files
            "description": "Old description",
            "categories": ["category1"],
            "tags": ["tag1", "tag2"],
            "view_count": 1000,
            "like_count": 50,
        }

        state = VideoState.from_dict(data)

        # Should load correctly without error
        assert state.video_id == "test"
        assert state.title == "Test Video"

        # Should NOT have the deprecated fields
        assert not hasattr(state, "description")
        assert not hasattr(state, "categories")
        assert not hasattr(state, "tags")
        assert not hasattr(state, "view_count")
        assert not hasattr(state, "like_count")

    def test_from_metadata_excludes_queryable_fields(self):
        """from_metadata should not store queryable fields in state."""
        meta = {
            "title": "Test Video",
            "duration": 120.5,
            "description": "Video description",
            "categories": ["Education"],
            "tags": ["test", "video"],
            "view_count": 1000,
            "like_count": 50,
            "channel": "TestChannel",
        }

        state = VideoState.from_metadata(
            "test123", "https://example.com/v/test123", meta
        )

        # Should have basic metadata
        assert state.video_id == "test123"
        assert state.title == "Test Video"
        assert state.duration == 120.5
        assert state.channel == "TestChannel"

        # Should NOT have queryable fields
        assert not hasattr(state, "description")
        assert not hasattr(state, "categories")
        assert not hasattr(state, "tags")
        assert not hasattr(state, "view_count")
        assert not hasattr(state, "like_count")


class TestVideoStatePreserved:
    """Tests that essential fields are preserved in slim VideoState."""

    def test_has_essential_fields(self):
        """VideoState should have all essential fast-access fields."""
        state = VideoState(
            video_id="test",
            url="https://example.com/v/test",
            title="Test Video",
            duration=120.5,
            duration_string="2:00",
            channel="TestChannel",
            upload_date="20240101",
            language="en",
            thumbnail="https://example.com/thumb.jpg",
            chapters=[{"title": "Intro", "start_time": 0, "end_time": 10}],
            transcript_complete=True,
            transcript_source="whisper",
            scenes_processed=True,
            scene_count=5,
        )

        assert state.video_id == "test"
        assert state.url == "https://example.com/v/test"
        assert state.title == "Test Video"
        assert state.duration == 120.5
        assert state.duration_string == "2:00"
        assert state.channel == "TestChannel"
        assert state.upload_date == "20240101"
        assert state.language == "en"
        assert state.thumbnail == "https://example.com/thumb.jpg"
        assert state.chapters == [{"title": "Intro", "start_time": 0, "end_time": 10}]
        assert state.transcript_complete is True
        assert state.transcript_source == "whisper"
        assert state.scenes_processed is True
        assert state.scene_count == 5

    def test_to_dict_roundtrip(self):
        """to_dict and from_dict should roundtrip correctly."""
        original = VideoState(
            video_id="test",
            title="Test Video",
            duration=120.5,
            chapters=[{"title": "Ch1", "start_time": 0, "end_time": 60}],
            transcript_complete=True,
            scenes_processed=True,
            scene_count=3,
        )

        d = original.to_dict()
        restored = VideoState.from_dict(d)

        assert restored.video_id == original.video_id
        assert restored.title == original.title
        assert restored.duration == original.duration
        assert restored.chapters == original.chapters
        assert restored.transcript_complete == original.transcript_complete
        assert restored.scenes_processed == original.scenes_processed
        assert restored.scene_count == original.scene_count
