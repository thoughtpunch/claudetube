"""Tests for local file caching (symlink/copy)."""

import platform
from pathlib import Path
from unittest.mock import patch

import pytest

from claudetube.cache.manager import CacheManager
from claudetube.cache.storage import cache_local_file, check_cached_source
from claudetube.exceptions import CacheError
from claudetube.models.state import VideoState


class TestCacheLocalFile:
    """Tests for cache_local_file function."""

    def test_symlink_by_default(self, tmp_path):
        """Default behavior creates a symlink."""
        source = tmp_path / "videos" / "test.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"fake video content")

        cache_dir = tmp_path / "cache"
        dest, mode = cache_local_file(source, cache_dir)

        assert mode == "symlink"
        assert dest.name == "source.mp4"
        assert dest.is_symlink()
        assert dest.resolve() == source.resolve()

    def test_copy_mode(self, tmp_path):
        """Copy mode creates a copy of the file."""
        source = tmp_path / "videos" / "test.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"fake video content")

        cache_dir = tmp_path / "cache"
        dest, mode = cache_local_file(source, cache_dir, copy=True)

        assert mode == "copy"
        assert dest.name == "source.mp4"
        assert not dest.is_symlink()
        assert dest.read_bytes() == b"fake video content"

    def test_preserves_extension(self, tmp_path):
        """Cached file preserves the original extension."""
        for ext in [".mp4", ".mkv", ".mov", ".webm"]:
            source = tmp_path / f"test{ext}"
            source.write_bytes(b"content")

            cache_dir = tmp_path / f"cache_{ext}"
            dest, _ = cache_local_file(source, cache_dir)

            assert dest.suffix == ext
            assert dest.name == f"source{ext}"

    def test_creates_cache_dir(self, tmp_path):
        """Creates cache directory if it doesn't exist."""
        source = tmp_path / "test.mp4"
        source.write_bytes(b"content")

        cache_dir = tmp_path / "nested" / "cache" / "dir"
        assert not cache_dir.exists()

        dest, _ = cache_local_file(source, cache_dir)

        assert cache_dir.exists()
        assert dest.exists()

    def test_replaces_existing_symlink(self, tmp_path):
        """Replaces existing symlink if present."""
        source1 = tmp_path / "video1.mp4"
        source2 = tmp_path / "video2.mp4"
        source1.write_bytes(b"video 1")
        source2.write_bytes(b"video 2")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Cache first file
        dest1, _ = cache_local_file(source1, cache_dir)
        assert dest1.resolve() == source1.resolve()

        # Cache second file - should replace
        dest2, _ = cache_local_file(source2, cache_dir)
        assert dest2.resolve() == source2.resolve()

    def test_raises_on_copy_failure(self, tmp_path):
        """Raises CacheError if copy fails."""
        source = tmp_path / "test.mp4"
        source.write_bytes(b"content")

        cache_dir = tmp_path / "cache"

        # Mock shutil.copy2 to fail
        with (
            patch(
                "claudetube.cache.storage.shutil.copy2",
                side_effect=OSError("Copy failed"),
            ),
            pytest.raises(CacheError, match="Failed to copy"),
        ):
            cache_local_file(source, cache_dir, copy=True)

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only test")
    def test_windows_fallback_to_copy(self, tmp_path):
        """On Windows, falls back to copy if symlink fails."""
        source = tmp_path / "test.mp4"
        source.write_bytes(b"content")
        cache_dir = tmp_path / "cache"

        # Mock symlink to fail
        with patch.object(Path, "symlink_to", side_effect=OSError("Symlink failed")):
            dest, mode = cache_local_file(source, cache_dir)

        assert mode == "copy"
        assert not dest.is_symlink()


class TestCheckCachedSource:
    """Tests for check_cached_source function."""

    def test_valid_symlink(self, tmp_path):
        """Valid symlink returns (True, None)."""
        source = tmp_path / "original.mp4"
        source.write_bytes(b"content")

        cache_dir = tmp_path / "cache"
        dest, _ = cache_local_file(source, cache_dir)

        is_valid, warning = check_cached_source(cache_dir, "source.mp4")
        assert is_valid is True
        assert warning is None

    def test_valid_copy(self, tmp_path):
        """Valid copy returns (True, None)."""
        source = tmp_path / "original.mp4"
        source.write_bytes(b"content")

        cache_dir = tmp_path / "cache"
        dest, _ = cache_local_file(source, cache_dir, copy=True)

        is_valid, warning = check_cached_source(cache_dir, "source.mp4")
        assert is_valid is True
        assert warning is None

    def test_broken_symlink(self, tmp_path):
        """Broken symlink returns (False, warning)."""
        source = tmp_path / "original.mp4"
        source.write_bytes(b"content")

        cache_dir = tmp_path / "cache"
        dest, _ = cache_local_file(source, cache_dir)

        # Delete the source file to break the symlink
        source.unlink()

        is_valid, warning = check_cached_source(cache_dir, "source.mp4")
        assert is_valid is False
        assert "Broken symlink" in warning
        assert "moved or deleted" in warning

    def test_missing_cached_file(self, tmp_path):
        """Missing cached file returns (False, warning)."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        is_valid, warning = check_cached_source(cache_dir, "source.mp4")
        assert is_valid is False
        assert "not found" in warning

    def test_no_cached_file_recorded(self, tmp_path):
        """None cached_file returns (False, warning)."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        is_valid, warning = check_cached_source(cache_dir, None)
        assert is_valid is False
        assert "No cached file" in warning


class TestCacheManagerLocalFile:
    """Tests for CacheManager local file methods."""

    def test_cache_local_file(self, tmp_path):
        """CacheManager.cache_local_file works correctly."""
        source = tmp_path / "video.mp4"
        source.write_bytes(b"video content")

        cache_base = tmp_path / "cache"
        manager = CacheManager(cache_base)

        dest, mode = manager.cache_local_file("test_video_123", source)

        assert mode == "symlink"
        assert dest.parent == cache_base / "test_video_123"
        assert dest.is_symlink()

    def test_cache_local_file_copy(self, tmp_path):
        """CacheManager.cache_local_file with copy=True."""
        source = tmp_path / "video.mp4"
        source.write_bytes(b"video content")

        cache_base = tmp_path / "cache"
        manager = CacheManager(cache_base)

        dest, mode = manager.cache_local_file("test_video_123", source, copy=True)

        assert mode == "copy"
        assert not dest.is_symlink()

    def test_get_source_path(self, tmp_path):
        """CacheManager.get_source_path returns correct path."""
        source = tmp_path / "video.mp4"
        source.write_bytes(b"video content")

        cache_base = tmp_path / "cache"
        manager = CacheManager(cache_base)

        # Create and cache file
        dest, mode = manager.cache_local_file("test_video_123", source)

        # Create and save state
        state = VideoState(
            video_id="test_video_123",
            source_type="local",
            source_path=str(source),
            cache_mode=mode,
            cached_file="source.mp4",
        )
        manager.save_state("test_video_123", state)

        # Get source path
        result = manager.get_source_path("test_video_123")
        assert result == cache_base / "test_video_123" / "source.mp4"

    def test_get_source_path_none_for_url(self, tmp_path):
        """CacheManager.get_source_path returns None for URL videos."""
        cache_base = tmp_path / "cache"
        manager = CacheManager(cache_base)

        # Create URL-type state
        state = VideoState(
            video_id="url_video_123",
            source_type="url",
            url="https://youtube.com/watch?v=abc123",
        )
        manager.save_state("url_video_123", state)

        result = manager.get_source_path("url_video_123")
        assert result is None

    def test_check_source_valid_symlink(self, tmp_path):
        """CacheManager.check_source_valid for valid symlink."""
        source = tmp_path / "video.mp4"
        source.write_bytes(b"video content")

        cache_base = tmp_path / "cache"
        manager = CacheManager(cache_base)

        dest, mode = manager.cache_local_file("test_video_123", source)

        state = VideoState(
            video_id="test_video_123",
            source_type="local",
            source_path=str(source),
            cache_mode=mode,
            cached_file="source.mp4",
        )
        manager.save_state("test_video_123", state)

        is_valid, warning = manager.check_source_valid("test_video_123")
        assert is_valid is True
        assert warning is None

    def test_check_source_valid_broken_symlink(self, tmp_path):
        """CacheManager.check_source_valid detects broken symlink."""
        source = tmp_path / "video.mp4"
        source.write_bytes(b"video content")

        cache_base = tmp_path / "cache"
        manager = CacheManager(cache_base)

        dest, mode = manager.cache_local_file("test_video_123", source)

        state = VideoState(
            video_id="test_video_123",
            source_type="local",
            source_path=str(source),
            cache_mode=mode,
            cached_file="source.mp4",
        )
        manager.save_state("test_video_123", state)

        # Delete source to break symlink
        source.unlink()

        is_valid, warning = manager.check_source_valid("test_video_123")
        assert is_valid is False
        assert "Broken symlink" in warning

    def test_check_source_valid_url_video(self, tmp_path):
        """CacheManager.check_source_valid returns (True, None) for URL videos."""
        cache_base = tmp_path / "cache"
        manager = CacheManager(cache_base)

        state = VideoState(
            video_id="url_video_123",
            source_type="url",
            url="https://youtube.com/watch?v=abc123",
        )
        manager.save_state("url_video_123", state)

        is_valid, warning = manager.check_source_valid("url_video_123")
        assert is_valid is True
        assert warning is None


class TestVideoStateLocalFields:
    """Tests for VideoState local file fields."""

    def test_cache_mode_field(self):
        """VideoState has cache_mode field."""
        state = VideoState(
            video_id="test",
            source_type="local",
            cache_mode="symlink",
        )
        assert state.cache_mode == "symlink"

    def test_cached_file_field(self):
        """VideoState has cached_file field."""
        state = VideoState(
            video_id="test",
            source_type="local",
            cached_file="source.mp4",
        )
        assert state.cached_file == "source.mp4"

    def test_to_dict_includes_cache_fields(self):
        """VideoState.to_dict includes cache_mode and cached_file."""
        state = VideoState(
            video_id="test",
            source_type="local",
            source_path="/path/to/video.mp4",
            cache_mode="symlink",
            cached_file="source.mp4",
        )
        d = state.to_dict()
        assert d["cache_mode"] == "symlink"
        assert d["cached_file"] == "source.mp4"

    def test_from_dict_loads_cache_fields(self):
        """VideoState.from_dict loads cache_mode and cached_file."""
        data = {
            "video_id": "test",
            "source_type": "local",
            "source_path": "/path/to/video.mp4",
            "cache_mode": "copy",
            "cached_file": "source.mp4",
        }
        state = VideoState.from_dict(data)
        assert state.cache_mode == "copy"
        assert state.cached_file == "source.mp4"

    def test_defaults_to_none(self):
        """Cache fields default to None."""
        state = VideoState(video_id="test")
        assert state.cache_mode is None
        assert state.cached_file is None
