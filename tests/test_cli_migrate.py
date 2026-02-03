"""Tests for the migrate CLI command."""

import json
from unittest.mock import MagicMock, patch


class TestMigrateCommand:
    """Tests for claudetube migrate."""

    def test_no_flat_dirs_found(self, tmp_path, capsys):
        """Test exits gracefully when no flat directories found."""
        from claudetube.cli import main

        with (
            patch("sys.argv", ["claudetube", "migrate"]),
            patch(
                "claudetube.config.loader.get_cache_dir",
                return_value=tmp_path,
            ),
        ):
            main()

        captured = capsys.readouterr()
        assert "No flat video directories found" in captured.out

    def test_dry_run_shows_planned_moves(self, tmp_path, capsys):
        """Test --dry-run shows what would be moved without moving."""
        from claudetube.cli import main

        # Create a flat video directory
        video_id = "test123"
        video_dir = tmp_path / video_id
        video_dir.mkdir()
        state = {
            "video_id": video_id,
            "url": "https://www.youtube.com/watch?v=test123",
            "channel": "Test Channel",
            "title": "Test Video",
        }
        (video_dir / "state.json").write_text(json.dumps(state))

        with (
            patch("sys.argv", ["claudetube", "migrate", "--dry-run"]),
            patch(
                "claudetube.config.loader.get_cache_dir",
                return_value=tmp_path,
            ),
        ):
            main()

        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert video_id in captured.out
        # Should show the new hierarchical path
        assert "youtube" in captured.out
        # Original dir should still exist
        assert video_dir.exists()

    def test_migrate_moves_flat_to_hierarchical(self, tmp_path):
        """Test migration moves flat directories to hierarchical paths."""
        from claudetube.cli import main

        # Create a flat video directory
        video_id = "abc123"
        video_dir = tmp_path / video_id
        video_dir.mkdir()
        state = {
            "video_id": video_id,
            "url": "https://www.youtube.com/watch?v=abc123",
            "channel": "My Channel",
            "title": "Test Video",
        }
        (video_dir / "state.json").write_text(json.dumps(state))
        (video_dir / "audio.mp3").write_text("audio content")

        with (
            patch("sys.argv", ["claudetube", "migrate"]),
            patch(
                "claudetube.config.loader.get_cache_dir",
                return_value=tmp_path,
            ),
            patch("claudetube.db.get_database", return_value=None),
        ):
            main()

        # Original flat dir should be gone
        assert not video_dir.exists()

        # New hierarchical dir should exist
        new_dir = tmp_path / "youtube" / "My_Channel" / "no_playlist" / video_id
        assert new_dir.exists()
        assert (new_dir / "state.json").exists()
        assert (new_dir / "audio.mp3").exists()

    def test_migrate_unknown_domain(self, tmp_path):
        """Test videos without URL get domain='unknown'."""
        from claudetube.cli import main

        video_id = "local123"
        video_dir = tmp_path / video_id
        video_dir.mkdir()
        state = {
            "video_id": video_id,
            "title": "Local Video",
            # No URL field
        }
        (video_dir / "state.json").write_text(json.dumps(state))

        with (
            patch("sys.argv", ["claudetube", "migrate"]),
            patch(
                "claudetube.config.loader.get_cache_dir",
                return_value=tmp_path,
            ),
            patch("claudetube.db.get_database", return_value=None),
        ):
            main()

        # Should use 'unknown' domain
        new_dir = tmp_path / "unknown" / "no_channel" / "no_playlist" / video_id
        assert new_dir.exists()

    def test_migrate_skips_existing_target(self, tmp_path, capsys):
        """Test migration skips when target already exists."""
        from claudetube.cli import main

        video_id = "dup123"
        flat_dir = tmp_path / video_id
        flat_dir.mkdir()
        state = {
            "video_id": video_id,
            "url": "https://www.youtube.com/watch?v=dup123",
            "title": "Test Video",
        }
        (flat_dir / "state.json").write_text(json.dumps(state))

        # Create target directory that already exists
        target_dir = tmp_path / "youtube" / "no_channel" / "no_playlist" / video_id
        target_dir.mkdir(parents=True)
        (target_dir / "state.json").write_text(json.dumps(state))

        with (
            patch("sys.argv", ["claudetube", "migrate"]),
            patch(
                "claudetube.config.loader.get_cache_dir",
                return_value=tmp_path,
            ),
            patch("claudetube.db.get_database", return_value=None),
        ):
            main()

        captured = capsys.readouterr()
        assert "Skipped" in captured.out
        # Original should still exist since we skipped
        assert flat_dir.exists()

    def test_migrate_idempotent(self, tmp_path, capsys):
        """Test migration is idempotent - safe to run multiple times."""
        from claudetube.cli import main

        video_id = "idem123"
        video_dir = tmp_path / video_id
        video_dir.mkdir()
        state = {
            "video_id": video_id,
            "url": "https://www.youtube.com/watch?v=idem123",
            "channel": "Test",
            "title": "Test Video",
        }
        (video_dir / "state.json").write_text(json.dumps(state))

        # Run migration twice
        for _ in range(2):
            with (
                patch("sys.argv", ["claudetube", "migrate"]),
                patch(
                    "claudetube.config.loader.get_cache_dir",
                    return_value=tmp_path,
                ),
                patch("claudetube.db.get_database", return_value=None),
            ):
                main()

        # After second run, should report nothing to migrate
        captured = capsys.readouterr()
        assert "No flat video directories found" in captured.out

    def test_migrate_with_channel_and_playlist(self, tmp_path):
        """Test migration correctly extracts channel and playlist."""
        from claudetube.cli import main

        video_id = "full123"
        video_dir = tmp_path / video_id
        video_dir.mkdir()
        state = {
            "video_id": video_id,
            "url": "https://www.youtube.com/watch?v=full123",
            "channel_id": "UCxxxxxxxxxxxx",
            "playlist_id": "PLyyyyyyyyyyyy",
            "title": "Test Video",
        }
        (video_dir / "state.json").write_text(json.dumps(state))

        with (
            patch("sys.argv", ["claudetube", "migrate"]),
            patch(
                "claudetube.config.loader.get_cache_dir",
                return_value=tmp_path,
            ),
            patch("claudetube.db.get_database", return_value=None),
        ):
            main()

        # Should use channel_id and playlist_id
        new_dir = tmp_path / "youtube" / "UCxxxxxxxxxxxx" / "PLyyyyyyyyyyyy" / video_id
        assert new_dir.exists()

    def test_migrate_updates_sqlite(self, tmp_path):
        """Test migration updates SQLite with new cache paths."""
        from claudetube.cli import main

        video_id = "sql123"
        video_dir = tmp_path / video_id
        video_dir.mkdir()
        state = {
            "video_id": video_id,
            "url": "https://www.youtube.com/watch?v=sql123",
            "title": "Test Video",
        }
        (video_dir / "state.json").write_text(json.dumps(state))

        # Mock database
        mock_db = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_by_video_id.return_value = {
            "id": "uuid-123",
            "video_id": video_id,
        }

        with (
            patch("sys.argv", ["claudetube", "migrate"]),
            patch(
                "claudetube.config.loader.get_cache_dir",
                return_value=tmp_path,
            ),
            patch("claudetube.db.get_database", return_value=mock_db),
            patch(
                "claudetube.db.repos.videos.VideoRepository",
                return_value=mock_repo,
            ),
        ):
            main()

        # Should have called update_cache_path
        mock_repo.update_cache_path.assert_called_once()
        call_args = mock_repo.update_cache_path.call_args
        assert call_args[0][0] == video_id
        assert "youtube" in call_args[0][1]

    def test_migrate_reports_errors(self, tmp_path, capsys):
        """Test migration reports errors without stopping."""
        from claudetube.cli import main

        # Create a valid video
        video_id = "good123"
        good_dir = tmp_path / video_id
        good_dir.mkdir()
        state = {
            "video_id": video_id,
            "url": "https://www.youtube.com/watch?v=good123",
            "title": "Good Video",
        }
        (good_dir / "state.json").write_text(json.dumps(state))

        # Create a video with invalid JSON
        bad_id = "bad456"
        bad_dir = tmp_path / bad_id
        bad_dir.mkdir()
        (bad_dir / "state.json").write_text("not valid json")

        with (
            patch("sys.argv", ["claudetube", "migrate"]),
            patch(
                "claudetube.config.loader.get_cache_dir",
                return_value=tmp_path,
            ),
            patch("claudetube.db.get_database", return_value=None),
        ):
            main()

        captured = capsys.readouterr()
        # Should report the error
        assert "Error" in captured.out or "ERROR" in captured.err
        # Good video should still be migrated
        new_good_dir = tmp_path / "youtube" / "no_channel" / "no_playlist" / video_id
        assert new_good_dir.exists()

    def test_migrate_sanitizes_channel_name(self, tmp_path):
        """Test migration sanitizes channel names for filesystem safety."""
        from claudetube.cli import main

        video_id = "sanitize123"
        video_dir = tmp_path / video_id
        video_dir.mkdir()
        state = {
            "video_id": video_id,
            "url": "https://www.youtube.com/watch?v=sanitize123",
            "channel": "Bad/Channel:Name*Here?",
            "title": "Test Video",
        }
        (video_dir / "state.json").write_text(json.dumps(state))

        with (
            patch("sys.argv", ["claudetube", "migrate"]),
            patch(
                "claudetube.config.loader.get_cache_dir",
                return_value=tmp_path,
            ),
            patch("claudetube.db.get_database", return_value=None),
        ):
            main()

        # Channel name should be sanitized (unsafe chars replaced with _)
        new_dir = (
            tmp_path / "youtube" / "Bad_Channel_Name_Here_" / "no_playlist" / video_id
        )
        assert new_dir.exists()

    def test_migrate_progress_reported(self, tmp_path, capsys):
        """Test migration reports progress."""
        from claudetube.cli import main

        # Create multiple flat video directories
        for i in range(3):
            video_id = f"vid{i}"
            video_dir = tmp_path / video_id
            video_dir.mkdir()
            state = {
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "title": f"Video {i}",
            }
            (video_dir / "state.json").write_text(json.dumps(state))

        with (
            patch("sys.argv", ["claudetube", "migrate"]),
            patch(
                "claudetube.config.loader.get_cache_dir",
                return_value=tmp_path,
            ),
            patch("claudetube.db.get_database", return_value=None),
        ):
            main()

        captured = capsys.readouterr()
        assert "Found 3 flat directories" in captured.out
        assert "Migrated 3/3" in captured.out


class TestVideoRepositoryUpdateCachePath:
    """Tests for VideoRepository.update_cache_path method."""

    def test_update_cache_path_existing_video(self, tmp_path):
        """Test updating cache_path for an existing video."""
        from claudetube.db.connection import Database
        from claudetube.db.migrate import run_migrations
        from claudetube.db.repos.videos import VideoRepository

        db = Database(":memory:")
        run_migrations(db)
        repo = VideoRepository(db)

        # Insert a video
        repo.insert(
            video_id="test123",
            domain="youtube",
            cache_path="test123",
        )

        # Update the cache path
        result = repo.update_cache_path("test123", "youtube/channel/playlist/test123")
        assert result is True

        # Verify the update
        video = repo.get_by_video_id("test123")
        assert video["cache_path"] == "youtube/channel/playlist/test123"

    def test_update_cache_path_nonexistent_video(self, tmp_path):
        """Test updating cache_path for a non-existent video."""
        from claudetube.db.connection import Database
        from claudetube.db.migrate import run_migrations
        from claudetube.db.repos.videos import VideoRepository

        db = Database(":memory:")
        run_migrations(db)
        repo = VideoRepository(db)

        # Try to update non-existent video
        result = repo.update_cache_path("nonexistent", "some/path")
        assert result is False
