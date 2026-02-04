"""Tests for the database connection manager, migration runner, and module init."""

import sqlite3
import threading

import pytest

from claudetube.db.connection import Database
from claudetube.db.migrate import (
    _discover_from_directory,
    _ensure_schema_version_table,
    _get_applied_versions,
    run_migrations,
)

# ============================================================
# Database class tests
# ============================================================


class TestDatabaseConnection:
    """Tests for Database class connection management."""

    def test_in_memory_database(self):
        """Test creating an in-memory database."""
        db = Database(":memory:")
        cursor = db.execute("SELECT 1 AS val")
        row = cursor.fetchone()
        assert row["val"] == 1
        db.close()

    def test_file_database(self, tmp_path):
        """Test creating a file-based database."""
        db_path = tmp_path / "test.db"
        db = Database(db_path)
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        db.commit()
        assert db_path.exists()
        db.close()

    def test_row_factory(self):
        """Test that row_factory returns sqlite3.Row (dict-like access)."""
        db = Database(":memory:")
        db.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        db.execute("INSERT INTO t VALUES (1, 'alice')")
        db.commit()

        cursor = db.execute("SELECT id, name FROM t")
        row = cursor.fetchone()
        assert row["id"] == 1
        assert row["name"] == "alice"
        # Also works by index
        assert row[0] == 1
        assert row[1] == "alice"
        db.close()

    def test_close_then_execute_raises(self):
        """Test that executing after close raises RuntimeError."""
        db = Database(":memory:")
        db.close()
        with pytest.raises(RuntimeError, match="Database is closed"):
            db.execute("SELECT 1")

    def test_close_is_idempotent(self):
        """Test that closing multiple times doesn't raise."""
        db = Database(":memory:")
        db.close()
        db.close()  # Should not raise


class TestDatabasePragmas:
    """Tests for SQLite PRAGMA configuration."""

    def test_foreign_keys_enabled(self):
        """Test that PRAGMA foreign_keys is ON."""
        db = Database(":memory:")
        cursor = db.execute("PRAGMA foreign_keys")
        row = cursor.fetchone()
        assert row[0] == 1
        db.close()

    def test_busy_timeout_set(self):
        """Test that PRAGMA busy_timeout is 5000ms."""
        db = Database(":memory:")
        cursor = db.execute("PRAGMA busy_timeout")
        row = cursor.fetchone()
        assert row[0] == 5000
        db.close()

    def test_wal_mode_for_file_database(self, tmp_path):
        """Test that WAL mode is enabled for file-based databases."""
        db_path = tmp_path / "wal_test.db"
        db = Database(db_path)
        # Force a write to ensure WAL is set up
        db.execute("CREATE TABLE t (id INTEGER)")
        db.commit()
        cursor = db.execute("PRAGMA journal_mode")
        row = cursor.fetchone()
        assert row[0] == "wal"
        db.close()

    def test_foreign_key_enforcement(self):
        """Test that foreign key violations are caught."""
        db = Database(":memory:")
        db.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        db.execute(
            "CREATE TABLE child (id INTEGER PRIMARY KEY, "
            "parent_id INTEGER REFERENCES parent(id))"
        )
        db.commit()

        # Insert child with non-existent parent should fail
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO child VALUES (1, 999)")
        db.close()


class TestDatabaseExecute:
    """Tests for execute, executemany, executescript."""

    def test_execute_with_params_tuple(self):
        """Test execute with tuple parameters."""
        db = Database(":memory:")
        db.execute("CREATE TABLE t (id INTEGER, val TEXT)")
        db.execute("INSERT INTO t VALUES (?, ?)", (1, "hello"))
        db.commit()

        cursor = db.execute("SELECT val FROM t WHERE id = ?", (1,))
        assert cursor.fetchone()["val"] == "hello"
        db.close()

    def test_execute_with_params_dict(self):
        """Test execute with dict parameters."""
        db = Database(":memory:")
        db.execute("CREATE TABLE t (id INTEGER, val TEXT)")
        db.execute("INSERT INTO t VALUES (:id, :val)", {"id": 1, "val": "hello"})
        db.commit()

        cursor = db.execute("SELECT val FROM t WHERE id = :id", {"id": 1})
        assert cursor.fetchone()["val"] == "hello"
        db.close()

    def test_execute_without_params(self):
        """Test execute with no parameters."""
        db = Database(":memory:")
        cursor = db.execute("SELECT 42 AS answer")
        assert cursor.fetchone()["answer"] == 42
        db.close()

    def test_executemany(self):
        """Test executemany inserts multiple rows."""
        db = Database(":memory:")
        db.execute("CREATE TABLE t (id INTEGER, val TEXT)")
        db.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(1, "a"), (2, "b"), (3, "c")],
        )
        db.commit()

        cursor = db.execute("SELECT COUNT(*) AS cnt FROM t")
        assert cursor.fetchone()["cnt"] == 3
        db.close()

    def test_executescript(self):
        """Test executescript runs multiple statements."""
        db = Database(":memory:")
        db.executescript("""
            CREATE TABLE t1 (id INTEGER);
            CREATE TABLE t2 (id INTEGER);
            INSERT INTO t1 VALUES (1);
            INSERT INTO t2 VALUES (2);
        """)

        cursor = db.execute("SELECT id FROM t1")
        assert cursor.fetchone()["id"] == 1
        cursor = db.execute("SELECT id FROM t2")
        assert cursor.fetchone()["id"] == 2
        db.close()


class TestDatabaseTransactions:
    """Tests for transaction support via context manager."""

    def test_context_manager_commits(self):
        """Test that context manager commits on success."""
        db = Database(":memory:")
        db.execute("CREATE TABLE t (id INTEGER)")
        db.commit()

        with db:
            db.execute("INSERT INTO t VALUES (1)")

        cursor = db.execute("SELECT COUNT(*) AS cnt FROM t")
        assert cursor.fetchone()["cnt"] == 1
        db.close()

    def test_context_manager_rollback_on_exception(self):
        """Test that context manager rolls back on exception."""
        db = Database(":memory:")
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        db.commit()

        # First insert should succeed
        with db:
            db.execute("INSERT INTO t VALUES (1)")

        # Second block should fail and rollback
        with pytest.raises(sqlite3.IntegrityError), db:
            db.execute("INSERT INTO t VALUES (2)")
            db.execute("INSERT INTO t VALUES (2)")  # Duplicate PK

        # Only the first row should exist
        cursor = db.execute("SELECT COUNT(*) AS cnt FROM t")
        # Note: executescript implicit commits make this tricky.
        # With BEGIN-based transactions, the rollback should work.
        count = cursor.fetchone()["cnt"]
        # Row 1 committed in first block, row 2 rolled back
        assert count == 1
        db.close()

    def test_commit_and_rollback_methods(self):
        """Test explicit commit and rollback."""
        db = Database(":memory:")
        db.execute("CREATE TABLE t (id INTEGER)")
        db.commit()

        db.execute("INSERT INTO t VALUES (1)")
        db.rollback()

        cursor = db.execute("SELECT COUNT(*) AS cnt FROM t")
        assert cursor.fetchone()["cnt"] == 0

        db.execute("INSERT INTO t VALUES (2)")
        db.commit()

        cursor = db.execute("SELECT COUNT(*) AS cnt FROM t")
        assert cursor.fetchone()["cnt"] == 1
        db.close()


class TestDatabaseThreadSafety:
    """Tests for thread-local connection isolation."""

    def test_different_threads_get_different_connections(self):
        """Test that each thread gets its own connection object."""
        db = Database(":memory:")
        connections = []

        def get_conn():
            connections.append(db.connection)

        t1 = threading.Thread(target=get_conn)
        t2 = threading.Thread(target=get_conn)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Two threads should get different connection objects
        assert len(connections) == 2
        assert connections[0] is not connections[1]
        db.close()

    def test_same_thread_gets_same_connection(self):
        """Test that the same thread always gets the same connection."""
        db = Database(":memory:")
        conn1 = db.connection
        conn2 = db.connection
        assert conn1 is conn2
        db.close()

    def test_concurrent_reads(self, tmp_path):
        """Test that concurrent reads work correctly with WAL mode."""
        db_path = tmp_path / "concurrent.db"
        db = Database(db_path)
        db.execute("CREATE TABLE t (id INTEGER, val TEXT)")
        db.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(i, f"val_{i}") for i in range(100)],
        )
        db.commit()

        results = []
        errors = []

        def read_rows():
            try:
                # Each thread creates its own connection via thread-local
                cursor = db.execute("SELECT COUNT(*) AS cnt FROM t")
                results.append(cursor.fetchone()["cnt"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_rows) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in threads: {errors}"
        assert all(r == 100 for r in results)
        db.close()


# ============================================================
# Migration runner tests
# ============================================================


class TestMigrationDiscovery:
    """Tests for migration file discovery."""

    def test_discover_empty_directory(self, tmp_path):
        """Test discovering migrations from an empty directory."""
        migrations = _discover_from_directory(tmp_path)
        assert migrations == []

    def test_discover_nonexistent_directory(self, tmp_path):
        """Test discovering migrations from a non-existent directory."""
        migrations = _discover_from_directory(tmp_path / "nonexistent")
        assert migrations == []

    def test_discover_single_migration(self, tmp_path):
        """Test discovering a single migration file."""
        sql_file = tmp_path / "001_initial.sql"
        sql_file.write_text("CREATE TABLE t (id INTEGER);")

        migrations = _discover_from_directory(tmp_path)
        assert len(migrations) == 1
        assert migrations[0][0] == 1  # version
        assert migrations[0][1] == "initial"  # description
        assert "CREATE TABLE t" in migrations[0][2]  # SQL

    def test_discover_multiple_migrations_ordered(self, tmp_path):
        """Test that migrations are returned in version order."""
        (tmp_path / "003_third.sql").write_text("CREATE TABLE t3 (id INTEGER);")
        (tmp_path / "001_first.sql").write_text("CREATE TABLE t1 (id INTEGER);")
        (tmp_path / "002_second.sql").write_text("CREATE TABLE t2 (id INTEGER);")

        migrations = _discover_from_directory(tmp_path)
        assert len(migrations) == 3
        assert [m[0] for m in migrations] == [1, 2, 3]

    def test_skips_non_migration_files(self, tmp_path):
        """Test that non-matching files are skipped."""
        (tmp_path / "001_valid.sql").write_text("SELECT 1;")
        (tmp_path / "README.md").write_text("not a migration")
        (tmp_path / "__init__.py").write_text("")
        (tmp_path / "invalid.sql").write_text("SELECT 2;")

        migrations = _discover_from_directory(tmp_path)
        assert len(migrations) == 1
        assert migrations[0][0] == 1


class TestSchemaVersionTable:
    """Tests for schema_version table management."""

    def test_ensure_creates_table(self):
        """Test that schema_version table is created."""
        db = Database(":memory:")
        _ensure_schema_version_table(db)

        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        assert cursor.fetchone() is not None
        db.close()

    def test_ensure_is_idempotent(self):
        """Test that calling ensure multiple times doesn't fail."""
        db = Database(":memory:")
        _ensure_schema_version_table(db)
        _ensure_schema_version_table(db)  # Should not raise
        db.close()

    def test_get_applied_versions_empty(self):
        """Test getting versions from empty schema_version table."""
        db = Database(":memory:")
        _ensure_schema_version_table(db)
        versions = _get_applied_versions(db)
        assert versions == set()
        db.close()

    def test_get_applied_versions_with_data(self):
        """Test getting versions after applying migrations."""
        db = Database(":memory:")
        _ensure_schema_version_table(db)
        db.execute(
            "INSERT INTO schema_version (version, description) VALUES (1, 'first')"
        )
        db.execute(
            "INSERT INTO schema_version (version, description) VALUES (3, 'third')"
        )
        db.commit()

        versions = _get_applied_versions(db)
        assert versions == {1, 3}
        db.close()


class TestRunMigrations:
    """Tests for the migration runner."""

    def test_run_migrations_empty_dir(self, tmp_path):
        """Test running with no migration files."""
        db = Database(":memory:")
        count = run_migrations(db, tmp_path)
        assert count == 0
        db.close()

    def test_run_single_migration(self, tmp_path):
        """Test running a single migration."""
        sql = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"
        (tmp_path / "001_create_users.sql").write_text(sql)

        db = Database(":memory:")
        count = run_migrations(db, tmp_path)
        assert count == 1

        # Verify table was created
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        )
        assert cursor.fetchone() is not None

        # Verify version was recorded
        cursor = db.execute(
            "SELECT version, description FROM schema_version WHERE version = 1"
        )
        row = cursor.fetchone()
        assert row["version"] == 1
        assert row["description"] == "create users"
        db.close()

    def test_run_multiple_migrations(self, tmp_path):
        """Test running multiple migrations in order."""
        (tmp_path / "001_create_users.sql").write_text(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"
        )
        (tmp_path / "002_create_posts.sql").write_text(
            "CREATE TABLE posts (id INTEGER PRIMARY KEY, title TEXT);"
        )

        db = Database(":memory:")
        count = run_migrations(db, tmp_path)
        assert count == 2

        # Verify both tables exist
        for table in ("users", "posts"):
            cursor = db.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
            )
            assert cursor.fetchone() is not None
        db.close()

    def test_skip_already_applied(self, tmp_path):
        """Test that re-running migrations skips already applied."""
        (tmp_path / "001_create_users.sql").write_text(
            "CREATE TABLE users (id INTEGER PRIMARY KEY);"
        )

        db = Database(":memory:")

        # First run
        count1 = run_migrations(db, tmp_path)
        assert count1 == 1

        # Second run - should skip
        count2 = run_migrations(db, tmp_path)
        assert count2 == 0
        db.close()

    def test_apply_only_new_migrations(self, tmp_path):
        """Test that only new migrations are applied when new files appear."""
        (tmp_path / "001_create_users.sql").write_text(
            "CREATE TABLE users (id INTEGER PRIMARY KEY);"
        )

        db = Database(":memory:")
        count1 = run_migrations(db, tmp_path)
        assert count1 == 1

        # Add a new migration
        (tmp_path / "002_create_posts.sql").write_text(
            "CREATE TABLE posts (id INTEGER PRIMARY KEY);"
        )

        count2 = run_migrations(db, tmp_path)
        assert count2 == 1  # Only the new one

        versions = _get_applied_versions(db)
        assert versions == {1, 2}
        db.close()

    def test_failed_migration_rolls_back(self, tmp_path):
        """Test that a failed migration doesn't leave partial state."""
        (tmp_path / "001_valid.sql").write_text(
            "CREATE TABLE t1 (id INTEGER PRIMARY KEY);"
        )
        (tmp_path / "002_invalid.sql").write_text("THIS IS NOT VALID SQL AT ALL;")

        db = Database(":memory:")

        # First migration should succeed
        # Second should fail
        with pytest.raises(Exception):  # noqa: B017 - testing migration failure
            run_migrations(db, tmp_path)

        # First migration should have been applied
        versions = _get_applied_versions(db)
        assert 1 in versions
        # Second should NOT have been recorded
        assert 2 not in versions
        db.close()

    def test_migration_with_multiple_statements(self, tmp_path):
        """Test migration with multiple SQL statements."""
        sql = """
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id));
        CREATE INDEX idx_posts_user ON posts(user_id);
        """
        (tmp_path / "001_initial.sql").write_text(sql)

        db = Database(":memory:")
        count = run_migrations(db, tmp_path)
        assert count == 1

        # Verify all objects were created
        cursor = db.execute(
            "SELECT COUNT(*) AS cnt FROM sqlite_master WHERE type IN ('table', 'index')"
        )
        row = cursor.fetchone()
        # users + posts + schema_version tables + idx_posts_user index + sqlite_autoindex
        assert row["cnt"] >= 4
        db.close()


# ============================================================
# Module-level get_database / close_database tests
# ============================================================


class TestGetDatabase:
    """Tests for the module-level singleton functions."""

    def test_get_database_in_memory(self):
        """Test get_database with explicit :memory: path."""
        from claudetube.db import close_database, get_database, reset_database

        reset_database()
        db = get_database(":memory:")
        assert db is not None

        # Should return same instance
        db2 = get_database()
        assert db is db2

        close_database()

    def test_get_database_creates_file(self, tmp_path):
        """Test get_database creates the database file."""
        from claudetube.db import close_database, get_database, reset_database

        reset_database()
        db_path = tmp_path / "test.db"
        get_database(db_path)  # Creates the database file
        assert db_path.exists()
        close_database()

    def test_close_database(self):
        """Test close_database cleans up."""
        from claudetube.db import close_database, get_database, reset_database

        reset_database()
        db = get_database(":memory:")
        close_database()

        # After close, get_database should create a new instance
        db2 = get_database(":memory:")
        assert db is not db2
        close_database()

    def test_close_database_idempotent(self):
        """Test closing when already closed doesn't raise."""
        from claudetube.db import close_database, reset_database

        reset_database()
        close_database()  # Nothing to close
        close_database()  # Still nothing

    def test_default_path_resolution(self, tmp_path, monkeypatch):
        """Test that default path is {cache_dir}/claudetube.db."""
        from claudetube.db import _default_db_path

        monkeypatch.setenv("CLAUDETUBE_CACHE_DIR", str(tmp_path))

        from claudetube.config.loader import clear_config_cache

        clear_config_cache()

        path = _default_db_path()
        assert path == tmp_path / "claudetube.db"

    def test_get_database_runs_migrations(self, tmp_path):
        """Test that get_database runs migrations automatically."""
        from claudetube.db import close_database, get_database, reset_database

        reset_database()
        db = get_database(":memory:")

        # schema_version table should exist (created by migration runner)
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        assert cursor.fetchone() is not None
        close_database()

    def test_singleton_thread_safety(self):
        """Test that get_database is thread-safe."""
        from claudetube.db import close_database, get_database, reset_database

        reset_database()
        instances = []
        errors = []

        def get_db():
            try:
                instances.append(get_database(":memory:"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_db) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All threads should get the same instance
        assert all(i is instances[0] for i in instances)
        close_database()


class TestGetVectorsDatabase:
    """Tests for the vectors database singleton functions."""

    def test_get_vectors_database_in_memory(self):
        """Test get_vectors_database with explicit :memory: path."""
        from claudetube.db import (
            close_database,
            get_vectors_database,
            reset_database,
        )

        reset_database()
        vec_db = get_vectors_database(":memory:")
        assert vec_db is not None

        # Should return same instance
        vec_db2 = get_vectors_database()
        assert vec_db is vec_db2

        close_database()

    def test_get_vectors_database_creates_file(self, tmp_path):
        """Test get_vectors_database creates the database file."""
        from claudetube.db import (
            close_database,
            get_vectors_database,
            reset_database,
        )

        reset_database()
        db_path = tmp_path / "vectors.db"
        get_vectors_database(db_path)
        assert db_path.exists()
        close_database()

    def test_main_and_vectors_databases_are_separate(self):
        """Test that main db and vectors db are different instances."""
        from claudetube.db import (
            close_database,
            get_database,
            get_vectors_database,
            reset_database,
        )

        reset_database()
        db = get_database(":memory:")
        vec_db = get_vectors_database(":memory:")

        # They should be different instances
        assert db is not vec_db

        close_database()

    def test_vectors_database_has_vec_metadata_table(self):
        """Test that vectors database has vec_metadata table after migrations."""
        from claudetube.db import (
            close_database,
            get_vectors_database,
            reset_database,
        )

        reset_database()
        vec_db = get_vectors_database(":memory:")

        # vec_metadata table should exist
        cursor = vec_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_metadata'"
        )
        assert cursor.fetchone() is not None

        close_database()

    def test_main_database_does_not_have_vec_metadata(self):
        """Test that main database does NOT have vec_metadata table."""
        from claudetube.db import close_database, get_database, reset_database

        reset_database()
        db = get_database(":memory:")

        # vec_metadata table should NOT exist in main db
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_metadata'"
        )
        assert cursor.fetchone() is None

        close_database()

    def test_default_vectors_path_resolution(self, tmp_path, monkeypatch):
        """Test that default vectors path is {cache_dir}/claudetube-vectors.db."""
        from claudetube.db import _default_vectors_db_path

        monkeypatch.setenv("CLAUDETUBE_CACHE_DIR", str(tmp_path))

        from claudetube.config.loader import clear_config_cache

        clear_config_cache()

        path = _default_vectors_db_path()
        assert path == tmp_path / "claudetube-vectors.db"

    def test_close_database_closes_both(self):
        """Test close_database closes both main and vectors databases."""
        from claudetube.db import (
            close_database,
            get_database,
            get_vectors_database,
            reset_database,
        )

        reset_database()
        db = get_database(":memory:")
        vec_db = get_vectors_database(":memory:")

        close_database()

        # After close, both should create new instances
        reset_database()  # Clear the closed state
        db2 = get_database(":memory:")
        vec_db2 = get_vectors_database(":memory:")

        assert db is not db2
        assert vec_db is not vec_db2

        close_database()
