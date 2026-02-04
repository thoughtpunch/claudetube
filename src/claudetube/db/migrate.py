"""Database migration runner.

Reads SQL migration files from the migrations/ directory, tracks applied
versions in a schema_version table, and applies pending migrations in order.

Migration files are named NNN_description.sql (e.g., 001_initial.sql).
Each migration runs in its own transaction.

The database is split into two files:
- claudetube.db: Main migrations (NNN_description.sql)
- claudetube-vectors.db: Vector migrations (NNN_description.vec.sql)
"""

import logging
import re
from importlib import resources
from pathlib import Path

from claudetube.db.connection import Database

logger = logging.getLogger(__name__)

# Pattern for migration filenames: NNN_description.sql
_MIGRATION_PATTERN = re.compile(r"^(\d{3})_(.+)\.sql$")
# Pattern for vector migration filenames: NNN_description.vec.sql
_VEC_MIGRATION_PATTERN = re.compile(r"^(\d{3})_(.+)\.vec\.sql$")


def _ensure_schema_version_table(db: Database) -> None:
    """Create the schema_version table if it doesn't exist."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version     INTEGER PRIMARY KEY,
            applied_at  TEXT NOT NULL DEFAULT (datetime('now')),
            description TEXT
        )
    """)
    db.commit()


def _get_applied_versions(db: Database) -> set[int]:
    """Return the set of already-applied migration version numbers."""
    cursor = db.execute("SELECT version FROM schema_version ORDER BY version")
    return {row["version"] for row in cursor.fetchall()}


def _discover_migrations(
    migrations_dir: Path | None = None,
    vec_migrations: bool = False,
) -> list[tuple[int, str, str]]:
    """Discover migration SQL files sorted by version number.

    Args:
        migrations_dir: Directory containing .sql files. If None, uses the
            package's bundled migrations/ directory.
        vec_migrations: If True, discover .vec.sql files for the vectors database.

    Returns:
        List of (version, description, sql_content) tuples, sorted by version.
    """
    if migrations_dir is not None:
        return _discover_from_directory(migrations_dir, vec_migrations=vec_migrations)
    return _discover_from_package(vec_migrations=vec_migrations)


def _discover_from_directory(
    migrations_dir: Path, vec_migrations: bool = False
) -> list[tuple[int, str, str]]:
    """Discover migrations from a filesystem directory.

    Args:
        migrations_dir: Directory containing .sql files.
        vec_migrations: If True, discover .vec.sql files instead of .sql files.

    Returns:
        List of (version, description, sql_content) tuples.
    """
    migrations = []
    if not migrations_dir.is_dir():
        logger.warning("Migrations directory does not exist: %s", migrations_dir)
        return migrations

    pattern = _VEC_MIGRATION_PATTERN if vec_migrations else _MIGRATION_PATTERN
    glob_pattern = "*.vec.sql" if vec_migrations else "*.sql"

    for sql_file in sorted(migrations_dir.glob(glob_pattern)):
        # Skip .vec.sql files when looking for regular migrations
        if not vec_migrations and sql_file.name.endswith(".vec.sql"):
            continue
        match = pattern.match(sql_file.name)
        if not match:
            logger.debug("Skipping non-migration file: %s", sql_file.name)
            continue
        version = int(match.group(1))
        description = match.group(2).replace("_", " ")
        sql_content = sql_file.read_text(encoding="utf-8")
        migrations.append((version, description, sql_content))

    migrations.sort(key=lambda m: m[0])
    return migrations


def _discover_from_package(vec_migrations: bool = False) -> list[tuple[int, str, str]]:
    """Discover migrations bundled with the package.

    Args:
        vec_migrations: If True, discover .vec.sql files instead of .sql files.

    Returns:
        List of (version, description, sql_content) tuples.
    """
    migrations = []
    pattern = _VEC_MIGRATION_PATTERN if vec_migrations else _MIGRATION_PATTERN
    suffix = ".vec.sql" if vec_migrations else ".sql"

    try:
        migration_files = resources.files("claudetube.db.migrations")
        for item in migration_files.iterdir():
            if not item.name.endswith(suffix):
                continue
            # Skip .vec.sql when looking for regular migrations
            if not vec_migrations and item.name.endswith(".vec.sql"):
                continue
            match = pattern.match(item.name)
            if not match:
                continue
            version = int(match.group(1))
            description = match.group(2).replace("_", " ")
            sql_content = item.read_text(encoding="utf-8")
            migrations.append((version, description, sql_content))
    except (TypeError, FileNotFoundError):
        logger.debug("No bundled migrations found in package")

    migrations.sort(key=lambda m: m[0])
    return migrations


def run_migrations(db: Database, migrations_dir: Path | None = None) -> int:
    """Apply all pending migrations in order.

    Each migration runs in its own transaction. If a migration fails,
    it is rolled back and the error is raised (subsequent migrations
    are not attempted).

    Args:
        db: Database instance to migrate.
        migrations_dir: Optional directory with .sql files. If None,
            uses the package's bundled migrations/.

    Returns:
        Number of migrations applied.

    Raises:
        Exception: If any migration fails (after rolling back that migration).
    """
    _ensure_schema_version_table(db)
    applied = _get_applied_versions(db)
    available = _discover_migrations(migrations_dir)

    applied_count = 0
    for version, description, sql_content in available:
        if version in applied:
            logger.debug("Migration %03d already applied, skipping", version)
            continue

        logger.info("Applying migration %03d: %s", version, description)
        try:
            with db:
                # Execute the migration SQL
                db.executescript(sql_content)
                # Record the version (executescript auto-commits, so we
                # need a fresh transaction for the version insert)
            with db:
                db.execute(
                    "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                    (version, description),
                )
            applied_count += 1
            logger.info("Migration %03d applied successfully", version)
        except Exception:
            logger.exception("Migration %03d failed, rolling back", version)
            raise

    if applied_count == 0:
        logger.debug("No pending migrations")
    else:
        logger.info("Applied %d migration(s)", applied_count)

    return applied_count


def run_vectors_migrations(db: Database, migrations_dir: Path | None = None) -> int:
    """Apply all pending vector migrations in order.

    Vector migrations use .vec.sql extension and are applied to the
    vectors database (claudetube-vectors.db) which contains vec_metadata
    and the vec0 virtual table.

    Each migration runs in its own transaction. If a migration fails,
    it is rolled back and the error is raised.

    Args:
        db: Vectors Database instance to migrate.
        migrations_dir: Optional directory with .vec.sql files. If None,
            uses the package's bundled migrations/.

    Returns:
        Number of migrations applied.

    Raises:
        Exception: If any migration fails (after rolling back that migration).
    """
    _ensure_schema_version_table(db)
    applied = _get_applied_versions(db)
    available = _discover_migrations(migrations_dir, vec_migrations=True)

    applied_count = 0
    for version, description, sql_content in available:
        if version in applied:
            logger.debug("Vector migration %03d already applied, skipping", version)
            continue

        logger.info("Applying vector migration %03d: %s", version, description)
        try:
            with db:
                # Execute the migration SQL
                db.executescript(sql_content)
            with db:
                db.execute(
                    "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                    (version, description),
                )
            applied_count += 1
            logger.info("Vector migration %03d applied successfully", version)
        except Exception:
            logger.exception("Vector migration %03d failed, rolling back", version)
            raise

    if applied_count == 0:
        logger.debug("No pending vector migrations")
    else:
        logger.info("Applied %d vector migration(s)", applied_count)

    return applied_count
