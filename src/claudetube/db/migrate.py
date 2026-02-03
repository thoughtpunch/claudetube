"""Database migration runner.

Reads SQL migration files from the migrations/ directory, tracks applied
versions in a schema_version table, and applies pending migrations in order.

Migration files are named NNN_description.sql (e.g., 001_initial.sql).
Each migration runs in its own transaction.
"""

import logging
import re
from importlib import resources
from pathlib import Path

from claudetube.db.connection import Database

logger = logging.getLogger(__name__)

# Pattern for migration filenames: NNN_description.sql
_MIGRATION_PATTERN = re.compile(r"^(\d{3})_(.+)\.sql$")


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


def _discover_migrations(migrations_dir: Path | None = None) -> list[tuple[int, str, str]]:
    """Discover migration SQL files sorted by version number.

    Args:
        migrations_dir: Directory containing .sql files. If None, uses the
            package's bundled migrations/ directory.

    Returns:
        List of (version, description, sql_content) tuples, sorted by version.
    """
    if migrations_dir is not None:
        return _discover_from_directory(migrations_dir)
    return _discover_from_package()


def _discover_from_directory(migrations_dir: Path) -> list[tuple[int, str, str]]:
    """Discover migrations from a filesystem directory."""
    migrations = []
    if not migrations_dir.is_dir():
        logger.warning("Migrations directory does not exist: %s", migrations_dir)
        return migrations

    for sql_file in sorted(migrations_dir.glob("*.sql")):
        match = _MIGRATION_PATTERN.match(sql_file.name)
        if not match:
            logger.debug("Skipping non-migration file: %s", sql_file.name)
            continue
        version = int(match.group(1))
        description = match.group(2).replace("_", " ")
        sql_content = sql_file.read_text(encoding="utf-8")
        migrations.append((version, description, sql_content))

    migrations.sort(key=lambda m: m[0])
    return migrations


def _discover_from_package() -> list[tuple[int, str, str]]:
    """Discover migrations bundled with the package."""
    migrations = []
    try:
        migration_files = resources.files("claudetube.db.migrations")
        for item in migration_files.iterdir():
            if not item.name.endswith(".sql"):
                continue
            match = _MIGRATION_PATTERN.match(item.name)
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
