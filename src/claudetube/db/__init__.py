"""Database module for claudetube.

Provides a singleton Database connection and migration support.

Usage:
    from claudetube.db import get_database, close_database

    db = get_database()
    cursor = db.execute("SELECT * FROM videos WHERE video_id = ?", (vid,))
    row = cursor.fetchone()

    # Transactions via context manager:
    with db:
        db.execute("INSERT INTO ...", (...))
        db.execute("UPDATE ...", (...))
    # auto-commits on success, rolls back on exception

    # Cleanup on shutdown:
    close_database()
"""

import logging
import threading
from pathlib import Path

from claudetube.db.connection import Database
from claudetube.db.migrate import run_migrations

logger = logging.getLogger(__name__)

_db_instance: Database | None = None
_db_lock = threading.Lock()


def get_database(db_path: str | Path | None = None) -> Database:
    """Get the singleton Database instance, creating it if needed.

    On first call, creates the database, enables WAL mode and foreign keys,
    and runs any pending migrations. If this is a fresh database (no existing
    videos), it auto-imports from the JSON cache.

    Args:
        db_path: Path to the SQLite database file. If None, auto-discovers
            from config: {cache_dir}/claudetube.db

    Returns:
        The singleton Database instance.
    """
    global _db_instance

    if _db_instance is not None:
        return _db_instance

    with _db_lock:
        # Double-check after acquiring lock
        if _db_instance is not None:
            return _db_instance

        if db_path is None:
            db_path = _default_db_path()

        # Check if database file exists before creating
        db_path = Path(db_path)
        is_new_db = not db_path.exists()

        db = Database(db_path)
        run_migrations(db)

        # Auto-import from JSON cache on first creation
        if is_new_db:
            _auto_import_from_cache(db, db_path)

        _db_instance = db
        logger.info("Database initialized at %s", db_path)
        return db


def _auto_import_from_cache(db: Database, db_path: Path) -> None:
    """Auto-import existing JSON caches into the fresh database.

    Called only on first database creation to populate SQLite from
    existing video caches. This ensures the full video library is
    queryable immediately without re-processing.

    Args:
        db: Fresh Database instance.
        db_path: Path to the database file (used to infer cache_base).
    """
    try:
        # Infer cache_base from db_path (db is typically in cache dir)
        cache_base = db_path.parent

        # Import from cache
        from claudetube.db.importer import auto_import

        imported = auto_import(cache_base, db)
        if imported > 0:
            logger.info("Auto-imported %d videos from JSON cache", imported)
    except Exception:
        # Don't fail database init if import fails
        logger.warning("Auto-import from JSON cache failed", exc_info=True)


def close_database() -> None:
    """Close and release the singleton Database instance."""
    global _db_instance

    with _db_lock:
        if _db_instance is not None:
            _db_instance.close()
            _db_instance = None
            logger.info("Database closed")


def _default_db_path() -> Path:
    """Resolve the default database path from config."""
    from claudetube.config.loader import get_cache_dir

    cache_dir = get_cache_dir(ensure_exists=True)
    return cache_dir / "claudetube.db"


def reset_database() -> None:
    """Reset the singleton for testing purposes."""
    global _db_instance
    with _db_lock:
        if _db_instance is not None:
            _db_instance.close()
        _db_instance = None
