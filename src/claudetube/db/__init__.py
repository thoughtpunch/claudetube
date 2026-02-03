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
    and runs any pending migrations.

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

        db = Database(db_path)
        run_migrations(db)
        _db_instance = db
        logger.info("Database initialized at %s", db_path)
        return db


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
