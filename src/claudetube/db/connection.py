"""SQLite database connection manager.

Thread-safe Database class with WAL mode, foreign key enforcement,
and transaction support via context manager.

Each thread gets its own connection via threading.local().
"""

import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class Database:
    """Thread-safe SQLite database wrapper.

    Features:
    - WAL mode for concurrent reads during writes
    - Foreign key enforcement (PRAGMA foreign_keys = ON)
    - busy_timeout=5000 for concurrent access
    - Row factory for dict-like access
    - Thread-local connections (each thread gets its own)
    - Context manager for transactions
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        """Initialize the database.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory.
        """
        self.db_path = str(db_path)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._closed = False

    @property
    def connection(self) -> sqlite3.Connection:
        """Get the thread-local connection, creating one if needed."""
        if self._closed:
            raise RuntimeError("Database is closed")

        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = self._create_connection()
            self._local.connection = conn
        return conn

    def _create_connection(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        # WAL mode for concurrent reads; skip for :memory: (no effect)
        if self.db_path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        logger.debug("Created new SQLite connection for thread %s", threading.current_thread().name)
        return conn

    def execute(self, sql: str, params: tuple | dict | None = None) -> sqlite3.Cursor:
        """Execute a single SQL statement.

        Args:
            sql: SQL statement.
            params: Parameters for the SQL statement.

        Returns:
            sqlite3.Cursor with results.
        """
        if params is None:
            return self.connection.execute(sql)
        return self.connection.execute(sql, params)

    def executemany(self, sql: str, params_seq: list[tuple | dict]) -> sqlite3.Cursor:
        """Execute a SQL statement against multiple parameter sets.

        Args:
            sql: SQL statement.
            params_seq: Sequence of parameter sets.

        Returns:
            sqlite3.Cursor with results.
        """
        return self.connection.executemany(sql, params_seq)

    def executescript(self, sql: str) -> sqlite3.Cursor:
        """Execute multiple SQL statements as a script.

        Note: executescript issues an implicit COMMIT before running,
        so it cannot be used inside a transaction block.

        Args:
            sql: SQL script (multiple statements separated by ;).

        Returns:
            sqlite3.Cursor.
        """
        return self.connection.executescript(sql)

    def commit(self) -> None:
        """Commit the current transaction."""
        self.connection.commit()

    def rollback(self) -> None:
        """Roll back the current transaction."""
        self.connection.rollback()

    def close(self) -> None:
        """Close all connections and mark the database as closed."""
        self._closed = True
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            self._local.connection = None

    def __enter__(self) -> "Database":
        """Begin a transaction."""
        self.connection.execute("BEGIN")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Commit on success, rollback on exception."""
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
