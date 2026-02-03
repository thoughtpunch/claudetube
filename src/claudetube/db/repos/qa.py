"""Q&A repository for CRUD operations on qa_history and qa_scenes tables.

Manages question-answer pairs for videos with support for full-text search
and scene associations.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database


class QARepository:
    """Repository for Q&A operations.

    Q&A pairs represent questions asked about videos and their answers,
    supporting progressive learning. Each Q&A can be associated with
    specific scenes via the qa_scenes junction table.

    Full-text search is supported via the qa_fts virtual table.
    """

    def __init__(self, db: Database) -> None:
        """Initialize with a Database instance.

        Args:
            db: Database connection wrapper.
        """
        self.db = db

    def insert(
        self,
        video_uuid: str,
        question: str,
        answer: str,
        *,
        scene_ids: list[int] | None = None,
    ) -> str:
        """Insert a new Q&A pair with optional scene associations.

        Args:
            video_uuid: UUID of the video this Q&A is about.
            question: The question asked.
            answer: The answer given.
            scene_ids: Optional list of scene IDs this Q&A relates to.

        Returns:
            The generated UUID for the new Q&A record.

        Raises:
            ValueError: If question or answer is empty.
            sqlite3.IntegrityError: If video_uuid doesn't exist.
        """
        if not question or not question.strip():
            msg = "Question cannot be empty"
            raise ValueError(msg)

        if not answer or not answer.strip():
            msg = "Answer cannot be empty"
            raise ValueError(msg)

        question = question.strip()
        answer = answer.strip()

        new_id = str(uuid.uuid4())

        self.db.execute(
            """
            INSERT INTO qa_history (id, video_id, question, answer)
            VALUES (?, ?, ?, ?)
            """,
            (new_id, video_uuid, question, answer),
        )

        # Insert scene associations if provided
        if scene_ids:
            for scene_id in scene_ids:
                if scene_id < 0:
                    msg = f"scene_id must be >= 0, got {scene_id}"
                    raise ValueError(msg)
                self.db.execute(
                    "INSERT INTO qa_scenes (qa_id, scene_id) VALUES (?, ?)",
                    (new_id, scene_id),
                )

        self.db.commit()
        return new_id

    def get_by_uuid(self, uuid_: str) -> dict[str, Any] | None:
        """Get a Q&A pair by its UUID.

        Args:
            uuid_: The UUID primary key.

        Returns:
            Dict with Q&A data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM qa_history WHERE id = ?",
            (uuid_,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        qa = dict(row)

        # Also fetch associated scene IDs
        scene_cursor = self.db.execute(
            "SELECT scene_id FROM qa_scenes WHERE qa_id = ? ORDER BY scene_id",
            (uuid_,),
        )
        qa["scene_ids"] = [r["scene_id"] for r in scene_cursor.fetchall()]

        return qa

    def get_by_video(self, video_uuid: str) -> list[dict[str, Any]]:
        """Get all Q&A pairs for a video.

        Args:
            video_uuid: UUID of the video.

        Returns:
            List of Q&A dicts, ordered by creation time (newest first).
        """
        cursor = self.db.execute(
            """
            SELECT * FROM qa_history
            WHERE video_id = ?
            ORDER BY created_at DESC
            """,
            (video_uuid,),
        )
        results = []
        for row in cursor.fetchall():
            qa = dict(row)
            # Fetch scene associations
            scene_cursor = self.db.execute(
                "SELECT scene_id FROM qa_scenes WHERE qa_id = ? ORDER BY scene_id",
                (qa["id"],),
            )
            qa["scene_ids"] = [r["scene_id"] for r in scene_cursor.fetchall()]
            results.append(qa)
        return results

    def search_fts(self, query: str) -> list[dict[str, Any]]:
        """Search Q&A using full-text search.

        Searches across question and answer fields via the qa_fts
        virtual table.

        Args:
            query: Search query string. Supports FTS5 syntax.

        Returns:
            List of matching Q&A pairs with rank, ordered by relevance.
        """
        escaped_query = self._escape_fts_query(query)

        cursor = self.db.execute(
            """
            SELECT
                q.*,
                v.video_id as video_natural_id,
                v.title as video_title,
                rank
            FROM qa_fts
            JOIN qa_history q ON qa_fts.rowid = q.rowid
            JOIN videos v ON q.video_id = v.id
            WHERE qa_fts MATCH ?
            ORDER BY rank
            """,
            (escaped_query,),
        )
        results = []
        for row in cursor.fetchall():
            qa = dict(row)
            # Fetch scene associations
            scene_cursor = self.db.execute(
                "SELECT scene_id FROM qa_scenes WHERE qa_id = ? ORDER BY scene_id",
                (qa["id"],),
            )
            qa["scene_ids"] = [r["scene_id"] for r in scene_cursor.fetchall()]
            results.append(qa)
        return results

    def get_for_scene(self, video_uuid: str, scene_id: int) -> list[dict[str, Any]]:
        """Get Q&A pairs related to a specific scene.

        Args:
            video_uuid: UUID of the video.
            scene_id: The scene identifier (0-indexed).

        Returns:
            List of Q&A dicts related to this scene.
        """
        cursor = self.db.execute(
            """
            SELECT q.*
            FROM qa_history q
            JOIN qa_scenes qs ON q.id = qs.qa_id
            WHERE q.video_id = ? AND qs.scene_id = ?
            ORDER BY q.created_at DESC
            """,
            (video_uuid, scene_id),
        )
        results = []
        for row in cursor.fetchall():
            qa = dict(row)
            # Fetch all scene associations for this Q&A
            scene_cursor = self.db.execute(
                "SELECT scene_id FROM qa_scenes WHERE qa_id = ? ORDER BY scene_id",
                (qa["id"],),
            )
            qa["scene_ids"] = [r["scene_id"] for r in scene_cursor.fetchall()]
            results.append(qa)
        return results

    def add_scene_association(self, qa_uuid: str, scene_id: int) -> bool:
        """Add a scene association to an existing Q&A.

        Args:
            qa_uuid: UUID of the Q&A record.
            scene_id: Scene ID to associate.

        Returns:
            True if added, False if association already exists.

        Raises:
            ValueError: If scene_id is negative.
        """
        if scene_id < 0:
            msg = f"scene_id must be >= 0, got {scene_id}"
            raise ValueError(msg)

        try:
            self.db.execute(
                "INSERT INTO qa_scenes (qa_id, scene_id) VALUES (?, ?)",
                (qa_uuid, scene_id),
            )
            self.db.commit()
            return True
        except Exception:
            # Already exists (PRIMARY KEY violation)
            return False

    def remove_scene_association(self, qa_uuid: str, scene_id: int) -> bool:
        """Remove a scene association from a Q&A.

        Args:
            qa_uuid: UUID of the Q&A record.
            scene_id: Scene ID to remove.

        Returns:
            True if removed, False if association didn't exist.
        """
        cursor = self.db.execute(
            "DELETE FROM qa_scenes WHERE qa_id = ? AND scene_id = ?",
            (qa_uuid, scene_id),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def delete(self, qa_uuid: str) -> bool:
        """Delete a Q&A pair by its UUID.

        Note: Due to ON DELETE CASCADE, this also removes
        all qa_scenes associations.

        Args:
            qa_uuid: UUID of the Q&A to delete.

        Returns:
            True if a Q&A was deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM qa_history WHERE id = ?",
            (qa_uuid,),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def delete_by_video(self, video_uuid: str) -> int:
        """Delete all Q&A pairs for a video.

        Args:
            video_uuid: UUID of the video.

        Returns:
            Number of Q&A records deleted.
        """
        cursor = self.db.execute(
            "DELETE FROM qa_history WHERE video_id = ?",
            (video_uuid,),
        )
        self.db.commit()
        return cursor.rowcount

    def count_by_video(self, video_uuid: str) -> int:
        """Count Q&A pairs for a video.

        Args:
            video_uuid: UUID of the video.

        Returns:
            Number of Q&A pairs.
        """
        cursor = self.db.execute(
            "SELECT COUNT(*) as cnt FROM qa_history WHERE video_id = ?",
            (video_uuid,),
        )
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    def _escape_fts_query(self, query: str) -> str:
        """Escape a query string for FTS5 MATCH syntax.

        Wraps terms in double quotes to prevent FTS5 syntax errors
        from special characters.

        Args:
            query: Raw search query.

        Returns:
            Escaped query safe for FTS5 MATCH.
        """
        terms = query.split()
        if not terms:
            return '""'
        escaped_terms = ['"' + term.replace('"', '""') + '"' for term in terms]
        return " ".join(escaped_terms)
