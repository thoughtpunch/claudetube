"""Entity repository for CRUD operations on entity-related tables.

Manages entities, entity_appearances, and entity_video_summary tables.
Replaces the in-memory VideoKnowledgeGraph (cache/knowledge_graph.py).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database


# Valid entity types per schema CHECK constraint
VALID_ENTITY_TYPES = frozenset(
    ["object", "concept", "person", "technology", "organization"]
)


class EntityRepository:
    """Repository for entity operations.

    Entities represent named concepts, people, technologies, objects, and
    organizations that appear in videos. The repository tracks:

    - entities: Global entity definitions with UNIQUE(name, entity_type)
    - entity_appearances: Where entities appear (video, scene, timestamp)
    - entity_video_summary: Aggregated stats per entity-video pair

    This replaces the file-based VideoKnowledgeGraph with SQLite storage.
    """

    def __init__(self, db: Database) -> None:
        """Initialize with a Database instance.

        Args:
            db: Database connection wrapper.
        """
        self.db = db

    def insert_entity(self, name: str, entity_type: str) -> str:
        """Insert or get an entity (upsert pattern).

        Uses INSERT OR IGNORE + SELECT to handle the UNIQUE(name, entity_type)
        constraint gracefully. Always returns the entity UUID.

        Args:
            name: Entity name (e.g., "Python", "Alice", "machine learning").
            entity_type: One of: object, concept, person, technology, organization.

        Returns:
            The UUID for the entity (existing or newly created).

        Raises:
            ValueError: If entity_type is invalid or name is empty.
        """
        if not name or not name.strip():
            msg = "Entity name cannot be empty"
            raise ValueError(msg)

        name = name.strip()

        if entity_type not in VALID_ENTITY_TYPES:
            msg = f"Invalid entity_type: {entity_type}. Must be one of {sorted(VALID_ENTITY_TYPES)}"
            raise ValueError(msg)

        new_id = str(uuid.uuid4())

        # Try to insert (will be ignored if duplicate)
        self.db.execute(
            """
            INSERT OR IGNORE INTO entities (id, name, entity_type)
            VALUES (?, ?, ?)
            """,
            (new_id, name, entity_type),
        )
        self.db.commit()

        # Always retrieve the ID (either new or existing)
        cursor = self.db.execute(
            "SELECT id FROM entities WHERE name = ? AND entity_type = ?",
            (name, entity_type),
        )
        row = cursor.fetchone()
        return row["id"]

    def insert_appearance(
        self,
        entity_uuid: str,
        video_uuid: str,
        scene_id: int,
        timestamp: float,
        *,
        score: float | None = None,
    ) -> str:
        """Record an entity appearance in a video scene.

        Uses INSERT OR IGNORE to handle the UNIQUE(entity_id, video_id, scene_id)
        constraint gracefully.

        Args:
            entity_uuid: UUID of the entity.
            video_uuid: UUID of the video.
            scene_id: Scene index (0-based).
            timestamp: Timestamp in seconds.
            score: Optional confidence score (0-1).

        Returns:
            The UUID for the appearance (existing or newly created).

        Raises:
            ValueError: If scene_id < 0, timestamp < 0, or score out of range.
        """
        if scene_id < 0:
            msg = f"scene_id must be >= 0, got {scene_id}"
            raise ValueError(msg)

        if timestamp < 0:
            msg = f"timestamp must be >= 0, got {timestamp}"
            raise ValueError(msg)

        if score is not None and (score < 0 or score > 1):
            msg = f"score must be between 0 and 1, got {score}"
            raise ValueError(msg)

        new_id = str(uuid.uuid4())

        # Try to insert (will be ignored if duplicate)
        self.db.execute(
            """
            INSERT OR IGNORE INTO entity_appearances (
                id, entity_id, video_id, scene_id, timestamp, score
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (new_id, entity_uuid, video_uuid, scene_id, timestamp, score),
        )
        self.db.commit()

        # Retrieve the ID (either new or existing)
        cursor = self.db.execute(
            """
            SELECT id FROM entity_appearances
            WHERE entity_id = ? AND video_id = ? AND scene_id = ?
            """,
            (entity_uuid, video_uuid, scene_id),
        )
        row = cursor.fetchone()
        return row["id"]

    def insert_video_summary(
        self,
        entity_uuid: str,
        video_uuid: str,
        frequency: int = 1,
        *,
        avg_score: float | None = None,
    ) -> str:
        """Record or update entity-video aggregated summary.

        Uses INSERT OR REPLACE to upsert the summary record.

        Args:
            entity_uuid: UUID of the entity.
            video_uuid: UUID of the video.
            frequency: Number of appearances (default: 1).
            avg_score: Optional average confidence score (0-1).

        Returns:
            The UUID for the summary record.

        Raises:
            ValueError: If frequency < 1 or avg_score out of range.
        """
        if frequency < 1:
            msg = f"frequency must be >= 1, got {frequency}"
            raise ValueError(msg)

        if avg_score is not None and (avg_score < 0 or avg_score > 1):
            msg = f"avg_score must be between 0 and 1, got {avg_score}"
            raise ValueError(msg)

        # Check for existing record
        cursor = self.db.execute(
            """
            SELECT id FROM entity_video_summary
            WHERE entity_id = ? AND video_id = ?
            """,
            (entity_uuid, video_uuid),
        )
        existing = cursor.fetchone()

        if existing:
            # Update existing record
            self.db.execute(
                """
                UPDATE entity_video_summary
                SET frequency = ?, avg_score = ?
                WHERE id = ?
                """,
                (frequency, avg_score, existing["id"]),
            )
            self.db.commit()
            return existing["id"]
        else:
            # Insert new record
            new_id = str(uuid.uuid4())
            self.db.execute(
                """
                INSERT INTO entity_video_summary (
                    id, entity_id, video_id, frequency, avg_score
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (new_id, entity_uuid, video_uuid, frequency, avg_score),
            )
            self.db.commit()
            return new_id

    def get_by_uuid(self, uuid_: str) -> dict[str, Any] | None:
        """Get an entity by its UUID.

        Args:
            uuid_: The UUID primary key.

        Returns:
            Dict with entity data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM entities WHERE id = ?",
            (uuid_,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_name_and_type(
        self, name: str, entity_type: str
    ) -> dict[str, Any] | None:
        """Get an entity by name and type.

        Args:
            name: Entity name.
            entity_type: Entity type.

        Returns:
            Dict with entity data, or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM entities WHERE name = ? AND entity_type = ?",
            (name.strip(), entity_type),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def find_by_name(self, query: str) -> list[dict[str, Any]]:
        """Find entities by name (case-insensitive substring match).

        Args:
            query: Search query string.

        Returns:
            List of matching entities.
        """
        pattern = f"%{query}%"
        cursor = self.db.execute(
            """
            SELECT * FROM entities
            WHERE name LIKE ? COLLATE NOCASE
            ORDER BY name
            """,
            (pattern,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_video_entities(self, video_uuid: str) -> list[dict[str, Any]]:
        """Get all entities for a video via entity_video_summary.

        Args:
            video_uuid: UUID of the video.

        Returns:
            List of entities with frequency and avg_score.
        """
        cursor = self.db.execute(
            """
            SELECT
                e.id,
                e.name,
                e.entity_type,
                evs.frequency,
                evs.avg_score
            FROM entity_video_summary evs
            JOIN entities e ON evs.entity_id = e.id
            WHERE evs.video_id = ?
            ORDER BY evs.frequency DESC, e.name
            """,
            (video_uuid,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_entity_videos(self, entity_name: str) -> list[dict[str, Any]]:
        """Get all videos containing an entity by name.

        Args:
            entity_name: Entity name to search for.

        Returns:
            List of videos with entity frequency and metadata.
        """
        cursor = self.db.execute(
            """
            SELECT
                v.id as video_uuid,
                v.video_id,
                v.title,
                v.domain,
                e.entity_type,
                evs.frequency,
                evs.avg_score
            FROM entity_video_summary evs
            JOIN entities e ON evs.entity_id = e.id
            JOIN videos v ON evs.video_id = v.id
            WHERE e.name = ? COLLATE NOCASE
            ORDER BY evs.frequency DESC
            """,
            (entity_name.strip(),),
        )
        return [dict(row) for row in cursor.fetchall()]

    def find_related_videos(self, query: str) -> list[dict[str, Any]]:
        """Find videos related to a topic (entity name substring match).

        Replaces VideoKnowledgeGraph.find_related_videos().

        Args:
            query: Search query (case-insensitive substring match).

        Returns:
            List of matching videos with entity context.
        """
        pattern = f"%{query}%"
        cursor = self.db.execute(
            """
            SELECT DISTINCT
                v.id as video_uuid,
                v.video_id,
                v.title as video_title,
                e.name as matched_term,
                e.entity_type as match_type
            FROM entity_video_summary evs
            JOIN entities e ON evs.entity_id = e.id
            JOIN videos v ON evs.video_id = v.id
            WHERE e.name LIKE ? COLLATE NOCASE
            ORDER BY v.title
            """,
            (pattern,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_connections(self, video_uuid: str) -> list[str]:
        """Get videos sharing entities with a specific video.

        Replaces VideoKnowledgeGraph.get_video_connections().

        Args:
            video_uuid: UUID of the video to find connections for.

        Returns:
            List of connected video UUIDs (excluding the input video).
        """
        cursor = self.db.execute(
            """
            SELECT DISTINCT evs2.video_id
            FROM entity_video_summary evs1
            JOIN entity_video_summary evs2 ON evs1.entity_id = evs2.entity_id
            WHERE evs1.video_id = ?
              AND evs2.video_id != ?
            """,
            (video_uuid, video_uuid),
        )
        return [row["video_id"] for row in cursor.fetchall()]

    def get_appearances(
        self,
        video_uuid: str,
        scene_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get entity appearances for a video, optionally filtered by scene.

        Args:
            video_uuid: UUID of the video.
            scene_id: Optional scene to filter by.

        Returns:
            List of appearances with entity details.
        """
        if scene_id is not None:
            cursor = self.db.execute(
                """
                SELECT
                    ea.*,
                    e.name,
                    e.entity_type
                FROM entity_appearances ea
                JOIN entities e ON ea.entity_id = e.id
                WHERE ea.video_id = ? AND ea.scene_id = ?
                ORDER BY ea.timestamp
                """,
                (video_uuid, scene_id),
            )
        else:
            cursor = self.db.execute(
                """
                SELECT
                    ea.*,
                    e.name,
                    e.entity_type
                FROM entity_appearances ea
                JOIN entities e ON ea.entity_id = e.id
                WHERE ea.video_id = ?
                ORDER BY ea.timestamp
                """,
                (video_uuid,),
            )
        return [dict(row) for row in cursor.fetchall()]

    def delete_entity(self, entity_uuid: str) -> bool:
        """Delete an entity by UUID.

        Note: Due to ON DELETE CASCADE, this removes all appearances
        and video summaries for this entity.

        Args:
            entity_uuid: UUID of the entity to delete.

        Returns:
            True if an entity was deleted, False if not found.
        """
        cursor = self.db.execute(
            "DELETE FROM entities WHERE id = ?",
            (entity_uuid,),
        )
        self.db.commit()
        return cursor.rowcount > 0

    def delete_video_entities(self, video_uuid: str) -> int:
        """Delete all entity data for a video.

        Removes appearances and video summaries, but not the entities
        themselves (which may be referenced by other videos).

        Args:
            video_uuid: UUID of the video.

        Returns:
            Number of records deleted (appearances + summaries).
        """
        cursor1 = self.db.execute(
            "DELETE FROM entity_appearances WHERE video_id = ?",
            (video_uuid,),
        )
        count1 = cursor1.rowcount

        cursor2 = self.db.execute(
            "DELETE FROM entity_video_summary WHERE video_id = ?",
            (video_uuid,),
        )
        count2 = cursor2.rowcount

        self.db.commit()
        return count1 + count2

    def count_entities(self) -> int:
        """Count total entities in the database.

        Returns:
            Total number of entities.
        """
        cursor = self.db.execute("SELECT COUNT(*) as cnt FROM entities")
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    def count_video_summaries(self) -> int:
        """Count total entity-video relationships.

        Returns:
            Total number of entity_video_summary records.
        """
        cursor = self.db.execute("SELECT COUNT(*) as cnt FROM entity_video_summary")
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    def get_stats(self) -> dict[str, int]:
        """Get statistics about the entity tables.

        Returns:
            Dict with counts for entities, appearances, and summaries.
        """
        entities = self.count_entities()

        cursor = self.db.execute("SELECT COUNT(*) as cnt FROM entity_appearances")
        appearances = cursor.fetchone()["cnt"]

        summaries = self.count_video_summaries()

        # Count unique videos with entities
        cursor = self.db.execute(
            "SELECT COUNT(DISTINCT video_id) as cnt FROM entity_video_summary"
        )
        videos = cursor.fetchone()["cnt"]

        return {
            "entity_count": entities,
            "appearance_count": appearances,
            "summary_count": summaries,
            "video_count": videos,
        }
