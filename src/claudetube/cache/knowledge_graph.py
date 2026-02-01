"""
Cross-video knowledge graph for tracking concepts and entities.

Links videos through shared concepts, entities, and topics, enabling:
- Finding videos related to a topic
- Discovering connections between videos
- Building a knowledge base across all processed videos

Storage: ~/.claude/video_knowledge/graph.json

Architecture: Cheap First, Expensive Last
1. CACHE - Load existing graph from disk instantly
2. CHEAP - Index from already-processed entities/concepts
3. NO NEW COMPUTE - Never triggers expensive operations
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Default location for the knowledge graph
DEFAULT_KNOWLEDGE_DIR = Path.home() / ".claude" / "video_knowledge"


@dataclass
class VideoNode:
    """A video indexed in the knowledge graph."""

    video_id: str
    title: str = ""
    channel: str = ""
    indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "video_id": self.video_id,
            "title": self.title,
            "channel": self.channel,
            "indexed_at": self.indexed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VideoNode:
        """Create from dictionary."""
        return cls(
            video_id=data["video_id"],
            title=data.get("title", ""),
            channel=data.get("channel", ""),
            indexed_at=data.get("indexed_at", ""),
        )


@dataclass
class EntityNode:
    """An entity (person, organization, etc.) tracked across videos."""

    name: str
    entity_type: str  # 'person', 'organization', 'technology', etc.
    video_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.entity_type,
            "videos": self.video_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EntityNode:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            entity_type=data.get("type", "unknown"),
            video_ids=data.get("videos", []),
        )


@dataclass
class ConceptNode:
    """A concept tracked across videos."""

    name: str
    video_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "videos": self.video_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConceptNode:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            video_ids=data.get("videos", []),
        )


@dataclass
class RelatedVideoMatch:
    """A video matching a search query."""

    video_id: str
    video_title: str
    match_type: str  # 'entity' or 'concept'
    matched_term: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "match_type": self.match_type,
            "matched": self.matched_term,
        }


class VideoKnowledgeGraph:
    """Track concepts and entities across all videos.

    Maintains a global graph of:
    - Videos: Basic metadata (title, channel)
    - Entities: Named entities shared across videos
    - Concepts: Key concepts/topics shared across videos

    The graph enables cross-video search and discovery.

    Example usage:
        graph = VideoKnowledgeGraph()

        # Index a video's entities and concepts
        graph.add_video(
            video_id="abc123",
            metadata={"title": "Python Tutorial", "channel": "CodeChannel"},
            entities={"technology": ["Python", "Django"], "person": ["Guido"]},
            concepts=["web development", "programming", "frameworks"]
        )

        # Find videos about Python
        matches = graph.find_related_videos("python")

        # Find videos connected to a specific video
        connected = graph.get_video_connections("abc123")
    """

    def __init__(self, graph_dir: Path | None = None):
        """Initialize knowledge graph.

        Args:
            graph_dir: Directory for graph storage.
                      Defaults to ~/.claude/video_knowledge/
        """
        self.graph_dir = graph_dir or DEFAULT_KNOWLEDGE_DIR
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.graph_path = self.graph_dir / "graph.json"

        self._videos: dict[str, VideoNode] = {}
        self._entities: dict[str, EntityNode] = {}
        self._concepts: dict[str, ConceptNode] = {}

        self._load()

    def _load(self) -> None:
        """Load graph from disk."""
        if not self.graph_path.exists():
            return

        try:
            data = json.loads(self.graph_path.read_text())

            # Load videos
            for vid, vdata in data.get("videos", {}).items():
                self._videos[vid] = VideoNode.from_dict({"video_id": vid, **vdata})

            # Load entities
            for key, edata in data.get("entities", {}).items():
                self._entities[key] = EntityNode.from_dict({"name": key, **edata})

            # Load concepts
            for key, cdata in data.get("concepts", {}).items():
                self._concepts[key] = ConceptNode.from_dict({"name": key, **cdata})

            logger.info(
                f"Loaded knowledge graph: {len(self._videos)} videos, "
                f"{len(self._entities)} entities, {len(self._concepts)} concepts"
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load knowledge graph: {e}")

    def _save(self) -> None:
        """Save graph to disk."""
        data = {
            "videos": {
                vid: {"title": v.title, "channel": v.channel, "indexed_at": v.indexed_at}
                for vid, v in self._videos.items()
            },
            "entities": {
                key: {"type": e.entity_type, "videos": e.video_ids}
                for key, e in self._entities.items()
            },
            "concepts": {key: {"videos": c.video_ids} for key, c in self._concepts.items()},
        }
        self.graph_path.write_text(json.dumps(data, indent=2))
        logger.debug(f"Saved knowledge graph to {self.graph_path}")

    def add_video(
        self,
        video_id: str,
        metadata: dict,
        entities: dict[str, list[str]],
        concepts: list[str],
    ) -> None:
        """Index a video into the knowledge graph.

        Args:
            video_id: Unique video identifier
            metadata: Video metadata with 'title' and 'channel' keys
            entities: Dict mapping entity type to list of entity names
                     e.g., {"person": ["Alice", "Bob"], "technology": ["Python"]}
            concepts: List of concept terms
        """
        # Add or update video node
        self._videos[video_id] = VideoNode(
            video_id=video_id,
            title=metadata.get("title", ""),
            channel=metadata.get("channel", ""),
        )

        # Link entities
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_key = entity.lower().strip()
                if not entity_key:
                    continue

                if entity_key not in self._entities:
                    self._entities[entity_key] = EntityNode(
                        name=entity_key,
                        entity_type=entity_type,
                    )

                if video_id not in self._entities[entity_key].video_ids:
                    self._entities[entity_key].video_ids.append(video_id)

        # Link concepts
        for concept in concepts:
            concept_key = concept.lower().strip()
            if not concept_key:
                continue

            if concept_key not in self._concepts:
                self._concepts[concept_key] = ConceptNode(name=concept_key)

            if video_id not in self._concepts[concept_key].video_ids:
                self._concepts[concept_key].video_ids.append(video_id)

        self._save()
        logger.info(f"Indexed video {video_id} into knowledge graph")

    def remove_video(self, video_id: str) -> bool:
        """Remove a video from the knowledge graph.

        Args:
            video_id: Video to remove

        Returns:
            True if video was removed, False if not found
        """
        if video_id not in self._videos:
            return False

        # Remove from videos
        del self._videos[video_id]

        # Remove from entities
        empty_entities = []
        for key, entity in self._entities.items():
            if video_id in entity.video_ids:
                entity.video_ids.remove(video_id)
            if not entity.video_ids:
                empty_entities.append(key)

        for key in empty_entities:
            del self._entities[key]

        # Remove from concepts
        empty_concepts = []
        for key, concept in self._concepts.items():
            if video_id in concept.video_ids:
                concept.video_ids.remove(video_id)
            if not concept.video_ids:
                empty_concepts.append(key)

        for key in empty_concepts:
            del self._concepts[key]

        self._save()
        logger.info(f"Removed video {video_id} from knowledge graph")
        return True

    def find_related_videos(self, query: str) -> list[RelatedVideoMatch]:
        """Find videos related to a concept or entity.

        Args:
            query: Search query (case-insensitive substring match)

        Returns:
            List of RelatedVideoMatch objects, deduplicated by video_id
        """
        query_lower = query.lower()
        matches: list[RelatedVideoMatch] = []
        seen_videos: set[str] = set()

        # Search entities
        for entity_key, entity in self._entities.items():
            if query_lower in entity_key:
                for video_id in entity.video_ids:
                    if video_id not in seen_videos:
                        seen_videos.add(video_id)
                        video = self._videos.get(video_id)
                        matches.append(
                            RelatedVideoMatch(
                                video_id=video_id,
                                video_title=video.title if video else "",
                                match_type="entity",
                                matched_term=entity_key,
                            )
                        )

        # Search concepts
        for concept_key, concept in self._concepts.items():
            if query_lower in concept_key:
                for video_id in concept.video_ids:
                    if video_id not in seen_videos:
                        seen_videos.add(video_id)
                        video = self._videos.get(video_id)
                        matches.append(
                            RelatedVideoMatch(
                                video_id=video_id,
                                video_title=video.title if video else "",
                                match_type="concept",
                                matched_term=concept_key,
                            )
                        )

        return matches

    def get_video_connections(self, video_id: str) -> list[str]:
        """Get other videos sharing entities/concepts with this one.

        Args:
            video_id: Video to find connections for

        Returns:
            List of connected video IDs
        """
        if video_id not in self._videos:
            return []

        # Find shared entities
        shared_keys: set[str] = set()
        for key, entity in self._entities.items():
            if video_id in entity.video_ids:
                shared_keys.add(key)

        # Find shared concepts
        for key, concept in self._concepts.items():
            if video_id in concept.video_ids:
                shared_keys.add(key)

        # Find connected videos
        connected: set[str] = set()
        for key in shared_keys:
            if key in self._entities:
                connected.update(self._entities[key].video_ids)
            if key in self._concepts:
                connected.update(self._concepts[key].video_ids)

        # Remove self
        connected.discard(video_id)

        return list(connected)

    def get_video(self, video_id: str) -> VideoNode | None:
        """Get video node by ID."""
        return self._videos.get(video_id)

    def get_all_videos(self) -> list[VideoNode]:
        """Get all indexed videos."""
        return list(self._videos.values())

    def get_entity(self, name: str) -> EntityNode | None:
        """Get entity by name (lowercase)."""
        return self._entities.get(name.lower())

    def get_concept(self, name: str) -> ConceptNode | None:
        """Get concept by name (lowercase)."""
        return self._concepts.get(name.lower())

    def get_stats(self) -> dict:
        """Get statistics about the knowledge graph."""
        return {
            "video_count": len(self._videos),
            "entity_count": len(self._entities),
            "concept_count": len(self._concepts),
            "graph_path": str(self.graph_path),
        }

    def clear(self) -> None:
        """Clear all data from the knowledge graph."""
        self._videos = {}
        self._entities = {}
        self._concepts = {}
        if self.graph_path.exists():
            self.graph_path.unlink()
        logger.info("Cleared knowledge graph")

    @property
    def video_count(self) -> int:
        """Number of indexed videos."""
        return len(self._videos)

    @property
    def entity_count(self) -> int:
        """Number of tracked entities."""
        return len(self._entities)

    @property
    def concept_count(self) -> int:
        """Number of tracked concepts."""
        return len(self._concepts)


def get_knowledge_graph(graph_dir: Path | None = None) -> VideoKnowledgeGraph:
    """Get or create the global knowledge graph.

    Args:
        graph_dir: Optional custom directory for graph storage

    Returns:
        VideoKnowledgeGraph instance
    """
    return VideoKnowledgeGraph(graph_dir)


def index_video_to_graph(
    video_id: str,
    cache_dir: Path,
    graph_dir: Path | None = None,
    force: bool = False,
) -> dict:
    """Index a video's entities and concepts into the knowledge graph.

    Reads entities and concepts from the video's cache directory
    and adds them to the global knowledge graph.

    Follows "Cheap First, Expensive Last":
    1. Check if video already indexed (skip unless force)
    2. Load from entities/*.json (already processed)
    3. Never triggers new entity extraction

    Args:
        video_id: Video ID
        cache_dir: Video cache directory
        graph_dir: Optional custom directory for graph storage
        force: Re-index even if already present

    Returns:
        Dict with indexing results
    """
    from claudetube.cache.entities import load_concepts, load_objects
    from claudetube.cache.storage import load_state

    graph = get_knowledge_graph(graph_dir)

    # Check if already indexed
    if not force and graph.get_video(video_id):
        return {
            "video_id": video_id,
            "status": "already_indexed",
            "from_cache": True,
        }

    # Load video state for metadata
    state_file = cache_dir / "state.json"
    state = load_state(state_file)
    if not state:
        return {
            "video_id": video_id,
            "error": "Video not cached. Run process_video first.",
        }

    # Load entities
    objects = load_objects(cache_dir) or {}
    concepts_data = load_concepts(cache_dir) or {}

    # Convert objects to entity format
    entities: dict[str, list[str]] = {"object": list(objects.keys())}

    # Load people from entities/people.json if available
    people_path = cache_dir / "entities" / "people.json"
    if people_path.exists():
        try:
            people_data = json.loads(people_path.read_text())
            entities["person"] = [p.get("name", f"Person {i}") for i, p in enumerate(people_data.get("people", []))]
        except (json.JSONDecodeError, KeyError):
            pass

    # Get concept terms
    concept_terms = list(concepts_data.keys())

    # Add to graph
    graph.add_video(
        video_id=video_id,
        metadata={
            "title": state.title or "",
            "channel": state.channel or state.uploader or "",
        },
        entities=entities,
        concepts=concept_terms,
    )

    return {
        "video_id": video_id,
        "status": "indexed",
        "entities_count": sum(len(v) for v in entities.values()),
        "concepts_count": len(concept_terms),
        "from_cache": False,
    }
