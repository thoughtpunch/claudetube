"""
Chapter dataclass for video chapter/segment information.
"""

from dataclasses import dataclass


@dataclass
class Chapter:
    """A video chapter or segment with timing and metadata.

    Attributes:
        title: Chapter title/name
        start: Start time in seconds
        end: End time in seconds (None if unknown)
        source: Where this chapter came from (e.g., "youtube_chapters", "description_parsed")
        confidence: Confidence score 0.0-1.0 (higher = more reliable)
    """

    title: str
    start: float
    end: float | None = None
    source: str = "unknown"
    confidence: float = 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "start": self.start,
            "end": self.end,
            "source": self.source,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chapter":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            title=data.get("title", ""),
            start=data.get("start", 0.0),
            end=data.get("end"),
            source=data.get("source", "unknown"),
            confidence=data.get("confidence", 0.5),
        )
