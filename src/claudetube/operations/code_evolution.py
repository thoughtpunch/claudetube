"""
Code evolution tracking across video scenes.

Tracks how code changes throughout a coding tutorial video.
Identifies code units (functions, classes, files) and tracks their evolution:
- shown: first appearance
- modified: content changed
- added_lines: new lines added
- removed_lines: lines removed
- deleted: no longer appears

Follows the "Cheap First, Expensive Last" principle:
1. CACHE - Return instantly if code_evolution.json already exists
2. TECHNICAL - Use cached technical.json data (already generated)
3. CODE - Use code block detection (already implemented)
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from claudetube.cache.manager import CacheManager
from claudetube.cache.scenes import (
    SceneBoundary,
    get_technical_json_path,
    load_scenes_data,
)
from claudetube.config.loader import get_cache_dir
from claudetube.utils.logging import log_timed

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CodeSnapshot:
    """A snapshot of code at a specific point in the video."""

    scene_id: int
    timestamp: float  # Seconds from video start
    content: str
    language: str | None
    change_type: str  # 'shown', 'modified', 'added_lines', 'removed_lines', 'unchanged'
    diff_summary: str | None = None  # Brief description of changes

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scene_id": self.scene_id,
            "timestamp": self.timestamp,
            "content": self.content,
            "language": self.language,
            "change_type": self.change_type,
            "diff_summary": self.diff_summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CodeSnapshot:
        """Create from dictionary."""
        return cls(
            scene_id=data["scene_id"],
            timestamp=data["timestamp"],
            content=data["content"],
            language=data.get("language"),
            change_type=data["change_type"],
            diff_summary=data.get("diff_summary"),
        )


@dataclass
class CodeUnit:
    """A tracked code unit (function, class, file, or snippet)."""

    unit_id: str  # e.g., "function:validate_token", "class:UserAuth"
    unit_type: str  # 'function', 'class', 'file', 'snippet'
    name: str  # Human-readable name
    snapshots: list[CodeSnapshot] = field(default_factory=list)

    @property
    def first_seen(self) -> float:
        """Timestamp when this code unit first appeared."""
        if not self.snapshots:
            return 0.0
        return self.snapshots[0].timestamp

    @property
    def last_seen(self) -> float:
        """Timestamp when this code unit last appeared."""
        if not self.snapshots:
            return 0.0
        return self.snapshots[-1].timestamp

    @property
    def change_count(self) -> int:
        """Number of times this code was modified."""
        return sum(
            1 for s in self.snapshots if s.change_type not in ("shown", "unchanged")
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "name": self.name,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "change_count": self.change_count,
            "snapshots": [s.to_dict() for s in self.snapshots],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CodeUnit:
        """Create from dictionary."""
        return cls(
            unit_id=data["unit_id"],
            unit_type=data["unit_type"],
            name=data["name"],
            snapshots=[CodeSnapshot.from_dict(s) for s in data.get("snapshots", [])],
        )


@dataclass
class CodeEvolutionData:
    """Container for all code evolution tracking data for a video."""

    video_id: str
    method: str  # "technical_json", "code_analysis"
    code_units: list[CodeUnit] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_id": self.video_id,
            "method": self.method,
            "unit_count": len(self.code_units),
            "code_units": {u.unit_id: u.to_dict() for u in self.code_units},
            "summary": {
                "total_units": len(self.code_units),
                "by_type": self._count_by_type(),
                "most_modified": self._get_most_modified(),
            },
        }

    def _count_by_type(self) -> dict[str, int]:
        """Count code units by type."""
        counts: dict[str, int] = {}
        for unit in self.code_units:
            counts[unit.unit_type] = counts.get(unit.unit_type, 0) + 1
        return counts

    def _get_most_modified(self, limit: int = 5) -> list[dict]:
        """Get the most frequently modified code units."""
        sorted_units = sorted(
            self.code_units, key=lambda u: u.change_count, reverse=True
        )
        return [
            {"unit_id": u.unit_id, "name": u.name, "change_count": u.change_count}
            for u in sorted_units[:limit]
            if u.change_count > 0
        ]

    @classmethod
    def from_dict(cls, data: dict) -> CodeEvolutionData:
        """Create from dictionary."""
        units_dict = data.get("code_units", {})
        units = [CodeUnit.from_dict(u) for u in units_dict.values()]
        return cls(
            video_id=data["video_id"],
            method=data.get("method", "unknown"),
            code_units=units,
        )


# Patterns for identifying code units
FUNCTION_PATTERNS = [
    re.compile(r"\bdef\s+(\w+)\s*\("),  # Python
    re.compile(r"\bfunction\s+(\w+)\s*\("),  # JavaScript
    re.compile(r"\bfn\s+(\w+)\s*\("),  # Rust
    re.compile(r"\bfunc\s+(\w+)\s*\("),  # Go
    re.compile(r"(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*\{"),  # Java/C#
    re.compile(r"\bconst\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"),  # JS arrow function
    re.compile(r"\blet\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"),  # JS arrow function
]

CLASS_PATTERNS = [
    re.compile(r"\bclass\s+(\w+)"),  # Python, JS, Java, etc.
    re.compile(r"\bstruct\s+(\w+)"),  # Rust, Go, C
    re.compile(r"\binterface\s+(\w+)"),  # TypeScript, Java
    re.compile(r"\benum\s+(\w+)"),  # Various languages
]

FILE_PATTERNS = [
    # Look for file paths in comments or strings
    re.compile(r"[#//*\s]+[\w/\\]+\.(\w{2,4})\b"),  # File extensions in comments
    re.compile(r'["\'][\w/\\]+\.(\w{2,4})["\']'),  # File paths in strings
]


def get_code_evolution_path(cache_dir: Path) -> Path:
    """Get path to entities/code_evolution.json for a video.

    Args:
        cache_dir: Video cache directory

    Returns:
        Path to entities/code_evolution.json
    """
    entities_dir = cache_dir / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    return entities_dir / "code_evolution.json"


def identify_code_unit(content: str) -> tuple[str, str, str]:
    """Identify what code unit a code block represents.

    Args:
        content: Code content to analyze

    Returns:
        Tuple of (unit_id, unit_type, name)
    """
    # Try to identify class/struct/interface FIRST
    # (classes often contain methods, so check class patterns before function patterns)
    for pattern in CLASS_PATTERNS:
        match = pattern.search(content)
        if match:
            name = match.group(1)
            unit_type = "class"
            if "struct" in pattern.pattern:
                unit_type = "struct"
            elif "interface" in pattern.pattern:
                unit_type = "interface"
            elif "enum" in pattern.pattern:
                unit_type = "enum"
            return f"{unit_type}:{name}", unit_type, name

    # Try to identify standalone function
    for pattern in FUNCTION_PATTERNS:
        match = pattern.search(content)
        if match:
            name = match.group(1)
            return f"function:{name}", "function", name

    # Fallback: create a snippet ID based on first significant line
    lines = [line.strip() for line in content.split("\n") if line.strip() and not line.strip().startswith(("#", "//", "/*", "*"))]
    if lines:
        first_line = lines[0][:40]
        # Create a stable hash
        snippet_hash = abs(hash(first_line)) % 100000
        return f"snippet:{snippet_hash}", "snippet", first_line[:30]

    return "snippet:unknown", "snippet", "unknown code"


def detect_change_type(old_content: str, new_content: str) -> tuple[str, str | None]:
    """Detect the type of change between two code versions.

    Args:
        old_content: Previous version of code
        new_content: New version of code

    Returns:
        Tuple of (change_type, diff_summary)
    """
    if old_content == new_content:
        return "unchanged", None

    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()

    diff = list(
        difflib.unified_diff(old_lines, new_lines, lineterm="", n=0)
    )

    additions = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

    # Build summary
    summary_parts = []
    if additions > 0:
        summary_parts.append(f"+{additions} lines")
    if deletions > 0:
        summary_parts.append(f"-{deletions} lines")
    summary = ", ".join(summary_parts) if summary_parts else None

    # Determine change type
    if additions > 0 and deletions == 0:
        return "added_lines", summary
    elif deletions > 0 and additions == 0:
        return "removed_lines", summary
    else:
        return "modified", summary


def _normalize_content(content: str) -> str:
    """Normalize code content for comparison.

    Strips extra whitespace and normalizes line endings.
    """
    lines = [line.rstrip() for line in content.splitlines()]
    # Remove empty lines at start and end
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _extract_code_blocks_from_technical(technical_path: Path) -> list[dict]:
    """Extract code blocks from a technical.json file.

    Args:
        technical_path: Path to technical.json

    Returns:
        List of code block dicts with 'content', 'language', 'timestamp'
    """
    if not technical_path.exists():
        return []

    try:
        data = json.loads(technical_path.read_text())
        code_blocks = []

        for frame in data.get("frames", []):
            timestamp = frame.get("timestamp", 0.0)

            # Check for code_blocks field (from code analysis)
            for block in frame.get("code_blocks", []):
                code_blocks.append({
                    "content": block.get("content", ""),
                    "language": block.get("language"),
                    "timestamp": timestamp,
                })

            # Also check for raw text regions that look like code
            # (in case code analysis hasn't been run)
            if not frame.get("code_blocks"):
                for region in frame.get("regions", []):
                    text = region.get("text", "")
                    # Simple heuristic - if content_type is 'code'
                    if frame.get("content_type") == "code" and len(text) > 20:
                        code_blocks.append({
                            "content": text,
                            "language": None,
                            "timestamp": timestamp,
                        })

        return code_blocks
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse technical.json: {e}")
        return []


def _track_from_technical_json(
    scenes: list[SceneBoundary],
    cache_dir: Path,
) -> CodeEvolutionData:
    """Track code evolution using existing technical.json data.

    This is the CHEAP path - uses already-generated technical analysis
    to identify and track code without re-running OCR or code detection.

    Args:
        scenes: List of scene boundaries
        cache_dir: Video cache directory

    Returns:
        CodeEvolutionData with tracked code units
    """
    # Map unit_id -> CodeUnit
    units_by_id: dict[str, CodeUnit] = {}

    for scene in scenes:
        technical_path = get_technical_json_path(cache_dir, scene.scene_id)
        code_blocks = _extract_code_blocks_from_technical(technical_path)

        scene_timestamp = scene.start_time + scene.duration() / 2  # Middle of scene

        for block in code_blocks:
            content = _normalize_content(block.get("content", ""))
            if not content or len(content) < 10:
                continue

            unit_id, unit_type, name = identify_code_unit(content)
            language = block.get("language")

            if unit_id not in units_by_id:
                # First appearance
                units_by_id[unit_id] = CodeUnit(
                    unit_id=unit_id,
                    unit_type=unit_type,
                    name=name,
                )
                change_type = "shown"
                diff_summary = None
            else:
                # Compare to previous version
                unit = units_by_id[unit_id]
                prev_content = unit.snapshots[-1].content
                change_type, diff_summary = detect_change_type(prev_content, content)

            units_by_id[unit_id].snapshots.append(
                CodeSnapshot(
                    scene_id=scene.scene_id,
                    timestamp=scene_timestamp,
                    content=content,
                    language=language,
                    change_type=change_type,
                    diff_summary=diff_summary,
                )
            )

    return CodeEvolutionData(
        video_id="",  # Will be set by caller
        method="technical_json",
        code_units=list(units_by_id.values()),
    )


def track_code_evolution(
    video_id: str,
    force: bool = False,
    output_base: Path | None = None,
) -> dict:
    """Track code evolution across scenes in a video.

    Follows "Cheap First, Expensive Last" principle:
    1. CACHE - Return entities/code_evolution.json instantly if exists
    2. TECHNICAL - Use technical.json data (already generated via OCR/code analysis)

    Args:
        video_id: Video ID
        force: Re-generate even if cached
        output_base: Cache directory

    Returns:
        Dict with tracking results
    """
    t0 = time.time()
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    if not cache_dir.exists():
        return {"error": "Video not cached. Run process_video first.", "video_id": video_id}

    # 1. CACHE - Return instantly if already exists
    evolution_path = get_code_evolution_path(cache_dir)
    if not force and evolution_path.exists():
        try:
            data = json.loads(evolution_path.read_text())
            log_timed(f"Code evolution: loaded from cache ({data.get('unit_count', 0)} units)", t0)
            return data
        except json.JSONDecodeError:
            pass  # Re-generate if cached data is invalid

    # Load scenes data
    scenes_data = load_scenes_data(cache_dir)
    if not scenes_data:
        return {"error": "No scenes found. Run get_scenes first.", "video_id": video_id}

    # 2. TECHNICAL - Track from technical.json data
    log_timed("Code evolution: analyzing technical data...", t0)
    evolution_data = _track_from_technical_json(scenes_data.scenes, cache_dir)
    evolution_data.video_id = video_id

    # Save to cache
    result = evolution_data.to_dict()
    evolution_path.write_text(json.dumps(result, indent=2))

    # Update state.json
    state_file = cache_dir / "state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        state["code_evolution_complete"] = True
        state_file.write_text(json.dumps(state, indent=2))

    log_timed(f"Code evolution complete: {len(evolution_data.code_units)} units tracked", t0)

    return result


def get_code_evolution(
    video_id: str,
    output_base: Path | None = None,
) -> dict:
    """Get cached code evolution data for a video.

    Does NOT generate new tracking - use track_code_evolution for that.

    Args:
        video_id: Video ID
        output_base: Cache directory

    Returns:
        Dict with cached code evolution data
    """
    cache = CacheManager(output_base or get_cache_dir())
    cache_dir = cache.get_cache_dir(video_id)

    if not cache_dir.exists():
        return {"error": "Video not cached", "video_id": video_id}

    evolution_path = get_code_evolution_path(cache_dir)
    if not evolution_path.exists():
        return {"error": "No code evolution data. Run track_code_evolution first.", "video_id": video_id}

    try:
        data = json.loads(evolution_path.read_text())
        return data
    except json.JSONDecodeError:
        return {"error": "Invalid code_evolution.json", "video_id": video_id}


def query_code_evolution(
    video_id: str,
    query: str,
    output_base: Path | None = None,
) -> dict:
    """Query code evolution for a specific code unit.

    Example: "How did the auth middleware evolve?"

    Args:
        video_id: Video ID
        query: Search query (function/class name)
        output_base: Cache directory

    Returns:
        Dict with matching code units and their evolution
    """
    data = get_code_evolution(video_id, output_base)

    if "error" in data:
        return data

    query_lower = query.lower()
    matches = []

    for unit_id, unit_data in data.get("code_units", {}).items():
        unit_name = unit_data.get("name", "").lower()
        if query_lower in unit_name or query_lower in unit_id.lower():
            matches.append(unit_data)

    return {
        "video_id": video_id,
        "query": query,
        "matches": matches,
        "match_count": len(matches),
    }
