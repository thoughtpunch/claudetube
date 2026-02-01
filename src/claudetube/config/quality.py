"""
Video quality tier definitions for frame extraction.
"""

# Quality tiers for frame extraction
# Lower tiers = faster download, smaller files, lower resolution
QUALITY_TIERS: dict[str, dict] = {
    "lowest": {
        "sort": "+res,+size,+br,+fps",
        "width": 480,
        "jpeg_q": 5,
        "concurrent_fragments": 1,
    },
    "low": {
        "sort": "res:360,+size,+br",
        "width": 640,
        "jpeg_q": 4,
        "concurrent_fragments": 2,
    },
    "medium": {
        "sort": "res:480",
        "width": 854,
        "jpeg_q": 3,
        "concurrent_fragments": 4,
    },
    "high": {
        "sort": "res:720",
        "width": 1280,
        "jpeg_q": 2,
        "concurrent_fragments": 4,
    },
    "highest": {
        "sort": "res:1080",
        "width": 1280,
        "jpeg_q": 2,
        "concurrent_fragments": 4,
    },
}

# Ordered list of quality tiers (lowest to highest)
QUALITY_LADDER: list[str] = ["lowest", "low", "medium", "high", "highest"]


def next_quality(current: str) -> str | None:
    """Return the next quality tier, or None if already at highest."""
    try:
        idx = QUALITY_LADDER.index(current)
        return QUALITY_LADDER[idx + 1] if idx + 1 < len(QUALITY_LADDER) else None
    except ValueError:
        return None
