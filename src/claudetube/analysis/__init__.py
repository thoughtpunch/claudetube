"""
Analysis module for claudetube.

Provides boundary detection and transcript analysis utilities.
"""

from claudetube.analysis.linguistic import (
    Boundary,
    detect_linguistic_boundaries,
)

__all__ = [
    "Boundary",
    "detect_linguistic_boundaries",
]
