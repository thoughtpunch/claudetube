"""
Navigation module for playlist-aware video navigation.

Provides progress tracking and navigation tools for moving through playlists.
"""

from claudetube.navigation.context import PlaylistContext
from claudetube.navigation.progress import PlaylistProgress

__all__ = ["PlaylistContext", "PlaylistProgress"]
