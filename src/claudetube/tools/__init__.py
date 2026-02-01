"""
External tool wrappers for claudetube.

Provides clean interfaces to yt-dlp, ffmpeg, ffprobe, and whisper.
"""

from claudetube.tools.base import ToolResult, VideoTool
from claudetube.tools.ffmpeg import FFmpegTool
from claudetube.tools.ffprobe import FFprobeTool, VideoMetadata
from claudetube.tools.whisper import WhisperTool
from claudetube.tools.yt_dlp import YtDlpTool

__all__ = [
    "VideoTool",
    "ToolResult",
    "YtDlpTool",
    "FFmpegTool",
    "FFprobeTool",
    "VideoMetadata",
    "WhisperTool",
]
