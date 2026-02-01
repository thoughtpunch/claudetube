"""
Text formatting utilities.
"""


def format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM-SS timestamp for filenames.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string (e.g., "01-30" for 90 seconds)
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}-{secs:02d}"


def format_duration(seconds: float | None) -> str | None:
    """Format seconds as human-readable duration string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string (e.g., "1:30" or "1:05:30"), or None
    """
    if seconds is None:
        return None

    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
