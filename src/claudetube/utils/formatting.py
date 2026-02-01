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
