"""
Logging utilities.
"""

import logging
import time

logger = logging.getLogger("claudetube")


def log_timed(msg: str, start_time: float | None = None) -> None:
    """Log timestamped message.

    Args:
        msg: Message to log
        start_time: Start time from time.time(), or None for [START]
    """
    elapsed = f"[{time.time() - start_time:.1f}s]" if start_time else "[START]"
    logger.info(f"{elapsed} {msg}")
