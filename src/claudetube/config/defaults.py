"""
Default configuration values for claudetube.

Note: Cache directory is configured via config/loader.py which supports
environment variables (CLAUDETUBE_CACHE_DIR), project config, and user config.
"""

# Default whisper model (tiny is fastest, small is good balance)
DEFAULT_WHISPER_MODEL = "tiny"

# Transcript truncation limit for inline responses
TRANSCRIPT_INLINE_CAP = 50_000

# Timeouts (seconds)
METADATA_TIMEOUT = 30
SUBTITLE_TIMEOUT = 30
THUMBNAIL_TIMEOUT = 15

# Audio quality for transcription (lower = smaller file, faster download)
AUDIO_QUALITY = "64K"

# Batch size for whisper batched inference
WHISPER_BATCH_SIZE = 16

# Minimum coverage ratio to accept batched transcription
# (fallback to non-batched if coverage is too low)
MIN_TRANSCRIPT_COVERAGE = 0.25
