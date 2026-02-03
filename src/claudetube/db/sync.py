"""Dual-write sync module for progressive enrichment.

Every JSON write has a companion SQLite sync. All sync functions are
fire-and-forget: they never raise exceptions, ensuring JSON remains
the authoritative source while SQLite is populated as a best-effort index.

Progressive enrichment: when metadata improves (NULL channel/playlist
replaced with real values), this module moves the cache directory and
updates the database path atomically.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database
    from claudetube.models.state import VideoState

logger = logging.getLogger(__name__)


def _get_db() -> Database | None:
    """Get the database instance, or None if unavailable.

    Uses lazy import to avoid import-time database initialization.
    Returns None if the database module is unavailable or fails.
    """
    try:
        from claudetube.db import get_database

        return get_database()
    except Exception:
        logger.debug("Database unavailable", exc_info=True)
        return None


def _cleanup_empty_parents(dir_path: Path, cache_base: Path) -> None:
    """Remove empty parent directories up to cache_base.

    After moving a directory from no_channel/no_playlist to a real path,
    clean up the now-empty placeholder directories.

    Args:
        dir_path: Directory that was just vacated (may not exist).
        cache_base: Stop cleaning up at this level (never delete cache_base).
    """
    try:
        # Resolve both paths for consistent comparison
        cache_base_resolved = cache_base.resolve()
        current = dir_path.resolve() if dir_path.exists() else dir_path

        # Work upward from the given directory, stopping at cache_base
        while True:
            # Stop if we've reached or passed the cache_base
            if current == cache_base_resolved:
                break

            # Stop if current is not under cache_base
            try:
                current.relative_to(cache_base_resolved)
            except ValueError:
                break

            # Check if directory exists and is empty
            if current.exists() and current.is_dir():
                try:
                    # rmdir only works on empty directories
                    current.rmdir()
                    logger.debug("Removed empty directory: %s", current)
                except OSError:
                    # Directory not empty or permission error - stop here
                    break
            else:
                # Directory doesn't exist, move to parent
                pass

            current = current.parent
    except Exception:
        # Best effort - don't fail if cleanup fails
        logger.debug("Error cleaning up empty parents", exc_info=True)


def sync_video(state: VideoState, cache_path: str) -> None:
    """Sync a video record from VideoState to SQLite.

    Creates or updates the video record using UPSERT semantics:
    - If video doesn't exist, creates it
    - If video exists, fills NULL fields without overwriting existing data

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        state: VideoState dataclass with video metadata.
        cache_path: Relative cache path for this video (e.g., "youtube/UCxx/PLxx/vid").
    """
    try:
        db = _get_db()
        if db is None:
            return

        from claudetube.db.repos.videos import VideoRepository

        repo = VideoRepository(db)

        # Determine domain - required for insert
        domain = state.domain
        if domain is None:
            # Fallback: try to extract from cache_path
            parts = Path(cache_path).parts
            domain = parts[0] if parts else "unknown"

        # Build metadata dict from VideoState
        metadata: dict[str, Any] = {}

        if state.channel_id is not None:
            metadata["channel"] = state.channel_id
        if state.playlist_id is not None:
            metadata["playlist"] = state.playlist_id
        if state.url is not None:
            metadata["url"] = state.url
        if state.title is not None:
            metadata["title"] = state.title
        if state.duration is not None:
            metadata["duration"] = state.duration
        if state.duration_string is not None:
            metadata["duration_string"] = state.duration_string
        if state.uploader is not None:
            metadata["uploader"] = state.uploader
        if state.channel is not None:
            metadata["channel_name"] = state.channel
        if state.upload_date is not None:
            metadata["upload_date"] = state.upload_date
        if state.description is not None:
            metadata["description"] = state.description
        if state.language is not None:
            metadata["language"] = state.language
        if state.view_count is not None:
            metadata["view_count"] = state.view_count
        if state.like_count is not None:
            metadata["like_count"] = state.like_count
        if state.source_type is not None:
            metadata["source_type"] = state.source_type

        repo.upsert(
            video_id=state.video_id,
            domain=domain,
            cache_path=cache_path,
            **metadata,
        )
        logger.debug("Synced video %s to SQLite", state.video_id)

    except Exception:
        # Fire-and-forget: log and continue
        logger.debug("Failed to sync video to SQLite", exc_info=True)


def enrich_video(video_id: str, metadata: dict[str, Any], cache_base: Path) -> None:
    """Progressively enrich a video with new metadata.

    When new metadata provides channel or playlist info that was previously
    NULL, this function:
    1. Builds the new hierarchical path from the metadata
    2. Moves the cache directory to the new location (if path improved)
    3. Updates the database with new metadata and cache_path

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_id: Natural key (e.g., YouTube video ID).
        metadata: yt-dlp metadata dict with keys like channel_id, playlist_id, etc.
        cache_base: Base cache directory (e.g., ~/.claude/video_cache/).
    """
    try:
        db = _get_db()
        if db is None:
            return

        from claudetube.db.repos.videos import VideoRepository
        from claudetube.models.video_path import VideoPath

        repo = VideoRepository(db)
        existing = repo.get_by_video_id(video_id)

        if existing is None:
            logger.debug("Cannot enrich video %s: not found in database", video_id)
            return

        old_cache_path = existing["cache_path"]
        old_dir = cache_base / old_cache_path

        # Build new VideoPath from current + new metadata
        # Start with existing values, overlay with new metadata
        domain = existing["domain"]
        channel = existing["channel"]
        playlist = existing["playlist"]

        # Check if metadata provides improved values
        if channel is None:
            new_channel = _extract_channel_from_metadata(metadata)
            if new_channel:
                channel = new_channel

        if playlist is None:
            new_playlist = _extract_playlist_from_metadata(metadata)
            if new_playlist:
                playlist = new_playlist

        # Build new path
        new_path = VideoPath(
            domain=domain,
            channel=channel,
            playlist=playlist,
            video_id=video_id,
        )
        new_cache_path = str(new_path.relative_path())

        # Check if path actually improved
        if new_cache_path == old_cache_path:
            # Path unchanged, but still UPSERT metadata
            _upsert_metadata_only(db, video_id, domain, old_cache_path, metadata)
            return

        new_dir = cache_base / new_cache_path

        # Attempt to move the directory
        if old_dir.exists() and not new_dir.exists():
            try:
                # Create parent directories
                new_dir.parent.mkdir(parents=True, exist_ok=True)

                # Move directory (atomic on same filesystem)
                shutil.move(str(old_dir), str(new_dir))
                logger.info("Moved video cache: %s -> %s", old_cache_path, new_cache_path)

                # Clean up empty parent directories
                _cleanup_empty_parents(old_dir.parent, cache_base)

                # Update database with new path
                _upsert_with_new_path(db, video_id, domain, new_cache_path, channel, playlist, metadata)

            except OSError as e:
                # Move failed - keep database at old path
                logger.warning(
                    "Failed to move cache directory %s -> %s: %s. "
                    "Database retains old path.",
                    old_cache_path,
                    new_cache_path,
                    e,
                )
                # Still UPSERT metadata, but keep old cache_path
                _upsert_metadata_only(db, video_id, domain, old_cache_path, metadata)
        else:
            # Can't move (source doesn't exist or dest already exists)
            # Just UPSERT metadata
            if new_dir.exists():
                # New location already exists - use it
                _upsert_with_new_path(db, video_id, domain, new_cache_path, channel, playlist, metadata)
            else:
                # Source doesn't exist - keep old path
                _upsert_metadata_only(db, video_id, domain, old_cache_path, metadata)

    except Exception:
        logger.debug("Failed to enrich video in SQLite", exc_info=True)


def _extract_channel_from_metadata(metadata: dict[str, Any]) -> str | None:
    """Extract channel from yt-dlp metadata.

    Priority: channel_id > uploader_id > sanitized channel name.
    """
    from claudetube.models.video_path import _sanitize_path_component

    for key in ("channel_id", "uploader_id"):
        value = metadata.get(key)
        if value:
            return _sanitize_path_component(str(value))

    # Fallback: channel display name
    channel_name = metadata.get("channel")
    if channel_name:
        return _sanitize_path_component(str(channel_name))

    return None


def _extract_playlist_from_metadata(metadata: dict[str, Any]) -> str | None:
    """Extract playlist from yt-dlp metadata.

    Priority: playlist_id > sanitized playlist_title.
    """
    from claudetube.models.video_path import _sanitize_path_component

    playlist_id = metadata.get("playlist_id")
    if playlist_id:
        return _sanitize_path_component(str(playlist_id))

    playlist_title = metadata.get("playlist_title")
    if playlist_title:
        return _sanitize_path_component(str(playlist_title))

    return None


def _upsert_metadata_only(
    db: Database,
    video_id: str,
    domain: str,
    cache_path: str,
    metadata: dict[str, Any],
) -> None:
    """UPSERT video metadata without changing cache_path."""
    from claudetube.db.repos.videos import VideoRepository

    repo = VideoRepository(db)

    fields = _build_metadata_fields(metadata)
    repo.upsert(video_id, domain, cache_path, **fields)


def _upsert_with_new_path(
    db: Database,
    video_id: str,
    domain: str,
    cache_path: str,
    channel: str | None,
    playlist: str | None,
    metadata: dict[str, Any],
) -> None:
    """UPSERT video with new cache_path and channel/playlist."""
    from claudetube.db.repos.videos import VideoRepository

    repo = VideoRepository(db)

    fields = _build_metadata_fields(metadata)
    if channel:
        fields["channel"] = channel
    if playlist:
        fields["playlist"] = playlist

    # For path changes, we need to force-update cache_path
    # The standard upsert uses COALESCE which won't replace non-NULL
    # So we do a direct update for cache_path
    existing = repo.get_by_video_id(video_id)
    if existing:
        db.execute(
            "UPDATE videos SET cache_path = ?, updated_at = datetime('now') WHERE video_id = ?",
            (cache_path, video_id),
        )
        db.commit()
        # Then upsert the rest of the metadata
        repo.upsert(video_id, domain, cache_path, **fields)
    else:
        # New record - just insert
        repo.upsert(video_id, domain, cache_path, **fields)


def _build_metadata_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build fields dict from yt-dlp metadata for upsert."""
    fields: dict[str, Any] = {}

    mappings = [
        ("url", "url"),
        ("title", "title"),
        ("duration", "duration"),
        ("duration_string", "duration_string"),
        ("uploader", "uploader"),
        ("channel", "channel_name"),
        ("upload_date", "upload_date"),
        ("description", "description"),
        ("language", "language"),
        ("view_count", "view_count"),
        ("like_count", "like_count"),
    ]

    for meta_key, field_key in mappings:
        value = metadata.get(meta_key)
        if value is not None:
            # Truncate description to reasonable length
            if meta_key == "description" and isinstance(value, str):
                value = value[:1500]
            fields[field_key] = value

    return fields


def sync_audio_track(
    video_uuid: str,
    format_: str,
    file_path: str,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
    bitrate_kbps: int | None = None,
    duration: float | None = None,
    file_size_bytes: int | None = None,
) -> str | None:
    """Sync an audio track record to SQLite.

    Creates a new audio_track record for the given video.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_uuid: UUID of the parent video record (from videos table).
        format_: Audio format (mp3, wav, etc.).
        file_path: Relative path to the audio file in cache.
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.
        bitrate_kbps: Bitrate in kbps.
        duration: Duration in seconds.
        file_size_bytes: File size in bytes.

    Returns:
        The generated UUID for the audio track, or None if sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return None

        from claudetube.db.repos.audio_tracks import AudioTrackRepository

        repo = AudioTrackRepository(db)

        # Check if this audio track already exists (by video + format)
        existing = repo.get_by_video_and_format(video_uuid, format_)
        if existing:
            logger.debug(
                "Audio track already exists for video %s format %s",
                video_uuid,
                format_,
            )
            return existing["id"]

        track_id = repo.insert(
            video_uuid=video_uuid,
            format_=format_,
            file_path=file_path,
            sample_rate=sample_rate,
            channels=channels,
            bitrate_kbps=bitrate_kbps,
            duration=duration,
            file_size_bytes=file_size_bytes,
        )
        logger.debug("Synced audio track %s for video %s", track_id, video_uuid)
        return track_id

    except Exception:
        logger.debug("Failed to sync audio track to SQLite", exc_info=True)
        return None


def sync_transcription(
    video_uuid: str,
    provider: str,
    format_: str,
    file_path: str,
    *,
    audio_track_id: str | None = None,
    model: str | None = None,
    language: str | None = None,
    full_text: str | None = None,
    word_count: int | None = None,
    duration: float | None = None,
    confidence: float | None = None,
    file_size_bytes: int | None = None,
    is_primary: bool = True,
) -> str | None:
    """Sync a transcription record to SQLite.

    Creates a new transcription record for the given video. If is_primary
    is True, any existing primary transcription for the video will be
    demoted.

    The full_text field is indexed by FTS5 for cross-video search.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_uuid: UUID of the parent video record (from videos table).
        provider: Transcription provider (whisper, youtube_subtitles, etc.).
        format_: Transcript format (srt, txt, vtt).
        file_path: Relative path to the transcript file in cache.
        audio_track_id: Optional UUID of the source audio track.
        model: Model used (e.g., 'small' for Whisper).
        language: Language code (e.g., 'en').
        full_text: Complete transcript text for FTS indexing.
        word_count: Number of words.
        duration: Duration in seconds.
        confidence: Confidence score (0.0 to 1.0).
        file_size_bytes: File size in bytes.
        is_primary: Whether this is the primary transcript (default True).

    Returns:
        The generated UUID for the transcription, or None if sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return None

        from claudetube.db.repos.transcriptions import TranscriptionRepository

        repo = TranscriptionRepository(db)

        transcription_id = repo.insert(
            video_uuid=video_uuid,
            provider=provider,
            format_=format_,
            file_path=file_path,
            audio_track_id=audio_track_id,
            model=model,
            language=language,
            full_text=full_text,
            word_count=word_count,
            duration=duration,
            confidence=confidence,
            file_size_bytes=file_size_bytes,
            is_primary=is_primary,
        )
        logger.debug(
            "Synced transcription %s for video %s (primary=%s)",
            transcription_id,
            video_uuid,
            is_primary,
        )
        return transcription_id

    except Exception:
        logger.debug("Failed to sync transcription to SQLite", exc_info=True)
        return None


def get_video_uuid(video_id: str) -> str | None:
    """Get the UUID for a video by its natural key.

    Helper function for callers that need the video UUID for sync
    operations (audio tracks, transcriptions, etc.).

    This is fire-and-forget: returns None on any error.

    Args:
        video_id: Natural key (e.g., YouTube video ID).

    Returns:
        The video's UUID, or None if not found or database unavailable.
    """
    try:
        db = _get_db()
        if db is None:
            return None

        from claudetube.db.repos.videos import VideoRepository

        repo = VideoRepository(db)
        existing = repo.get_by_video_id(video_id)
        return existing["id"] if existing else None

    except Exception:
        logger.debug("Failed to get video UUID", exc_info=True)
        return None
