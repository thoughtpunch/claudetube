"""Dual-write sync module for progressive enrichment.

Every JSON write has a companion SQLite sync. All sync functions are
fire-and-forget: they never raise exceptions, ensuring JSON remains
the authoritative source while SQLite is populated as a best-effort index.

Progressive enrichment: when metadata improves (NULL channel/playlist
replaced with real values), this module moves the cache directory and
updates the database path atomically.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database
    from claudetube.models.state import VideoState

logger = logging.getLogger(__name__)


def safe_update_pipeline_step(
    step_uuid: str | None,
    status: str,
    *,
    error_message: str | None = None,
) -> None:
    """Fire-and-forget wrapper for update_pipeline_step.

    Handles None step_uuid gracefully and suppresses all exceptions.
    Use this from processor.py to avoid try-except-pass patterns.

    Args:
        step_uuid: UUID of the step to update, or None to skip.
        status: New status value.
        error_message: Optional error message (for failed status).
    """
    if step_uuid is None:
        return
    with contextlib.suppress(Exception):
        update_pipeline_step(step_uuid, status, error_message=error_message)


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

    Note: This syncs ONLY the fields present in VideoState. Queryable metadata
    (description, categories, tags, view_count, like_count) should be synced
    separately using sync_video_metadata() with the raw yt-dlp metadata dict.

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

        # Build metadata dict from VideoState (fast-access fields only)
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
        if state.language is not None:
            metadata["language"] = state.language
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


def sync_video_metadata(video_id: str, raw_metadata: dict[str, Any]) -> None:
    """Sync queryable metadata from raw yt-dlp metadata dict to SQLite.

    This syncs fields that are NOT stored in VideoState (description, categories,
    tags, view_count, like_count). SQLite is the source of truth for these fields.

    Call this after creating a VideoState from metadata to ensure all queryable
    fields are persisted to the database.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_id: Natural key (e.g., YouTube video ID).
        raw_metadata: Raw yt-dlp metadata dict (or similar structure).
    """
    try:
        db = _get_db()
        if db is None:
            return

        from claudetube.db.repos.videos import VideoRepository

        repo = VideoRepository(db)

        # Check if video exists in DB
        existing = repo.get_by_video_id(video_id)
        if existing is None:
            logger.debug("Cannot sync metadata for %s: video not in database", video_id)
            return

        # Build queryable metadata dict
        updates: dict[str, Any] = {}

        description = raw_metadata.get("description")
        if description is not None:
            # Truncate description to reasonable length
            updates["description"] = (description or "")[:1500]

        view_count = raw_metadata.get("view_count")
        if view_count is not None:
            updates["view_count"] = view_count

        like_count = raw_metadata.get("like_count")
        if like_count is not None:
            updates["like_count"] = like_count

        if not updates:
            return

        # Update using raw SQL for simplicity (upsert only fills NULLs)
        set_clauses = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [video_id]
        db.execute(
            f"UPDATE videos SET {set_clauses}, updated_at = datetime('now') WHERE video_id = ?",
            tuple(values),
        )
        db.commit()
        logger.debug("Synced queryable metadata for %s to SQLite", video_id)

    except Exception:
        # Fire-and-forget: log and continue
        logger.debug("Failed to sync video metadata to SQLite", exc_info=True)


def sync_video_tags(video_id: str, tags: list[str] | None) -> None:
    """Sync video tags to SQLite (video_tags table).

    Tags are NOT stored in VideoState. SQLite is the source of truth.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_id: Natural key (e.g., YouTube video ID).
        tags: List of tag strings (truncated to first 15).
    """
    if not tags:
        return

    try:
        db = _get_db()
        if db is None:
            return

        from claudetube.db.repos.videos import VideoRepository

        repo = VideoRepository(db)

        # Get video UUID
        existing = repo.get_by_video_id(video_id)
        if existing is None:
            logger.debug("Cannot sync tags for %s: video not in database", video_id)
            return

        video_uuid = existing["id"]

        # Limit to first 15 tags
        tags = tags[:15]

        # Insert tags (ignore duplicates)
        import uuid

        for tag in tags:
            if not tag or not tag.strip():
                continue
            with contextlib.suppress(Exception):
                db.execute(
                    "INSERT OR IGNORE INTO video_tags (id, video_id, tag) VALUES (?, ?, ?)",
                    (str(uuid.uuid4()), video_uuid, tag.strip()),
                )

        db.commit()
        logger.debug("Synced %d tags for %s to SQLite", len(tags), video_id)

    except Exception:
        # Fire-and-forget: log and continue
        logger.debug("Failed to sync video tags to SQLite", exc_info=True)


def sync_local_file_metadata(
    video_id: str,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    codec: str | None = None,
) -> None:
    """Sync local file metadata (dimensions, codec) to SQLite.

    For local video files, stores video dimensions in the description field.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_id: Natural key (e.g., generated video_id for local file).
        width: Video width in pixels.
        height: Video height in pixels.
        fps: Frames per second.
        codec: Video codec name.
    """
    if not width or not height:
        return

    try:
        db = _get_db()
        if db is None:
            return

        # Build dimensions string
        dimensions = f"{width}x{height}"
        if fps:
            dimensions += f" @ {fps:.1f}fps"
        if codec:
            dimensions += f" ({codec})"

        db.execute(
            "UPDATE videos SET description = ?, updated_at = datetime('now') WHERE video_id = ?",
            (dimensions, video_id),
        )
        db.commit()
        logger.debug("Synced local file metadata for %s to SQLite", video_id)

    except Exception:
        # Fire-and-forget: log and continue
        logger.debug("Failed to sync local file metadata to SQLite", exc_info=True)


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

        # Check if old_dir has actual content (state.json)
        old_has_content = old_dir.exists() and (old_dir / "state.json").exists()
        new_has_content = new_dir.exists() and (new_dir / "state.json").exists()

        # Attempt to move/merge the directory
        if old_has_content and not new_has_content:
            # Old has content, new doesn't (or doesn't exist) - move old to new
            try:
                if new_dir.exists():
                    # New dir exists but is empty shell - remove it first
                    shutil.rmtree(str(new_dir))
                    logger.debug("Removed empty shell directory: %s", new_dir)

                # Create parent directories
                new_dir.parent.mkdir(parents=True, exist_ok=True)

                # Move directory (atomic on same filesystem)
                shutil.move(str(old_dir), str(new_dir))
                logger.info(
                    "Moved video cache: %s -> %s", old_cache_path, new_cache_path
                )

                # Clean up empty parent directories
                _cleanup_empty_parents(old_dir.parent, cache_base)

                # Update database with new path
                _upsert_with_new_path(
                    db, video_id, domain, new_cache_path, channel, playlist, metadata
                )

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
        elif new_has_content:
            # New location has content - use it, clean up old if it exists
            if old_has_content and old_dir != new_dir:
                # Both have content - this shouldn't happen, but prefer new
                logger.warning(
                    "Both old and new cache dirs have content: %s, %s. Using new.",
                    old_cache_path,
                    new_cache_path,
                )
            _upsert_with_new_path(
                db, video_id, domain, new_cache_path, channel, playlist, metadata
            )
        elif old_has_content:
            # Old has content, can't determine new state - keep old path
            _upsert_metadata_only(db, video_id, domain, old_cache_path, metadata)
        else:
            # Neither has content - just update metadata with new path
            _upsert_with_new_path(
                db, video_id, domain, new_cache_path, channel, playlist, metadata
            )

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

        # Auto-embed the full text (fire-and-forget)
        if full_text:
            embed_transcription(video_uuid, full_text, duration)

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


def record_pipeline_step(
    video_id: str,
    step_type: str,
    status: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    scene_id: int | None = None,
    config: str | None = None,
    error_message: str | None = None,
) -> str | None:
    """Record a pipeline step for a video.

    Looks up the video UUID by natural key and records the step.
    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_id: Natural key (e.g., YouTube video ID).
        step_type: Type of processing step (download, audio_extract, transcribe, etc.).
        status: Current status (pending, running, completed, failed, skipped).
        provider: Optional provider name (e.g., 'whisper', 'yt-dlp').
        model: Optional model name (e.g., 'small', 'tiny').
        scene_id: Optional scene ID if step is per-scene.
        config: Optional JSON config string.
        error_message: Optional error message (for failed status).

    Returns:
        The generated UUID for the pipeline step, or None if sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return None

        # Get video UUID first
        from claudetube.db.repos.videos import VideoRepository

        video_repo = VideoRepository(db)
        existing = video_repo.get_by_video_id(video_id)
        if existing is None:
            logger.debug("Cannot record pipeline step: video %s not found", video_id)
            return None

        video_uuid = existing["id"]

        from claudetube.db.repos.pipeline import PipelineRepository

        pipeline_repo = PipelineRepository(db)
        step_uuid = pipeline_repo.record_step(
            video_uuid=video_uuid,
            step_type=step_type,
            status=status,
            provider=provider,
            model=model,
            scene_id=scene_id,
            config=config,
            error_message=error_message,
        )
        logger.debug(
            "Recorded pipeline step %s for video %s: %s -> %s",
            step_uuid,
            video_id,
            step_type,
            status,
        )
        return step_uuid

    except Exception:
        logger.debug("Failed to record pipeline step", exc_info=True)
        return None


def update_pipeline_step(
    step_uuid: str,
    status: str,
    *,
    error_message: str | None = None,
) -> bool:
    """Update an existing pipeline step.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        step_uuid: UUID of the step to update.
        status: New status value.
        error_message: Optional error message (for failed status).

    Returns:
        True if step was updated, False if not found or sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return False

        from claudetube.db.repos.pipeline import PipelineRepository

        pipeline_repo = PipelineRepository(db)
        result = pipeline_repo.update_step(
            step_uuid=step_uuid,
            status=status,
            error_message=error_message,
        )
        logger.debug("Updated pipeline step %s -> %s", step_uuid, status)
        return result

    except Exception:
        logger.debug("Failed to update pipeline step", exc_info=True)
        return False


def sync_scene(
    video_uuid: str,
    scene_id: int,
    start_time: float,
    end_time: float,
    *,
    title: str | None = None,
    transcript_text: str | None = None,
    method: str | None = None,
    relevance_boost: float = 1.0,
) -> str | None:
    """Sync a scene record to SQLite.

    Creates a new scene record for the given video. The scene's transcript_text
    is indexed by FTS5 for per-scene search.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_uuid: UUID of the parent video record (from videos table).
        scene_id: Sequential scene identifier (0-indexed).
        start_time: Scene start time in seconds.
        end_time: Scene end time in seconds.
        title: Optional scene title (e.g., from chapter).
        transcript_text: Transcript text for this scene segment.
        method: Segmentation method (transcript, visual, hybrid, chapters).
        relevance_boost: Relevance boost multiplier (default: 1.0).

    Returns:
        The generated UUID for the scene, or None if sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return None

        from claudetube.db.repos.scenes import SceneRepository

        repo = SceneRepository(db)

        # Check if scene already exists (by video + scene_id)
        existing = repo.get_scene(video_uuid, scene_id)
        if existing:
            logger.debug(
                "Scene %d already exists for video %s",
                scene_id,
                video_uuid,
            )
            return existing["id"]

        scene_uuid = repo.insert(
            video_uuid=video_uuid,
            scene_id=scene_id,
            start_time=start_time,
            end_time=end_time,
            title=title,
            transcript_text=transcript_text,
            method=method,
            relevance_boost=relevance_boost,
        )
        logger.debug("Synced scene %d for video %s", scene_id, video_uuid)

        # Auto-embed scene transcript (fire-and-forget)
        if transcript_text:
            embed_scene_transcript(
                video_uuid, scene_id, transcript_text, start_time, end_time
            )

        return scene_uuid

    except Exception:
        logger.debug("Failed to sync scene to SQLite", exc_info=True)
        return None


def sync_scenes_bulk(
    video_uuid: str,
    scenes: list[dict[str, Any]],
) -> list[str] | None:
    """Sync multiple scenes to SQLite efficiently.

    Uses bulk insert for better performance when syncing all scenes at once.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_uuid: UUID of the parent video record.
        scenes: List of scene dicts with scene_id, start_time, end_time, etc.

    Returns:
        List of generated UUIDs for the scenes, or None if sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return None

        from claudetube.db.repos.scenes import SceneRepository

        repo = SceneRepository(db)

        # Check if any scenes already exist for this video
        existing_count = repo.count_by_video(video_uuid)
        if existing_count > 0:
            logger.debug(
                "Scenes already exist for video %s, skipping bulk insert",
                video_uuid,
            )
            return None

        uuids = repo.bulk_insert(video_uuid, scenes)
        logger.debug("Synced %d scenes for video %s", len(uuids), video_uuid)

        # Auto-embed each scene's transcript (fire-and-forget)
        for scene in scenes:
            transcript_text = scene.get("transcript_text")
            if transcript_text:
                embed_scene_transcript(
                    video_uuid,
                    scene.get("scene_id", 0),
                    transcript_text,
                    scene.get("start_time"),
                    scene.get("end_time"),
                )

        return uuids

    except Exception:
        logger.debug("Failed to bulk sync scenes to SQLite", exc_info=True)
        return None


def sync_frame(
    video_uuid: str,
    timestamp: float,
    extraction_type: str,
    file_path: str,
    *,
    scene_id: int | None = None,
    quality_tier: str | None = None,
    is_thumbnail: bool = False,
    width: int | None = None,
    height: int | None = None,
    file_size_bytes: int | None = None,
) -> str | None:
    """Sync a frame record to SQLite.

    Creates a new frame record for the given video.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_uuid: UUID of the parent video record (from videos table).
        timestamp: Frame timestamp in seconds.
        extraction_type: Type of extraction (drill, hq, keyframe, thumbnail).
        file_path: Relative path to the frame file in cache.
        scene_id: Scene identifier (NULL for video-level extractions).
        quality_tier: Quality tier (lowest, low, medium, high, highest).
        is_thumbnail: Whether this is the video's thumbnail image.
        width: Frame width in pixels.
        height: Frame height in pixels.
        file_size_bytes: File size in bytes.

    Returns:
        The generated UUID for the frame, or None if sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return None

        from claudetube.db.repos.frames import FrameRepository

        repo = FrameRepository(db)

        frame_uuid = repo.insert(
            video_uuid=video_uuid,
            timestamp=timestamp,
            extraction_type=extraction_type,
            file_path=file_path,
            scene_id=scene_id,
            quality_tier=quality_tier,
            is_thumbnail=is_thumbnail,
            width=width,
            height=height,
            file_size_bytes=file_size_bytes,
        )
        logger.debug(
            "Synced %s frame at %.1fs for video %s",
            extraction_type,
            timestamp,
            video_uuid,
        )
        return frame_uuid

    except Exception:
        logger.debug("Failed to sync frame to SQLite", exc_info=True)
        return None


def sync_visual_description(
    video_uuid: str,
    scene_id: int,
    description: str,
    *,
    provider: str | None = None,
    file_path: str | None = None,
) -> str | None:
    """Sync a visual description record to SQLite.

    Creates a new visual description record for the given video scene.
    The description is indexed by FTS5 for cross-video search.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_uuid: UUID of the parent video record (from videos table).
        scene_id: The scene identifier (0-indexed).
        description: The visual description text.
        provider: Optional provider name (e.g., 'anthropic', 'openai').
        file_path: Optional path to the JSON file.

    Returns:
        The generated UUID for the visual description, or None if sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return None

        from claudetube.db.repos.visual_descriptions import VisualDescriptionRepository

        repo = VisualDescriptionRepository(db)

        # Check if visual description already exists (by video + scene_id)
        existing = repo.get_by_scene(video_uuid, scene_id)
        if existing:
            logger.debug(
                "Visual description already exists for video %s scene %d",
                video_uuid,
                scene_id,
            )
            return existing["id"]

        visual_uuid = repo.insert(
            video_uuid=video_uuid,
            scene_id=scene_id,
            description=description,
            provider=provider,
            file_path=file_path,
        )
        logger.debug(
            "Synced visual description for video %s scene %d",
            video_uuid,
            scene_id,
        )

        # Auto-embed visual description (fire-and-forget)
        # Note: We don't have start/end time here, but scene_id links to scenes table
        embed_visual_description(video_uuid, scene_id, description)

        return visual_uuid

    except Exception:
        logger.debug("Failed to sync visual description to SQLite", exc_info=True)
        return None


def sync_technical_content(
    video_uuid: str,
    scene_id: int,
    has_code: bool,
    has_text: bool,
    *,
    provider: str | None = None,
    ocr_text: str | None = None,
    code_language: str | None = None,
    file_path: str | None = None,
) -> str | None:
    """Sync a technical content record to SQLite.

    Creates a new technical content record for the given video scene.
    The ocr_text is indexed by FTS5 for cross-video search.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_uuid: UUID of the parent video record (from videos table).
        scene_id: The scene identifier (0-indexed).
        has_code: Whether code was detected in the scene.
        has_text: Whether text was detected in the scene.
        provider: Optional provider name (e.g., 'anthropic', 'openai').
        ocr_text: Extracted OCR text for FTS search.
        code_language: Detected programming language.
        file_path: Optional path to the JSON file.

    Returns:
        The generated UUID for the technical content, or None if sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return None

        from claudetube.db.repos.technical_content import TechnicalContentRepository

        repo = TechnicalContentRepository(db)

        # Check if technical content already exists (by video + scene_id)
        existing = repo.get_by_scene(video_uuid, scene_id)
        if existing:
            logger.debug(
                "Technical content already exists for video %s scene %d",
                video_uuid,
                scene_id,
            )
            return existing["id"]

        tech_uuid = repo.insert(
            video_uuid=video_uuid,
            scene_id=scene_id,
            has_code=has_code,
            has_text=has_text,
            provider=provider,
            ocr_text=ocr_text,
            code_language=code_language,
            file_path=file_path,
        )
        logger.debug(
            "Synced technical content for video %s scene %d",
            video_uuid,
            scene_id,
        )

        # Auto-embed OCR text if non-empty (fire-and-forget)
        if ocr_text:
            embed_technical_content(video_uuid, scene_id, ocr_text)

        return tech_uuid

    except Exception:
        logger.debug("Failed to sync technical content to SQLite", exc_info=True)
        return None


def sync_entity(
    name: str,
    entity_type: str,
) -> str | None:
    """Sync an entity record to SQLite.

    Creates a new entity record or returns the existing one. Uses INSERT OR IGNORE
    to handle the UNIQUE(name, entity_type) constraint gracefully.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        name: Entity name (e.g., "Python", "Alice", "machine learning").
        entity_type: One of: object, concept, person, technology, organization.

    Returns:
        The UUID for the entity (existing or newly created), or None if sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return None

        from claudetube.db.repos.entities import EntityRepository

        repo = EntityRepository(db)
        entity_uuid = repo.insert_entity(name, entity_type)
        logger.debug("Synced entity %s (%s)", name, entity_type)
        return entity_uuid

    except Exception:
        logger.debug("Failed to sync entity to SQLite", exc_info=True)
        return None


def sync_entity_appearance(
    entity_uuid: str,
    video_uuid: str,
    scene_id: int,
    timestamp: float,
    *,
    score: float | None = None,
) -> str | None:
    """Sync an entity appearance record to SQLite.

    Records where an entity appears (video, scene, timestamp). Uses INSERT OR IGNORE
    to handle the UNIQUE(entity_id, video_id, scene_id) constraint gracefully.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        entity_uuid: UUID of the entity.
        video_uuid: UUID of the video.
        scene_id: Scene index (0-based).
        timestamp: Timestamp in seconds.
        score: Optional confidence score (0-1).

    Returns:
        The UUID for the appearance, or None if sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return None

        from claudetube.db.repos.entities import EntityRepository

        repo = EntityRepository(db)
        appearance_uuid = repo.insert_appearance(
            entity_uuid=entity_uuid,
            video_uuid=video_uuid,
            scene_id=scene_id,
            timestamp=timestamp,
            score=score,
        )
        logger.debug(
            "Synced entity appearance for entity %s at scene %d",
            entity_uuid,
            scene_id,
        )
        return appearance_uuid

    except Exception:
        logger.debug("Failed to sync entity appearance to SQLite", exc_info=True)
        return None


def sync_entity_video_summary(
    entity_uuid: str,
    video_uuid: str,
    frequency: int = 1,
    *,
    avg_score: float | None = None,
) -> str | None:
    """Sync an entity-video summary record to SQLite.

    Records or updates aggregated stats for an entity in a video. Uses INSERT OR REPLACE
    to upsert the summary record.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        entity_uuid: UUID of the entity.
        video_uuid: UUID of the video.
        frequency: Number of appearances (default: 1).
        avg_score: Optional average confidence score (0-1).

    Returns:
        The UUID for the summary record, or None if sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return None

        from claudetube.db.repos.entities import EntityRepository

        repo = EntityRepository(db)
        summary_uuid = repo.insert_video_summary(
            entity_uuid=entity_uuid,
            video_uuid=video_uuid,
            frequency=frequency,
            avg_score=avg_score,
        )
        logger.debug(
            "Synced entity video summary for entity %s in video %s",
            entity_uuid,
            video_uuid,
        )
        return summary_uuid

    except Exception:
        logger.debug("Failed to sync entity video summary to SQLite", exc_info=True)
        return None


def sync_qa(
    video_uuid: str,
    question: str,
    answer: str,
    *,
    scene_ids: list[int] | None = None,
) -> str | None:
    """Sync a Q&A record to SQLite.

    Creates a new Q&A record for the given video with optional scene associations.
    The question and answer are indexed by FTS5 for cross-video search.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_uuid: UUID of the video this Q&A is about.
        question: The question asked.
        answer: The answer given.
        scene_ids: Optional list of scene IDs this Q&A relates to.

    Returns:
        The UUID for the Q&A record, or None if sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return None

        from claudetube.db.repos.qa import QARepository

        repo = QARepository(db)
        qa_uuid = repo.insert(
            video_uuid=video_uuid,
            question=question,
            answer=answer,
            scene_ids=scene_ids,
        )
        logger.debug(
            "Synced Q&A for video %s with %d scenes",
            video_uuid,
            len(scene_ids) if scene_ids else 0,
        )

        # Auto-embed Q&A (fire-and-forget)
        embed_qa(video_uuid, question, answer)

        return qa_uuid

    except Exception:
        logger.debug("Failed to sync Q&A to SQLite", exc_info=True)
        return None


def sync_observation(
    video_uuid: str,
    scene_id: int,
    obs_type: str,
    content: str,
) -> str | None:
    """Sync an observation record to SQLite.

    Creates a new observation record for the given video scene.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_uuid: UUID of the video this observation is about.
        scene_id: Scene index this observation relates to (0-based).
        obs_type: Type of observation (e.g., "visual", "technical", "note").
        content: The observation content.

    Returns:
        The UUID for the observation, or None if sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return None

        from claudetube.db.repos.observations import ObservationRepository

        repo = ObservationRepository(db)
        obs_uuid = repo.insert(
            video_uuid=video_uuid,
            scene_id=scene_id,
            obs_type=obs_type,
            content=content,
        )
        logger.debug(
            "Synced observation for video %s scene %d",
            video_uuid,
            scene_id,
        )

        # Auto-embed observation content (fire-and-forget)
        embed_observation(video_uuid, scene_id, content)

        return obs_uuid

    except Exception:
        logger.debug("Failed to sync observation to SQLite", exc_info=True)
        return None


def update_scene_relevance_boost(
    video_uuid: str,
    scene_id: int,
    boost: float,
) -> bool:
    """Update the relevance boost for a scene in SQLite.

    This is fire-and-forget: exceptions are caught and logged.

    Args:
        video_uuid: UUID of the parent video.
        scene_id: The scene identifier.
        boost: New relevance boost value (must be >= 0).

    Returns:
        True if the scene was updated, False if not found or sync failed.
    """
    try:
        db = _get_db()
        if db is None:
            return False

        from claudetube.db.repos.scenes import SceneRepository

        repo = SceneRepository(db)
        result = repo.update_relevance_boost(
            video_uuid=video_uuid,
            scene_id=scene_id,
            boost=boost,
        )
        logger.debug(
            "Updated relevance boost for video %s scene %d to %.2f",
            video_uuid,
            scene_id,
            boost,
        )
        return result

    except Exception:
        logger.debug("Failed to update scene relevance boost in SQLite", exc_info=True)
        return False


# ============================================================
# Auto-Embedding Helpers
# ============================================================


def _embed_async_fire_and_forget(
    video_uuid: str,
    scene_id: int | None,
    source: str,
    text: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> None:
    """Fire-and-forget async embedding in a background thread.

    This function spawns embedding in a background thread so it doesn't
    block the main sync path. If embedding fails for any reason (no
    embedder configured, API error, etc.), it's silently ignored.

    The embedding is best-effort on top of best-effort DB sync.
    When embeddings are successfully created, a pipeline step is recorded.

    Args:
        video_uuid: UUID of the video (videos.id).
        scene_id: Scene index, or None for video-level.
        source: Content source type (must be in VALID_SOURCES).
        text: Text to embed.
        start_time: Optional start timestamp.
        end_time: Optional end timestamp.
    """
    import asyncio
    import concurrent.futures

    def _run_embed():
        """Run embedding in a new event loop."""
        try:
            from claudetube.db.vec import embed_text

            result = asyncio.run(
                embed_text(
                    video_uuid=video_uuid,
                    scene_id=scene_id,
                    source=source,
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                )
            )
            logger.debug(
                "Auto-embedded %s for video %s scene %s",
                source,
                video_uuid,
                scene_id,
            )

            # Record pipeline step if embedding succeeded
            if result is not None:
                _record_embed_pipeline_step(
                    video_uuid=video_uuid,
                    scene_id=scene_id,
                    source=source,
                )
        except Exception:
            # Embedding is best-effort, never fail
            logger.debug("Auto-embed failed (ignored)", exc_info=True)

    # Check if text is empty before spawning thread
    if not text or not text.strip():
        return

    try:
        # Use a thread pool to avoid blocking
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_run_embed)
    except Exception:
        # Don't fail if thread pool fails
        logger.debug("Failed to spawn embed thread (ignored)", exc_info=True)


def _record_embed_pipeline_step(
    video_uuid: str,
    scene_id: int | None,
    source: str,
) -> None:
    """Record a pipeline step for successful embedding.

    Fire-and-forget - never raises.

    Args:
        video_uuid: UUID of the video.
        scene_id: Scene index, or None for video-level.
        source: Embedding source type.
    """
    import json

    try:
        db = _get_db()
        if db is None:
            return

        from claudetube.db.repos.pipeline import PipelineRepository

        repo = PipelineRepository(db)
        config = json.dumps({"source": source})
        repo.record_step(
            video_uuid=video_uuid,
            step_type="embed",
            status="completed",
            scene_id=scene_id,
            config=config,
        )
        logger.debug(
            "Recorded embed pipeline step for video %s scene %s source %s",
            video_uuid,
            scene_id,
            source,
        )
    except Exception:
        # Fire-and-forget
        logger.debug("Failed to record embed pipeline step (ignored)", exc_info=True)


def embed_transcription(
    video_uuid: str,
    full_text: str,
    duration: float | None = None,
) -> None:
    """Auto-embed transcription full text after sync.

    Fire-and-forget wrapper for transcription embedding.

    Args:
        video_uuid: UUID of the video.
        full_text: Complete transcript text.
        duration: Optional video duration for end_time.
    """
    _embed_async_fire_and_forget(
        video_uuid=video_uuid,
        scene_id=None,  # Video-level
        source="transcription",
        text=full_text,
        start_time=0.0,
        end_time=duration,
    )


def embed_scene_transcript(
    video_uuid: str,
    scene_id: int,
    transcript_text: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> None:
    """Auto-embed scene transcript text after sync.

    Fire-and-forget wrapper for scene transcript embedding.

    Args:
        video_uuid: UUID of the video.
        scene_id: Scene index (0-based).
        transcript_text: Transcript text for this scene.
        start_time: Scene start time.
        end_time: Scene end time.
    """
    _embed_async_fire_and_forget(
        video_uuid=video_uuid,
        scene_id=scene_id,
        source="scene_transcript",
        text=transcript_text,
        start_time=start_time,
        end_time=end_time,
    )


def embed_visual_description(
    video_uuid: str,
    scene_id: int,
    description: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> None:
    """Auto-embed visual description after sync.

    Fire-and-forget wrapper for visual description embedding.

    Args:
        video_uuid: UUID of the video.
        scene_id: Scene index (0-based).
        description: Visual description text.
        start_time: Scene start time.
        end_time: Scene end time.
    """
    _embed_async_fire_and_forget(
        video_uuid=video_uuid,
        scene_id=scene_id,
        source="visual",
        text=description,
        start_time=start_time,
        end_time=end_time,
    )


def embed_technical_content(
    video_uuid: str,
    scene_id: int,
    ocr_text: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> None:
    """Auto-embed OCR text from technical content after sync.

    Fire-and-forget wrapper for technical content embedding.

    Args:
        video_uuid: UUID of the video.
        scene_id: Scene index (0-based).
        ocr_text: Extracted OCR text.
        start_time: Scene start time.
        end_time: Scene end time.
    """
    _embed_async_fire_and_forget(
        video_uuid=video_uuid,
        scene_id=scene_id,
        source="technical",
        text=ocr_text,
        start_time=start_time,
        end_time=end_time,
    )


def embed_qa(
    video_uuid: str,
    question: str,
    answer: str,
) -> None:
    """Auto-embed Q&A pair after sync.

    Combines question and answer into a single text for embedding.
    Fire-and-forget wrapper.

    Args:
        video_uuid: UUID of the video.
        question: The question asked.
        answer: The answer given.
    """
    combined = f"Q: {question}\nA: {answer}"
    _embed_async_fire_and_forget(
        video_uuid=video_uuid,
        scene_id=None,  # Q&A is video-level
        source="qa",
        text=combined,
    )


def embed_observation(
    video_uuid: str,
    scene_id: int,
    content: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> None:
    """Auto-embed observation content after sync.

    Fire-and-forget wrapper for observation embedding.

    Args:
        video_uuid: UUID of the video.
        scene_id: Scene index (0-based).
        content: Observation content text.
        start_time: Scene start time.
        end_time: Scene end time.
    """
    _embed_async_fire_and_forget(
        video_uuid=video_uuid,
        scene_id=scene_id,
        source="observation",
        text=content,
        start_time=start_time,
        end_time=end_time,
    )
