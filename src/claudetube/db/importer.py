"""Auto-import module for populating SQLite from existing JSON caches.

This module scans the video cache directory and imports all existing
video data into SQLite when the database is first created. This ensures
the full video library is queryable immediately without re-processing.

The import is idempotent: running twice won't create duplicates because
video_id is UNIQUE and most tables use INSERT OR IGNORE patterns.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claudetube.db.connection import Database

logger = logging.getLogger(__name__)


def auto_import(cache_base: Path, db: Database) -> int:
    """Scan all cached videos and populate SQLite database.

    This function discovers video directories in the cache, reads their
    JSON state files, and inserts the data into the database. It handles
    both flat (video_id/) and hierarchical (domain/channel/playlist/video_id/)
    directory structures.

    Args:
        cache_base: Base cache directory (e.g., ~/.claude/video_cache/).
        db: Database instance to populate.

    Returns:
        Number of videos imported.
    """
    if not cache_base.exists():
        logger.info("Cache directory does not exist: %s", cache_base)
        return 0

    # Discover all video directories (those with state.json)
    video_dirs = list(_discover_video_dirs(cache_base))

    if not video_dirs:
        logger.info("No cached videos found in %s", cache_base)
        return 0

    logger.info("Found %d cached videos to import", len(video_dirs))

    imported = 0
    for video_dir in video_dirs:
        try:
            if _import_video(video_dir, cache_base, db):
                imported += 1
        except Exception:
            # Log and continue - don't let one bad video stop the import
            logger.warning(
                "Failed to import video from %s",
                video_dir,
                exc_info=True,
            )

    logger.info("Imported %d/%d videos into SQLite", imported, len(video_dirs))
    return imported


def _discover_video_dirs(cache_base: Path) -> list[Path]:
    """Discover all video directories in the cache.

    Supports both flat (video_id/) and hierarchical paths by looking
    for state.json files at any depth under cache_base.

    Args:
        cache_base: Base cache directory.

    Yields:
        Paths to video directories (parent of state.json).
    """
    # Use glob to find all state.json files
    for state_file in cache_base.rglob("state.json"):
        # Skip if parent is cache_base (shouldn't happen, but safety check)
        if state_file.parent == cache_base:
            continue
        # Skip playlists directory
        if "playlists" in state_file.parts:
            continue
        yield state_file.parent


def _import_video(video_dir: Path, cache_base: Path, db: Database) -> bool:
    """Import a single video from its cache directory.

    Args:
        video_dir: Path to the video's cache directory.
        cache_base: Base cache directory for computing relative paths.
        db: Database instance.

    Returns:
        True if video was imported successfully.
    """
    state_file = video_dir / "state.json"
    if not state_file.exists():
        logger.debug("No state.json in %s, skipping", video_dir)
        return False

    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("Failed to read state.json in %s: %s", video_dir, e)
        return False

    video_id = state.get("video_id")
    if not video_id:
        logger.warning("No video_id in state.json at %s", video_dir)
        return False

    # Compute relative cache path
    try:
        cache_path = str(video_dir.relative_to(cache_base))
    except ValueError:
        # video_dir is not under cache_base (shouldn't happen)
        cache_path = video_dir.name

    # Import the video record and get its UUID
    video_uuid = _import_video_record(state, cache_path, db)
    if not video_uuid:
        return False

    # Import child data
    _import_audio_track(video_dir, cache_path, video_uuid, state, db)
    _import_transcription(video_dir, cache_path, video_uuid, state, db)
    _import_thumbnail(video_dir, cache_path, video_uuid, db)
    _import_scenes(video_dir, cache_path, video_uuid, db)
    _import_frames(video_dir, cache_path, video_uuid, db)
    _import_enrichment(video_dir, video_uuid, db)
    _import_entities(video_dir, video_uuid, db)
    _import_narrative(video_dir, cache_path, video_uuid, db)
    _import_code_evolution(video_dir, cache_path, video_uuid, db)

    return True


def _import_video_record(
    state: dict[str, Any],
    cache_path: str,
    db: Database,
) -> str | None:
    """Import the video record from state.json.

    Args:
        state: Parsed state.json data.
        cache_path: Relative path to video cache directory.
        db: Database instance.

    Returns:
        The video UUID, or None if import failed.
    """
    from claudetube.db.repos.videos import VideoRepository

    repo = VideoRepository(db)

    video_id = state["video_id"]

    # Check if already exists
    existing = repo.get_by_video_id(video_id)
    if existing:
        logger.debug("Video %s already in database", video_id)
        return existing["id"]

    # Determine domain
    domain = state.get("domain")
    if not domain:
        # Try to extract from cache_path (hierarchical: domain/channel/playlist/video_id)
        # For flat paths, assume youtube for backward compatibility
        parts = Path(cache_path).parts
        domain = parts[0] if len(parts) > 1 else "youtube"

    # Build metadata
    try:
        video_uuid = repo.insert(
            video_id=video_id,
            domain=domain,
            cache_path=cache_path,
            channel=state.get("channel_id"),
            playlist=state.get("playlist_id"),
            url=state.get("url"),
            title=state.get("title"),
            duration=state.get("duration"),
            duration_string=state.get("duration_string"),
            uploader=state.get("uploader"),
            channel_name=state.get("channel"),
            upload_date=state.get("upload_date"),
            description=_truncate(state.get("description"), 1500),
            language=state.get("language"),
            view_count=state.get("view_count"),
            like_count=state.get("like_count"),
            source_type=state.get("source_type", "url"),
        )
        logger.debug("Imported video %s -> %s", video_id, video_uuid)
        return video_uuid
    except Exception as e:
        logger.warning("Failed to insert video %s: %s", video_id, e)
        return None


def _import_audio_track(
    video_dir: Path,
    cache_path: str,
    video_uuid: str,
    state: dict[str, Any],
    db: Database,
) -> None:
    """Import audio track from audio files on disk.

    Args:
        video_dir: Path to video cache directory.
        cache_path: Relative cache path.
        video_uuid: UUID of the parent video.
        state: Parsed state.json data.
        db: Database instance.
    """
    from claudetube.db.repos.audio_tracks import AudioTrackRepository

    repo = AudioTrackRepository(db)

    # Look for audio files
    audio_extensions = ["mp3", "wav", "aac", "m4a", "opus", "flac", "ogg"]
    for ext in audio_extensions:
        audio_file = video_dir / f"audio.{ext}"
        if audio_file.exists():
            # Check if already exists
            existing = repo.get_by_video_and_format(video_uuid, ext)
            if existing:
                logger.debug("Audio track already exists for %s", video_uuid)
                return

            try:
                file_size = audio_file.stat().st_size
                repo.insert(
                    video_uuid=video_uuid,
                    format_=ext,
                    file_path=f"{cache_path}/audio.{ext}",
                    duration=state.get("duration"),
                    file_size_bytes=file_size,
                )
                logger.debug("Imported audio track for video %s", video_uuid)
                return
            except Exception as e:
                logger.debug("Failed to insert audio track: %s", e)
                return


def _import_transcription(
    video_dir: Path,
    cache_path: str,
    video_uuid: str,
    state: dict[str, Any],
    db: Database,
) -> None:
    """Import transcription from audio.txt/audio.srt files.

    Args:
        video_dir: Path to video cache directory.
        cache_path: Relative cache path.
        video_uuid: UUID of the parent video.
        state: Parsed state.json data.
        db: Database instance.
    """
    from claudetube.db.repos.transcriptions import TranscriptionRepository

    repo = TranscriptionRepository(db)

    # Check if already has a transcription
    existing = repo.get_by_video(video_uuid)
    if existing:
        logger.debug("Transcription already exists for %s", video_uuid)
        return

    # Determine provider from state
    transcript_source = state.get("transcript_source", "")
    if transcript_source == "whisper":
        provider = "whisper"
    elif transcript_source in ("youtube_subtitles", "auto-generated", "uploaded"):
        provider = "youtube_subtitles"
    else:
        provider = "manual"

    # Read the full text from audio.txt
    full_text = None
    txt_file = video_dir / "audio.txt"
    if txt_file.exists():
        with contextlib.suppress(Exception):
            full_text = txt_file.read_text(encoding="utf-8")

    # Prefer SRT file for the record
    srt_file = video_dir / "audio.srt"
    if srt_file.exists():
        try:
            file_size = srt_file.stat().st_size
            word_count = len(full_text.split()) if full_text else None
            repo.insert(
                video_uuid=video_uuid,
                provider=provider,
                format_="srt",
                file_path=f"{cache_path}/audio.srt",
                model=state.get("whisper_model"),
                language=state.get("language"),
                full_text=full_text,
                word_count=word_count,
                duration=state.get("duration"),
                file_size_bytes=file_size,
                is_primary=True,
            )
            logger.debug("Imported transcription for video %s", video_uuid)
            return
        except Exception as e:
            logger.debug("Failed to insert transcription: %s", e)

    # Fall back to TXT file
    if txt_file.exists() and full_text:
        try:
            file_size = txt_file.stat().st_size
            word_count = len(full_text.split())
            repo.insert(
                video_uuid=video_uuid,
                provider=provider,
                format_="txt",
                file_path=f"{cache_path}/audio.txt",
                model=state.get("whisper_model"),
                language=state.get("language"),
                full_text=full_text,
                word_count=word_count,
                duration=state.get("duration"),
                file_size_bytes=file_size,
                is_primary=True,
            )
            logger.debug("Imported transcription (txt) for video %s", video_uuid)
        except Exception as e:
            logger.debug("Failed to insert transcription (txt): %s", e)


def _import_thumbnail(
    video_dir: Path,
    cache_path: str,
    video_uuid: str,
    db: Database,
) -> None:
    """Import thumbnail image as a frame record.

    Args:
        video_dir: Path to video cache directory.
        cache_path: Relative cache path.
        video_uuid: UUID of the parent video.
        db: Database instance.
    """
    from claudetube.db.repos.frames import FrameRepository

    repo = FrameRepository(db)

    # Check if thumbnail already exists
    existing = repo.get_thumbnail(video_uuid)
    if existing:
        logger.debug("Thumbnail already exists for %s", video_uuid)
        return

    # Look for thumbnail file
    for ext in ["jpg", "jpeg", "png", "webp"]:
        thumb_file = video_dir / f"thumbnail.{ext}"
        if thumb_file.exists():
            try:
                file_size = thumb_file.stat().st_size
                repo.insert(
                    video_uuid=video_uuid,
                    timestamp=0.0,
                    extraction_type="thumbnail",
                    file_path=f"{cache_path}/thumbnail.{ext}",
                    is_thumbnail=True,
                    file_size_bytes=file_size,
                )
                logger.debug("Imported thumbnail for video %s", video_uuid)
                return
            except Exception as e:
                logger.debug("Failed to insert thumbnail: %s", e)
                return


def _import_scenes(
    video_dir: Path,
    cache_path: str,
    video_uuid: str,
    db: Database,
) -> None:
    """Import scenes from scenes/scenes.json.

    Args:
        video_dir: Path to video cache directory.
        cache_path: Relative cache path.
        video_uuid: UUID of the parent video.
        db: Database instance.
    """
    from claudetube.db.repos.scenes import SceneRepository
    from claudetube.db.repos.technical_content import TechnicalContentRepository
    from claudetube.db.repos.visual_descriptions import VisualDescriptionRepository

    scenes_file = video_dir / "scenes" / "scenes.json"
    if not scenes_file.exists():
        return

    try:
        scenes_data = json.loads(scenes_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.debug("Failed to read scenes.json for %s", video_uuid)
        return

    scene_repo = SceneRepository(db)

    # Check if scenes already exist
    if scene_repo.count_by_video(video_uuid) > 0:
        logger.debug("Scenes already exist for %s", video_uuid)
        return

    # scenes_data is a list of scene dicts
    scenes_list = scenes_data if isinstance(scenes_data, list) else scenes_data.get("scenes", [])

    for i, scene in enumerate(scenes_list):
        try:
            scene_id = scene.get("scene_id", i)
            start_time = scene.get("start_time", scene.get("start", 0))
            end_time = scene.get("end_time", scene.get("end", start_time + 1))

            # Ensure valid times
            if start_time >= end_time:
                continue

            scene_repo.insert(
                video_uuid=video_uuid,
                scene_id=scene_id,
                start_time=start_time,
                end_time=end_time,
                title=scene.get("title"),
                transcript_text=scene.get("transcript_text", scene.get("transcript")),
                method=_normalize_method(scene.get("method")),
            )
        except Exception as e:
            logger.debug("Failed to insert scene %d: %s", i, e)

    logger.debug("Imported %d scenes for video %s", len(scenes_list), video_uuid)

    # Import per-scene data (visual.json, technical.json, entities.json)
    scenes_dir = video_dir / "scenes"
    visual_repo = VisualDescriptionRepository(db)
    tech_repo = TechnicalContentRepository(db)

    for scene_subdir in sorted(scenes_dir.iterdir()):
        if not scene_subdir.is_dir() or not scene_subdir.name.startswith("scene_"):
            continue

        try:
            scene_id = int(scene_subdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        # Import visual.json
        visual_file = scene_subdir / "visual.json"
        if visual_file.exists():
            _import_visual_description(
                visual_file, cache_path, video_uuid, scene_id, visual_repo
            )

        # Import technical.json
        tech_file = scene_subdir / "technical.json"
        if tech_file.exists():
            _import_technical_content(
                tech_file, cache_path, video_uuid, scene_id, tech_repo
            )

        # Import keyframes
        _import_scene_keyframes(
            scene_subdir, cache_path, video_uuid, scene_id, db
        )


def _import_visual_description(
    visual_file: Path,
    cache_path: str,
    video_uuid: str,
    scene_id: int,
    repo: Any,
) -> None:
    """Import visual description from visual.json."""
    try:
        data = json.loads(visual_file.read_text(encoding="utf-8"))
        description = data.get("description", "")
        if not description:
            return

        # Check if exists
        existing = repo.get_by_scene(video_uuid, scene_id)
        if existing:
            return

        # Determine relative file path
        file_path = f"{cache_path}/scenes/scene_{scene_id:03d}/visual.json"

        repo.insert(
            video_uuid=video_uuid,
            scene_id=scene_id,
            description=description,
            provider=data.get("provider"),
            file_path=file_path,
        )
    except Exception as e:
        logger.debug("Failed to import visual description: %s", e)


def _import_technical_content(
    tech_file: Path,
    cache_path: str,
    video_uuid: str,
    scene_id: int,
    repo: Any,
) -> None:
    """Import technical content from technical.json."""
    try:
        data = json.loads(tech_file.read_text(encoding="utf-8"))

        # Check if exists
        existing = repo.get_by_scene(video_uuid, scene_id)
        if existing:
            return

        file_path = f"{cache_path}/scenes/scene_{scene_id:03d}/technical.json"

        repo.insert(
            video_uuid=video_uuid,
            scene_id=scene_id,
            has_code=data.get("has_code", False),
            has_text=data.get("has_text", False),
            provider=data.get("provider"),
            ocr_text=data.get("ocr_text", data.get("text")),
            code_language=data.get("code_language", data.get("language")),
            file_path=file_path,
        )
    except Exception as e:
        logger.debug("Failed to import technical content: %s", e)


def _import_scene_keyframes(
    scene_dir: Path,
    cache_path: str,
    video_uuid: str,
    scene_id: int,
    db: Database,
) -> None:
    """Import keyframes from a scene's keyframes/ directory."""
    from claudetube.db.repos.frames import FrameRepository

    keyframes_dir = scene_dir / "keyframes"
    if not keyframes_dir.exists():
        return

    repo = FrameRepository(db)

    for frame_file in keyframes_dir.glob("*.jpg"):
        try:
            # Extract timestamp from filename (e.g., frame_123.5.jpg)
            name = frame_file.stem
            if name.startswith("frame_"):
                timestamp = float(name.replace("frame_", ""))
            else:
                # Try to parse the whole name as a number
                try:
                    timestamp = float(name)
                except ValueError:
                    timestamp = 0.0

            file_path = f"{cache_path}/scenes/scene_{scene_id:03d}/keyframes/{frame_file.name}"
            file_size = frame_file.stat().st_size

            repo.insert(
                video_uuid=video_uuid,
                timestamp=timestamp,
                extraction_type="keyframe",
                file_path=file_path,
                scene_id=scene_id,
                file_size_bytes=file_size,
            )
        except Exception as e:
            logger.debug("Failed to import keyframe %s: %s", frame_file, e)


def _import_frames(
    video_dir: Path,
    cache_path: str,
    video_uuid: str,
    db: Database,
) -> None:
    """Import frames from drill/ and hq/ directories.

    Args:
        video_dir: Path to video cache directory.
        cache_path: Relative cache path.
        video_uuid: UUID of the parent video.
        db: Database instance.
    """
    from claudetube.db.repos.frames import FrameRepository

    repo = FrameRepository(db)

    # Import drill frames
    for drill_dir in video_dir.glob("drill*"):
        if not drill_dir.is_dir():
            continue

        # Extract quality tier from directory name (e.g., drill_lowest)
        quality_tier = None
        if "_" in drill_dir.name:
            tier = drill_dir.name.split("_")[1]
            if tier in ["lowest", "low", "medium", "high", "highest"]:
                quality_tier = tier

        for frame_file in drill_dir.glob("*.jpg"):
            try:
                timestamp = _extract_timestamp_from_filename(frame_file.stem)
                file_path = f"{cache_path}/{drill_dir.name}/{frame_file.name}"
                file_size = frame_file.stat().st_size

                repo.insert(
                    video_uuid=video_uuid,
                    timestamp=timestamp,
                    extraction_type="drill",
                    file_path=file_path,
                    quality_tier=quality_tier,
                    file_size_bytes=file_size,
                )
            except Exception:
                pass

    # Import hq frames
    hq_dir = video_dir / "hq"
    if hq_dir.is_dir():
        for frame_file in hq_dir.glob("*.jpg"):
            try:
                timestamp = _extract_timestamp_from_filename(frame_file.stem)
                file_path = f"{cache_path}/hq/{frame_file.name}"
                file_size = frame_file.stat().st_size

                repo.insert(
                    video_uuid=video_uuid,
                    timestamp=timestamp,
                    extraction_type="hq",
                    file_path=file_path,
                    file_size_bytes=file_size,
                )
            except Exception:
                pass


def _import_enrichment(
    video_dir: Path,
    video_uuid: str,
    db: Database,
) -> None:
    """Import Q&A history and observations from enrichment/ directory.

    Args:
        video_dir: Path to video cache directory.
        video_uuid: UUID of the parent video.
        db: Database instance.
    """
    enrichment_dir = video_dir / "enrichment"
    if not enrichment_dir.exists():
        return

    # Import Q&A history
    qa_file = enrichment_dir / "qa.json"
    if qa_file.exists():
        _import_qa_history(qa_file, video_uuid, db)

    # Import observations
    obs_file = enrichment_dir / "observations.json"
    if obs_file.exists():
        _import_observations(obs_file, video_uuid, db)


def _import_qa_history(qa_file: Path, video_uuid: str, db: Database) -> None:
    """Import Q&A records from qa.json."""
    from claudetube.db.repos.qa import QARepository

    try:
        data = json.loads(qa_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    repo = QARepository(db)

    # Check if Q&A already exists
    if repo.count_by_video(video_uuid) > 0:
        logger.debug("Q&A history already exists for %s", video_uuid)
        return

    # data can be a list or dict with "qa_pairs" key
    qa_list = data if isinstance(data, list) else data.get("qa_pairs", [])

    for qa in qa_list:
        try:
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            if not question or not answer:
                continue

            scene_ids = qa.get("scene_ids", qa.get("scenes"))
            repo.insert(
                video_uuid=video_uuid,
                question=question,
                answer=answer,
                scene_ids=scene_ids,
            )
        except Exception as e:
            logger.debug("Failed to import Q&A: %s", e)

    logger.debug("Imported Q&A history for video %s", video_uuid)


def _import_observations(obs_file: Path, video_uuid: str, db: Database) -> None:
    """Import observations from observations.json."""
    from claudetube.db.repos.observations import ObservationRepository

    try:
        data = json.loads(obs_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    repo = ObservationRepository(db)

    # Check if observations already exist
    if repo.count_by_video(video_uuid) > 0:
        logger.debug("Observations already exist for %s", video_uuid)
        return

    # data can be a list or dict with "observations" key
    obs_list = data if isinstance(data, list) else data.get("observations", [])

    for obs in obs_list:
        try:
            scene_id = obs.get("scene_id", 0)
            obs_type = obs.get("type", "note")
            content = obs.get("content", "")
            if not content:
                continue

            repo.insert(
                video_uuid=video_uuid,
                scene_id=scene_id,
                obs_type=obs_type,
                content=content,
            )
        except Exception as e:
            logger.debug("Failed to import observation: %s", e)

    logger.debug("Imported observations for video %s", video_uuid)


def _import_entities(
    video_dir: Path,
    video_uuid: str,
    db: Database,
) -> None:
    """Import entities from entities/ directory.

    Reads:
    - concepts.json: Global concepts for the video
    - scenes/scene_NNN/entities.json: Per-scene entity appearances

    Args:
        video_dir: Path to video cache directory.
        video_uuid: UUID of the parent video.
        db: Database instance.
    """
    from claudetube.db.repos.entities import EntityRepository

    repo = EntityRepository(db)
    entities_dir = video_dir / "entities"

    # Import concepts.json (video-level entities)
    concepts_file = entities_dir / "concepts.json"
    if concepts_file.exists():
        _import_concepts(concepts_file, video_uuid, repo)

    # Import graph.json (entity-video summary)
    graph_file = entities_dir / "graph.json"
    if graph_file.exists():
        _import_entity_graph(graph_file, video_uuid, repo)

    # Import per-scene entities.json files
    scenes_dir = video_dir / "scenes"
    if scenes_dir.exists():
        for scene_subdir in sorted(scenes_dir.iterdir()):
            if not scene_subdir.is_dir() or not scene_subdir.name.startswith("scene_"):
                continue

            try:
                scene_id = int(scene_subdir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            entities_file = scene_subdir / "entities.json"
            if entities_file.exists():
                _import_scene_entities(entities_file, video_uuid, scene_id, repo)


def _import_concepts(concepts_file: Path, video_uuid: str, repo: Any) -> None:
    """Import concepts from concepts.json."""
    try:
        data = json.loads(concepts_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    # data can be a list of concepts or dict with "concepts" key
    concepts = data if isinstance(data, list) else data.get("concepts", [])

    for concept in concepts:
        try:
            if isinstance(concept, str):
                name = concept
                entity_type = "concept"
            else:
                name = concept.get("name", concept.get("term", ""))
                entity_type = concept.get("type", "concept")

            if not name:
                continue

            # Normalize entity type
            entity_type = _normalize_entity_type(entity_type)

            # Insert entity (upsert pattern)
            entity_uuid = repo.insert_entity(name, entity_type)

            # Insert video summary
            frequency = concept.get("frequency", 1) if isinstance(concept, dict) else 1
            repo.insert_video_summary(
                entity_uuid=entity_uuid,
                video_uuid=video_uuid,
                frequency=frequency,
            )
        except Exception as e:
            logger.debug("Failed to import concept: %s", e)


def _import_entity_graph(graph_file: Path, video_uuid: str, repo: Any) -> None:
    """Import entity-video summary from graph.json."""
    try:
        data = json.loads(graph_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    # data should have "entities" key with entity summaries
    entities = data.get("entities", [])

    for entity in entities:
        try:
            name = entity.get("name", "")
            entity_type = _normalize_entity_type(entity.get("type", "concept"))
            frequency = entity.get("frequency", entity.get("count", 1))

            if not name:
                continue

            entity_uuid = repo.insert_entity(name, entity_type)
            repo.insert_video_summary(
                entity_uuid=entity_uuid,
                video_uuid=video_uuid,
                frequency=frequency,
                avg_score=entity.get("avg_score"),
            )
        except Exception as e:
            logger.debug("Failed to import entity from graph: %s", e)


def _import_scene_entities(
    entities_file: Path,
    video_uuid: str,
    scene_id: int,
    repo: Any,
) -> None:
    """Import entities from a scene's entities.json."""
    try:
        data = json.loads(entities_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    # data can be a list or dict with "entities" key
    entities = data if isinstance(data, list) else data.get("entities", [])

    for entity in entities:
        try:
            if isinstance(entity, str):
                name = entity
                entity_type = "concept"
                timestamp = 0.0
                score = None
            else:
                name = entity.get("name", entity.get("term", ""))
                entity_type = entity.get("type", "concept")
                timestamp = entity.get("timestamp", 0.0)
                score = entity.get("score", entity.get("confidence"))

            if not name:
                continue

            entity_type = _normalize_entity_type(entity_type)

            entity_uuid = repo.insert_entity(name, entity_type)
            repo.insert_appearance(
                entity_uuid=entity_uuid,
                video_uuid=video_uuid,
                scene_id=scene_id,
                timestamp=timestamp,
                score=score,
            )
        except Exception as e:
            logger.debug("Failed to import scene entity: %s", e)


def _import_narrative(
    video_dir: Path,
    cache_path: str,
    video_uuid: str,
    db: Database,
) -> None:
    """Import narrative structure from structure/narrative.json.

    Args:
        video_dir: Path to video cache directory.
        cache_path: Relative cache path.
        video_uuid: UUID of the parent video.
        db: Database instance.
    """
    from claudetube.db.repos.narrative import NarrativeRepository

    narrative_file = video_dir / "structure" / "narrative.json"
    if not narrative_file.exists():
        return

    repo = NarrativeRepository(db)

    # Check if already exists
    if repo.exists(video_uuid):
        logger.debug("Narrative structure already exists for %s", video_uuid)
        return

    try:
        data = json.loads(narrative_file.read_text(encoding="utf-8"))

        video_type = data.get("video_type")
        if video_type and video_type not in repo.VALID_VIDEO_TYPES:
            video_type = "other"

        section_count = data.get("section_count")
        if section_count is None:
            sections = data.get("sections", [])
            section_count = len(sections) if sections else None

        file_path = f"{cache_path}/structure/narrative.json"

        repo.insert(
            video_uuid=video_uuid,
            video_type=video_type,
            section_count=section_count,
            file_path=file_path,
        )
        logger.debug("Imported narrative structure for video %s", video_uuid)
    except Exception as e:
        logger.debug("Failed to import narrative structure: %s", e)


def _import_code_evolution(
    video_dir: Path,
    cache_path: str,
    video_uuid: str,
    db: Database,
) -> None:
    """Import code evolution from entities/code_evolution.json.

    Args:
        video_dir: Path to video cache directory.
        cache_path: Relative cache path.
        video_uuid: UUID of the parent video.
        db: Database instance.
    """
    from claudetube.db.repos.code_evolution import CodeEvolutionRepository

    code_file = video_dir / "entities" / "code_evolution.json"
    if not code_file.exists():
        return

    repo = CodeEvolutionRepository(db)

    # Check if already exists
    if repo.exists(video_uuid):
        logger.debug("Code evolution already exists for %s", video_uuid)
        return

    try:
        data = json.loads(code_file.read_text(encoding="utf-8"))

        files_tracked = data.get("files_tracked")
        if files_tracked is None:
            files = data.get("files", [])
            files_tracked = len(files) if files else None

        total_changes = data.get("total_changes")
        if total_changes is None:
            changes = data.get("changes", [])
            total_changes = len(changes) if changes else None

        file_path = f"{cache_path}/entities/code_evolution.json"

        repo.insert(
            video_uuid=video_uuid,
            files_tracked=files_tracked,
            total_changes=total_changes,
            file_path=file_path,
        )
        logger.debug("Imported code evolution for video %s", video_uuid)
    except Exception as e:
        logger.debug("Failed to import code evolution: %s", e)


# Helper functions


def _truncate(s: str | None, max_len: int) -> str | None:
    """Truncate a string to max_len characters."""
    if s is None:
        return None
    return s[:max_len] if len(s) > max_len else s


def _normalize_method(method: str | None) -> str | None:
    """Normalize scene segmentation method to valid values."""
    if method is None:
        return None

    valid = {"transcript", "visual", "hybrid", "chapters"}
    method_lower = method.lower()

    if method_lower in valid:
        return method_lower

    # Map common variations
    if "chapter" in method_lower:
        return "chapters"
    if "visual" in method_lower or "video" in method_lower:
        return "visual"
    if "transcript" in method_lower or "text" in method_lower:
        return "transcript"

    return None


def _normalize_entity_type(entity_type: str) -> str:
    """Normalize entity type to valid values."""
    valid = {"object", "concept", "person", "technology", "organization"}
    entity_type_lower = entity_type.lower()

    if entity_type_lower in valid:
        return entity_type_lower

    # Map common variations
    type_map = {
        "people": "person",
        "tech": "technology",
        "org": "organization",
        "company": "organization",
        "tool": "technology",
        "library": "technology",
        "framework": "technology",
        "thing": "object",
        "item": "object",
        "topic": "concept",
        "term": "concept",
        "keyword": "concept",
    }

    return type_map.get(entity_type_lower, "concept")


def _extract_timestamp_from_filename(stem: str) -> float:
    """Extract timestamp from a frame filename.

    Handles formats like:
    - "frame_123.5" -> 123.5
    - "123.5" -> 123.5
    - "frame_0001" -> 1.0
    """
    import re

    # Remove "frame_" prefix if present
    clean = stem.replace("frame_", "")

    # Try to parse as float
    try:
        return float(clean)
    except ValueError:
        pass

    # Try to extract number from string
    match = re.search(r"(\d+(?:\.\d+)?)", clean)
    if match:
        return float(match.group(1))

    return 0.0
