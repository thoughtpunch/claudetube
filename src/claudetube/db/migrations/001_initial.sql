-- 001_initial.sql
-- Initial schema: videos, audio_tracks, transcriptions, scenes, frames (incl.
-- thumbnails), visual_descriptions, technical_content, audio_descriptions,
-- narrative_structures, code_evolutions, entities, qa, observations, playlists,
-- pipeline_steps, FTS5, video_processing_status view.
--
-- Note: vec_metadata and vec_embeddings are in the separate vectors database
-- (claudetube-vectors.db) to allow this database to be opened with standard
-- SQLite tools without requiring the sqlite-vec extension.

-- ============================================================
-- VIDEOS (metadata only -- processing state in pipeline_steps)
-- ============================================================
CREATE TABLE videos (
    id                  TEXT PRIMARY KEY CHECK(length(id) = 36),  -- UUID
    video_id            TEXT NOT NULL UNIQUE CHECK(length(video_id) > 0),
    -- Path components (hierarchical structure, progressively enriched)
    domain              TEXT NOT NULL CHECK(domain GLOB '[a-z]*'),  -- always required
    channel             TEXT CHECK(channel IS NULL OR length(channel) > 0),  -- NULL = unknown
    playlist            TEXT CHECK(playlist IS NULL OR length(playlist) > 0),  -- NULL = unknown
    cache_path          TEXT NOT NULL CHECK(length(cache_path) > 0),
    -- Metadata (enriched via UPSERT when yt-dlp returns data)
    url                 TEXT,
    title               TEXT,
    duration            REAL CHECK(duration IS NULL OR duration >= 0),
    duration_string     TEXT,
    uploader            TEXT,
    channel_name        TEXT,  -- display name (vs channel which is the path-safe ID)
    upload_date         TEXT,
    description         TEXT,
    language            TEXT,
    view_count          INTEGER CHECK(view_count IS NULL OR view_count >= 0),
    like_count          INTEGER CHECK(like_count IS NULL OR like_count >= 0),
    source_type         TEXT NOT NULL CHECK(source_type IN ('url', 'local')) DEFAULT 'url',
    -- Timestamps
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_videos_video_id ON videos(video_id);
CREATE INDEX idx_videos_domain ON videos(domain);
CREATE INDEX idx_videos_channel ON videos(channel);
CREATE INDEX idx_videos_playlist ON videos(playlist);
CREATE INDEX idx_videos_domain_channel ON videos(domain, channel);

-- Video tags (many-to-many)
CREATE TABLE video_tags (
    id       TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    tag      TEXT NOT NULL CHECK(length(tag) > 0),
    UNIQUE(video_id, tag)
);
CREATE INDEX idx_tags_video ON video_tags(video_id);
CREATE INDEX idx_tags_tag ON video_tags(tag);

-- ============================================================
-- AUDIO TRACKS (extracted audio files)
-- ============================================================
CREATE TABLE audio_tracks (
    id              TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id        TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    format          TEXT NOT NULL CHECK(format IN ('mp3', 'wav', 'aac', 'm4a', 'opus', 'flac', 'ogg')),
    sample_rate     INTEGER CHECK(sample_rate IS NULL OR sample_rate > 0),
    channels        INTEGER CHECK(channels IS NULL OR (channels > 0 AND channels <= 16)),
    bitrate_kbps    INTEGER CHECK(bitrate_kbps IS NULL OR bitrate_kbps > 0),
    duration        REAL CHECK(duration IS NULL OR duration >= 0),
    file_size_bytes INTEGER CHECK(file_size_bytes IS NULL OR file_size_bytes > 0),
    file_path       TEXT NOT NULL CHECK(length(file_path) > 0),  -- relative to cache dir
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_audio_video ON audio_tracks(video_id);

-- ============================================================
-- TRANSCRIPTIONS (first-class entity for FTS/RAG)
--
-- A video can have MULTIPLE transcriptions (different providers,
-- languages, models). One is marked is_primary for default use.
-- full_text is indexed via FTS5 for cross-video transcript search.
-- ============================================================
CREATE TABLE transcriptions (
    id              TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id        TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    audio_track_id  TEXT REFERENCES audio_tracks(id) ON DELETE SET NULL,
    provider        TEXT NOT NULL CHECK(provider IN (
        'youtube_subtitles', 'whisper', 'deepgram', 'openai', 'manual'
    )),
    model           TEXT CHECK(model IS NULL OR length(model) > 0),
    language        TEXT CHECK(language IS NULL OR length(language) > 0),
    format          TEXT NOT NULL CHECK(format IN ('srt', 'txt', 'vtt')),
    full_text       TEXT,  -- complete transcript text for FTS/RAG
    word_count      INTEGER CHECK(word_count IS NULL OR word_count >= 0),
    duration        REAL CHECK(duration IS NULL OR duration >= 0),
    confidence      REAL CHECK(confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    file_path       TEXT NOT NULL CHECK(length(file_path) > 0),
    file_size_bytes INTEGER CHECK(file_size_bytes IS NULL OR file_size_bytes > 0),
    is_primary      INTEGER NOT NULL CHECK(is_primary IN (0, 1)) DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_transcriptions_video ON transcriptions(video_id);
CREATE INDEX idx_transcriptions_provider ON transcriptions(provider);
CREATE INDEX idx_transcriptions_primary ON transcriptions(video_id, is_primary);

-- ============================================================
-- SCENES (segmentation structure -- no processing state flags)
-- ============================================================
CREATE TABLE scenes (
    id              TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id        TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    scene_id        INTEGER NOT NULL CHECK(scene_id >= 0),
    start_time      REAL NOT NULL CHECK(start_time >= 0),
    end_time        REAL NOT NULL CHECK(end_time > start_time),
    title           TEXT,
    transcript_text TEXT,
    method          TEXT CHECK(method IS NULL OR method IN (
        'transcript', 'visual', 'hybrid', 'chapters'
    )),
    relevance_boost REAL NOT NULL CHECK(relevance_boost >= 0) DEFAULT 1.0,
    UNIQUE(video_id, scene_id)
);
CREATE INDEX idx_scenes_video ON scenes(video_id);

-- ============================================================
-- FRAMES (all extracted images -- thumbnails, keyframes, drill, hq)
--
-- Covers all extraction types: drill (quick), hq (high-quality),
-- keyframes (per-scene representative frames), and thumbnails.
-- scene_id is NULL when frames are extracted without scene context.
-- is_thumbnail flags any frame as the video's thumbnail image.
-- ============================================================
CREATE TABLE frames (
    id              TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id        TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    scene_id        INTEGER CHECK(scene_id IS NULL OR scene_id >= 0),
    timestamp       REAL NOT NULL CHECK(timestamp >= 0),
    extraction_type TEXT NOT NULL CHECK(extraction_type IN ('drill', 'hq', 'keyframe', 'thumbnail')),
    quality_tier    TEXT CHECK(quality_tier IS NULL OR quality_tier IN (
        'lowest', 'low', 'medium', 'high', 'highest'
    )),
    is_thumbnail    INTEGER NOT NULL CHECK(is_thumbnail IN (0, 1)) DEFAULT 0,
    width           INTEGER CHECK(width IS NULL OR width > 0),
    height          INTEGER CHECK(height IS NULL OR height > 0),
    file_size_bytes INTEGER CHECK(file_size_bytes IS NULL OR file_size_bytes > 0),
    file_path       TEXT NOT NULL CHECK(length(file_path) > 0),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_frames_video ON frames(video_id);
CREATE INDEX idx_frames_scene ON frames(video_id, scene_id);
CREATE INDEX idx_frames_type ON frames(extraction_type);
CREATE INDEX idx_frames_timestamp ON frames(video_id, timestamp);
CREATE INDEX idx_frames_thumbnail ON frames(video_id, is_thumbnail);

-- ============================================================
-- VISUAL DESCRIPTIONS (per-scene AI visual analysis)
-- ============================================================
CREATE TABLE visual_descriptions (
    id              TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id        TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    scene_id        INTEGER NOT NULL CHECK(scene_id >= 0),
    provider        TEXT CHECK(provider IS NULL OR length(provider) > 0),
    description     TEXT NOT NULL CHECK(length(description) > 0),
    file_path       TEXT CHECK(file_path IS NULL OR length(file_path) > 0),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(video_id, scene_id)
);
CREATE INDEX idx_visual_video ON visual_descriptions(video_id);

-- ============================================================
-- TECHNICAL CONTENT (per-scene OCR, code detection)
-- ============================================================
CREATE TABLE technical_content (
    id              TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id        TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    scene_id        INTEGER NOT NULL CHECK(scene_id >= 0),
    provider        TEXT CHECK(provider IS NULL OR length(provider) > 0),
    has_code        INTEGER NOT NULL CHECK(has_code IN (0, 1)) DEFAULT 0,
    has_text        INTEGER NOT NULL CHECK(has_text IN (0, 1)) DEFAULT 0,
    ocr_text        TEXT,  -- extracted text for FTS
    code_language   TEXT CHECK(code_language IS NULL OR length(code_language) > 0),
    file_path       TEXT CHECK(file_path IS NULL OR length(file_path) > 0),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(video_id, scene_id)
);
CREATE INDEX idx_technical_video ON technical_content(video_id);

-- ============================================================
-- AUDIO DESCRIPTIONS (accessibility)
-- ============================================================
CREATE TABLE audio_descriptions (
    id              TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id        TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    format          TEXT NOT NULL CHECK(format IN ('vtt', 'txt')),
    source          TEXT NOT NULL CHECK(source IN ('generated', 'source_track', 'compiled')),
    provider        TEXT CHECK(provider IS NULL OR length(provider) > 0),
    file_path       TEXT NOT NULL CHECK(length(file_path) > 0),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_ad_video ON audio_descriptions(video_id);

-- ============================================================
-- NARRATIVE STRUCTURES (video type + section analysis)
-- ============================================================
CREATE TABLE narrative_structures (
    id              TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id        TEXT NOT NULL UNIQUE REFERENCES videos(id) ON DELETE CASCADE,
    video_type      TEXT CHECK(video_type IS NULL OR video_type IN (
        'coding_tutorial', 'lecture', 'demo', 'presentation', 'interview',
        'review', 'vlog', 'documentary', 'music_video', 'other'
    )),
    section_count   INTEGER CHECK(section_count IS NULL OR section_count >= 0),
    file_path       TEXT CHECK(file_path IS NULL OR length(file_path) > 0),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_narrative_video ON narrative_structures(video_id);

-- ============================================================
-- CODE EVOLUTIONS (code tracking across scenes)
-- ============================================================
CREATE TABLE code_evolutions (
    id              TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id        TEXT NOT NULL UNIQUE REFERENCES videos(id) ON DELETE CASCADE,
    files_tracked   INTEGER CHECK(files_tracked IS NULL OR files_tracked >= 0),
    total_changes   INTEGER CHECK(total_changes IS NULL OR total_changes >= 0),
    file_path       TEXT CHECK(file_path IS NULL OR length(file_path) > 0),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_code_evo_video ON code_evolutions(video_id);

-- ============================================================
-- ENTITIES (unified: objects + concepts + people)
-- ============================================================
CREATE TABLE entities (
    id          TEXT PRIMARY KEY CHECK(length(id) = 36),
    name        TEXT NOT NULL CHECK(length(name) > 0),
    entity_type TEXT NOT NULL CHECK(entity_type IN (
        'object', 'concept', 'person', 'technology', 'organization'
    )),
    UNIQUE(name, entity_type)
);
CREATE INDEX idx_entities_name ON entities(name);
CREATE INDEX idx_entities_type ON entities(entity_type);

-- Entity appearances in scenes
CREATE TABLE entity_appearances (
    id          TEXT PRIMARY KEY CHECK(length(id) = 36),
    entity_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    video_id    TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    scene_id    INTEGER NOT NULL CHECK(scene_id >= 0),
    timestamp   REAL NOT NULL CHECK(timestamp >= 0),
    score       REAL CHECK(score IS NULL OR (score >= 0 AND score <= 1)),
    UNIQUE(entity_id, video_id, scene_id)
);
CREATE INDEX idx_ea_entity ON entity_appearances(entity_id);
CREATE INDEX idx_ea_video ON entity_appearances(video_id);

-- Cross-video entity summary (replaces graph.json)
CREATE TABLE entity_video_summary (
    id        TEXT PRIMARY KEY CHECK(length(id) = 36),
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    video_id  TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    frequency INTEGER NOT NULL CHECK(frequency > 0) DEFAULT 1,
    avg_score REAL CHECK(avg_score IS NULL OR (avg_score >= 0 AND avg_score <= 1)),
    UNIQUE(entity_id, video_id)
);
CREATE INDEX idx_evs_entity ON entity_video_summary(entity_id);
CREATE INDEX idx_evs_video ON entity_video_summary(video_id);

-- ============================================================
-- Q&A HISTORY
-- ============================================================
CREATE TABLE qa_history (
    id        TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id  TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    question  TEXT NOT NULL CHECK(length(question) > 0),
    answer    TEXT NOT NULL CHECK(length(answer) > 0),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_qa_video ON qa_history(video_id);

-- Q&A to scene mapping
CREATE TABLE qa_scenes (
    qa_id    TEXT NOT NULL REFERENCES qa_history(id) ON DELETE CASCADE,
    scene_id INTEGER NOT NULL CHECK(scene_id >= 0),
    PRIMARY KEY (qa_id, scene_id)
);

-- ============================================================
-- OBSERVATIONS
-- ============================================================
CREATE TABLE observations (
    id        TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id  TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    scene_id  INTEGER NOT NULL CHECK(scene_id >= 0),
    type      TEXT NOT NULL CHECK(length(type) > 0),
    content   TEXT NOT NULL CHECK(length(content) > 0),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_obs_video ON observations(video_id);

-- ============================================================
-- PLAYLISTS (first-class construct)
-- ============================================================
CREATE TABLE playlists (
    id            TEXT PRIMARY KEY CHECK(length(id) = 36),
    playlist_id   TEXT NOT NULL UNIQUE CHECK(length(playlist_id) > 0),
    domain        TEXT NOT NULL CHECK(domain GLOB '[a-z]*'),
    channel       TEXT CHECK(channel IS NULL OR length(channel) > 0),  -- NULL = unknown
    title         TEXT,  -- NULL until yt-dlp provides it
    description   TEXT,
    url           TEXT,
    video_count   INTEGER CHECK(video_count IS NULL OR video_count >= 0),
    playlist_type TEXT CHECK(playlist_type IS NULL OR playlist_type IN (
        'course', 'series', 'conference', 'collection'
    )),
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Playlist-video membership (many-to-many, ordered)
CREATE TABLE playlist_videos (
    id          TEXT PRIMARY KEY CHECK(length(id) = 36),
    playlist_id TEXT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
    video_id    TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL CHECK(position >= 0) DEFAULT 0,
    UNIQUE(playlist_id, video_id)
);
CREATE INDEX idx_pv_playlist ON playlist_videos(playlist_id);
CREATE INDEX idx_pv_video ON playlist_videos(video_id);

-- ============================================================
-- PIPELINE STEPS (processing state for ALL operations)
--
-- This is the single source of truth for "what has been processed?"
-- Replaces scattered boolean flags on videos and scenes tables.
-- Each step tracks: what was done, by whom, when, with what config.
-- ============================================================
CREATE TABLE pipeline_steps (
    id              TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id        TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    step_type       TEXT NOT NULL CHECK(step_type IN (
        'download', 'audio_extract', 'transcribe', 'scene_detect',
        'keyframe_extract', 'visual_analyze', 'entity_extract',
        'deep_analyze', 'focus_analyze', 'narrative_detect',
        'change_detect', 'code_track', 'people_track',
        'ad_generate', 'knowledge_index', 'embed'
    )),
    status          TEXT NOT NULL CHECK(status IN (
        'pending', 'running', 'completed', 'failed', 'skipped'
    )),
    provider        TEXT CHECK(provider IS NULL OR length(provider) > 0),
    model           TEXT CHECK(model IS NULL OR length(model) > 0),
    scene_id        INTEGER CHECK(scene_id IS NULL OR scene_id >= 0),
    config          TEXT,  -- JSON blob for step-specific params
    error_message   TEXT,  -- populated on failure
    started_at      TEXT,
    completed_at    TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_pipeline_video ON pipeline_steps(video_id);
CREATE INDEX idx_pipeline_type ON pipeline_steps(step_type);
CREATE INDEX idx_pipeline_status ON pipeline_steps(status);
CREATE INDEX idx_pipeline_video_type ON pipeline_steps(video_id, step_type);
CREATE INDEX idx_pipeline_video_scene ON pipeline_steps(video_id, scene_id);

-- ============================================================
-- FTS5 VIRTUAL TABLES + SYNC TRIGGERS
-- ============================================================

-- Transcriptions FTS (full transcript text for cross-video RAG)
CREATE VIRTUAL TABLE transcriptions_fts USING fts5(
    full_text,
    content=transcriptions, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TRIGGER transcriptions_ai AFTER INSERT ON transcriptions BEGIN
    INSERT INTO transcriptions_fts(rowid, full_text)
    VALUES (new.rowid, new.full_text);
END;
CREATE TRIGGER transcriptions_ad AFTER DELETE ON transcriptions BEGIN
    INSERT INTO transcriptions_fts(transcriptions_fts, rowid, full_text)
    VALUES ('delete', old.rowid, old.full_text);
END;
CREATE TRIGGER transcriptions_au AFTER UPDATE ON transcriptions BEGIN
    INSERT INTO transcriptions_fts(transcriptions_fts, rowid, full_text)
    VALUES ('delete', old.rowid, old.full_text);
    INSERT INTO transcriptions_fts(rowid, full_text)
    VALUES (new.rowid, new.full_text);
END;

-- Scene transcripts FTS (per-scene segments)
CREATE VIRTUAL TABLE scenes_fts USING fts5(
    transcript_text,
    content=scenes, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TRIGGER scenes_ai AFTER INSERT ON scenes BEGIN
    INSERT INTO scenes_fts(rowid, transcript_text)
    VALUES (new.rowid, new.transcript_text);
END;
CREATE TRIGGER scenes_ad AFTER DELETE ON scenes BEGIN
    INSERT INTO scenes_fts(scenes_fts, rowid, transcript_text)
    VALUES ('delete', old.rowid, old.transcript_text);
END;
CREATE TRIGGER scenes_au AFTER UPDATE ON scenes BEGIN
    INSERT INTO scenes_fts(scenes_fts, rowid, transcript_text)
    VALUES ('delete', old.rowid, old.transcript_text);
    INSERT INTO scenes_fts(rowid, transcript_text)
    VALUES (new.rowid, new.transcript_text);
END;

-- Visual descriptions FTS (AI-generated scene descriptions)
CREATE VIRTUAL TABLE visual_fts USING fts5(
    description,
    content=visual_descriptions, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TRIGGER visual_ai AFTER INSERT ON visual_descriptions BEGIN
    INSERT INTO visual_fts(rowid, description)
    VALUES (new.rowid, new.description);
END;
CREATE TRIGGER visual_ad AFTER DELETE ON visual_descriptions BEGIN
    INSERT INTO visual_fts(visual_fts, rowid, description)
    VALUES ('delete', old.rowid, old.description);
END;
CREATE TRIGGER visual_au AFTER UPDATE ON visual_descriptions BEGIN
    INSERT INTO visual_fts(visual_fts, rowid, description)
    VALUES ('delete', old.rowid, old.description);
    INSERT INTO visual_fts(rowid, description)
    VALUES (new.rowid, new.description);
END;

-- Technical content FTS (OCR text)
CREATE VIRTUAL TABLE technical_fts USING fts5(
    ocr_text,
    content=technical_content, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TRIGGER technical_ai AFTER INSERT ON technical_content BEGIN
    INSERT INTO technical_fts(rowid, ocr_text)
    VALUES (new.rowid, new.ocr_text);
END;
CREATE TRIGGER technical_ad AFTER DELETE ON technical_content BEGIN
    INSERT INTO technical_fts(technical_fts, rowid, ocr_text)
    VALUES ('delete', old.rowid, old.ocr_text);
END;
CREATE TRIGGER technical_au AFTER UPDATE ON technical_content BEGIN
    INSERT INTO technical_fts(technical_fts, rowid, ocr_text)
    VALUES ('delete', old.rowid, old.ocr_text);
    INSERT INTO technical_fts(rowid, ocr_text)
    VALUES (new.rowid, new.ocr_text);
END;

-- Q&A FTS
CREATE VIRTUAL TABLE qa_fts USING fts5(
    question, answer,
    content=qa_history, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TRIGGER qa_ai AFTER INSERT ON qa_history BEGIN
    INSERT INTO qa_fts(rowid, question, answer)
    VALUES (new.rowid, new.question, new.answer);
END;
CREATE TRIGGER qa_ad AFTER DELETE ON qa_history BEGIN
    INSERT INTO qa_fts(qa_fts, rowid, question, answer)
    VALUES ('delete', old.rowid, old.question, old.answer);
END;
CREATE TRIGGER qa_au AFTER UPDATE ON qa_history BEGIN
    INSERT INTO qa_fts(qa_fts, rowid, question, answer)
    VALUES ('delete', old.rowid, old.question, old.answer);
    INSERT INTO qa_fts(rowid, question, answer)
    VALUES (new.rowid, new.question, new.answer);
END;

-- Videos FTS (titles, descriptions, channels)
CREATE VIRTUAL TABLE videos_fts USING fts5(
    title, description, channel_name,
    content=videos, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TRIGGER videos_fts_ai AFTER INSERT ON videos BEGIN
    INSERT INTO videos_fts(rowid, title, description, channel_name)
    VALUES (new.rowid, new.title, new.description, new.channel_name);
END;
CREATE TRIGGER videos_fts_ad AFTER DELETE ON videos BEGIN
    INSERT INTO videos_fts(videos_fts, rowid, title, description, channel_name)
    VALUES ('delete', old.rowid, old.title, old.description, old.channel_name);
END;
CREATE TRIGGER videos_fts_au AFTER UPDATE ON videos BEGIN
    INSERT INTO videos_fts(videos_fts, rowid, title, description, channel_name)
    VALUES ('delete', old.rowid, old.title, old.description, old.channel_name);
    INSERT INTO videos_fts(rowid, title, description, channel_name)
    VALUES (new.rowid, new.title, new.description, new.channel_name);
END;

-- ============================================================
-- CONVENIENCE VIEW: video processing status
--
-- Derives processing state from child tables and pipeline_steps
-- instead of relying on denormalized boolean flags.
-- ============================================================
CREATE VIEW video_processing_status AS
SELECT
    v.id,
    v.video_id,
    v.title,
    v.domain,
    v.duration,
    (SELECT COUNT(*) FROM audio_tracks a WHERE a.video_id = v.id) as audio_track_count,
    (SELECT COUNT(*) FROM transcriptions t WHERE t.video_id = v.id) as transcription_count,
    EXISTS(SELECT 1 FROM transcriptions t WHERE t.video_id = v.id AND t.is_primary = 1) as has_primary_transcript,
    (SELECT t.provider FROM transcriptions t WHERE t.video_id = v.id AND t.is_primary = 1) as transcript_provider,
    (SELECT COUNT(*) FROM scenes s WHERE s.video_id = v.id) as scene_count,
    (SELECT COUNT(*) FROM frames f WHERE f.video_id = v.id) as frame_count,
    (SELECT COUNT(*) FROM frames f WHERE f.video_id = v.id AND f.extraction_type = 'keyframe') as keyframe_count,
    EXISTS(SELECT 1 FROM frames f WHERE f.video_id = v.id AND f.is_thumbnail = 1) as has_thumbnail,
    (SELECT COUNT(*) FROM visual_descriptions vd WHERE vd.video_id = v.id) as visual_description_count,
    (SELECT COUNT(*) FROM technical_content tc WHERE tc.video_id = v.id) as technical_content_count,
    EXISTS(SELECT 1 FROM narrative_structures ns WHERE ns.video_id = v.id) as has_narrative,
    EXISTS(SELECT 1 FROM code_evolutions ce WHERE ce.video_id = v.id) as has_code_evolution,
    EXISTS(SELECT 1 FROM audio_descriptions ad WHERE ad.video_id = v.id) as has_audio_description,
    (SELECT COUNT(*) FROM entity_video_summary evs WHERE evs.video_id = v.id) as entity_count,
    (SELECT COUNT(*) FROM qa_history q WHERE q.video_id = v.id) as qa_count,
    (SELECT COUNT(*) FROM pipeline_steps p WHERE p.video_id = v.id AND p.status = 'completed') as completed_steps,
    (SELECT COUNT(*) FROM pipeline_steps p WHERE p.video_id = v.id AND p.status = 'failed') as failed_steps,
    (SELECT COUNT(*) FROM pipeline_steps p WHERE p.video_id = v.id AND p.status = 'running') as running_steps
FROM videos v;
