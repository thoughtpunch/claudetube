-- 001_initial.vec.sql
-- Initial vectors database schema: vec_metadata table for vector embeddings.
--
-- This database is separate from the main claudetube.db to allow
-- standard SQLite tools to open the main database without needing
-- the sqlite-vec extension.
--
-- The vec0 virtual table (vec_embeddings) is created at runtime after
-- loading the sqlite-vec extension via VecStore._ensure_vec_table().

-- ============================================================
-- VECTOR EMBEDDINGS METADATA
-- ============================================================
-- Maps rowids in vec_embeddings to video/scene identifiers.
-- Note: video_id here is the UUID from the main database's videos.id
-- We cannot enforce a foreign key across databases, so we rely on
-- application-level integrity.
CREATE TABLE vec_metadata (
    id         TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id   TEXT NOT NULL CHECK(length(video_id) = 36),  -- UUID from videos.id
    scene_id   INTEGER CHECK(scene_id IS NULL OR scene_id >= 0),
    start_time REAL CHECK(start_time IS NULL OR start_time >= 0),
    end_time   REAL CHECK(end_time IS NULL OR end_time > start_time),
    source     TEXT NOT NULL CHECK(source IN (
        'transcription', 'scene_transcript', 'visual', 'technical',
        'entity', 'qa', 'observation', 'audio_description'
    )),
    UNIQUE(video_id, scene_id, source)
);

CREATE INDEX idx_vec_video ON vec_metadata(video_id);
CREATE INDEX idx_vec_source ON vec_metadata(source);
