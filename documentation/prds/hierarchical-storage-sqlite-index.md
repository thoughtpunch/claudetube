[← Documentation](../README.md)

# PRD: Hierarchical Storage + SQLite Index for claudetube

## Overview

Two complementary changes:
1. **Hierarchical folder structure** -- `<domain>/<channel>/<playlist>/<video_id>/` replacing the flat `<video_id>/` layout
2. **SQLite index layer** -- Queryable database for cross-video search, RAG, and analytics (JSON stays authoritative)

These reinforce each other: the hierarchical tree makes the cache human-navigable; SQLite makes it machine-queryable.

### Core Principles

1. **UUID primary keys** on all tables -- stable identity that survives path changes and metadata enrichment
2. **Standardize on sqlite-vec** for vector storage -- one database, one dependency (drop ChromaDB)
3. **Progressive enrichment** -- when yt-dlp returns richer metadata, UPSERT the database AND `shutil.move()` the directory to the correct hierarchical path. Paths are living -- they improve as we learn more.
4. **NULL = unknown** -- `channel` and `playlist` are NULL in the database until populated. Filesystem uses `no_channel`/`no_playlist` as human-readable placeholders; the `VideoPath` model translates between the two representations. Domain is always required (every URL has one).
5. **Strict validation** -- Pydantic `strict=True` + SQLite CHECK constraints. Fail on construction, fail on INSERT. No silent data corruption.

---

## Part 1: Hierarchical Folder Structure

### New Path Model

```
{cache_dir}/<domain>/<channel>/<playlist>/<video_id>/
```

**Examples:**
```
~/.claude/video_cache/youtube/UC_x5XG1OV2P6uZZ5FSM9Ttw/PLRqwX-V7Uu6ZiZxtDDRCi6uhfTH4FilpH/dQw4w9WgXcQ/
~/.claude/video_cache/twitter/elikiowa/no_playlist/1879432010/
~/.claude/video_cache/dailymotion/cnn/no_playlist/x8fgh12/
~/.claude/video_cache/vimeo/no_channel/no_playlist/912345678/
~/.claude/video_cache/local/no_channel/no_playlist/screen_rec_a3f2dd1e/
```

When channel or playlist is unknown, the filesystem uses `no_channel`/`no_playlist` as human-readable placeholder directory names. The database stores NULL for these -- translation happens in the `VideoPath` model (`no_channel` on disk <-> NULL in DB).

### Component Extraction Rules

| Component | Source Priority | Sanitization | Default |
|-----------|---------------|-------------|---------|
| **domain** | 1. URL hostname 2. yt-dlp `extractor_key` | Lowercase, strip TLD, strip non-`\w` chars. `youtube.com` -> `youtube`, `clips.twitch.tv` -> `twitch` | **Required** -- every URL has a domain. Raises `ValueError` if missing. |
| **channel** | 1. URL named capture `(?P<channel>...)` 2. yt-dlp `channel_id` 3. yt-dlp `uploader_id` 4. yt-dlp `channel` (sanitized) | Regex `[^\w-]` -> `_`, truncate to 60 chars | `None` (NULL in DB, `no_channel` on disk) |
| **playlist** | 1. URL named capture `(?P<playlist>...)` (e.g., `list=PLxxx`) 2. yt-dlp `playlist_id` 3. yt-dlp `playlist_title` (slugified) | Regex `[^\w-]` -> `_`, truncate to 60 chars | `None` (NULL in DB, `no_playlist` on disk) |
| **video_id** | 1. URL named captures (existing `video_id`, `clip_id`, etc.) 2. yt-dlp `id` | Existing `cache_key` logic (safe or SHA256) | **Required** |

### New Data Model: `VideoPath`

**New file: `src/claudetube/models/video_path.py`**

```python
from pydantic import BaseModel, ConfigDict, field_validator

class VideoPath(BaseModel):
    """Hierarchical path components for a cached video."""
    model_config = ConfigDict(strict=True, frozen=True)

    domain: str        # "youtube", "twitter", "local" -- REQUIRED
    channel: str | None = None  # channel_id or None if unknown
    playlist: str | None = None  # playlist_id or None if unknown
    video_id: str      # unique video identifier -- REQUIRED

    @field_validator("domain")
    @classmethod
    def domain_must_be_lowercase_alpha(cls, v: str) -> str:
        import re
        if not re.match(r'^[a-z][a-z0-9]*$', v):
            raise ValueError(f"domain must be lowercase alphanumeric, got: {v!r}")
        return v

    @field_validator("video_id")
    @classmethod
    def video_id_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("video_id must be non-empty")
        return v

    @classmethod
    def from_url(cls, url: str, metadata: dict | None = None) -> "VideoPath":
        """Extract path from URL, optionally augmenting with yt-dlp metadata.

        Strategy: parse URL with named captures first, then
        overwrite/fill gaps from yt-dlp metadata dict.
        """
        ...

    @classmethod
    def from_local(cls, local_file: "LocalFile") -> "VideoPath":
        """Create path for a local file."""
        return cls(domain="local", channel=None,
                   playlist=None, video_id=local_file.video_id)

    def relative_path(self) -> "Path":
        """Return domain/channel/playlist/video_id as a Path.

        Uses 'no_channel'/'no_playlist' as filesystem placeholders for NULL.
        """
        from pathlib import Path
        return (Path(self.domain)
                / (self.channel or "no_channel")
                / (self.playlist or "no_playlist")
                / self.video_id)

    @classmethod
    def from_cache_path(cls, cache_path: str) -> "VideoPath":
        """Reconstruct from a relative cache path string.

        Translates filesystem sentinels back to None:
        'no_channel' -> None, 'no_playlist' -> None
        """
        from pathlib import Path
        parts = Path(cache_path).parts
        domain, channel, playlist, video_id = parts[0], parts[1], parts[2], parts[3]
        return cls(
            domain=domain,
            channel=None if channel == "no_channel" else channel,
            playlist=None if playlist == "no_playlist" else playlist,
            video_id=video_id,
        )

    def cache_dir(self, cache_base: "Path") -> "Path":
        """Full absolute cache directory."""
        return cache_base / self.relative_path()
```

### URL Regex Enhancement

**Modify: `src/claudetube/config/providers.py`**

Add named capture groups for `channel` and `playlist` to the existing provider patterns where extractable from URL alone. For example:

```python
# YouTube - add channel and playlist captures
VideoProvider(
    name="YouTube",
    domains=["youtube.com", "youtu.be", ...],
    pattern=r"""(?x)
        (?:youtube\.com/
            (?:watch\?.*?v=|embed/|v/|shorts/|live/)
            (?P<video_id>[a-zA-Z0-9_-]{11})
            (?:.*?[&?]list=(?P<playlist>[a-zA-Z0-9_-]+))?
        |youtu\.be/(?P<video_id>[a-zA-Z0-9_-]{11})
            (?:\?list=(?P<playlist>[a-zA-Z0-9_-]+))?
        )
    """,
    ...
)
```

The `channel` capture isn't available from most YouTube URLs (it's in `/c/` or `/@` URLs, not `/watch?v=`), so it falls through to yt-dlp metadata augmentation. This is exactly the "try URL first, augment with yt-dlp" pattern.

### Domain Sanitization Function

**New: `src/claudetube/models/video_path.py`**

```python
def sanitize_domain(hostname: str) -> str:
    """youtube.com -> youtube, clips.twitch.tv -> twitch, m.facebook.com -> facebook"""
    # Strip common prefixes
    for prefix in ("www.", "m.", "mobile.", "clips.", "player.", "music."):
        if hostname.startswith(prefix):
            hostname = hostname[len(prefix):]
    # Strip TLD(s)
    parts = hostname.split(".")
    if len(parts) >= 2:
        hostname = parts[0]  # "youtube" from "youtube.com"
    # Sanitize
    result = re.sub(r"\W+", "", hostname).lower()
    if not result:
        raise ValueError(f"Cannot extract domain from hostname: {hostname!r}")
    return result
```

### URL Query Parameter Extraction

Currently only `list=` is extracted from URLs (via a hardcoded regex in `parsing/utils.py`). Well-known video providers embed useful metadata in query parameters that should be captured alongside the regex-based path extraction.

**Strategy**: After the provider regex matches the URL path, run `urllib.parse.parse_qs()` on the query string and extract known params per-provider. Store results in `provider_data` dict alongside the regex captures.

#### Parameters to Extract

**YouTube** (highest priority -- most popular provider):
| Parameter | Example | Field | Notes |
|-----------|---------|-------|-------|
| `list=` | `PLRqwX-V7Uu6Z...` | `playlist` | Already extracted separately; unify into provider pattern |
| `t=` / `start=` | `t=120`, `t=2m30s`, `start=300` | `start_time` | Seconds or duration string. Parse to float seconds. |
| `index=` | `index=3` | `playlist_position` | 1-based position in playlist |
| `ab_channel=` | `ab_channel=3Blue1Brown` | `channel_hint` | Channel name hint (not ID). Use as fallback for display name. |
| `feature=` | `feature=shared` | `referral_source` | Discovery context: `shared`, `youtu.be`, `emb_title`, etc. |

**Vimeo**:
| Parameter | Example | Field | Notes |
|-----------|---------|-------|-------|
| `h=` | `h=a1b2c3d4e5` | `private_hash` | Required for password-protected videos. Must be passed to yt-dlp. |
| `time=` | `time=1m30s` | `start_time` | Same as YouTube `t=` |

**Twitch**:
| Parameter | Example | Field | Notes |
|-----------|---------|-------|-------|
| `t=` | `t=1h30m45s` | `start_time` | Duration string format (`XhYmZs`) |

**Generic** (all providers):
| Parameter | Example | Field | Notes |
|-----------|---------|-------|-------|
| `utm_source` | `utm_source=twitter` | `referral_source` | How the URL was shared |
| `utm_medium` | `utm_medium=social` | `referral_medium` | Sharing channel |

#### Timestamp Parsing

Multiple formats need to be normalized to `float` seconds:

```python
def parse_timestamp(value: str) -> float | None:
    """Parse various timestamp formats to seconds.

    Supports:
      - Pure seconds: "120", "300"
      - YouTube compact: "2m30s", "1h2m3s"
      - Twitch format: "1h30m45s"
      - Colon format: "1:30:45", "2:30"
    """
    ...
```

#### Implementation

**New: `src/claudetube/parsing/params.py`**

```python
from urllib.parse import parse_qs, urlparse

# Per-provider param extraction rules
PROVIDER_PARAMS: dict[str, dict[str, str]] = {
    "youtube": {
        "list": "playlist",
        "t": "start_time",
        "start": "start_time",
        "index": "playlist_position",
        "ab_channel": "channel_hint",
        "feature": "referral_source",
    },
    "vimeo": {
        "h": "private_hash",
        "time": "start_time",
    },
    "twitch": {
        "t": "start_time",
    },
}

def extract_query_params(url: str, provider_name: str) -> dict[str, str]:
    """Extract well-known query parameters for a given provider.

    Returns dict mapping canonical field names to raw string values.
    Timestamp fields are NOT parsed here -- caller normalizes.
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query, keep_blank_values=False)

    param_map = PROVIDER_PARAMS.get(provider_name.lower(), {})
    result: dict[str, str] = {}

    for param_key, field_name in param_map.items():
        if param_key in qs:
            result[field_name] = qs[param_key][0]

    # Generic UTM params (all providers)
    for utm_key in ("utm_source", "utm_medium"):
        if utm_key in qs:
            result[utm_key.replace("utm_", "referral_")] = qs[utm_key][0]

    return result
```

#### Integration Point

In `VideoURL.parse()` or `extract_url_context()`, after the provider regex matches:

```python
# After regex extraction
provider_data = match.groupdict()

# Extract query params for this provider
from claudetube.parsing.params import extract_query_params
query_data = extract_query_params(url, provider.name)

# Merge: regex captures take priority, query params fill gaps
for key, value in query_data.items():
    if key not in provider_data or provider_data[key] is None:
        provider_data[key] = value
```

This keeps the regex as the primary extraction mechanism and uses query params to augment with metadata that can't be captured from the URL path alone.

#### What Gets Stored Where

| Field | `VideoPath` | `VideoState` | SQLite `videos` | Notes |
|-------|-------------|-------------|-----------------|-------|
| `playlist` | Yes (path component) | Yes | Yes | From `list=` param or yt-dlp |
| `channel_hint` | No | Yes (`channel_name`) | Yes (`channel_name`) | Display name fallback from `ab_channel=` |
| `start_time` | No | Yes | No (session-specific) | Useful for frame extraction, not stored long-term |
| `playlist_position` | No | No | Yes (`playlist_videos.position`) | Position in playlist context |
| `private_hash` | No | Yes | No | Passed to yt-dlp for authentication |
| `referral_source` | No | No | No (analytics, future) | Could be stored if analytics needed later |

### Files to Create/Modify for Part 1

| File | Action | Description |
|------|--------|-------------|
| `src/claudetube/models/video_path.py` | **CREATE** | `VideoPath` Pydantic model (strict) + `sanitize_domain()` + `from_url()` + `from_local()` |
| `src/claudetube/parsing/params.py` | **CREATE** | `extract_query_params()` + `parse_timestamp()` + `PROVIDER_PARAMS` mapping |
| `src/claudetube/config/providers.py` | **MODIFY** | Add `(?P<channel>...)` and `(?P<playlist>...)` named captures to provider patterns where possible |
| `src/claudetube/models/video_url.py` | **MODIFY** | `VideoURL.parse()` returns a `VideoPath` (or stores one). Add `video_path` property that constructs `VideoPath` from parsed data. Update `ID_GROUPS` to include `channel`, `playlist`. Merge query params into `provider_data`. |
| `src/claudetube/models/state.py` | **MODIFY** | Add `domain`, `channel_id`, `playlist_id` fields to `VideoState`. Update `from_metadata()` to populate from yt-dlp |
| `src/claudetube/cache/manager.py` | **MODIFY** | `get_cache_dir(video_id)` -> `get_cache_dir(video_path)`. All path methods accept `VideoPath` or resolve via lookup |
| `src/claudetube/cache/storage.py` | **MODIFY** | `save_state()` / `load_state()` handle new fields |
| `src/claudetube/operations/processor.py` | **MODIFY** | Build `VideoPath` from URL + yt-dlp metadata, pass to cache manager |
| `src/claudetube/operations/playlist.py` | **MODIFY** | Use `VideoPath` for playlist video paths, populate `playlist` component |
| `src/claudetube/mcp_server.py` | **MODIFY** | Tool handlers that accept `video_id` need to resolve to `VideoPath` via SQLite lookup or scan |

### Migration of Existing Flat Caches

Existing caches at `{cache_dir}/{video_id}/` need to be discoverable. Three approaches (all needed):

1. **SQLite index** maps `video_id -> full_path` so tools can find videos by ID regardless of directory structure
2. **Lazy migration**: When a video is accessed by bare `video_id` and isn't found at the new hierarchical path, check the flat legacy path. If found, optionally move it (or just index it in SQLite where it is)
3. **Bulk migration CLI**: `claudetube migrate` command that reads each `state.json`, constructs the `VideoPath`, and moves the directory

---

## Part 2: SQLite Index Layer

### Design Decisions

- **Location**: Config-driven, `{cache_dir}/claudetube.db`
- **Role**: Index + cross-video queries. JSON stays authoritative.
- **Vectors**: Standardize on sqlite-vec (drop ChromaDB)
- **PKs**: UUID on all tables (Python `uuid.uuid4()`, stored as TEXT in SQLite, `CHECK(length(id) = 36)`)
- **Migration**: Auto-import on first use
- **Progressive enrichment**: UPSERT replaces NULLs with real data when available, AND moves directories to match
- **NULL = unknown**: `channel` and `playlist` are NULL until populated. Domain is NOT NULL (every URL has one).
- **Strict schema**: CHECK constraints on every column. Fail on INSERT/UPDATE rather than storing bad data.

### Schema (DDL)

All PKs are UUIDs (TEXT, `CHECK(length(id) = 36)`, generated in Python via `uuid.uuid4()`). `video_id` is a natural key (YouTube ID, etc.) used for lookups but NOT the PK. NULL means "not yet known". Every column has CHECK constraints enforcing data integrity.

```sql
-- Schema versioning
CREATE TABLE schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);

-- ============================================================
-- VIDEOS
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
    -- Processing state
    transcript_complete INTEGER NOT NULL CHECK(transcript_complete IN (0, 1)) DEFAULT 0,
    transcript_source   TEXT CHECK(transcript_source IS NULL OR transcript_source IN (
        'youtube_subtitles', 'whisper', 'deepgram', 'openai', 'manual'
    )),
    scenes_processed    INTEGER NOT NULL CHECK(scenes_processed IN (0, 1)) DEFAULT 0,
    scene_count         INTEGER CHECK(scene_count IS NULL OR scene_count >= 0),
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
-- SCENES
-- ============================================================
CREATE TABLE scenes (
    id              TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id        TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    scene_id        INTEGER NOT NULL CHECK(scene_id >= 0),
    start_time      REAL NOT NULL CHECK(start_time >= 0),
    end_time        REAL NOT NULL CHECK(end_time > start_time),
    title           TEXT,
    transcript_text TEXT,
    has_keyframes   INTEGER NOT NULL CHECK(has_keyframes IN (0, 1)) DEFAULT 0,
    has_visual      INTEGER NOT NULL CHECK(has_visual IN (0, 1)) DEFAULT 0,
    has_technical   INTEGER NOT NULL CHECK(has_technical IN (0, 1)) DEFAULT 0,
    relevance_boost REAL NOT NULL CHECK(relevance_boost >= 0) DEFAULT 1.0,
    UNIQUE(video_id, scene_id)
);
CREATE INDEX idx_scenes_video ON scenes(video_id);

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
-- VECTOR EMBEDDINGS (sqlite-vec)
-- ============================================================
-- The vec0 virtual table is created at runtime after loading
-- the sqlite-vec extension. This metadata table maps rowids
-- to video/scene identifiers and tracks embedding source.
CREATE TABLE vec_metadata (
    id         TEXT PRIMARY KEY CHECK(length(id) = 36),
    video_id   TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    scene_id   INTEGER NOT NULL CHECK(scene_id >= 0),
    start_time REAL NOT NULL CHECK(start_time >= 0),
    end_time   REAL NOT NULL CHECK(end_time > start_time),
    source     TEXT NOT NULL CHECK(source IN (
        'transcript', 'visual', 'entity', 'qa', 'observation'
    )),
    UNIQUE(video_id, scene_id, source)
);
CREATE INDEX idx_vec_video ON vec_metadata(video_id);
CREATE INDEX idx_vec_source ON vec_metadata(source);

-- ============================================================
-- FTS5 VIRTUAL TABLES + SYNC TRIGGERS
-- ============================================================

-- Scene transcripts FTS
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

INSERT INTO schema_version (version, description)
VALUES (1, 'Initial: strict UUID PKs, videos, scenes, entities, qa, observations, playlists, vec_metadata, FTS5');
```

### Progressive Enrichment: UPSERT + Directory Move

When yt-dlp returns richer metadata than what we initially had (e.g., URL-only processing that started with NULL channel), the system:

1. **UPSERTs the database** -- replaces NULL channel/playlist with real values
2. **Moves the directory** -- `shutil.move(old_path, new_path)` to match the enriched path
3. **Updates `cache_path`** in the videos table

```python
# In db/sync.py:
def enrich_video(video_id: str, metadata: dict, cache_base: Path) -> None:
    """Progressively enrich a video with new metadata from yt-dlp.

    If path components improve (NULL -> real channel), moves the
    directory and updates the database.
    """
    db = _get_db()
    if db is None:
        return

    repo = VideoRepository(db)
    existing = repo.get_by_video_id(video_id)
    if existing is None:
        return

    # Build new VideoPath from metadata
    new_path = VideoPath.from_metadata(video_id, metadata)
    old_cache_path = existing["cache_path"]
    new_cache_path = str(new_path.relative_path())

    # Check if path improved (NULL replaced with real data)
    if new_cache_path != old_cache_path:
        old_dir = cache_base / old_cache_path
        new_dir = cache_base / new_cache_path
        if old_dir.exists() and not new_dir.exists():
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_dir), str(new_dir))
            # Clean up empty parent dirs
            _cleanup_empty_parents(old_dir.parent, cache_base)

    # UPSERT all metadata (title, channel, playlist, etc.)
    repo.upsert(video_id, new_path, metadata)
```

This means a video's lifecycle looks like:
```
1. URL parsed -> VideoPath(youtube, None, None, dQw4w9WgXcQ)
   Dir created: youtube/no_channel/no_playlist/dQw4w9WgXcQ/

2. yt-dlp returns metadata -> channel_id=UCxxx
   UPSERT: channel = 'UCxxx' (was NULL)
   mv: youtube/no_channel/no_playlist/dQw4w9WgXcQ/ -> youtube/UCxxx/no_playlist/dQw4w9WgXcQ/

3. Later, video found in playlist -> playlist_id=PLxxx
   UPSERT: playlist = 'PLxxx' (was NULL)
   mv: youtube/UCxxx/no_playlist/dQw4w9WgXcQ/ -> youtube/UCxxx/PLxxx/dQw4w9WgXcQ/
```

### Why sqlite-vec: Product/UX Impact

Every pipeline step auto-embeds its text into sqlite-vec. Here's what that unlocks:

| Capability | Before (JSON + ChromaDB) | After (sqlite-vec in every pipeline step) |
|---|---|---|
| **"Find videos about auth"** | Keyword match only. Misses videos that say "login", "credentials", "SSO" | Semantic match across all vocabulary. "auth" finds "login flow" videos. |
| **Cross-video search** | Load every `graph.json` + substring match. O(n) file scans. | Single vec query across all videos. Sub-100ms regardless of library size. |
| **"What did I already ask about X?"** | Exact keyword search on Q&A history per-video | Semantic search across ALL Q&A history. "How does auth work?" finds "What's the login flow?" |
| **Find related moments** | FTS keyword -> falls back to separate ChromaDB (requires separate build step) | FTS + vec in one query, always available. No separate index build. |
| **Infrastructure** | ChromaDB sqlite file per video + Python server code | One `.db` file. Zero extra processes. Portable, inspectable, backupable. |
| **Index freshness** | Must explicitly run `build_scene_index()` to populate ChromaDB | Every `save_scenes_data()`, `record_qa()`, `save_entities()` auto-embeds. Always current. |
| **RAG for AI agents** | Agent must know which video to search, then use video-specific ChromaDB | Agent queries entire library: "What have I learned about distributed systems?" Returns ranked results across all videos. |

**The key insight**: By embedding at write-time (not query-time), the entire video library becomes a semantic knowledge base that gets richer with every interaction. The second time you ask about a topic, the system knows not just the keyword match but the *meaning* overlap across everything it's ever processed.

### Strict Validation Philosophy

**Pydantic models**: Use `model_config = ConfigDict(strict=True)`, required fields have no defaults, types are enforced. Fail on construction if data doesn't match.

**SQLite schema**: NOT NULL everywhere meaningful, CHECK constraints for enums and ranges, UNIQUE constraints to prevent duplicates. Fail on INSERT/UPDATE rather than storing bad data.

**Examples of strictness:**
- `domain` must match `CHECK(domain GLOB '[a-z]*')` -- lowercase alpha, always present, no default
- `channel` and `playlist` are nullable: `CHECK(channel IS NULL OR length(channel) > 0)` -- either NULL or non-empty, never empty string
- `scene_id` must be `CHECK(scene_id >= 0)` -- non-negative
- `entity_type` must be `CHECK(entity_type IN ('object','concept','person','technology','organization'))`
- UUIDs must match `CHECK(length(id) = 36)` -- proper UUID format
- `start_time < end_time` enforced at both Pydantic and SQLite levels
- Boolean columns: `CHECK(col IN (0, 1))` -- no truthy integers
- Score columns: `CHECK(score IS NULL OR (score >= 0 AND score <= 1))` -- bounded range
- `source_type`: `CHECK(source_type IN ('url', 'local'))` -- closed enum
- `transcript_source`: `CHECK(... IN ('youtube_subtitles', 'whisper', 'deepgram', 'openai', 'manual'))` -- closed enum
- `playlist_type`: `CHECK(... IN ('course', 'series', 'conference', 'collection'))` -- closed enum

### Module Structure

```
src/claudetube/db/
    __init__.py            # get_database(), close_database()
    connection.py          # Database class (WAL, foreign keys, thread-safe)
    migrations.py          # Migration runner
    migrations/
        __init__.py
        001_initial.sql    # Schema above
    sync.py                # Dual-write + progressive enrichment (UPSERT + mv)
    importer.py            # Auto-import existing JSON caches
    vec.py                 # sqlite-vec integration (standard, not optional)
    repos/
        __init__.py
        videos.py          # VideoRepository (CRUD + FTS + path lookup + enrichment)
        scenes.py          # SceneRepository (CRUD + FTS)
        entities.py        # EntityRepository (CRUD + cross-video queries)
        qa.py              # QARepository (CRUD + FTS)
        observations.py    # ObservationRepository
        playlists.py       # PlaylistRepository (CRUD + membership)
```

### Key Design: Dual-Write (Fire-and-Forget)

Every JSON write gets a companion SQLite sync wrapped in try/except. If SQLite fails, JSON is unaffected:

```python
# In save_state(), save_scenes_data(), save_objects(), etc:
try:
    from claudetube.db.sync import sync_state
    sync_state(state)
except Exception:
    pass  # JSON is authoritative; SQLite is best-effort
```

### Key Design: Video ID Resolution

With hierarchical paths, the MCP tools that accept bare `video_id` need to resolve it to a full path:

```python
# In VideoRepository:
def resolve_path(self, video_id: str) -> str | None:
    """Look up cache_path for a video_id. O(1) index lookup."""
    row = self.db.execute(
        "SELECT cache_path FROM videos WHERE video_id = ?", (video_id,)
    ).fetchone()
    return row["cache_path"] if row else None
```

Fallback chain when SQLite isn't available: check `{cache_dir}/{video_id}/state.json` (legacy flat), then glob `{cache_dir}/*/*/*/*/state.json` with matching video_id.

---

## Implementation Phases

### Phase 1: VideoPath Model + Domain Sanitization + Query Params
- Create `src/claudetube/models/video_path.py` with strict Pydantic validation
- Add `sanitize_domain()`, `VideoPath.from_url()`, `VideoPath.from_local()`
- Create `src/claudetube/parsing/params.py` with `extract_query_params()`, `parse_timestamp()`, and per-provider param maps (YouTube `t=`/`index=`/`ab_channel=`, Vimeo `h=`/`time=`, Twitch `t=`)
- Add named capture groups for `channel` and `playlist` to provider patterns in `providers.py`
- Integrate query param extraction into `VideoURL.parse()` -- merge into `provider_data` after regex match
- Add `domain`, `channel_id`, `playlist_id` fields to `VideoState`
- Unit tests for all path construction, sanitization, validation failures, timestamp parsing (seconds, `XhYmZs`, colon format), and query param extraction per provider
- **No changes to cache paths yet** -- just the models and parsing

### Phase 2: Core DB Module + Schema + sqlite-vec
- Create `src/claudetube/db/` package
- `connection.py`: Database class with WAL, foreign keys, thread-local connections
- `migrations.py`: Migration runner with versioned SQL files
- `migrations/001_initial.sql`: Full schema DDL with CHECK constraints (as specified above)
- `vec.py`: sqlite-vec integration (load extension, create vec0 table, embed + search)
- All 7 repository classes (videos, scenes, entities, qa, observations, playlists + vec)
- Add `sqlite-vec` to core dependencies in `pyproject.toml`
- Unit tests for each repo against in-memory SQLite, including vec operations and **CHECK constraint enforcement tests** (verify bad data is rejected)

### Phase 3: Dual-Write + Auto-Embed Pipeline
- Create `src/claudetube/db/sync.py` with progressive enrichment (UPSERT + `shutil.move`)
- Add fire-and-forget sync calls to every write path:
  - `cache/storage.py` -> `save_state()` -> sync video record
  - `cache/scenes.py` -> `save_scenes_data()` -> sync scenes + **embed transcripts**
  - `cache/entities.py` -> `save_objects()`, `save_concepts()` -> sync entities + **embed entity names**
  - `cache/memory.py` -> `_save_qa()` -> sync Q&A + **embed question+answer**
  - `cache/memory.py` -> `_save_observations()` -> sync observations + **embed observation content**
  - `cache/enrichment.py` -> `save_relevance_boosts()` -> sync boosts
  - `cache/knowledge_graph.py` -> `_save()` -> sync to entity_video_summary
- **Every pipeline step that produces text automatically embeds it into sqlite-vec**
- Embedding is async/fire-and-forget -- if embedder is unavailable, data is still stored in SQLite, just not vectorized
- Test: process a video end-to-end, verify JSON + SQLite + vec embeddings all populated

### Phase 4: Read-Side Queries + Auto-Import
- Create `src/claudetube/db/importer.py`
- Auto-import in `get_database()` on first DB creation (scan existing JSON, populate SQLite + vec)
- Replace file-scanning operations with SQL/FTS/vec queries:
  - `CacheManager.list_cached_videos()` -> SQL
  - `knowledge_graph.find_related_videos()` -> SQL on entity_video_summary
  - `enrichment.search_cached_qa()` -> FTS5 or vec semantic search
  - `search._search_transcript_text()` -> FTS5 fast path
  - `search._search_embedding()` -> sqlite-vec (replaces ChromaDB)
- **Semantic search is now unified**: one query searches both FTS5 (keyword) and sqlite-vec (semantic), merged by score
- All with graceful fallback to file-based if DB unavailable

### Phase 5: Hierarchical Path Activation
- Modify `CacheManager.get_cache_dir()` to use `VideoPath`
- Modify `process_video` to construct `VideoPath` from URL + yt-dlp metadata
- Implement progressive enrichment: when yt-dlp returns richer metadata, UPSERT + move directory
- Add video ID resolution via SQLite (bare `video_id` -> full path)
- Legacy path fallback for existing flat caches
- Add `claudetube migrate` CLI command for bulk migration (moves flat dirs to hierarchical)

---

## Files Summary

### New Files (17)
| File | Phase |
|------|-------|
| `src/claudetube/models/video_path.py` | 1 |
| `src/claudetube/parsing/params.py` | 1 |
| `src/claudetube/db/__init__.py` | 2 |
| `src/claudetube/db/connection.py` | 2 |
| `src/claudetube/db/migrations.py` | 2 |
| `src/claudetube/db/migrations/__init__.py` | 2 |
| `src/claudetube/db/migrations/001_initial.sql` | 2 |
| `src/claudetube/db/vec.py` | 2 |
| `src/claudetube/db/repos/__init__.py` | 2 |
| `src/claudetube/db/repos/videos.py` | 2 |
| `src/claudetube/db/repos/scenes.py` | 2 |
| `src/claudetube/db/repos/entities.py` | 2 |
| `src/claudetube/db/repos/qa.py` | 2 |
| `src/claudetube/db/repos/observations.py` | 2 |
| `src/claudetube/db/repos/playlists.py` | 2 |
| `src/claudetube/db/sync.py` | 3 |
| `src/claudetube/db/importer.py` | 4 |

### Modified Files
| File | Phase | Change |
|------|-------|--------|
| `src/claudetube/config/providers.py` | 1 | Add `channel`/`playlist` named captures |
| `src/claudetube/models/video_url.py` | 1 | Add `video_path` property, update `ID_GROUPS` |
| `src/claudetube/models/state.py` | 1 | Add `domain`, `channel_id`, `playlist_id` fields + strict validation |
| `src/claudetube/cache/storage.py` | 3 | Dual-write + embed in `save_state()` |
| `src/claudetube/cache/scenes.py` | 3 | Dual-write + embed transcripts in `save_scenes_data()` |
| `src/claudetube/cache/entities.py` | 3 | Dual-write + embed entities in `save_objects()`, `save_concepts()` |
| `src/claudetube/cache/memory.py` | 3 | Dual-write + embed in `_save_qa()`, `_save_observations()` |
| `src/claudetube/cache/enrichment.py` | 3,4 | Dual-write + FTS/vec read path |
| `src/claudetube/cache/knowledge_graph.py` | 3,4 | Dual-write + SQL read path (replaces graph.json) |
| `src/claudetube/cache/manager.py` | 4,5 | SQL `list_cached_videos()` + hierarchical `get_cache_dir()` |
| `src/claudetube/operations/processor.py` | 5 | Build `VideoPath`, progressive enrichment on yt-dlp response |
| `src/claudetube/analysis/search.py` | 4 | FTS5 + sqlite-vec unified search (replaces ChromaDB) |
| `src/claudetube/analysis/vector_index.py` | 4 | Rewrite to use sqlite-vec instead of ChromaDB |
| `pyproject.toml` | 2 | Add `sqlite-vec` to core deps, remove `chromadb` from core |

---

## Verification

1. **Phase 1**: `pytest tests/test_video_path.py` -- path construction, sanitization, strict validation failures (empty domain rejected, empty channel rejected, NULL channel accepted, bad UUID rejected)
2. **Phase 2**: `pytest tests/test_db_*.py` -- repos CRUD, FTS, migrations, vec operations, CHECK constraint enforcement (verify INSERT with bad data raises IntegrityError)
3. **Phase 3**: Process a video end-to-end, verify JSON + SQLite + vec embeddings all populated. Verify UPSERT replaces NULLs when yt-dlp data arrives.
4. **Phase 4**: Delete `claudetube.db`, restart, verify auto-import. Run `find_moments` -> verify FTS + vec semantic search returns results.
5. **Phase 5**: Process a new video, verify hierarchical path `youtube/UCxxx/PLxxx/vid123/`. Re-process with richer metadata, verify directory moved. Access by bare `video_id`, verify SQLite resolution works.

## Edge Cases

- **Concurrent access**: WAL mode + `busy_timeout=5000` handles writer contention
- **Corrupt DB**: All DB ops are fire-and-forget; delete DB and auto-import recreates it from JSON
- **Missing FTS5**: Detect at startup, skip FTS table creation, fall back to LIKE queries
- **Missing sqlite-vec**: If extension can't be loaded, vec operations degrade gracefully (no semantic search, FTS-only)
- **Legacy flat paths**: SQLite lookup -> flat path fallback -> glob scan. `claudetube migrate` for bulk move.
- **Directory move race**: Use atomic `shutil.move()` + update DB in single transaction. If move fails, DB stays at old path.
- **Video in multiple playlists**: `playlist_videos` is many-to-many. `cache_path` uses the *first* playlist. Subsequent associations add rows only.
- **Progressive enrichment**: When channel changes from NULL -> `UCxxx`, `no_channel` placeholder dirs cleaned up if empty
- **Strict validation**: Pydantic rejects bad data at construction time. SQLite CHECK constraints reject on INSERT. Both fail loudly -- no silent data corruption.
- **No yt-dlp metadata** (e.g., direct MP4 URL): domain from URL, channel and playlist stay NULL. Enriched later if metadata becomes available.
- **Embedder unavailable**: Data is still stored in SQLite (structured + FTS). Vec embeddings are best-effort. Semantic search degrades to FTS-only.
- **Empty string vs NULL**: Schema enforces `CHECK(col IS NULL OR length(col) > 0)` on nullable text fields. Empty strings are never stored -- only NULL or real values.
