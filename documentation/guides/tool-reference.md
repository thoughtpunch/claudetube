[← Documentation](../README.md)

# Tool Reference

claudetube is an **MCP server** that exposes 40+ tools for video understanding. When registered with Claude Code, these tools are automatically available - just describe what you want and Claude will use the appropriate tools.

## Quick Reference

| Capability | MCP Tool |
|---|---|
| **Simplest Q&A** | `ask_video` |
| Download & transcribe | `process_video_tool` |
| Get frames (480p) | `get_frames` |
| Get HQ frames (1280p) | `get_hq_frames` |
| Transcribe audio | `transcribe_video` |
| Full transcript | `get_transcript` |
| List cached videos | `list_cached_videos` |
| Scene structure | `get_scenes` |
| Find moments | `find_moments_tool` |
| Active watch | `watch_video_tool` |
| Extract entities | `extract_entities_tool` |
| Visual transcripts | `generate_visual_transcripts` |
| Track people | `track_people_tool` |
| Deep analysis | `analyze_deep_tool` |
| Focus analysis | `analyze_focus_tool` |
| Analysis status | `get_analysis_status_tool` |
| Record Q&A | `record_qa_tool` |
| Search Q&A history | `search_qa_history_tool` |
| Scene context | `get_scene_context_tool` |
| Enrichment stats | `get_enrichment_stats_tool` |
| Get playlist | `get_playlist` |
| List playlists | `list_playlists` |
| Audio descriptions | `get_descriptions` |
| Describe moment | `describe_moment` |
| Accessible transcript | `get_accessible_transcript` |
| Has audio description | `has_audio_description` |
| Find related videos | `find_related_videos_tool` |
| Index to graph | `index_video_to_graph_tool` |
| Video connections | `get_video_connections_tool` |
| Knowledge graph stats | `get_knowledge_graph_stats_tool` |
| List providers | `list_providers_tool` |
| Narrative structure | `detect_narrative_structure_tool` |
| Get narrative structure | `get_narrative_structure_tool` |
| Detect changes | `detect_changes_tool` |
| Get changes | `get_changes_tool` |
| Major transitions | `get_major_transitions_tool` |
| Track code evolution | `track_code_evolution_tool` |
| Get code evolution | `get_code_evolution_tool` |
| Query code evolution | `query_code_evolution_tool` |
| Build knowledge graph | `build_knowledge_graph_tool` |
| Playlist video context | `get_playlist_video_context_tool` |
| Validate config | -- | `claudetube validate-config` | -- |

## MCP Tools (Detail)

### Video Processing

#### `process_video_tool`
Download and transcribe a video from any supported site.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | str | required | Video URL from any supported site |
| `whisper_model` | str | `"tiny"` | Whisper model size |
| `copy` | bool | `False` | Copy local files instead of symlinking |

Returns metadata, transcript (capped at 50k chars), and file paths.

**CLI equivalent:** `claudetube <URL> [--model MODEL] [--frames] [-o OUTPUT]`
**Slash command:** `/yt <url> [question]`

#### `transcribe_video`
Transcribe a video's audio. Returns cached transcript instantly if available.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id_or_url` | str | required | Video ID or URL |
| `whisper_model` | str | `"small"` | Whisper model size |
| `force` | bool | `False` | Re-transcribe even if cached |
| `provider` | str | `None` | Override transcription provider |

**Slash command:** `/yt:transcribe <video_id>`

#### `get_transcript`
Get full transcript for a cached video (no 50k char limit).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `format` | str | `"txt"` | `"txt"` (plain) or `"srt"` (with timestamps) |

**Slash command:** `/yt:transcript <video_id>`

### Frame Extraction

#### `get_frames`
Extract frames at a specific timestamp (480p, fast).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id_or_url` | str | required | Video ID or URL |
| `start_time` | float | required | Seconds from start |
| `duration` | float | `5.0` | Seconds to capture |
| `interval` | float | `1.0` | Seconds between frames |
| `quality` | str | `"lowest"` | Quality tier |

**Slash command:** `/yt:see <video_id> <timestamp>`

#### `get_hq_frames`
Extract HIGH QUALITY frames (1280p) for reading text, code, or diagrams.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id_or_url` | str | required | Video ID or URL |
| `start_time` | float | required | Seconds from start |
| `duration` | float | `5.0` | Seconds to capture |
| `interval` | float | `1.0` | Seconds between frames |
| `width` | int | `1280` | Frame width in pixels |

**Slash command:** `/yt:hq <video_id> <timestamp>`

### Scene Analysis

#### `get_scenes`
Get scene/chapter structure. Runs smart segmentation if not cached.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `force` | bool | `False` | Re-segment even if cached |
| `enrich` | bool | `False` | Generate visual descriptions |

**Slash command:** `/yt:scenes <video_id>`

#### `generate_visual_transcripts`
Generate visual descriptions for scenes using vision AI.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `scene_id` | int | `None` | Specific scene (or all) |
| `force` | bool | `False` | Regenerate existing |
| `provider` | str | `None` | Override vision provider |

#### `extract_entities_tool`
Extract entities (objects, people, text, concepts) from scenes.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `scene_id` | int | `None` | Specific scene (or all) |
| `force` | bool | `False` | Re-extract |
| `generate_visual` | bool | `True` | Generate visual transcripts first |
| `provider` | str | `None` | Override provider |

**CLI equivalent:** `claudetube extract-entities <video_id> [--scene-id ID] [--force] [--no-visual] [--provider NAME]`

#### `track_people_tool`
Track distinct people across scenes with timestamps.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `force` | bool | `False` | Re-track |
| `use_face_recognition` | bool | `False` | Use face_recognition library (expensive) |
| `provider` | str | `None` | Override provider |

### Deep Analysis

#### `analyze_deep_tool`
Full deep analysis: segmentation + visual transcripts + OCR + code detection + entity extraction.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `force` | bool | `False` | Re-analyze |

#### `analyze_focus_tool`
Exhaustive frame-by-frame analysis of a specific time range. Use sparingly.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `start_time` | float | required | Start of range (seconds) |
| `end_time` | float | required | End of range (seconds) |
| `force` | bool | `False` | Re-analyze |

#### `get_analysis_status_tool`
Show what analysis is cached: transcript coverage, visuals, technical content, entities.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |

### Search & Discovery

#### `find_moments_tool`
Find moments matching a natural language query. Tiered: text match then semantic fallback.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `query` | str | required | Natural language query |
| `top_k` | int | `5` | Max results |
| `semantic_weight` | float | `0.5` | Weight for semantic vs text matching |

**Slash command:** `/yt:find <video_id> <query>`

#### `watch_video_tool`
Active reasoning: checks Q&A cache, identifies relevant scenes, progressive examination, hypothesis building.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `question` | str | required | Question to answer |
| `max_iterations` | int | `15` | Max examination rounds |

**Slash command:** `/yt:watch <video_id> <question>`

### Progressive Learning

#### `record_qa_tool`
Record a Q&A interaction. Auto-identifies relevant scenes and boosts them for future queries.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `question` | str | required | The question asked |
| `answer` | str | required | The answer given |

#### `search_qa_history_tool`
Search cached Q&A pairs about a video.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `query` | str | required | Search query |

#### `get_scene_context_tool`
Get all learned context for a scene: observations, Q&A, relevance boost.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `scene_id` | int | required | Scene index |

#### `get_enrichment_stats_tool`
Show progressive learning stats: observation count, Q&A count, boosted scenes.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |

### Playlists

#### `get_playlist`
Extract playlist metadata without downloading videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `playlist_url` | str | required | Playlist URL |

#### `list_playlists`
List all cached playlists with inferred type (course/series/conference).

No parameters.

### Audio Descriptions (Accessibility)

#### `get_descriptions`
Get audio descriptions. Cheap-first: cache, yt-dlp AD track, scene compilation, AI generation.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id_or_url` | str | required | Video ID or URL |
| `format` | str | `"vtt"` | Output format |
| `regenerate` | bool | `False` | Force regeneration |

#### `describe_moment`
Generate audio description for a specific moment. Expensive on-demand operation.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id_or_url` | str | required | Video ID or URL |
| `timestamp` | float | required | Timestamp in seconds |
| `context` | str | `None` | Additional context |

#### `get_accessible_transcript`
Merge spoken transcript with audio descriptions tagged `[AD]`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id_or_url` | str | required | Video ID or URL |
| `format` | str | `"txt"` | `"txt"` or `"srt"` |

#### `has_audio_description`
Check if video has audio descriptions available.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id_or_url` | str | required | Video ID or URL |

### Knowledge Graph (Cross-Video)

#### `find_related_videos_tool`
Find videos related to a topic across all cached videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | str | required | Topic to search |

#### `index_video_to_graph_tool`
Add video to the cross-video knowledge graph.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `force` | bool | `False` | Re-index |

#### `get_video_connections_tool`
Find other videos sharing entities/concepts with a video.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |

#### `get_knowledge_graph_stats_tool`
Show knowledge graph stats: indexed videos, entities, concepts.

No parameters.

### Narrative Structure

#### `detect_narrative_structure_tool`
Detect narrative structure: sections (intro, main content, conclusion, transitions) and video type classification.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `force` | bool | `False` | Re-detect even if cached |

Results are cached in `structure/narrative.json`.

#### `get_narrative_structure_tool`
Get cached narrative structure for a video.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |

Run `detect_narrative_structure_tool` first if no structure is cached.

### Change Detection

#### `detect_changes_tool`
Detect changes between consecutive scenes: visual changes, topic shifts, content type transitions.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `force` | bool | `False` | Re-detect even if cached |

Results are cached in `structure/changes.json`.

#### `get_changes_tool`
Get cached scene change data for a video.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |

Run `detect_changes_tool` first if no changes are cached.

#### `get_major_transitions_tool`
Get only significant transitions (topic shifts, content type changes) for a quick structural overview.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |

### Code Evolution

#### `track_code_evolution_tool`
Track how code evolves across scenes. Analyzes code snapshots from entity data to identify additions, modifications, and refactoring. Best for coding tutorials and live coding videos.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `force` | bool | `False` | Re-track even if cached |

Results are cached in `entities/code_evolution.json`.

#### `get_code_evolution_tool`
Get cached code evolution data for a video.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |

Run `track_code_evolution_tool` first if no evolution data is cached.

#### `query_code_evolution_tool`
Query code evolution for a specific file, function, or code pattern.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `query` | str | required | Filename, function name, or code pattern |

### Playlist Knowledge Graph

#### `build_knowledge_graph_tool`
Build a cross-video knowledge graph for a playlist. Analyzes shared topics, entities, and prerequisite chains.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `playlist_id` | str | required | Playlist ID (from `get_playlist` results) |

Requires the playlist to have been fetched with `get_playlist` first.

#### `get_playlist_video_context_tool`
Get contextual information for a video within a playlist: position, related topics, prerequisites, shared entities.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_id` | str | required | Video ID |
| `playlist_id` | str | required | Playlist ID |

Requires a knowledge graph to have been built with `build_knowledge_graph_tool`.

### Utility

#### `list_cached_videos`
List all processed and cached videos.

No parameters.

**Slash command:** `/yt:list`

#### `list_providers_tool`
List available AI providers organized by capability.

No parameters.

## CLI Commands (Detail)

### `claudetube <URL>`
Main entry point. Downloads and transcribes a video.

```
claudetube <URL> [--frames] [--model MODEL] [--interval SECONDS] [-o OUTPUT]
```

| Flag | Description |
|---|---|
| `--frames` | Also extract frames |
| `--model MODEL` | Whisper model size (tiny/base/small/medium/large) |
| `--interval N` | Seconds between extracted frames |
| `-o OUTPUT` | Output directory |

### `claudetube extract-entities <video_id>`
Extract entities from a processed video.

```
claudetube extract-entities <video_id> [--scene-id ID] [--force] [--no-visual] [--provider NAME] [-o OUTPUT]
```

### `claudetube validate-config`
Validate provider configuration file.

```
claudetube validate-config [--skip-availability]
```

## Slash Commands (Detail)

Slash commands are Claude Code shortcuts defined in `commands/`. They use a cache-first bash strategy for speed, falling back to Python when needed.

| Command | Usage | Notes |
|---|---|---|
| `/yt` | `/yt <url> [question]` | Process video, optionally answer a question about it |
| `/yt:see` | `/yt:see <video_id> <timestamp>` | Quick frames with quality escalation on re-request |
| `/yt:hq` | `/yt:hq <video_id> <timestamp>` | HQ frames for reading text/code/diagrams |
| `/yt:transcript` | `/yt:transcript <video_id>` | Full transcript, no character limit |
| `/yt:transcribe` | `/yt:transcribe <video_id>` | Force re-transcription with Whisper |
| `/yt:list` | `/yt:list` | List cached videos (pure bash, instant) |
| `/yt:scenes` | `/yt:scenes <video_id>` | Scene/chapter structure |
| `/yt:find` | `/yt:find <video_id> <query>` | Natural language moment search |
| `/yt:watch` | `/yt:watch <video_id> <question>` | Active reasoning over video content |
| `/yt:deep` | `/yt:deep <video_id>` | Deep analysis: OCR, entities, code detection |
| `/yt:focus` | `/yt:focus <video_id> <start> <end>` | Exhaustive frame-by-frame section analysis |

## Python API

All MCP tools map to functions importable from Python:

```python
from claudetube import process_video, get_frames_at, get_hq_frames_at, VideoURL, extract_video_id
from claudetube.operations.playlist import extract_playlist_metadata, list_cached_playlists
from claudetube.operations.segmentation import segment_video_smart
from claudetube.operations.visual_transcript import generate_visual_transcript
from claudetube.operations.entity_extraction import extract_entities_for_video
from claudetube.operations.person_tracking import track_people
from claudetube.operations.analysis_depth import analyze_video, get_analysis_status, AnalysisDepth
from claudetube.operations.watch import watch_video
from claudetube.operations.narrative_structure import detect_narrative_structure
from claudetube.operations.change_detection import detect_scene_changes
from claudetube.operations.audio_description import compile_scene_descriptions
from claudetube.analysis.search import find_moments
from claudetube.analysis.embeddings import embed_scenes
from claudetube.cache.enrichment import record_qa_interaction, search_cached_qa, get_scene_context
from claudetube.cache.knowledge_graph import get_knowledge_graph, index_video_to_graph
from claudetube.cache.manager import CacheManager
from claudetube.providers.registry import list_available, list_all
from claudetube.operations.factory import OperationFactory, get_factory
```

## Provider Override

These tools accept a `provider` parameter to override configured defaults:

- `transcribe_video(provider="openai")`
- `generate_visual_transcripts(provider="anthropic")`
- `extract_entities_tool(provider="google")`
- `track_people_tool(provider="anthropic")`

Slash commands always use configured defaults.
