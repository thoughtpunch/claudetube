[← Documentation](../README.md)

# PRD: Channel Surfing - Playlist-Aware Video Navigation

**Status:** Draft
**Author:** Claude
**Created:** 2026-02-04
**Last Updated:** 2026-02-04

---

## Executive Summary

Enable Claude to "channel surf" through video playlists and series—automatically navigating to the next video, tracking learning progress, and understanding content holistically across an entire playlist. This transforms claudetube from a single-video tool into a **continuous learning companion** that understands the knowledge-web of video series.

The goal: When a user says "I want to learn this topic," Claude can process a playlist, track what the user has watched, and proactively suggest or navigate to the most relevant next video—whether that's sequential, prerequisite-based, or topic-driven.

---

## Problem Statement

### Current State

claudetube can:
- Process individual videos
- Fetch playlist metadata (`get_playlist`)
- Build a knowledge graph linking videos by shared entities
- Get context for a video within a playlist (`get_playlist_video_context_tool`)

### What's Missing

| Human Behavior | Current Capability | Gap |
|----------------|-------------------|-----|
| "Watch the next video" | Can't trigger processing of next video | No `watch_next` action |
| Track what I've watched | No watch history | Can't resume learning |
| Know where I am in a course | Position is static metadata | No progress tracking |
| Jump to relevant chapter in another video | Cross-video chapter search missing | Chapters isolated per video |
| "Continue learning about X" | Manual URL entry | No topic-driven navigation |
| Understand how videos build on each other | Prerequisite data exists but unused | No learning path guidance |

### User Pain Points

1. **Manual URL management**: Users must copy/paste URLs for each video. No "watch next" workflow.

2. **Lost context**: When moving between videos in a series, Claude loses understanding of what was covered before.

3. **No curriculum guidance**: For courses, Claude can't say "you should watch video 3 before video 7 because it covers prerequisites."

4. **Redundant processing**: If a user asks "where was X covered?", Claude can't search across multiple videos efficiently.

5. **No bookmarks or progress**: Users can't say "where did I leave off in this course?"

---

## Goals

### Primary Goals

1. **Seamless video navigation**: Claude can automatically process the next video in a playlist without manual URL entry.

2. **Holistic playlist awareness**: Claude understands the entire playlist structure—not just the current video—and uses this to inform responses.

3. **Progress tracking**: Track which videos a user has watched and where they left off.

4. **Cross-video intelligence**: Search, navigate, and answer questions across all videos in a playlist.

5. **Learning path guidance**: For courses/series, recommend the optimal viewing order based on prerequisites and user goals.

### Non-Goals

- Building a full LMS (learning management system)
- User accounts or multi-user progress sync
- Automated playlist discovery (users provide playlist URLs)
- Real-time video playback coordination

---

## User Stories

### Story 1: Sequential Learning

> **As a** learner watching a tutorial series,
> **I want** Claude to automatically continue to the next video when I'm ready,
> **So that** I can learn continuously without managing URLs.

**Acceptance Criteria:**
- Claude can process the next video in a playlist with a simple command
- Context from previous videos is retained
- Claude acknowledges position ("This is video 5 of 12 in the Python course")

### Story 2: Topic-Driven Navigation

> **As a** learner with a specific question,
> **I want** Claude to find which video in a playlist covers that topic,
> **So that** I can jump directly to relevant content.

**Acceptance Criteria:**
- Cross-video search returns results with video + timestamp
- Claude can navigate to and process the identified video
- Results include context ("In video 7, the instructor covers this at 12:34")

### Story 3: Progress Tracking

> **As a** learner returning to a course,
> **I want** Claude to know where I left off,
> **So that** I can resume learning without manual tracking.

**Acceptance Criteria:**
- Claude tracks which videos have been watched
- Claude can report progress ("You've completed 3 of 12 videos")
- Claude can resume from last position

### Story 4: Prerequisite Awareness

> **As a** learner jumping into the middle of a course,
> **I want** Claude to warn me about prerequisites,
> **So that** I don't miss foundational concepts.

**Acceptance Criteria:**
- Claude identifies prerequisite videos
- Claude summarizes what I'd miss by skipping
- Claude can offer to process prerequisites first

### Story 5: Cross-Video Q&A

> **As a** learner with a question about a series,
> **I want** Claude to search across all videos I've watched,
> **So that** I get comprehensive answers referencing multiple sources.

**Acceptance Criteria:**
- Questions can search across all cached videos in a playlist
- Answers cite specific videos and timestamps
- Claude synthesizes information from multiple videos

---

## Proposed Solution

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CHANNEL SURFING ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │   Playlist   │───▶│  Learning Path  │───▶│    Navigation    │   │
│  │   Context    │    │     Engine      │    │      Actions     │   │
│  └──────────────┘    └─────────────────┘    └──────────────────┘   │
│         │                    │                       │              │
│         ▼                    ▼                       ▼              │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │   Progress   │◀──▶│  Cross-Video    │◀──▶│   Video Cache    │   │
│  │   Tracker    │    │     Search      │    │   (existing)     │   │
│  └──────────────┘    └─────────────────┘    └──────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Component 1: Playlist Context Manager

Maintains holistic understanding of a playlist across the session.

```python
# src/claudetube/navigation/playlist_context.py

@dataclass
class PlaylistContext:
    """Active playlist context for navigation."""

    playlist_id: str
    playlist_type: str  # course, series, conference, collection
    total_videos: int
    current_video_id: str | None
    current_position: int | None

    # Progress tracking
    watched_videos: list[str]  # video_ids that have been processed
    watch_times: dict[str, float]  # video_id -> last watched timestamp

    # Knowledge graph cache
    common_topics: list[str]
    shared_entities: list[dict]
    prerequisite_chain: dict[str, list[str]]  # video_id -> prerequisite video_ids

    # Navigation state
    next_video: str | None
    previous_video: str | None
    recommended_next: str | None  # May differ from sequential next

    def get_progress_summary(self) -> dict:
        """Return human-readable progress."""
        return {
            "completed": len(self.watched_videos),
            "total": self.total_videos,
            "percentage": len(self.watched_videos) / self.total_videos * 100,
            "current": self.current_position,
            "remaining": self.total_videos - len(self.watched_videos)
        }

    def get_prerequisites_for(self, video_id: str) -> list[dict]:
        """Get unwatched prerequisites for a video."""
        prereqs = self.prerequisite_chain.get(video_id, [])
        unwatched = [vid for vid in prereqs if vid not in self.watched_videos]
        return unwatched

    def mark_watched(self, video_id: str, timestamp: float = None):
        """Mark a video as watched."""
        if video_id not in self.watched_videos:
            self.watched_videos.append(video_id)
        self.watch_times[video_id] = timestamp or time.time()
```

### Component 2: Navigation Actions

New MCP tools for navigation.

```python
# New MCP Tools

@mcp.tool()
async def watch_next(
    playlist_id: str,
    auto_process: bool = True
) -> dict:
    """
    Navigate to and process the next video in a playlist.

    Args:
        playlist_id: The playlist to navigate within
        auto_process: Whether to automatically process the video (default: True)

    Returns:
        Video metadata and processing status
    """
    context = get_playlist_context(playlist_id)

    if not context.next_video:
        return {"status": "complete", "message": "You've reached the end of the playlist"}

    next_video_id = context.next_video

    # Check prerequisites (for courses)
    if context.playlist_type == "course":
        missing_prereqs = context.get_prerequisites_for(next_video_id)
        if missing_prereqs:
            return {
                "status": "prerequisites_missing",
                "next_video": next_video_id,
                "missing_prerequisites": missing_prereqs,
                "recommendation": f"Consider watching videos {missing_prereqs} first"
            }

    if auto_process:
        # Process the video
        result = await process_video_tool(next_video_id)
        context.mark_watched(next_video_id)
        context.current_video_id = next_video_id
        context.current_position += 1
        update_playlist_context(playlist_id, context)

        return {
            "status": "processed",
            "video": result,
            "progress": context.get_progress_summary(),
            "next_available": bool(context.next_video)
        }

    return {
        "status": "ready",
        "next_video": get_video_metadata(next_video_id),
        "progress": context.get_progress_summary()
    }


@mcp.tool()
async def watch_video_in_playlist(
    playlist_id: str,
    video_id: str | None = None,
    position: int | None = None,
    skip_prerequisites: bool = False
) -> dict:
    """
    Navigate to a specific video in a playlist by ID or position.

    Args:
        playlist_id: The playlist containing the video
        video_id: Video ID to navigate to (optional if position provided)
        position: 1-indexed position in playlist (optional if video_id provided)
        skip_prerequisites: Skip prerequisite warnings (default: False)
    """
    context = get_playlist_context(playlist_id)

    # Resolve video_id from position if needed
    if video_id is None and position is not None:
        video_id = get_video_at_position(playlist_id, position - 1)  # 0-indexed internally

    if video_id is None:
        return {"error": "Must provide either video_id or position"}

    # Check prerequisites unless skipped
    if not skip_prerequisites and context.playlist_type == "course":
        missing = context.get_prerequisites_for(video_id)
        unwatched = [v for v in missing if v not in context.watched_videos]
        if unwatched:
            return {
                "status": "prerequisites_warning",
                "video_id": video_id,
                "missing_prerequisites": unwatched,
                "hint": "Use skip_prerequisites=True to proceed anyway"
            }

    # Process the video
    result = await process_video_tool(video_id)
    context.mark_watched(video_id)
    context.current_video_id = video_id
    update_playlist_context(playlist_id, context)

    return {
        "status": "processed",
        "video": result,
        "progress": context.get_progress_summary()
    }


@mcp.tool()
async def get_playlist_progress(playlist_id: str) -> dict:
    """
    Get learning progress for a playlist.

    Returns watched videos, current position, and recommendations.
    """
    context = get_playlist_context(playlist_id)

    return {
        "playlist_id": playlist_id,
        "playlist_type": context.playlist_type,
        "progress": context.get_progress_summary(),
        "watched_videos": [
            get_video_summary(vid) for vid in context.watched_videos
        ],
        "current_video": context.current_video_id,
        "next_video": context.next_video,
        "recommended_next": context.recommended_next,
        "common_topics": context.common_topics[:10]
    }


@mcp.tool()
async def search_playlist(
    playlist_id: str,
    query: str,
    search_type: str = "all"  # all, watched, unwatched
) -> dict:
    """
    Search across all videos in a playlist for a topic or question.

    Args:
        playlist_id: Playlist to search within
        query: Natural language query
        search_type: Which videos to search (all, watched only, unwatched only)

    Returns:
        Results with video IDs, timestamps, and relevance scores
    """
    context = get_playlist_context(playlist_id)

    # Determine which videos to search
    if search_type == "watched":
        video_ids = context.watched_videos
    elif search_type == "unwatched":
        video_ids = [v for v in get_all_playlist_videos(playlist_id)
                     if v not in context.watched_videos]
    else:
        video_ids = get_all_playlist_videos(playlist_id)

    results = []
    for video_id in video_ids:
        if video_is_cached(video_id):
            # Search within this video
            moments = await find_moments_tool(video_id, query, top_k=3)
            for moment in moments:
                results.append({
                    "video_id": video_id,
                    "video_title": get_video_title(video_id),
                    "position_in_playlist": get_video_position(playlist_id, video_id),
                    "timestamp": moment["start"],
                    "end": moment["end"],
                    "relevance": moment["relevance"],
                    "preview": moment["preview"],
                    "is_watched": video_id in context.watched_videos
                })

    # Sort by relevance
    results.sort(key=lambda x: x["relevance"], reverse=True)

    return {
        "query": query,
        "total_results": len(results),
        "results": results[:10],
        "videos_searched": len(video_ids),
        "hint": "Use watch_video_in_playlist to navigate to a result"
    }


@mcp.tool()
async def find_chapter_across_playlist(
    playlist_id: str,
    topic: str
) -> dict:
    """
    Find chapters matching a topic across all videos in a playlist.

    Useful for navigating to specific concepts in a course.
    """
    videos = get_all_playlist_videos(playlist_id)
    matches = []

    for video_id in videos:
        chapters = get_video_chapters(video_id)
        for chapter in chapters:
            # Fuzzy match chapter title against topic
            if topic_matches_chapter(topic, chapter["title"]):
                matches.append({
                    "video_id": video_id,
                    "video_title": get_video_title(video_id),
                    "video_position": get_video_position(playlist_id, video_id),
                    "chapter_title": chapter["title"],
                    "start_time": chapter["start"],
                    "end_time": chapter["end"],
                    "match_score": calculate_match_score(topic, chapter["title"])
                })

    matches.sort(key=lambda x: x["match_score"], reverse=True)

    return {
        "topic": topic,
        "matches": matches,
        "hint": "Use watch_video_in_playlist with video_id to navigate"
    }


@mcp.tool()
async def get_learning_recommendations(
    playlist_id: str,
    goal: str | None = None
) -> dict:
    """
    Get recommendations for what to watch next based on progress and goals.

    Args:
        playlist_id: Playlist to get recommendations for
        goal: Optional learning goal to optimize for
    """
    context = get_playlist_context(playlist_id)

    recommendations = []

    # 1. Sequential next (if course/series)
    if context.next_video and context.playlist_type in ("course", "series"):
        recommendations.append({
            "type": "sequential",
            "video_id": context.next_video,
            "reason": "Next video in sequence",
            "priority": 1
        })

    # 2. Goal-driven (if goal provided)
    if goal:
        relevant = await search_playlist(playlist_id, goal, search_type="unwatched")
        if relevant["results"]:
            recommendations.append({
                "type": "goal_driven",
                "video_id": relevant["results"][0]["video_id"],
                "reason": f"Most relevant to your goal: '{goal}'",
                "priority": 2 if context.playlist_type == "collection" else 3
            })

    # 3. Fill prerequisites
    if context.playlist_type == "course" and context.next_video:
        missing = context.get_prerequisites_for(context.next_video)
        if missing:
            recommendations.append({
                "type": "prerequisite",
                "video_id": missing[0],
                "reason": "Prerequisite for next video",
                "priority": 0  # Highest priority
            })

    # 4. Related by entities
    if context.current_video_id:
        connections = get_video_connections_tool(context.current_video_id)
        for conn in connections.get("connections", [])[:2]:
            if conn["video_id"] not in context.watched_videos:
                recommendations.append({
                    "type": "related",
                    "video_id": conn["video_id"],
                    "reason": f"Shares topic: {conn['shared_entities'][0]}",
                    "priority": 4
                })

    # Sort by priority
    recommendations.sort(key=lambda x: x["priority"])

    return {
        "playlist_id": playlist_id,
        "progress": context.get_progress_summary(),
        "recommendations": recommendations,
        "top_recommendation": recommendations[0] if recommendations else None
    }
```

### Component 3: Progress Persistence

Store progress in the playlist cache directory.

```python
# src/claudetube/navigation/progress.py

@dataclass
class PlaylistProgress:
    """Persistent progress tracking for a playlist."""

    playlist_id: str
    watched_videos: list[str]
    watch_times: dict[str, float]  # video_id -> timestamp
    current_video: str | None
    bookmarks: list[dict]  # {video_id, timestamp, note}
    last_accessed: float

    @classmethod
    def load(cls, playlist_id: str) -> "PlaylistProgress":
        """Load progress from cache."""
        path = get_playlist_cache_path(playlist_id) / "progress.json"
        if path.exists():
            data = json.loads(path.read_text())
            return cls(**data)
        return cls(
            playlist_id=playlist_id,
            watched_videos=[],
            watch_times={},
            current_video=None,
            bookmarks=[],
            last_accessed=time.time()
        )

    def save(self):
        """Persist progress to cache."""
        path = get_playlist_cache_path(self.playlist_id) / "progress.json"
        self.last_accessed = time.time()
        path.write_text(json.dumps(asdict(self)))

    def add_bookmark(self, video_id: str, timestamp: float, note: str = ""):
        """Add a bookmark."""
        self.bookmarks.append({
            "video_id": video_id,
            "timestamp": timestamp,
            "note": note,
            "created_at": time.time()
        })
        self.save()


# Cache structure addition:
# ~/.claudetube/playlists/{playlist_id}/
#     ├── playlist.json           # Existing: playlist metadata
#     ├── knowledge_graph.json    # Existing: cross-video links
#     ├── progress.json           # NEW: user progress tracking
#     └── videos/                 # Existing: symlinks to video caches
```

### Component 4: Cross-Video Intelligence

Unified search and context across playlist.

```python
# src/claudetube/navigation/cross_video.py

class PlaylistIntelligence:
    """Cross-video search and synthesis for a playlist."""

    def __init__(self, playlist_id: str):
        self.playlist_id = playlist_id
        self.knowledge_graph = load_knowledge_graph(playlist_id)
        self.progress = PlaylistProgress.load(playlist_id)

    def search_all_transcripts(self, query: str) -> list[dict]:
        """FTS search across all video transcripts in playlist."""
        results = []
        for video_id in self.get_cached_videos():
            # Use existing FTS infrastructure
            matches = search_video_transcript_fts(video_id, query)
            for match in matches:
                match["video_id"] = video_id
                match["is_watched"] = video_id in self.progress.watched_videos
                results.append(match)
        return results

    def search_all_chapters(self, topic: str) -> list[dict]:
        """Find chapters matching topic across all videos."""
        results = []
        for video_id in self.get_all_videos():
            state = load_video_state(video_id)
            for chapter in state.get("chapters", []):
                score = semantic_similarity(topic, chapter["title"])
                if score > 0.5:
                    results.append({
                        "video_id": video_id,
                        "chapter": chapter,
                        "score": score
                    })
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def synthesize_topic_coverage(self, topic: str) -> dict:
        """Understand how a topic is covered across the playlist."""
        mentions = self.search_all_transcripts(topic)
        chapters = self.search_all_chapters(topic)

        # Group by video
        by_video = {}
        for mention in mentions:
            vid = mention["video_id"]
            if vid not in by_video:
                by_video[vid] = {
                    "video_id": vid,
                    "title": get_video_title(vid),
                    "mentions": [],
                    "chapters": []
                }
            by_video[vid]["mentions"].append(mention)

        for ch in chapters:
            vid = ch["video_id"]
            if vid in by_video:
                by_video[vid]["chapters"].append(ch)

        return {
            "topic": topic,
            "coverage": list(by_video.values()),
            "total_mentions": len(mentions),
            "videos_covering": len(by_video)
        }

    def get_context_for_current_video(self) -> dict:
        """Get relevant context from previously watched videos."""
        if not self.progress.current_video:
            return {}

        current = self.progress.current_video
        context = {
            "previous_videos_summary": [],
            "relevant_prior_content": [],
            "prerequisites_covered": []
        }

        # Summarize what was covered in watched videos
        for vid in self.progress.watched_videos:
            if vid == current:
                continue
            summary = get_video_summary(vid)
            context["previous_videos_summary"].append(summary)

        # Find relevant prior content (entities/topics shared with current)
        current_topics = get_video_topics(current)
        for vid in self.progress.watched_videos:
            if vid == current:
                continue
            shared = get_shared_topics(current, vid)
            if shared:
                context["relevant_prior_content"].append({
                    "video_id": vid,
                    "shared_topics": shared
                })

        return context
```

### Component 5: Skills Interface

New skills for user-friendly navigation.

```python
# Skills to add to mcp_server.py

CHANNEL_SURFING_SKILLS = """
## Channel Surfing Skills

- yt:next - Watch the next video in the current playlist
- yt:goto <position|title> - Jump to a specific video in playlist
- yt:progress - Show playlist progress and recommendations
- yt:search-playlist <query> - Search across all videos in playlist
- yt:chapters <topic> - Find chapters about a topic across playlist
- yt:bookmark [note] - Bookmark current position
- yt:resume - Resume from last position
"""
```

---

## Data Model Changes

### New: Progress Schema

```json
// ~/.claudetube/playlists/{playlist_id}/progress.json
{
  "playlist_id": "PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf",
  "watched_videos": ["abc123", "def456", "ghi789"],
  "watch_times": {
    "abc123": 1706832000,
    "def456": 1706918400,
    "ghi789": 1707004800
  },
  "current_video": "ghi789",
  "current_timestamp": 342.5,
  "bookmarks": [
    {
      "video_id": "def456",
      "timestamp": 125.0,
      "note": "Important explanation of async/await",
      "created_at": 1706918500
    }
  ],
  "last_accessed": 1707004900
}
```

### Enhanced: Knowledge Graph Schema

```json
// ~/.claudetube/playlists/{playlist_id}/knowledge_graph.json
{
  "playlist": { ... },
  "common_topics": [ ... ],
  "shared_entities": [ ... ],

  // NEW: Chapter index for cross-video search
  "chapter_index": [
    {
      "video_id": "abc123",
      "chapter_title": "Introduction to Functions",
      "start": 0,
      "end": 120,
      "keywords": ["functions", "def", "parameters"]
    }
  ],

  // NEW: Topic coverage map
  "topic_coverage": {
    "functions": ["abc123", "def456"],
    "async": ["ghi789", "jkl012"],
    "classes": ["mno345"]
  },

  "videos": [ ... ]
}
```

### Enhanced: Video State Schema

```json
// ~/.claudetube/cache/{video_id}/state.json
{
  // Existing fields...

  // NEW: Playlist membership (can be in multiple playlists)
  "playlists": [
    {
      "playlist_id": "PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf",
      "position": 3
    }
  ]
}
```

---

## User Experience

### Workflow 1: Start a Course

```
User: /yt:playlist https://youtube.com/playlist?list=PLxyz

Claude: I've loaded the playlist "Python Masterclass" (12 videos, ~8 hours total).

This appears to be a **course** with sequential content:
1. Introduction to Python (15:00)
2. Variables and Data Types (22:00)
3. Control Flow (28:00)
...

Would you like me to:
• Start from the beginning (/yt:next)
• Jump to a specific topic (/yt:search-playlist <topic>)
• See the full curriculum (/yt:progress)
```

### Workflow 2: Continue Learning

```
User: /yt:resume

Claude: Welcome back! You're 25% through "Python Masterclass":
• Completed: 3 of 12 videos
• Current: Video 4 - "Functions" (stopped at 5:42)

Would you like to:
• Continue "Functions" from 5:42
• Start the next video: "Modules and Packages"
• Search for a specific topic
```

### Workflow 3: Topic-Driven Navigation

```
User: Where does this course cover decorators?

Claude: "Decorators" is covered in 2 videos:

1. **Video 7 - "Advanced Functions"** (primary coverage)
   - Chapter "Understanding Decorators" at 8:45
   - Chapter "Writing Custom Decorators" at 15:20

2. **Video 10 - "Object-Oriented Python"** (mentions)
   - Used in context of class methods at 22:10

Note: Videos 4-6 are prerequisites for Video 7.
You've watched 3/6 prerequisites.

Would you like me to process Video 7 now? (Missing prerequisites: "Functions", "Closures", "Higher-Order Functions")
```

### Workflow 4: Cross-Video Q&A

```
User: How do I use *args and **kwargs? I remember it was covered somewhere.

Claude: Based on your watched videos, this was covered in:

**Video 4 - "Functions"** (you watched this)
- At 12:34: "The *args syntax allows passing variable positional arguments..."
- At 14:02: "Similarly, **kwargs captures keyword arguments as a dictionary..."

**Video 7 - "Advanced Functions"** (not yet watched)
- At 3:15: More advanced patterns combining both

Here's the summary from Video 4:
[Provides transcript excerpt and explanation]

Would you like me to show the advanced patterns from Video 7?
```

---

## Implementation Plan

### Phase 1: Core Navigation (MVP)

**Goal:** Basic "watch next" functionality with progress tracking.

| Task | Description | Effort |
|------|-------------|--------|
| `PlaylistContext` class | Session-level playlist state | S |
| `PlaylistProgress` persistence | Save/load progress.json | S |
| `watch_next` tool | Navigate to next video | M |
| `watch_video_in_playlist` tool | Navigate to specific video | M |
| `get_playlist_progress` tool | Report progress | S |
| Progress display | Show position/completion | S |

**Deliverables:**
- Users can say "watch next" and Claude processes the next video
- Progress is tracked across sessions
- Position is displayed ("Video 4 of 12")

### Phase 2: Cross-Video Search

**Goal:** Find content across all videos in a playlist.

| Task | Description | Effort |
|------|-------------|--------|
| Chapter indexing | Index all chapters when building knowledge graph | M |
| `search_playlist` tool | FTS across all transcripts | M |
| `find_chapter_across_playlist` tool | Topic → chapter mapping | M |
| Result ranking | Score by relevance and position | S |

**Deliverables:**
- Users can search for topics across entire playlist
- Chapter-level navigation across videos
- Results include video + timestamp

### Phase 3: Learning Intelligence

**Goal:** Smart recommendations and prerequisite awareness.

| Task | Description | Effort |
|------|-------------|--------|
| Prerequisite checking | Warn when skipping ahead | M |
| `get_learning_recommendations` tool | Suggest next video | M |
| Topic coverage analysis | Understand how topics span videos | L |
| Context bridging | Provide relevant context from prior videos | L |

**Deliverables:**
- Prerequisite warnings for courses
- Intelligent recommendations based on goals
- Cross-video context in responses

### Phase 4: Skills & Polish

**Goal:** User-friendly interface and bookmarks.

| Task | Description | Effort |
|------|-------------|--------|
| Navigation skills | /yt:next, /yt:goto, /yt:resume | M |
| Bookmark system | Save/retrieve bookmarks | S |
| Progress visualization | ASCII progress bar, completion % | S |
| Documentation | User guide, examples | S |

**Deliverables:**
- Intuitive slash commands
- Bookmark functionality
- Polished UX

---

## Technical Considerations

### Performance

1. **Lazy loading**: Don't process entire playlist upfront. Process videos on-demand.

2. **Incremental indexing**: Build chapter index incrementally as videos are processed.

3. **Cached searches**: Cache FTS results for repeated queries within a session.

### Storage

1. **Progress file size**: Progress.json should stay small (<1KB per playlist).

2. **Chapter index**: Store in knowledge_graph.json, built incrementally.

3. **No duplication**: Video data stays in video cache; playlists only store references.

### Concurrency

1. **Progress writes**: Use atomic writes (write to temp, rename) to prevent corruption.

2. **Multi-session**: Progress is per-playlist, not per-session. Multiple Claude sessions can update same progress.

### Edge Cases

1. **Playlist changes**: If playlist is updated on YouTube, detect and reconcile.

2. **Video removed**: Handle gracefully if a watched video is no longer in playlist.

3. **Multiple playlists**: Same video can appear in multiple playlists with different positions.

---

## Success Metrics

### Adoption

- Users invoke navigation tools (watch_next, search_playlist) regularly
- Multi-video sessions become common (3+ videos per session)

### Efficiency

- Time to navigate to next video < 5 seconds (excluding processing)
- Cross-playlist search returns results in < 2 seconds

### Learning Outcomes

- Users complete more videos in sequence (progress tracking)
- Users find relevant content faster (cross-video search)

### User Satisfaction

- Reduced manual URL management
- Coherent learning experience across videos
- Accurate prerequisite guidance

---

## Open Questions

1. **Progress sync**: Should progress sync across devices? (Currently local-only)

2. **Playlist updates**: How to handle when YouTube playlist is modified?

3. **Multiple users**: Should we support per-user progress on shared machines?

4. **Watch history export**: Should users be able to export/import progress?

5. **Automatic processing**: Should we pre-process the next video in background?

---

## Appendix: Skill Reference

### Navigation Skills

| Skill | Description | Example |
|-------|-------------|---------|
| `/yt:next` | Process next video in playlist | `/yt:next` |
| `/yt:goto` | Jump to specific video | `/yt:goto 5` or `/yt:goto "Functions"` |
| `/yt:progress` | Show progress and recommendations | `/yt:progress` |
| `/yt:resume` | Resume from last position | `/yt:resume` |

### Search Skills

| Skill | Description | Example |
|-------|-------------|---------|
| `/yt:search-playlist` | Search across all videos | `/yt:search-playlist async await` |
| `/yt:chapters` | Find chapters by topic | `/yt:chapters decorators` |

### Bookmark Skills

| Skill | Description | Example |
|-------|-------------|---------|
| `/yt:bookmark` | Bookmark current position | `/yt:bookmark "Important concept"` |
| `/yt:bookmarks` | List all bookmarks | `/yt:bookmarks` |

---

*Document version: 1.0*
*Last updated: February 2026*
