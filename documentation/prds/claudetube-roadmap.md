[← Documentation](../README.md)

# ClaudeTube Evolution Roadmap

## Vision: Agents That Truly Understand Video

The goal is not "AI that can answer questions about video" but **AI that watches, comprehends, and learns from video the way humans do**—building mental models, tracking entities over time, noticing what changed, remembering what it learned, and integrating new information with existing knowledge.

Current state: Claude gets a transcript and can request static frames. This is like reading a screenplay while occasionally glancing at photographs. We want Claude to *watch* the video.

---

## The Problem with Current Approaches

| What Humans Do | What Current Tools Do |
|----------------|----------------------|
| Perceive continuous motion and change | See isolated static frames |
| Build persistent mental models | Stateless—forgets between queries |
| Integrate audio + visual simultaneously | Process separately, correlate manually |
| Notice "something changed" | No change detection |
| Track people/objects across time | Each frame is independent |
| Learn progressively with each viewing | Same analysis every time |
| Connect to prior knowledge | No memory across videos |

The fundamental shift: **from frame retrieval to temporal comprehension**.

---

## Phase 1: Structural Understanding

**Goal:** Give the agent a semantic map of the video before it asks any questions.

### 1.0 Playlist Awareness & Cross-Video Context

Before analyzing individual videos, capture playlist-level context for semantic linking.

```python
def extract_playlist_context(playlist_url):
    """Extract playlist metadata for cross-video semantic linking."""
    
    playlist_info = yt_dlp.extract_info(playlist_url, download=False)
    
    return {
        "playlist_id": playlist_info["id"],
        "title": playlist_info["title"],
        "description": playlist_info.get("description", ""),
        "channel": playlist_info.get("channel", ""),
        "video_count": len(playlist_info["entries"]),
        "videos": [
            {
                "video_id": entry["id"],
                "title": entry["title"],
                "description": entry.get("description", ""),
                "duration": entry.get("duration"),
                "position": idx,
                "upload_date": entry.get("upload_date")
            }
            for idx, entry in enumerate(playlist_info["entries"])
        ],
        # Inferred metadata
        "inferred_type": classify_playlist_type(playlist_info),  # course, series, compilation, etc.
        "topic_keywords": extract_topic_keywords(playlist_info)
    }

def classify_playlist_type(playlist_info):
    """Infer playlist type from metadata patterns."""
    
    title = playlist_info["title"].lower()
    video_titles = [v["title"].lower() for v in playlist_info["entries"]]
    
    # Course/tutorial detection
    course_signals = ["course", "tutorial", "lesson", "part", "chapter", "module", "lecture"]
    if any(s in title for s in course_signals):
        return "course"
    
    # Numbered series detection (Part 1, Episode 2, etc.)
    import re
    numbered_pattern = r'(part|ep|episode|#)\s*\d+'
    if sum(1 for t in video_titles if re.search(numbered_pattern, t)) > len(video_titles) * 0.5:
        return "series"
    
    # Conference/event detection
    if any(s in title for s in ["conference", "summit", "meetup", "talks"]):
        return "conference"
    
    return "collection"

def build_playlist_knowledge_graph(playlist_context):
    """Build semantic links between videos in a playlist."""
    
    videos = playlist_context["videos"]
    
    # Extract shared entities across video titles/descriptions
    all_text = " ".join([v["title"] + " " + v["description"] for v in videos])
    common_entities = extract_named_entities(all_text)
    
    # Build prerequisite chain for courses
    if playlist_context["inferred_type"] == "course":
        for i, video in enumerate(videos):
            video["prerequisites"] = [videos[j]["video_id"] for j in range(i)]
            video["next"] = videos[i + 1]["video_id"] if i < len(videos) - 1 else None
    
    # Link by shared topics
    for video in videos:
        video["related"] = find_related_videos(video, videos, common_entities)
    
    return {
        "playlist": playlist_context,
        "common_entities": common_entities,
        "videos": videos
    }
```

**Cache structure for playlists:**
```
~/.claude/video_cache/
├── playlists/
│   └── {PLAYLIST_ID}/
│       ├── playlist.json       # Playlist metadata
│       ├── knowledge_graph.json # Cross-video links
│       └── videos/             # Symlinks to video caches
│           ├── VIDEO_ID_1 -> ../../VIDEO_ID_1/
│           └── VIDEO_ID_2 -> ../../VIDEO_ID_2/
```

**Why this matters:** 
- "In the previous video, we covered..." now has context
- Course progression is explicit (prerequisites, next steps)
- Shared terminology/entities are pre-identified
- Agent can answer "What video in this series covers X?"

**Commands:**
```
/yt:playlist <playlist_url>           # Analyze entire playlist
/yt:playlist:list <playlist_id>       # List videos with summaries
/yt:playlist:find <playlist_id> <query>  # Find video in playlist matching query
```

### 1.1 Cheap Boundary Detection (Transcript-First)

**Key Insight:** Visual scene detection (PySceneDetect) is computationally expensive and often misses *semantic* boundaries. For educational/tutorial content, **transcript analysis is cheaper and often better**.

#### YouTube Chapters (Free, Human-Curated)

```python
def extract_youtube_chapters(video_info):
    """Extract chapters—free structure from creator or YouTube."""
    
    # yt-dlp extracts chapters automatically
    if "chapters" in video_info and video_info["chapters"]:
        return [
            {
                "title": ch["title"],
                "start": ch["start_time"],
                "end": ch.get("end_time"),
                "source": "youtube_chapters",
                "confidence": 0.95  # Human-labeled = high confidence
            }
            for ch in video_info["chapters"]
        ]
    
    # Fallback: Parse from description (format: "0:00 Introduction")
    description = video_info.get("description", "")
    chapter_pattern = r"(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]?\s*(.+?)(?:\n|$)"
    
    return [
        {
            "title": title.strip(),
            "start": parse_timestamp(ts),
            "source": "description_parsed",
            "confidence": 0.9
        }
        for ts, title in re.findall(chapter_pattern, description)
    ]
```

#### Linguistic Transition Cues

Detect when the speaker signals a topic change.

```python
TRANSITION_PATTERNS = [
    # Explicit transitions
    r"\b(next|now)\s+(let'?s|we('ll)?|i('ll)?)\b",
    r"\b(moving on|let's move|let's talk about)\b",
    r"\bnow\s+(that|we|i)\b",
    r"\b(first|second|third|finally|lastly)\b",
    r"\bso\s+(now|let's|we)\b",
    r"\b(okay|alright|all right)\s*,?\s*(so|now|let's)\b",
    
    # Section markers
    r"\b(step\s+\d+|part\s+\d+)\b",
    r"\bin\s+this\s+(section|part|video)\b",
    r"\b(to\s+summarize|in\s+summary|to\s+recap)\b",
    
    # Topic shifts
    r"\b(another\s+(thing|way|approach|important))\b",
    r"\b(the\s+(next|last|final)\s+(thing|step|part))\b",
]

def detect_linguistic_boundaries(transcript_segments):
    """Find topic boundaries from speaker cues."""
    
    boundaries = []
    for seg in transcript_segments:
        text = seg["text"].lower()
        
        for pattern in TRANSITION_PATTERNS:
            if re.search(pattern, text):
                boundaries.append({
                    "timestamp": seg["start"],
                    "type": "linguistic_cue",
                    "trigger_text": seg["text"][:50],
                    "confidence": 0.7
                })
                break
    
    return boundaries
```

#### Pause Detection

Significant pauses (>2s) often indicate topic boundaries.

```python
def detect_pause_boundaries(transcript_segments):
    """Detect boundaries from significant pauses."""
    
    boundaries = []
    for i in range(1, len(transcript_segments)):
        gap = transcript_segments[i]["start"] - transcript_segments[i-1]["end"]
        
        if gap > 2.0:  # >2 second pause
            boundaries.append({
                "timestamp": transcript_segments[i]["start"],
                "type": "pause",
                "gap_seconds": gap,
                "confidence": 0.5 + min(gap / 10, 0.3)  # Longer pause = higher confidence
            })
    
    return boundaries
```

#### Vocabulary Shift Detection

Detect when the word distribution suddenly changes.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def detect_vocabulary_shifts(transcript_segments, window_seconds=30):
    """Find boundaries where vocabulary suddenly changes."""
    
    # Group transcript into time windows
    windows = group_into_windows(transcript_segments, window_seconds)
    
    if len(windows) < 2:
        return []
    
    # Compute TF-IDF vectors for each window
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf = vectorizer.fit_transform([w["text"] for w in windows])
    
    boundaries = []
    for i in range(1, len(windows)):
        similarity = cosine_similarity(tfidf[i-1], tfidf[i])[0][0]
        
        if similarity < 0.3:  # Low similarity = topic shift
            boundaries.append({
                "timestamp": windows[i]["start"],
                "type": "vocabulary_shift",
                "similarity": similarity,
                "confidence": 0.6,
                "keywords_before": get_top_terms(tfidf[i-1], vectorizer, 5),
                "keywords_after": get_top_terms(tfidf[i], vectorizer, 5)
            })
    
    return boundaries
```

#### Unified Cheap Boundary Detection

Combine all cheap signals before falling back to expensive visual analysis.

```python
def detect_boundaries_cheap(video_id, transcript, video_info):
    """Multi-signal boundary detection without visual analysis."""
    
    all_boundaries = []
    
    # 1. YouTube chapters (highest confidence—human labeled)
    chapters = extract_youtube_chapters(video_info)
    all_boundaries.extend(chapters)
    
    # 2. Linguistic transition cues
    linguistic = detect_linguistic_boundaries(transcript)
    all_boundaries.extend(linguistic)
    
    # 3. Significant pauses
    pauses = detect_pause_boundaries(transcript)
    all_boundaries.extend(pauses)
    
    # 4. Vocabulary shifts
    vocab_shifts = detect_vocabulary_shifts(transcript)
    all_boundaries.extend(vocab_shifts)
    
    # Merge nearby boundaries (within 5s), keeping highest confidence
    merged = merge_nearby_boundaries(all_boundaries, threshold=5.0)
    
    return sorted(merged, key=lambda x: x["timestamp"])

def merge_nearby_boundaries(boundaries, threshold=5.0):
    """Merge boundaries within threshold, boost confidence for agreement."""
    
    boundaries = sorted(boundaries, key=lambda x: x["timestamp"])
    merged = []
    
    for b in boundaries:
        if merged and b["timestamp"] - merged[-1]["timestamp"] < threshold:
            # Multiple signals agree = boost confidence
            merged[-1]["confidence"] = min(
                merged[-1]["confidence"] + 0.1, 
                0.95
            )
            merged[-1]["sources"] = merged[-1].get("sources", []) + [b["type"]]
        else:
            merged.append({**b, "sources": [b["type"]]})
    
    return merged
```

**Performance comparison** (30-minute tutorial video):
| Method | Time | Boundaries Found | Quality |
|--------|------|------------------|---------|
| Visual (PySceneDetect) | 60-120s | ~40 (visual cuts) | Good for visual-heavy |
| Transcript-based | 1-2s | ~15 (semantic) | Better for tutorials |
| Combined | 65-125s | ~50 (both) | Best coverage |

**Fallback strategy:** Only use visual scene detection when:
- Transcript unavailable or poor quality
- Content is primarily visual (music videos, demos without narration)
- Cheap methods find <3 boundaries in a 10+ minute video

### 1.2 Visual Scene Segmentation (Fallback)

When cheap methods don't provide enough boundaries, fall back to visual analysis.

```python
from scenedetect import detect, ContentDetector, AdaptiveDetector

def segment_video(video_path):
    """Detect semantic scene boundaries."""
    scenes = detect(video_path, AdaptiveDetector())
    return [
        {
            "scene_id": i,
            "start": scene[0].get_seconds(),
            "end": scene[1].get_seconds(),
            "duration": scene[1].get_seconds() - scene[0].get_seconds(),
            "keyframe_times": extract_keyframes(scene)
        }
        for i, scene in enumerate(scenes)
    ]
```

**Why this matters:** Scenes are the atomic units of video meaning. A scene = consistent visual context. Scene boundaries = something changed (cut, camera move, new content).

**Deliverable:** `scenes.json` in video cache with scene boundaries and metadata.

#### Unified Segmentation Strategy

The main entry point—tries cheap first, falls back to visual.

```python
def segment_video_smart(video_id, video_path, transcript, video_info):
    """Segment video using cheapest effective method."""
    
    video_duration = video_info.get("duration", 0)
    
    # Step 1: Try cheap methods first
    cheap_boundaries = detect_boundaries_cheap(video_id, transcript, video_info)
    
    # Step 2: Evaluate if we have enough coverage
    if cheap_boundaries:
        avg_segment = video_duration / (len(cheap_boundaries) + 1)
        has_chapters = any(b.get("source") == "youtube_chapters" for b in cheap_boundaries)
    else:
        avg_segment = video_duration
        has_chapters = False
    
    # Step 3: Decide if we need visual fallback
    need_visual = (
        len(cheap_boundaries) < 3 and video_duration > 300  # <3 boundaries in 5+ min
        or avg_segment > 300  # Segments longer than 5 min
        or not transcript  # No transcript available
    )
    
    # Step 4: If we have good chapters, skip visual entirely
    if has_chapters and len(cheap_boundaries) >= 5:
        need_visual = False
    
    # Step 5: Run visual if needed, merge with cheap
    if need_visual:
        visual_boundaries = detect_visual_boundaries(video_path)
        all_boundaries = merge_boundary_sources(cheap_boundaries, visual_boundaries)
    else:
        all_boundaries = cheap_boundaries
    
    # Step 6: Convert boundaries to segments
    segments = boundaries_to_segments(all_boundaries, video_duration)
    
    return {
        "segments": segments,
        "method": "visual" if need_visual else "transcript",
        "boundary_count": len(all_boundaries),
        "avg_segment_duration": video_duration / len(segments) if segments else video_duration
    }

def boundaries_to_segments(boundaries, video_duration):
    """Convert boundary timestamps to segment objects."""
    
    if not boundaries:
        return [{"start": 0, "end": video_duration, "segment_id": 0}]
    
    segments = []
    boundaries = sorted(boundaries, key=lambda x: x["timestamp"])
    
    # First segment: start to first boundary
    segments.append({
        "segment_id": 0,
        "start": 0,
        "end": boundaries[0]["timestamp"],
        "boundary_info": None
    })
    
    # Middle segments
    for i, b in enumerate(boundaries[:-1]):
        segments.append({
            "segment_id": i + 1,
            "start": b["timestamp"],
            "end": boundaries[i + 1]["timestamp"],
            "boundary_info": b
        })
    
    # Last segment: last boundary to end
    segments.append({
        "segment_id": len(boundaries),
        "start": boundaries[-1]["timestamp"],
        "end": video_duration,
        "boundary_info": boundaries[-1]
    })
    
    return segments
```

### 1.3 Transcript-Scene Alignment

Align the Whisper transcript to detected scenes, not arbitrary time chunks.

```python
def align_transcript_to_scenes(transcript_segments, scenes):
    """Map transcript segments to their containing scenes."""
    for scene in scenes:
        scene["transcript"] = [
            seg for seg in transcript_segments
            if seg["start"] >= scene["start"] and seg["end"] <= scene["end"]
        ]
        scene["transcript_text"] = " ".join(
            seg["text"] for seg in scene["transcript"]
        )
    return scenes
```

**Why this matters:** "What did they say when showing the diagram" becomes answerable. Audio and visual are now linked at the scene level.

### 1.4 Visual Transcripts (Dense Captioning)

For each scene, generate a natural language description of what's visually happening.

```python
def generate_visual_transcript(scene, keyframes):
    """Generate description of visual content for a scene."""
    # Option 1: Local model (Molmo 2, LLaVA, Qwen-VL)
    # Option 2: Claude API with vision
    
    prompt = """Describe what is visually happening in these frames from a video.
    Focus on: actions, objects, text on screen, people, settings, changes between frames.
    Be specific and factual. This will be used for search and retrieval."""
    
    response = vision_model.describe(keyframes, prompt)
    return {
        "scene_id": scene["scene_id"],
        "visual_description": response,
        "detected_elements": extract_elements(response)  # people, objects, text
    }
```

**Output structure:**
```json
{
  "scene_id": 4,
  "start": 45.2,
  "end": 62.8,
  "visual_description": "Person at desk typing on laptop. Screen shows VS Code with Python file open. Terminal visible in bottom panel showing pytest output with 3 failed tests. Person appears frustrated, rubs forehead.",
  "transcript_text": "So these tests are failing because we didn't handle the edge case...",
  "detected_elements": {
    "people": ["person_1"],
    "objects": ["laptop", "desk", "monitor"],
    "text_on_screen": ["pytest", "FAILED", "test_edge_case"],
    "applications": ["VS Code", "terminal"],
    "emotions": ["frustrated"]
  }
}
```

**Deliverable:** `visual_transcript.json` parallel to `audio.srt`.

### 1.5 Technical Content Extraction (OCR + Code Detection)

For technical videos, extract text and code visible on screen.

```python
def extract_technical_content(keyframe):
    """Extract code, terminal output, slide text from frames."""
    
    # OCR for general text
    text_blocks = ocr_engine.extract(keyframe)
    
    # Code detection (look for syntax patterns, IDE chrome)
    code_blocks = detect_code_regions(keyframe)
    
    # Classify content type
    content_type = classify_frame(keyframe)  # code, slides, terminal, diagram, talking_head
    
    return {
        "content_type": content_type,
        "text_blocks": text_blocks,
        "code_blocks": [
            {
                "language": detect_language(block),
                "content": block.text,
                "bounding_box": block.bbox
            }
            for block in code_blocks
        ]
    }
```

**Why this matters:** Technical tutorials are full of code that's spoken *about* but shown *visually*. The transcript says "and then we add the forward method" but the actual code is on screen. Without OCR, Claude is blind to the implementation.

**Deliverable:** `technical_content.json` with extracted code/text per scene.

---

## Phase 2: Semantic Search & Retrieval

**Goal:** Enable "find the part where..." queries without scanning the entire video.

### 2.1 Multimodal Scene Embeddings

Embed each scene as a single vector combining visual + audio + text.

```python
import voyageai  # or use local model

def embed_scene(scene):
    """Create unified embedding for a scene."""
    
    # Combine modalities into single input
    text_content = f"""
    Scene {scene['scene_id']} ({scene['start']:.1f}s - {scene['end']:.1f}s)
    
    AUDIO: {scene['transcript_text']}
    
    VISUAL: {scene['visual_description']}
    
    DETECTED TEXT: {scene.get('ocr_text', '')}
    """
    
    # Load keyframe images
    keyframe_images = [Image.open(kf) for kf in scene['keyframe_paths']]
    
    # Multimodal embedding (text + images → single vector)
    input_payload = [text_content] + keyframe_images
    
    embedding = voyage.multimodal_embed(
        inputs=[input_payload],
        model="voyage-multimodal-3",
        input_type="document"
    )
    
    return embedding[0]
```

**Alternative for local/offline:** Use CLIP for image embeddings + sentence-transformers for text, then concatenate/average. Less good but no API dependency.

### 2.2 Vector Index

Store embeddings in a searchable index.

```python
import chromadb  # or faiss, or qdrant

def build_scene_index(video_id, scenes, embeddings):
    """Create searchable index of video scenes."""
    
    client = chromadb.PersistentClient(path=f"~/.claude/video_cache/{video_id}/vectordb")
    collection = client.create_collection("scenes")
    
    collection.add(
        ids=[f"scene_{s['scene_id']}" for s in scenes],
        embeddings=embeddings,
        metadatas=[{
            "start": s["start"],
            "end": s["end"],
            "transcript": s["transcript_text"][:500],
            "visual": s["visual_description"][:500]
        } for s in scenes],
        documents=[s["visual_description"] for s in scenes]
    )
    
    return collection
```

### 2.3 Temporal Grounding Tool

New command: find moments matching a query.

```
/yt:find <video_id> <query>
```

```python
def find_moments(video_id, query, top_k=5):
    """Find scenes matching a natural language query."""
    
    # Embed query
    query_embedding = voyage.multimodal_embed(
        inputs=[[query]],
        model="voyage-multimodal-3",
        input_type="query"
    )[0]
    
    # Search index
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    return [
        {
            "scene_id": r["id"],
            "start": r["metadata"]["start"],
            "end": r["metadata"]["end"],
            "relevance": r["distance"],
            "preview": r["metadata"]["transcript"][:100]
        }
        for r in results
    ]
```

**Example:**
```
> /yt:find abc123 "when they fix the authentication bug"

Found 3 relevant moments:
1. [4:32-5:15] "...so the issue was we weren't validating the token expiry..."
2. [12:08-12:45] "...and that's how we patched the auth middleware..."
3. [8:22-8:50] "...let me show you the failing test for authentication..."
```

Claude can now jump directly to relevant sections instead of guessing timestamps.

---

## Phase 3: Temporal Reasoning

**Goal:** Understand *change over time*, not just static moments.

### 3.1 Entity Tracking

Track people, objects, and concepts across scenes.

```python
def track_entities(scenes):
    """Identify and track entities across the video."""
    
    entities = {
        "people": {},      # person_id → appearances
        "objects": {},     # object_id → appearances  
        "concepts": {},    # concept → mentions
        "code_evolution": {}  # function/file → changes over time
    }
    
    for scene in scenes:
        # People tracking (face recognition or description matching)
        for person in scene["detected_elements"].get("people", []):
            if person not in entities["people"]:
                entities["people"][person] = []
            entities["people"][person].append({
                "scene_id": scene["scene_id"],
                "timestamp": scene["start"],
                "action": extract_action(scene, person)
            })
        
        # Code evolution tracking
        for code_block in scene.get("code_blocks", []):
            file_or_function = identify_code_unit(code_block)
            if file_or_function:
                if file_or_function not in entities["code_evolution"]:
                    entities["code_evolution"][file_or_function] = []
                entities["code_evolution"][file_or_function].append({
                    "scene_id": scene["scene_id"],
                    "timestamp": scene["start"],
                    "content": code_block["content"],
                    "change_type": detect_change_type(code_block)  # added, modified, deleted
                })
    
    return entities
```

**Output:**
```json
{
  "code_evolution": {
    "auth_middleware.py": [
      {"timestamp": 120.5, "change_type": "shown", "content": "def validate(token):..."},
      {"timestamp": 245.2, "change_type": "modified", "content": "def validate(token, check_expiry=True):..."},
      {"timestamp": 380.1, "change_type": "modified", "content": "def validate(token, check_expiry=True):\n    if check_expiry and token.expired:..."}
    ]
  }
}
```

Now Claude can answer: "How did the auth middleware evolve during this video?"

### 3.2 Change Detection

Identify what changed between scenes.

```python
def detect_changes(scene_a, scene_b):
    """Identify what changed between two consecutive scenes."""
    
    changes = {
        "visual_changes": [],
        "audio_changes": [],
        "content_changes": []
    }
    
    # Visual diff (new objects, people left/entered, screen content changed)
    elements_a = set(scene_a["detected_elements"].get("objects", []))
    elements_b = set(scene_b["detected_elements"].get("objects", []))
    
    changes["visual_changes"] = {
        "added": list(elements_b - elements_a),
        "removed": list(elements_a - elements_b),
        "scene_type_change": scene_a.get("content_type") != scene_b.get("content_type")
    }
    
    # Topic shift detection (via embedding similarity)
    topic_shift = 1 - cosine_similarity(scene_a["embedding"], scene_b["embedding"])
    changes["topic_shift_score"] = topic_shift
    
    return changes
```

### 3.3 Narrative Structure Detection

Identify the high-level structure of the video.

```python
def detect_narrative_structure(scenes):
    """Identify introduction, main sections, conclusion, etc."""
    
    # Cluster scenes by topic
    embeddings = [s["embedding"] for s in scenes]
    clusters = cluster_embeddings(embeddings, method="agglomerative")
    
    # Detect structure patterns
    structure = {
        "sections": [],
        "type": None  # tutorial, lecture, demo, interview, etc.
    }
    
    # Label sections based on content
    for cluster_id in sorted(set(clusters)):
        cluster_scenes = [s for s, c in zip(scenes, clusters) if c == cluster_id]
        section_summary = summarize_scenes(cluster_scenes)
        
        structure["sections"].append({
            "cluster_id": cluster_id,
            "start": cluster_scenes[0]["start"],
            "end": cluster_scenes[-1]["end"],
            "summary": section_summary,
            "scene_ids": [s["scene_id"] for s in cluster_scenes]
        })
    
    # Detect video type
    structure["type"] = classify_video_type(scenes)
    
    return structure
```

**Output:**
```json
{
  "type": "coding_tutorial",
  "sections": [
    {"summary": "Introduction and problem statement", "start": 0, "end": 45},
    {"summary": "Environment setup and dependencies", "start": 45, "end": 120},
    {"summary": "Implementing the solution", "start": 120, "end": 480},
    {"summary": "Testing and debugging", "start": 480, "end": 620},
    {"summary": "Recap and next steps", "start": 620, "end": 680}
  ]
}
```

---

## Phase 4: Progressive Learning

**Goal:** The agent gets smarter about a video with each interaction, and retains learning across sessions.

### 4.1 Interaction-Driven Enrichment

When Claude examines frames or asks about a section, cache what it learns.

```python
class VideoMemory:
    """Persistent memory of what the agent has learned about a video."""
    
    def __init__(self, video_id):
        self.video_id = video_id
        self.memory_path = f"~/.claude/video_cache/{video_id}/agent_memory.json"
        self.memory = self.load()
    
    def record_observation(self, scene_id, observation_type, content):
        """Record something the agent noticed or concluded."""
        
        if scene_id not in self.memory["observations"]:
            self.memory["observations"][scene_id] = []
        
        self.memory["observations"][scene_id].append({
            "type": observation_type,  # "code_explanation", "person_identified", "error_found", etc.
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.save()
    
    def record_question_answer(self, question, answer, relevant_scenes):
        """Cache Q&A for future reference."""
        
        self.memory["qa_history"].append({
            "question": question,
            "answer": answer,
            "scenes": relevant_scenes,
            "timestamp": datetime.now().isoformat()
        })
        self.save()
    
    def get_context_for_scene(self, scene_id):
        """Retrieve everything the agent has learned about a scene."""
        
        return {
            "observations": self.memory["observations"].get(scene_id, []),
            "related_qa": [
                qa for qa in self.memory["qa_history"]
                if scene_id in qa["scenes"]
            ]
        }
```

**Example flow:**

1. User asks: "What's the bug in this video?"
2. Claude searches, finds scene 12, examines frames
3. Claude concludes: "The bug is an off-by-one error in the loop at 5:32"
4. This gets cached in `agent_memory.json`
5. Later, user asks: "How did they fix it?"
6. Claude already knows scene 12 has the bug → searches for "fix" near that context
7. Faster, more accurate answer

### 4.2 Multi-Pass Analysis

Support iterative deepening of understanding.

```python
class AnalysisDepth(Enum):
    QUICK = "quick"      # Scene detection + transcript only
    STANDARD = "standard"  # + visual transcripts
    DEEP = "deep"        # + OCR, code extraction, entity tracking
    EXHAUSTIVE = "exhaustive"  # + frame-by-frame for specific sections

def analyze_video(video_path, depth=AnalysisDepth.STANDARD, focus_sections=None):
    """Analyze video at specified depth, optionally focusing on sections."""
    
    # Always do quick pass first
    scenes = segment_video(video_path)
    transcript = transcribe_audio(video_path)
    align_transcript_to_scenes(transcript, scenes)
    
    if depth == AnalysisDepth.QUICK:
        return scenes
    
    # Standard: add visual transcripts
    for scene in scenes:
        if focus_sections and scene["scene_id"] not in focus_sections:
            continue
        scene["visual_description"] = generate_visual_transcript(scene)
    
    if depth == AnalysisDepth.STANDARD:
        return scenes
    
    # Deep: add technical content extraction
    for scene in scenes:
        if focus_sections and scene["scene_id"] not in focus_sections:
            continue
        scene["technical_content"] = extract_technical_content(scene)
        scene["entities"] = extract_entities(scene)
    
    if depth == AnalysisDepth.DEEP:
        return scenes
    
    # Exhaustive: frame-by-frame analysis for focus sections
    for scene in scenes:
        if scene["scene_id"] in (focus_sections or []):
            scene["frame_analysis"] = analyze_all_frames(scene)
    
    return scenes
```

**Command interface:**
```
/yt <url> <question>                    # Standard analysis
/yt:deep <video_id> <question>          # Deep analysis
/yt:focus <video_id> <timestamp> <question>  # Exhaustive analysis of specific section
```

### 4.3 Cross-Video Learning

Connect knowledge across multiple videos.

```python
class VideoKnowledgeGraph:
    """Track concepts, code patterns, and entities across videos."""
    
    def __init__(self):
        self.graph_path = "~/.claude/video_knowledge/graph.json"
        self.graph = self.load()
    
    def add_video(self, video_id, metadata, entities):
        """Index a video's concepts into the knowledge graph."""
        
        # Add video node
        self.graph["videos"][video_id] = {
            "title": metadata["title"],
            "topics": extract_topics(metadata),
            "indexed_at": datetime.now().isoformat()
        }
        
        # Link entities
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_key = normalize_entity(entity)
                if entity_key not in self.graph["entities"]:
                    self.graph["entities"][entity_key] = {
                        "type": entity_type,
                        "appearances": []
                    }
                self.graph["entities"][entity_key]["appearances"].append({
                    "video_id": video_id,
                    "context": entity.get("context", "")
                })
        
        self.save()
    
    def find_related_videos(self, query):
        """Find videos related to a concept or entity."""
        
        # Search entities
        matches = []
        for entity_key, entity_data in self.graph["entities"].items():
            if query.lower() in entity_key.lower():
                matches.extend(entity_data["appearances"])
        
        return matches
```

**Example:**
```
User: "I watched a video about React hooks last week. How does that relate to this Redux video?"

Claude: [Searches knowledge graph for "React hooks" and "Redux"]
        [Finds both videos, identifies shared concepts: state management, useEffect, dispatch]
        [Generates comparative analysis]
```

---

## Phase 5: Human-Like Video Comprehension

**Goal:** The agent watches video like a human expert would.

### 5.1 Active Watching Strategy

Instead of passive analysis, the agent actively decides where to focus.

```python
class ActiveVideoWatcher:
    """Agent that actively decides what to examine in a video."""
    
    def __init__(self, video_id, user_goal):
        self.video_id = video_id
        self.user_goal = user_goal
        self.examined = set()
        self.hypotheses = []
        self.confidence = {}
    
    def decide_next_action(self):
        """Decide what to examine next based on current understanding."""
        
        # If we have high-confidence answer, stop
        if self.has_sufficient_confidence():
            return {"action": "answer", "content": self.formulate_answer()}
        
        # Find most promising unexplored scene
        candidates = self.rank_unexplored_scenes()
        
        if not candidates:
            return {"action": "answer", "content": self.formulate_answer()}
        
        best_candidate = candidates[0]
        
        # Decide examination depth based on relevance
        if best_candidate["relevance"] > 0.8:
            return {"action": "examine_deep", "scene_id": best_candidate["scene_id"]}
        else:
            return {"action": "examine_quick", "scene_id": best_candidate["scene_id"]}
    
    def update_understanding(self, scene_id, findings):
        """Update hypotheses based on new observations."""
        
        self.examined.add(scene_id)
        
        # Update or create hypotheses
        for finding in findings:
            matching_hypothesis = self.find_matching_hypothesis(finding)
            if matching_hypothesis:
                matching_hypothesis["evidence"].append(finding)
                matching_hypothesis["confidence"] = self.calculate_confidence(matching_hypothesis)
            else:
                self.hypotheses.append({
                    "claim": finding["claim"],
                    "evidence": [finding],
                    "confidence": finding["initial_confidence"]
                })
    
    def formulate_answer(self):
        """Generate answer from current understanding."""
        
        # Rank hypotheses
        ranked = sorted(self.hypotheses, key=lambda h: h["confidence"], reverse=True)
        
        # Generate response with evidence
        return {
            "main_answer": ranked[0]["claim"] if ranked else "Unable to determine",
            "confidence": ranked[0]["confidence"] if ranked else 0,
            "evidence": [
                {"timestamp": e["timestamp"], "observation": e["description"]}
                for e in ranked[0]["evidence"]
            ] if ranked else [],
            "alternative_interpretations": [h["claim"] for h in ranked[1:3]]
        }
```

### 5.2 Attention Modeling

Model where a human expert would focus attention.

```python
def calculate_attention_priority(scene, user_goal, video_type):
    """Calculate how much attention a scene deserves."""
    
    factors = {
        "relevance_to_goal": semantic_similarity(scene["content"], user_goal),
        "information_density": estimate_information_density(scene),
        "novelty": 1 - similarity_to_previous_scenes(scene),
        "visual_salience": detect_visual_salience(scene),  # text on screen, diagrams, etc.
        "audio_emphasis": detect_audio_emphasis(scene),    # "importantly", "key point", etc.
        "structural_importance": get_structural_weight(scene, video_type)  # intro/conclusion weight
    }
    
    # Video-type-specific weighting
    if video_type == "coding_tutorial":
        weights = {"relevance_to_goal": 0.3, "information_density": 0.25, 
                   "visual_salience": 0.25, "novelty": 0.1, "audio_emphasis": 0.1}
    elif video_type == "lecture":
        weights = {"relevance_to_goal": 0.25, "audio_emphasis": 0.25,
                   "structural_importance": 0.2, "novelty": 0.15, "information_density": 0.15}
    else:
        weights = {k: 1/len(factors) for k in factors}
    
    priority = sum(factors[k] * weights[k] for k in factors)
    return priority
```

### 5.3 Comprehension Verification

The agent checks its own understanding.

```python
def verify_comprehension(video_understanding, verification_questions=None):
    """Verify the agent actually understood the video."""
    
    if verification_questions is None:
        # Generate self-test questions
        verification_questions = [
            f"What is the main topic of this video?",
            f"What problem does scene {random_scene} address?",
            f"How does the content at {random_timestamp} relate to the overall topic?",
            f"What would someone learn from watching this video?",
        ]
    
    results = []
    for question in verification_questions:
        # Answer without re-examining video (from understanding only)
        answer = generate_answer_from_understanding(video_understanding, question)
        
        # Verify answer against video content
        verification = verify_answer_against_content(answer, question, video_understanding)
        
        results.append({
            "question": question,
            "answer": answer,
            "verified": verification["correct"],
            "confidence": verification["confidence"]
        })
    
    comprehension_score = sum(r["verified"] for r in results) / len(results)
    
    return {
        "score": comprehension_score,
        "details": results,
        "gaps": [r["question"] for r in results if not r["verified"]]
    }
```

---

## Implementation Phases

### MVP (2-3 weeks)
- [ ] Scene detection with PySceneDetect
- [ ] Transcript-scene alignment
- [ ] Basic visual transcripts (1 keyframe per scene → Claude vision)
- [ ] Updated cache structure
- [ ] `/yt:scenes` command to dump scene list

### V1 (4-6 weeks)
- [ ] Full visual transcript generation
- [ ] OCR/code extraction for technical content
- [ ] Scene embeddings (voyage-multimodal-3 or local CLIP)
- [ ] Vector search with ChromaDB
- [ ] `/yt:find` command for temporal grounding

### V2 (6-10 weeks)
- [ ] Entity tracking across scenes
- [ ] Change detection between scenes
- [ ] Narrative structure detection
- [ ] Progressive analysis depth (quick/standard/deep)
- [ ] Agent memory per video

### V3 (10+ weeks)
- [ ] Cross-video knowledge graph
- [ ] Active watching strategy
- [ ] Attention modeling
- [ ] Comprehension verification
- [ ] Multi-video Q&A

---

## Technical Requirements

### New Dependencies

```txt
# Scene detection
scenedetect>=0.6

# OCR
pytesseract>=0.3
easyocr>=1.7  # better for code/technical text

# Vector search
chromadb>=0.4
# OR faiss-cpu>=1.7

# Embeddings (choose one)
voyageai>=0.2  # API-based, best quality
sentence-transformers>=2.2  # local, good quality
open-clip-torch>=2.24  # local, for images

# Optional: local vision models
transformers>=4.36
accelerate>=0.25
```

### Cache Structure (Updated)

```
~/.claude/video_cache/
└── {VIDEO_ID}/
    ├── state.json              # Existing metadata
    ├── audio.mp3               # Existing
    ├── audio.srt               # Existing
    ├── audio.txt               # Existing
    │
    ├── scenes/
    │   ├── scenes.json         # Scene boundaries and metadata
    │   ├── scene_001/
    │   │   ├── keyframes/      # Representative frames
    │   │   ├── visual.json     # Visual transcript
    │   │   └── technical.json  # OCR, code extraction
    │   └── ...
    │
    ├── embeddings/
    │   ├── scene_embeddings.npy
    │   └── chroma/             # Vector DB
    │
    ├── entities/
    │   ├── people.json
    │   ├── objects.json
    │   └── code_evolution.json
    │
    ├── structure/
    │   ├── narrative.json      # Sections, video type
    │   └── changes.json        # Scene-to-scene changes
    │
    └── memory/
        ├── observations.json   # Agent's cached insights
        └── qa_history.json     # Past Q&A pairs
```

---

## Success Metrics

### Accuracy
- Can correctly answer "when does X happen?" with timestamp within 10 seconds
- Can correctly identify main topics/sections of a video
- Can track entity changes over time (e.g., "how did the code change?")

### Efficiency  
- Time to first answer < 30s for standard queries (after initial processing)
- Reduces frame examination by 80%+ vs. current approach
- Progressive analysis means second query is faster than first

### Comprehension Quality
- Agent can generate accurate video summary without hallucination
- Agent can answer follow-up questions without re-examining video
- Agent can connect information across multiple videos

### Human-Likeness
- Agent focuses on relevant sections, not random sampling
- Agent builds on previous observations (memory)
- Agent can explain its reasoning with evidence (timestamps, quotes)

---

## Open Questions

1. **Local vs. API for vision models?** Molmo 2 is open and excellent, but requires GPU. Claude API is convenient but adds latency/cost. Hybrid approach?

2. **Embedding model choice?** voyage-multimodal-3 is best but API-only. CLIP+sentence-transformers is local but lower quality. How much does quality matter for retrieval?

3. **Granularity of visual transcripts?** One per scene? One per N seconds? Adaptive based on content density?

4. **Cross-video knowledge persistence?** Should the knowledge graph be opt-in? Privacy implications of persistent video memory?

5. **Real-time vs. batch processing?** Current approach is batch (process whole video upfront). Could we stream-process for very long videos?

---

## Research & Further Reading

This section covers cutting-edge research from 2024-2026 on video understanding, multimodal models, and related techniques. Organized by topic for agents and developers wanting to go deeper.

---

### Surveys & Overviews

| Paper | Description | Link |
|-------|-------------|------|
| **Video Understanding with LLMs: A Survey** (IEEE TCSVT 2025) | Comprehensive survey covering Vid-LLM architectures, training strategies, tasks, datasets, and benchmarks | [GitHub](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding) |
| **Video Temporal Grounding with MLLMs Survey** (Aug 2025) | First comprehensive survey on VTG-MLLMs—covers functional roles, training paradigms, and video feature processing | [arXiv:2508.10922](https://arxiv.org/abs/2508.10922) |
| **Dense Video Captioning Survey** (April 2025) | Techniques, datasets, evaluation protocols for generating temporally precise descriptions | [MDPI](https://www.mdpi.com/2076-3417/15/9/4990) |
| **MME-Survey** | Comprehensive survey on evaluation of Multimodal LLMs (MME, MMBench, LLaVA teams) | [GitHub](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) |

---

### Long Video Understanding (Hour-Scale)

The frontier problem: understanding videos that are minutes to hours long, not seconds.

| Paper/Model | Key Innovation | Link |
|-------------|----------------|------|
| **LongVILA** (NVIDIA) | 5-stage training pipeline with Multi-Modal Sequence Parallelism; processes 2048+ frames; 99.8% needle-in-haystack at 6K frames | [arXiv:2408.10188](https://arxiv.org/abs/2408.10188) |
| **LongVILA-R1** | Reinforcement learning for long video reasoning; Multi-modal RL Sequence Parallelism for 512+ frame training | [arXiv:2507.07966](https://arxiv.org/pdf/2507.07966) |
| **Long-VITA** | Scales to 1M tokens with Ring Attention; maintains short-context accuracy while enabling hour-length video | [arXiv:2502.05177](https://arxiv.org/abs/2502.05177) |
| **InternVideo2.5** | Empowering Video MLLMs with long and rich context modeling | [arXiv:2501.12386](https://arxiv.org/abs/2501.12386) |
| **STORM** (NVIDIA) | Mamba State Space Model for temporal encoding between vision encoder and LLM; 8× compute reduction, 2.4-2.9× latency reduction | [arXiv:2503.04130](https://arxiv.org/abs/2503.04130) |
| **LongVLM** | Hierarchical token merging for local+global features; decomposes long videos into short segments | [arXiv:2404.03384](https://arxiv.org/abs/2404.03384) |
| **Temporal Dynamic Context** | Query-based Transformer aggregating video/audio/text into temporal context tokens; chain-of-thought for extremely long videos | [arXiv:2504.10443](https://arxiv.org/abs/2504.10443) |
| **APVR** | Adaptive Pivot Visual Information Retrieval for hour-level video; processes 1024+ frames | [arXiv:2506.04953](https://arxiv.org/abs/2506.04953) |

---

### Temporal Grounding & Localization

The "find when X happens" problem—localizing moments in video from natural language queries.

| Paper/Model | Key Innovation | Link |
|-------------|----------------|------|
| **UniTime** | Universal video temporal grounding with adaptive frame scaling; interleaves timestamp tokens with video tokens | [arXiv:2506.18883](https://arxiv.org/abs/2506.18883) |
| **SpaceVLLM** | Spatio-temporal video grounding (where AND when); Uni-STG dataset with 480K instances | [arXiv:2503.13983](https://arxiv.org/abs/2503.13983) |
| **ED-VTG** (Enrich and Detect) | Two-stage: enrich query with missing details, then localize; MIL training for hallucination mitigation | [arXiv:2510.17023](https://arxiv.org/abs/2510.17023) |
| **StepVTG** | Step-by-step reasoning for long video (>10 min) temporal grounding; explainable reasoning steps | [Tsinghua](https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2025_Localizing%20Step-by-Step%20Multimodal%20Long%20Video%20Temporal%20Grounding%20with%20LLM.pdf) |
| **Grounded-VideoLLM** (EMNLP 2025) | Temporal tokens in unified embedding space with LLM; segment-wise encoding (spatial + temporal streams) | [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.50.pdf) |
| **Training-Free VTG** | LLM reasons about sub-events and relationships; VLM measures vision-text similarity | [arXiv:2408.16219](https://arxiv.org/abs/2408.16219) |

---

### Video Reasoning & Chain-of-Thought

Teaching models to *think* about video, not just perceive it.

| Paper/Model | Key Innovation | Link |
|-------------|----------------|------|
| **Chain-of-Frames** | Frame-aware reasoning; identifies which frames support conclusions | [arXiv:2506.00318](https://arxiv.org/abs/2506.00318) |
| **VideoCoT** | Active annotation tool for generating reasoning explanations on visual content | Referenced in Chain-of-Frames |
| **Chain-of-Shot** | Prompting strategy for long-form videos via key frame selection | Referenced in Chain-of-Frames |
| **Video-of-Thought** | Step-by-step video reasoning from perception to cognition (ICML 2024) | Referenced in surveys |
| **Visual Sketchpad** | Sketching as visual chain-of-thought for multimodal LLMs | Referenced in Chain-of-Frames |
| **Graph-VideoAgent** | LLM agent with dynamic graph memory tracking entity relations across time; iterative reasoning and self-reflection | [arXiv:2501.15953](https://arxiv.org/abs/2501.15953) |

---

### Visual Encoders & Representations

The foundation: how to turn pixels into tokens that models can reason over.

| Model | Key Innovation | Link |
|-------|----------------|------|
| **SigLIP 2** (Google, Feb 2025) | Multilingual vision-language encoder; captioning-based pretraining + self-distillation + masked prediction; improved localization and dense features | [arXiv:2502.14786](https://arxiv.org/abs/2502.14786) |
| **Molmo 2** (AI2, Dec 2025) | State-of-the-art for video understanding, pointing, and tracking; 128 frames at ≤2fps; new video-centric corpus of 9M examples | [AI2 Blog](https://allenai.org/blog/molmo2) |
| **VideoMAE** | Masked autoencoder for video; used by InternVideo as video encoder | Referenced in surveys |
| **Janus** | Decouples visual encoding for understanding (SigLIP) vs generation (VQ tokenizer) | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Janus_Decoupling_Visual_Encoding_for_Unified_Multimodal_Understanding_and_Generation_CVPR_2025_paper.pdf) |

---

### World Models & Physical Understanding

Beyond perception: models that understand how the world *works*.

| Model | Key Innovation | Link |
|-------|----------------|------|
| **Genie 3** (DeepMind, Aug 2025) | Foundation world model; generates interactive 3D environments at 720p/24fps for minutes; emergent consistency; promptable world events | [DeepMind Blog](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) |
| **Genie 2** (DeepMind, Dec 2024) | Trained on video data to simulate physically consistent worlds; emergent physics understanding | [DeepMind Blog](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) |
| **D4RT** (DeepMind, Jan 2026) | Dynamic 4D Reconstruction and Tracking; unified framework for point tracking, depth, pose from video | [DeepMind Blog](https://deepmind.google/blog/d4rt-teaching-ai-to-see-the-world-in-four-dimensions/) |
| **Runway GWM-1** | First commercial world model from video generation startup | Industry announcement |
| **World Labs Marble** | Fei-Fei Li's first commercial world model | Industry announcement |

---

### Benchmarks & Evaluation

How we measure progress.

| Benchmark | Focus | Link |
|-----------|-------|------|
| **Video-MME** (CVPR 2025) | Comprehensive evaluation for multi-modal LLMs in video analysis; used by GPT-5, Gemini 2.5/3 | [GitHub](https://github.com/MME-Benchmarks/Video-MME) |
| **LongVideoBench** | Long-context interleaved video-language understanding (NeurIPS 2024) | Referenced in surveys |
| **MLVU** | Multi-task long video understanding | Referenced in STORM paper |
| **MM-LVTG** | Long video temporal grounding with multimodal inputs | StepVTG paper |
| **Charades-STA** | Temporal sentence grounding benchmark | Standard benchmark |
| **ActivityNet Captions** | Dense video captioning benchmark | Standard benchmark |
| **YouCook2** | Cooking video understanding and captioning | Standard benchmark |

---

### Architectures & Techniques

Key architectural innovations driving progress.

| Technique | Description | Used By |
|-----------|-------------|---------|
| **Sequence Parallelism** | Distribute long sequences across GPUs for memory-intensive training | LongVILA, Long-VITA |
| **Ring Attention** | Efficient attention for very long sequences | Long-VITA |
| **Mamba State Space Models** | Linear-complexity alternative to transformers for temporal modeling | STORM |
| **Token Compression** | Reduce visual tokens via pooling, merging, or learned compression | LongVLM, STORM, many others |
| **Perceiver Resampler** | Reduce token counts while preserving information | Many VLMs |
| **Timestamp Tokens** | Interleave special time tokens with visual tokens for temporal grounding | UniTime, Grounded-VideoLLM |
| **Multi-stage Training** | Progressive curriculum (alignment → pretraining → SFT → extension → long SFT) | LongVILA, most large models |

---

### Curated Lists & Resources

| Resource | Description | Link |
|----------|-------------|------|
| **Awesome-LLMs-for-Video-Understanding** | Latest papers, codes, datasets on Vid-LLMs | [GitHub](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding) |
| **Awesome-Multimodal-Large-Language-Models** | Comprehensive MLLM resource list | [GitHub](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) |
| **CVPR 2025 Papers** | All CVPR 2025 papers including video understanding | [GitHub](https://github.com/52CV/CVPR-2025-Papers) |
| **From Video Generation to World Model** (CVPR 2025 Tutorial) | Deep dive into text-to-video, diffusion models, world modeling | [Tutorial Site](https://world-model-tutorial.github.io/) |

---

### Industry Trajectory (2025-2026)

Based on announcements and research directions:

**OpenAI**: GPT-5 with 272K context, PhD-level reasoning, smart model routing. Focus on reasoning + agentic capability as core features.

**Google DeepMind**: Gemini 3 with real-time video processing at 60fps, generative UI. Genie 3 world models for AGI path. Veo 3 for video generation with physics understanding.

**Anthropic**: Claude 4/4.5 with extended autonomous operation, Model Context Protocol (MCP) for agent-tool integration. Focus on reliability and safety over speed.

**Meta**: Llama 4 open-source with agentic capabilities. Yann LeCun's world model lab pursuing new architectures beyond transformers.

**Key Trends**:
- Scaling laws plateauing → shift to architectural innovation
- World models as path to embodied AI / AGI
- MCP becoming standard for agent-tool integration  
- Hour-scale video understanding becoming tractable
- Multimodal embeddings replacing separate image/text pipelines

---

### Original References

- [Dense Video Captioning Survey (2025)](https://www.mdpi.com/2076-3417/15/9/4990)
- [Molmo 2 - Video Understanding](https://allenai.org/blog/molmo2)
- [MMCTAgent - Multimodal Reasoning](https://www.microsoft.com/en-us/research/blog/mmctagent-enabling-multimodal-reasoning-over-large-video-and-image-collections/)
- [voyage-multimodal-3 for Video RAG](https://kx.com/blog/revolutionizing-video-search-with-multimodal-ai/)
- [Vid2Seq - Dense Video Captioning](https://antoyang.github.io/vid2seq.html)
- [ButterCut - Video Editing with Claude](https://github.com/barefootford/buttercut)
- [PySceneDetect](https://scenedetect.com)
- [Apollo: Video Understanding in LMMs](https://openaccess.thecvf.com/content/CVPR2025/papers/Zohar_Apollo__An_Exploration_of_Video_Understanding_in_Large_Multimodal_CVPR_2025_paper.pdf)
- [MVU: Understanding Long Videos with MLMs](https://arxiv.org/abs/2403.16998)

---

*Document version: 2.0*  
*Last updated: January 2026*
