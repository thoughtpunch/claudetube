[← Documentation](../README.md)

# The Processing Pipeline

> From URL to understanding: how claudetube processes video.

## Core Principle: Cheap First, Expensive Last

Every stage checks cache first and uses the cheapest available method:

```
┌─────────────────────────────────────────────────────────────┐
│  ALWAYS: Check cache first (instant if already processed)   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  PREFER: Use yt-dlp built-ins (subtitles, chapters, etc.)   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  FALLBACK: Local processing (ffprobe, transcript analysis)  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  LAST RESORT: Expensive compute (Whisper, visual analysis)  │
└─────────────────────────────────────────────────────────────┘
```

**Never do work that's already been done. Never use expensive methods when cheap ones suffice.**

See: [Architecture Principles](../architecture/principles.md)

## Overview

```
┌──────────┐    ┌───────────┐    ┌───────────┐    ┌──────────┐
│   URL    │ → │  Metadata  │ → │   Audio   │ → │ Transcript│
│          │    │  (yt-dlp)  │    │  (yt-dlp) │    │ (whisper) │
└──────────┘    └───────────┘    └───────────┘    └──────────┘
                      ↓                                 ↓
                ┌───────────┐                    ┌──────────┐
                │ Thumbnail │                    │   SRT    │
                │  (yt-dlp) │                    │   TXT    │
                └───────────┘                    └──────────┘
                                                       ↓
                                               ┌──────────────┐
                                               │    Cache     │
                                               │ (state.json) │
                                               └──────────────┘
```

## Stage 1: URL Resolution

**Input**: Any supported URL (YouTube, Vimeo, local file, etc.)

```python
from claudetube import VideoURL

parsed = VideoURL.parse("https://youtube.com/watch?v=dQw4w9WgXcQ")
# → provider: "youtube", video_id: "dQw4w9WgXcQ", is_known: True
```

The URL parser:
- Identifies the provider (70+ patterns)
- Extracts the video ID
- Handles edge cases (shorts, embeds, mobile URLs)

**See**: [Architecture Overview](../architecture/architecture.md)

## Stage 2: Metadata Fetch

**Tool**: yt-dlp

```python
metadata = fetch_metadata(url)
# → title, duration, uploader, chapters, formats, thumbnails...
```

This stage:
- Fetches video info without downloading
- Extracts chapters (if available)
- Identifies available subtitle tracks
- Checks for available audio formats

**See**: [Architecture Overview](../architecture/architecture.md)

## Stage 3: Audio Download

**Tool**: yt-dlp

```python
audio_path = download_audio(url, output_dir)
# → ~/.claudetube/cache/{video_id}/audio.mp3
```

Configuration:
- Format: MP3 (universal compatibility)
- Quality: 64K (optimized for speech)
- Fallback: Extract from video if no audio-only stream

## Stage 4: Transcription

**Priority order**:
1. **Uploaded subtitles** - Human-written, highest accuracy
2. **Auto-generated captions** - YouTube's ASR
3. **Whisper transcription** - Local fallback

```python
transcript = transcribe_audio(audio_path, model="small")
# → { "srt": "1\n00:00:00...", "txt": "Full text..." }
```

**See**: [Transcripts](transcripts.md)

## Stage 5: Caching

All outputs are persisted:

```
~/.claudetube/cache/{video_id}/
├── state.json      # Metadata + processing state
├── audio.mp3       # Audio track
├── audio.srt       # Timestamped transcript
├── audio.txt       # Plain text transcript
└── thumbnail.jpg   # Video thumbnail
```

Subsequent requests are instant—no re-processing.

## On-Demand: Frame Extraction

Frames are **not** part of the initial pipeline. They're extracted when requested:

```python
frames = get_frames(video_id, start_time=120, duration=5)
# → Downloads video segment → Extracts frames → Returns paths
```

**See**: [Frames](frames.md)

## The VideoResult Object

The pipeline returns a unified result:

```python
@dataclass
class VideoResult:
    video_id: str
    success: bool
    error: str | None
    metadata: dict
    transcript_srt: Path | None
    transcript_txt: Path | None
    transcript_text: str | None
    thumbnail: Path | None
```

## Error Handling

Each stage can fail independently:

| Stage | Common Errors | Fallback |
|-------|--------------|----------|
| URL Resolution | Unknown provider | Use yt-dlp generic |
| Metadata | Private video, geo-blocked | Error with message |
| Audio | No audio stream | Extract from video |
| Subtitles | No captions available | Use Whisper |
| Whisper | Model loading failure | Error with message |

Partial success is possible—you might get metadata but no transcript.

---

**Next**: [Transcripts](transcripts.md) - Speech-to-text details
