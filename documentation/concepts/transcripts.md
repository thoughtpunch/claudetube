# Transcripts

> Converting speech to searchable, timestamped text.

## Transcript Sources

claudetube tries multiple sources in priority order:

### 1. Uploaded Subtitles (Best Quality)

Human-written captions uploaded by the video creator.
- Highest accuracy
- Proper punctuation and formatting
- May include speaker labels
- Not always available

### 2. Auto-Generated Captions (Good Quality)

YouTube's automatic speech recognition.
- Available on most YouTube videos
- Generally good accuracy for English
- May have errors with names, technical terms
- No punctuation in older videos

### 3. Whisper Transcription (Universal Fallback)

Local speech-to-text using OpenAI's Whisper model.
- Works on any audio (YouTube, local files, etc.)
- Multiple model sizes for speed/accuracy tradeoff
- Runs locally—no API calls
- Requires audio download

## Output Formats

### SRT (SubRip Text)

Timestamped format with sequence numbers:

```srt
1
00:00:05,000 --> 00:00:08,500
Let me show you how the authentication flow works.

2
00:00:08,500 --> 00:00:12,000
First, the user clicks the login button here.

3
00:00:12,000 --> 00:00:15,500
This triggers the OAuth redirect.
```

**Use for**: Correlating speech with timestamps, finding specific moments.

### Plain Text (TXT)

Just the words, one line per segment:

```
Let me show you how the authentication flow works.
First, the user clicks the login button here.
This triggers the OAuth redirect.
```

**Use for**: Full-text search, summarization, quick reading.

## Whisper Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| tiny | ~75MB | Fastest | Basic | Quick previews |
| base | ~150MB | Fast | Good | Most videos |
| small | ~500MB | Medium | Better | Important content |
| medium | ~1.5GB | Slow | Great | Technical content |
| large | ~3GB | Slowest | Best | Critical accuracy |

Default: `small` (good balance of speed and accuracy)

## Batched vs Standard Transcription

claudetube uses **faster-whisper** with two modes:

### Batched Inference (Default)
- Uses all CPU cores
- ~4-10x faster than standard
- May occasionally miss segments
- Falls back to standard if coverage is low

### Standard Inference
- Sequential processing
- Guaranteed complete coverage
- Slower but reliable
- Used as fallback

## API Usage

### Process Video (Includes Transcription)

```python
from claudetube import process_video

result = process_video("https://youtube.com/watch?v=...")
print(result.transcript_text)  # Plain text
# Or read files:
# result.transcript_srt → Path to .srt file
# result.transcript_txt → Path to .txt file
```

### Transcribe Cached Video

```python
from claudetube import transcribe_video

# Re-transcribe with different model
result = transcribe_video(video_id, model="medium", force=True)
```

### MCP Tools

```
process_video_tool(url, whisper_model="small")
transcribe_video(video_id, whisper_model="medium", force=true)
get_transcript(video_id, format="srt")
```

## Transcript Coverage Check

For long videos, claudetube verifies the transcript covers the full duration:

```python
# If batched transcription only covers 80% of audio:
# → Automatically falls back to standard transcription
```

This prevents silent failures where only part of the video is transcribed.

---

**See also**:
- [The Pipeline](pipeline.md) - Where transcription fits
- [Frames](frames.md) - Visual complement to audio
