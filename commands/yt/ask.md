---
description: Ask a question about a video - handles all processing automatically
argument-hint: <url> <question>
allowed-tools: ["mcp__claudetube__ask_video"]
---

# Ask Video (Streamlined)

The **simplest way to get answers from videos**. Just provide a URL and your question.

This tool handles everything automatically:
- Downloads and transcribes (if not cached)
- Generates scene structure (if not cached)
- Analyzes relevant scenes
- Returns a detailed answer with evidence

## Input: $ARGUMENTS

Arguments: `<url> <question>`
- **url**: Video URL or ID (YouTube, Vimeo, 1500+ sites)
- **question**: Your question about the video

Example: `/yt:ask https://youtube.com/watch?v=abc123 What is the main topic?`

## Step 1: Call ask_video MCP Tool

Use the `ask_video` MCP tool with:
- `url`: The video URL or ID from arguments
- `question`: The question from arguments

Parse arguments:
- First word/URL = video URL or ID
- Everything after = question

## Step 2: Present Results

The tool returns JSON with:
- `answer`: The answer to your question
- `confidence`: Confidence score (0-1)
- `evidence`: Supporting evidence with timestamps
- `steps`: What processing was done (cached vs fresh)
- `video_id`: The video identifier

Present the answer clearly, showing:
1. The answer
2. Confidence level
3. Key evidence with timestamps
4. Whether video was already cached or freshly processed

## Example Output

```
Question: What is the main topic?
Answer: The video explains how neural networks learn through backpropagation...
Confidence: 87%

Evidence:
  [2:15] "Today we'll explore how neural networks actually learn"
  [5:30] Diagram showing gradient descent process
  [8:45] Code walkthrough of backprop implementation

Processing: video_cached, scenes_cached (instant)
```

## Comparison with /yt:watch

| Command | Requires | Best For |
|---------|----------|----------|
| `/yt:ask` | Just URL + question | Quick answers, new videos |
| `/yt:watch` | Pre-processed video | Deep analysis, repeat queries |

Use `/yt:ask` when you want the simplest experience.
Use `/yt:watch` when you've already processed the video and want more control.

## Follow-up Actions

After getting an answer:
- `/yt:see <video_id> <timestamp>` - View frames at evidence timestamps
- `/yt:hq <video_id> <timestamp>` - HQ frames for code/text
- `/yt:find <video_id> <query>` - Find additional moments
- `/yt:transcript <video_id>` - Read full transcript
