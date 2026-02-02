[← Documentation](../README.md)

# Docker Setup

Run the claudetube MCP server in a container for isolation or for environments where native install is inconvenient.

## Building

```bash
git clone https://github.com/thoughtpunch/claudetube.git
cd claudetube
docker build -t claudetube-mcp .
```

## Running

### Basic

```bash
docker run -i claudetube-mcp
```

The `-i` flag is required because the MCP server communicates over stdin/stdout.

### With Cache Persistence

Mount a volume so processed videos persist across container restarts:

```bash
docker run -i \
  -v ~/.claude/video_cache:/home/claudetube/.claude/video_cache \
  claudetube-mcp
```

This shares the same cache directory used by native installs and slash commands.

## Configuring Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "claudetube": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/Users/YOU/.claude/video_cache:/home/claudetube/.claude/video_cache",
        "claudetube-mcp"
      ]
    }
  }
}
```

Replace `/Users/YOU` with your home directory.

## Configuring Claude Code

```bash
claude mcp add claudetube -- docker run -i --rm \
  -v ~/.claude/video_cache:/home/claudetube/.claude/video_cache \
  claudetube-mcp
```

## What's in the Image

- **Base:** `python:3.11-slim`
- **System deps:** ffmpeg (installed via apt)
- **Python deps:** yt-dlp, faster-whisper, mcp SDK
- **User:** Non-root `claudetube` user
- **Entrypoint:** `claudetube-mcp` (stdio MCP server)

## Cross-Platform Notes

### macOS / Linux

Works out of the box with the volume mount syntax shown above.

### Windows

Use Windows-style paths for the volume mount:

```powershell
docker run -i --rm `
  -v "${env:USERPROFILE}\.claude\video_cache:/home/claudetube/.claude/video_cache" `
  claudetube-mcp
```

### Apple Silicon (M1/M2/M3)

The image builds natively on ARM64. If you encounter issues with faster-whisper, the container will fall back to YouTube subtitles (which are faster anyway).

## Image Size

The image is approximately 1.5-2 GB due to:
- Python runtime (~150 MB)
- ffmpeg (~100 MB)
- faster-whisper + CTranslate2 (~1 GB)
- yt-dlp + other deps (~50 MB)

To reduce size, you can create a custom image without faster-whisper if you only need subtitle-based transcripts.
