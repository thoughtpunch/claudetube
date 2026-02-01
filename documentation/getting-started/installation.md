# Installation

> Get claudetube running on your system.

## Requirements

- **Python 3.10+**
- **ffmpeg** - For video/audio processing
- **yt-dlp** - For video downloading (installed automatically)

## Quick Install

```bash
pip install claudetube
```

Or with uv:

```bash
uv pip install claudetube
```

## Installing ffmpeg

### macOS

```bash
brew install ffmpeg
```

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install ffmpeg
```

### Windows

1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your PATH

Or use Chocolatey:
```powershell
choco install ffmpeg
```

## Verify Installation

```bash
# Check Python
python --version  # Should be 3.10+

# Check ffmpeg
ffmpeg -version

# Check yt-dlp (installed with claudetube)
yt-dlp --version

# Test claudetube
python -c "from claudetube import process_video; print('OK')"
```

## Optional: Faster Whisper

For transcription, claudetube uses faster-whisper (included). First-time transcription downloads the model (~500MB for "small").

Model storage: `~/.cache/huggingface/hub/`

## Development Install

```bash
git clone https://github.com/thoughtpunch/claudetube.git
cd claudetube
pip install -e ".[dev]"
```

## Troubleshooting

### "ffmpeg not found"

Make sure ffmpeg is in your PATH:
```bash
which ffmpeg  # Should show a path
```

### "yt-dlp not found"

Reinstall claudetube or install yt-dlp separately:
```bash
pip install yt-dlp
```

### Whisper model download fails

Check your internet connection and disk space. Models are stored in `~/.cache/huggingface/`.

---

**Next**: [Quick Start](quickstart.md) - Process your first video
