# Configuration Guide

> How to configure claudetube's cache directory and other settings.

## Cache Directory Configuration

By default, claudetube stores all cached data (videos, transcripts, frames) at `~/.claude/video_cache/`. You can change this location using several methods.

### Configuration Priority

claudetube checks these sources in order (highest priority first):

```
1. Environment variable  → CLAUDETUBE_CACHE_DIR
2. Project config        → .claudetube/config.yaml
3. User config           → ~/.config/claudetube/config.yaml (Linux/macOS)
                         → %APPDATA%/claudetube/config.yaml (Windows)
4. Default               → ~/.claude/video_cache
```

The first source that provides a value wins.

## Method 1: Environment Variable

Set `CLAUDETUBE_CACHE_DIR` to override all other configuration:

```bash
# Temporary (current shell only)
export CLAUDETUBE_CACHE_DIR=/path/to/my/cache

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export CLAUDETUBE_CACHE_DIR=/path/to/my/cache' >> ~/.zshrc
```

This is useful for:
- CI/CD pipelines
- Docker containers
- Quick testing with alternate cache locations

## Method 2: Project Config

Create `.claudetube/config.yaml` in your project root:

```yaml
# .claudetube/config.yaml
cache_dir: ./video_cache
```

claudetube walks up from the current directory looking for this file, so it works from any subdirectory.

**Path Resolution:**
- **Relative paths** (like `./video_cache`) are resolved relative to the config file location, not the current working directory
- **Absolute paths** (like `/data/cache`) are used as-is
- **Tilde paths** (like `~/my-cache`) expand to the user's home directory

### Example Project Structure

```
my-project/
├── .claudetube/
│   └── config.yaml    # cache_dir: ./video_cache
├── video_cache/       # ← Cache goes here
│   └── dYP2V_nK8o0/
│       ├── state.json
│       └── audio.mp3
└── src/
    └── app.py         # Can run from here, config still found
```

## Method 3: User Config

Create `~/.config/claudetube/config.yaml` for system-wide defaults:

```yaml
# ~/.config/claudetube/config.yaml
cache_dir: ~/my-video-cache
```

**Platform-specific paths:**
- **Linux/macOS**: `~/.config/claudetube/config.yaml` (respects `$XDG_CONFIG_HOME`)
- **Windows**: `%APPDATA%\claudetube\config.yaml`

## Config File Format

Currently supported settings:

```yaml
# Cache directory for all video data
cache_dir: /path/to/cache

# Future settings (not yet implemented)
# whisper_model: small
# default_quality: medium
```

## Common Use Cases

### Store Cache in Project Directory

Useful for keeping video analysis with the project:

```yaml
# .claudetube/config.yaml
cache_dir: ./.claudetube/cache
```

### Use External Drive

For large video collections:

```yaml
# ~/.config/claudetube/config.yaml
cache_dir: /Volumes/ExternalDrive/claudetube-cache
```

### CI/CD Pipeline

Use environment variable for ephemeral builds:

```yaml
# .github/workflows/ci.yml
env:
  CLAUDETUBE_CACHE_DIR: /tmp/claudetube-cache
```

### Docker Container

Mount a volume and set env:

```dockerfile
ENV CLAUDETUBE_CACHE_DIR=/data/cache
VOLUME /data/cache
```

## Verifying Configuration

Check which config source is active:

```python
from claudetube.config.loader import get_config

config = get_config()
print(f"Cache dir: {config.cache_dir}")
print(f"Source: {config.source.value}")  # 'env', 'project', 'user', or 'default'
```

## Performance Notes

- **Local SSD** is recommended for best performance
- **Network mounts** (NFS, SMB) will add latency to every operation
- **SSD vs HDD**: Transcript files are small, but frame extraction benefits from fast I/O

See also: [Architecture Principles](../architecture/principles.md) — Cache is the first line of defense in our "Cheap First, Expensive Last" philosophy.
