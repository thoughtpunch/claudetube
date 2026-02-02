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

```yaml
# Cache directory for all video data
cache_dir: /path/to/cache

# AI Provider configuration (see Provider Configuration below)
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o
  # ... see full reference below
```

## Provider Configuration

claudetube supports configurable AI providers for transcription, vision, reasoning, and embeddings. Without any configuration, it defaults to free/local options.

### Zero-Config Defaults

| Capability     | Default Provider | Cost |
|---------------|-----------------|------|
| Transcription | whisper-local   | Free |
| Vision        | claude-code     | Free |
| Reasoning     | claude-code     | Free |
| Embedding     | voyage          | $    |

### API Provider Credentials

Add API keys for cloud providers. Keys can reference environment variables:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
  google:
    api_key: ${GOOGLE_API_KEY}
    model: gemini-2.0-flash
  deepgram:
    api_key: ${DEEPGRAM_API_KEY}
  assemblyai:
    api_key: ${ASSEMBLYAI_API_KEY}
  voyage:
    api_key: ${VOYAGE_API_KEY}
```

### Local Provider Settings

```yaml
providers:
  local:
    whisper_model: small       # tiny, base, small, medium, large
    ollama_model: llava:13b    # Any Ollama model with vision support
```

### Capability Preferences

Set which provider to use for each capability:

```yaml
providers:
  preferences:
    transcription: whisper-local  # whisper-local, openai, deepgram, assemblyai
    vision: claude-code           # claude-code, anthropic, openai, google, ollama
    video: google                 # google (only Gemini supports native video)
    reasoning: claude-code        # claude-code, anthropic, openai, google, ollama
    embedding: voyage             # voyage, local-embedder
```

### Fallback Chains

If the preferred provider fails, try these in order:

```yaml
providers:
  fallbacks:
    transcription: [openai, whisper-local]
    vision: [anthropic, openai, claude-code]
    reasoning: [anthropic, openai, claude-code]
```

### Migrating from Environment Variables

If you previously used bare env vars, they still work as fallbacks:

| Environment Variable     | YAML Equivalent            |
|-------------------------|---------------------------|
| `OPENAI_API_KEY`        | `providers.openai.api_key` |
| `ANTHROPIC_API_KEY`     | `providers.anthropic.api_key` |
| `GOOGLE_API_KEY`        | `providers.google.api_key` |
| `DEEPGRAM_API_KEY`      | `providers.deepgram.api_key` |
| `ASSEMBLYAI_API_KEY`    | `providers.assemblyai.api_key` |
| `VOYAGE_API_KEY`        | `providers.voyage.api_key` |
| `CLAUDETUBE_CACHE_DIR`  | `cache_dir`               |

`CLAUDETUBE_*_API_KEY` variants also work. YAML config takes priority over env vars.

### Provider Capability Matrix

| Provider       | Transcribe | Vision | Video | Reason | Embed | Cost  |
|---------------|:----------:|:------:|:-----:|:------:|:-----:|:-----:|
| whisper-local  |     ✓      |        |       |        |       | Free  |
| openai         |     ✓      |   ✓    |       |   ✓    |       | $$    |
| anthropic      |            |   ✓    |       |   ✓    |       | $$    |
| google         |            |   ✓    |   ✓   |   ✓    |       | $     |
| deepgram       |     ✓      |        |       |        |       | $     |
| assemblyai     |     ✓      |        |       |        |       | $     |
| claude-code    |            |   ✓    |       |   ✓    |       | Free* |
| ollama         |            |   ✓    |       |   ✓    |       | Free  |
| voyage         |            |        |       |        |   ✓   | $     |
| local-embedder |            |        |       |        |   ✓   | Free  |

\* Included with Claude Code subscription.

### Validation

Configuration is validated on load. Errors are logged for:
- Invalid whisper model sizes
- Unknown provider names
- Capability mismatches (e.g., whisper-local set as vision preference)
- Structural issues (wrong types)

### Example Configs

See `examples/config.minimal.yaml` and `examples/config.full.yaml` for complete references.

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
