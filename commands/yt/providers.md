---
description: List available AI providers and their capabilities
argument-hint:
allowed-tools: ["Bash", "Read"]
---

# List AI Providers

Show available AI providers and their capabilities for transcription, vision, reasoning, and embedding.

## Step 1: Get Provider Information

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from claudetube.providers.registry import list_all, list_available
from claudetube.providers.capabilities import PROVIDER_INFO, Capability
from claudetube.providers import get_factory

available = list_available()
all_providers = list_all()

# Build capability -> provider mapping
capabilities = {}
for cap in Capability:
    cap_providers = []
    for name in all_providers:
        info = PROVIDER_INFO.get(name)
        if info and info.can(cap):
            cap_providers.append({
                "name": name,
                "available": name in available,
            })
    capabilities[cap.name.lower()] = {
        "providers": cap_providers,
    }

# Add configured preferences
try:
    factory = get_factory()
    config = factory.config
    capabilities["transcribe"]["preferred"] = config.transcription_provider
    capabilities["transcribe"]["fallbacks"] = config.transcription_fallbacks
    capabilities["vision"]["preferred"] = config.vision_provider
    capabilities["vision"]["fallbacks"] = config.vision_fallbacks
    capabilities["video"]["preferred"] = config.video_provider
    capabilities["reason"]["preferred"] = config.reasoning_provider
    capabilities["reason"]["fallbacks"] = config.reasoning_fallbacks
    capabilities["embed"]["preferred"] = config.embedding_provider
except (RuntimeError, ImportError):
    pass

result = {
    "available_providers": available,
    "all_providers": all_providers,
    "capabilities": capabilities,
}
print(json.dumps(result, indent=2))
PYTHON
```

## Output Format

Present the providers in a clear format:

```
## AI Providers

### Available Providers
- provider1
- provider2 (not configured)
...

### Capabilities

#### Transcription
- **Preferred**: whisper-local
- **Fallbacks**: [groq, openai]
- Providers: whisper-local (available), groq (available), openai (not configured)

#### Vision
- **Preferred**: claude-code
- **Fallbacks**: [anthropic, openai]
- Providers: claude-code (available), anthropic (available), openai (not configured)

#### Reasoning
- **Preferred**: claude-code
- Providers: claude-code (available), anthropic (available)

#### Embedding
- **Preferred**: voyage
- Providers: voyage (available), openai (not configured)
```

Indicate which providers are available (configured with API keys) vs just registered.
