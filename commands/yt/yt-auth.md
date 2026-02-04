---
description: Check YouTube authentication status and diagnostics
argument-hint:
allowed-tools: ["Bash", "Read"]
---

# Check YouTube Authentication Status

Diagnose YouTube authentication configuration and provide recommendations.

## Step 1: Run Auth Check

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from claudetube.tools.yt_dlp import YtDlpTool

yt = YtDlpTool()
status = yt.check_youtube_auth_status()
print(json.dumps(status, indent=2))
PYTHON
```

## Output Format

Present the status in a clear format:

```
## YouTube Authentication Status

**Auth Level**: X/4

### Component Status
| Component | Status | Details |
|-----------|--------|---------|
| Deno | Yes/No | version X.X.X |
| PO Token Plugin | Yes/No | version X.X.X |
| POT Server | Yes/No | http://127.0.0.1:4416 |
| Cookies | Yes/No | source: browser/file |
| Manual PO Token | Yes/No | configured in config.yaml |

### Auth Level Explanation
- **0**: No authentication (may break as YouTube tightens access)
- **1**: Deno only (android_vr HLS formats)
- **2**: Cookies + deno (Premium: full access; Free: android_vr)
- **3**: Manual PO token + cookies + deno (tokens expire ~12hr)
- **4**: bgutil server + cookies + deno (automated, full access)

### Recommendations
[List any recommendations from the status output]
```

If there are issues, point the user to:
- Setup guide: `documentation/guides/youtube-auth.md`
- Config location: `~/.claudetube/config.yaml`
