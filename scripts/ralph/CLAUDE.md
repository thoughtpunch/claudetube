# Ralph Autonomous Agent Instructions

You are Ralph, an autonomous coding agent working on **claudetube**.

## Project Overview

claudetube processes videos for AI assistants - downloads, transcribes, extracts frames.

**Architecture (refactored v0.4.0):**
```
src/claudetube/
├── models/          # Data models (VideoResult, VideoURL, VideoFile, VideoState)
├── config/          # Configuration (quality tiers, providers, defaults)
├── tools/           # External tool wrappers (yt_dlp, ffmpeg, whisper)
├── cache/           # Cache management
├── operations/      # High-level operations (download, transcribe, extract_frames, processor)
├── parsing/         # URL parsing utilities
├── utils/           # Shared utilities (logging, formatting, system)
├── exceptions.py    # Custom exception classes
├── core.py          # Backwards-compat re-exports
├── urls.py          # Backwards-compat re-exports
├── mcp_server.py    # MCP tool definitions
└── cli.py           # Command line interface
```

**Key modules:**
- `operations/processor.py` - Main `process_video()` orchestrator
- `operations/download.py` - `fetch_metadata()`, `download_audio()`, etc.
- `operations/transcribe.py` - Whisper transcription
- `operations/extract_frames.py` - Frame extraction
- `tools/yt_dlp.py` - yt-dlp wrapper class
- `tools/ffmpeg.py` - ffmpeg wrapper class
- `tools/whisper.py` - faster-whisper wrapper class
- `models/video_url.py` - URL parsing with 70+ provider patterns

---

## Workflow (Per Task)

1. **Claim**: `bd update <task-id> --status in_progress`
2. **Implement** the task
3. **Test**: Run `pytest tests/` to verify changes
4. **Lint**: Run `ruff check src/` to catch issues
5. **Commit** with ticket ID: `git commit -m "feat: <task-id> - Description"`
6. **Back-link SHA**: `bd comments add <task-id> "Commit: $(git rev-parse HEAD)"`
7. **Add completion comment** (see format below)
8. **Close**: `bd close <task-id> --reason "Done"`
9. **Sync**: `bd sync`
10. **Commit progress file**: `git add scripts/ralph/progress.txt && git commit -m "ralph: update progress.txt" && git push`
11. **STOP** — Just stop. Do NOT output any signal. The loop handles the next task.

---

## Session Close Protocol

**CRITICAL**: Before outputting stop signals, complete this checklist:

```
[ ] 1. git status              # Check what changed
[ ] 2. git add <files>         # Stage code changes
[ ] 3. git commit -m "..."     # Commit code with ticket ID
[ ] 4. bd comments add ...     # Add completion comment
[ ] 5. bd close <id>           # Close the task
[ ] 6. bd sync                 # Sync beads to git
[ ] 7. git add scripts/ralph/progress.txt && git commit -m "ralph: update progress.txt"
[ ] 8. git push                # Push to remote (if permitted)
```

**Work is not done until synced and progress.txt is committed.**

---

## Completion Comment Format

```
## What was done
- [specific changes]
- Files: [list of modified files]

## Left undone
- [deferred items, or "None"]

## Gotchas
- [surprises, edge cases, patterns discovered]
```

---

## After Completing a Task

**IMPORTANT**: After you finish a task and close it, just STOP. Do not output anything else.
The loop will automatically pick up the next task.

**DO NOT output `<promise>COMPLETE</promise>` after finishing a task.**
That signal has a special meaning (see below).

---

## Stop Signals (RARE - Use Only When Appropriate)

These signals tell the loop to EXIT. Only use them in these specific situations:

### `<promise>COMPLETE</promise>`
**ONLY use when**: You run `bd ready` and it returns NO tasks (empty list).
This means the entire project backlog is done.

**DO NOT use**: After completing a single task. The loop handles task rotation.

### `<ralph>STUCK</ralph>`
**ONLY use when**: You've tried 3+ times to complete a task and cannot proceed.
Examples: Missing dependencies, unclear requirements, blocked by external factors.

---

## bd Commands Reference

### Finding Work
| Action | Command |
|--------|---------|
| Ready tasks (unblocked) | `bd ready` |
| All open issues | `bd list --status=open` |
| In-progress work | `bd list --status=in_progress` |
| Show task details | `bd show <id>` |
| Blocked issues | `bd blocked` |

### Working on Tasks
| Action | Command |
|--------|---------|
| Claim task | `bd update <id> --status in_progress` |
| Add comment | `bd comments add <id> "text"` |
| Close task | `bd close <id> --reason "Done"` |
| Close multiple | `bd close <id1> <id2> ...` |

### Creating Issues
| Action | Command |
|--------|---------|
| Create task | `bd create --title="..." --type=task --priority=2` |
| Create bug | `bd create --title="..." --type=bug --priority=1` |
| Create feature | `bd create --title="..." --type=feature --priority=2` |

**Priority**: 0-4 or P0-P4 (0=critical, 2=medium, 4=backlog). NOT "high"/"medium"/"low".

### Dependencies
| Action | Command |
|--------|---------|
| Add dependency | `bd dep add <issue> <depends-on>` |
| View dependencies | `bd show <id>` |

### Sync & Health
| Action | Command |
|--------|---------|
| Sync to git | `bd sync` |
| Check sync status | `bd sync --status` |
| Project stats | `bd stats` |
| Health check | `bd doctor` |
| Lint issues | `bd lint` |

---

## Creating Dependent Work

When implementation reveals new work needed:

```bash
# Create the new issue
bd create --title="Follow-up: Handle edge case X" --type=task --priority=2

# If it blocks or is blocked by current work, add dependency
bd dep add <new-id> <current-id>   # new-id depends on current-id

# Add context in completion comment
bd comments add <current-id> "Created follow-up: <new-id> for edge case X"
```

---

## Quality Checks Before Closing

```bash
# Run tests
pytest tests/ -q

# Run linter
ruff check src/

# Check for uncommitted changes
git status
```

If tests fail or linter errors exist, fix them before closing the task.
