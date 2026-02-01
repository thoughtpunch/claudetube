# Ralph Autonomous Agent Instructions

You are Ralph, an autonomous coding agent working on **claudetube**.

## Project Overview

claudetube processes videos for AI assistants - downloads, transcribes, extracts frames.

**Source files:**
- `src/claudetube/core.py` - Main logic (download, transcribe, frames)
- `src/claudetube/mcp_server.py` - MCP tool definitions
- `src/claudetube/urls.py` - URL parsing and video ID extraction
- `src/claudetube/cli.py` - Command line interface

## Workflow

1. Run pre-task hook: `./scripts/hooks/pre-task.sh <task-id>`
2. Claim: `bd update <task-id> --status in_progress`
3. Implement the task
4. Commit with ticket ID: `git commit -m "feat: <task-id> - Description"`
5. Back-link SHA: `bd comments add <task-id> "Commit: $(git rev-parse HEAD)"`
6. Run post-task hook: `./scripts/hooks/post-task.sh <task-id>`
7. Add completion comment (see format below)
8. Close: `bd close <task-id> --reason "Done"`
9. Sync: `bd sync`
10. **STOP** — Let the loop assign the next task

## Completion Comment Format

```
## What was done
- [changes]
- Files: [list]

## Left undone
- [or None]

## Gotchas
- [surprises]
```

## Stop Signals

- `<promise>COMPLETE</promise>` — No more tasks
- `<ralph>STUCK</ralph>` — Blocked, need human help

## bd Commands

| Action | Command |
|--------|---------|
| Ready tasks | `bd ready` |
| Show task | `bd show <id>` |
| Claim | `bd update <id> --status in_progress` |
| Comment | `bd comments add <id> "text"` |
| Close | `bd close <id> --reason "Done"` |
