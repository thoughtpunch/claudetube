# Better Ralph Loops Using Beads

> A practical guide to building traceable, autonomous AI agent workflows

This document captures learnings from building "Ralph" — an autonomous coding agent that uses Beads for task tracking. The system enables AI agents to work independently while maintaining full audit trails.

---

## The Problem

Autonomous AI agents can write code, but without structure they produce:
- **Orphaned work** — Changes with no context for why they exist
- **Lost knowledge** — Learnings that disappear between sessions
- **Quality drift** — Standards that slip without enforcement
- **Untraceable bugs** — Issues you can't root-cause back to decisions

## The Solution: Traceable Autonomy

Every change links to a ticket. Every ticket links to its predecessors. Every closure requires documentation. This creates an audit trail so any future agent or human can trace from a commit back to the original design decision.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         RALPH LOOP                              │
│  (ralph-beads.sh)                                               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Get ready tasks: bd ready --json                    │   │
│  │  2. Filter for tasks (not epics), sort by priority      │   │
│  │  3. Pass task + CLAUDE.md to agent                      │   │
│  │  4. Agent executes workflow                             │   │
│  │  5. Check for COMPLETE or STUCK signals                 │   │
│  │  6. Sync to git: bd sync                                │   │
│  │  7. Sleep, continue                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  AGENT EXECUTION                                        │   │
│  │                                                         │   │
│  │  pre-task.sh ──► Implementation ──► post-task.sh       │   │
│  │       │                                    │            │   │
│  │       ▼                                    ▼            │   │
│  │  - Show related tickets            - Verify commit      │   │
│  │  - Suggest documentation           - Verify comment     │   │
│  │  - Display checklist               - Check DoD          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  BEADS (bd)                                             │   │
│  │                                                         │   │
│  │  .beads/beads.db ◄──► .beads/issues.jsonl ◄──► git     │   │
│  │       (local)              (synced)           (remote)  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: The Main Loop

The loop script (`ralph-beads.sh`) orchestrates everything:

```bash
#!/usr/bin/env bash
set -euo pipefail

MAX_ITERATIONS=${1:-10}
LOG_FILE="ralph.log"
exec > >(tee -a "$LOG_FILE") 2>&1

iteration=0
while [ $iteration -lt $MAX_ITERATIONS ]; do
    ((iteration++))
    echo "━━━ Iteration $iteration / $MAX_ITERATIONS ━━━"

    # 1. Get ready tasks (unblocked, not claimed)
    READY=$(bd ready --json --limit 100)

    # 2. Filter for tasks only, sort by priority
    TASK=$(echo "$READY" | jq -r '
        [.[] | select(.issue_type == "task")
             | select((.title | startswith("Epic:")) | not)]
        | sort_by(.priority // 99)
        | .[0] // empty
    ')

    if [ -z "$TASK" ]; then
        # Check for epic reviews...
        echo "No ready tasks. Checking epics..."
        # (epic review logic here)
        continue
    fi

    TASK_ID=$(echo "$TASK" | jq -r '.id')
    TITLE=$(echo "$TASK" | jq -r '.title')

    # 3. Build prompt with CLAUDE.md + task details
    TASK_DETAILS=$(bd show "$TASK_ID")
    PROMPT="## AUTONOMOUS MODE

$(cat scripts/ralph/CLAUDE.md)

---

## YOUR ASSIGNED TASK

**Task ID:** $TASK_ID
**Title:** $TITLE

### Task Details:
$TASK_DETAILS

Begin now. Follow the workflow exactly."

    # 4. Run agent
    echo "$PROMPT" | claude --dangerously-skip-permissions --print

    # 5. Check for stop signals
    if grep -q "<promise>COMPLETE</promise>" <<< "$OUTPUT"; then
        echo "All tasks complete!"
        exit 0
    fi
    if grep -q "<ralph>STUCK</ralph>" <<< "$OUTPUT"; then
        echo "Agent stuck. Human intervention needed."
        exit 2
    fi

    # 6. Sync
    bd sync 2>/dev/null || true

    sleep 2
done
```

**Key design choices:**
- **Limit 100** on `bd ready` — epics sort first alphabetically, limit ensures we see tasks
- **Filter with jq** — remove epics, sort by priority (P0 > P1 > P2)
- **Stdin-based Claude** — no project files, streaming prompts
- **Stop signals** — `<promise>COMPLETE</promise>` and `<ralph>STUCK</ralph>` exit the loop
- **Append logging** — `tee -a` preserves history across runs

---

## Component 2: The 6-Step Workflow

Every piece of work follows this sequence. No exceptions.

```
SCAN → CLAIM → DO → UPDATE → CLOSE → COMMIT
```

### Step 1: SCAN
Find related work before starting:
```bash
bd search "relevant keyword"
bd list --status in_progress   # What's already claimed?
```

### Step 2: CLAIM (or CREATE)
```bash
# Claim existing:
bd update <id> --status in_progress

# Or create new:
bd create "Description" -t task -p 2

# If related to existing work:
bd create "Follow-up to proj-xyz" --deps "discovered-from:proj-xyz"
```

### Step 3: DO
Implement the work.

### Step 4: UPDATE
Add a completion comment:
```bash
bd comments add <id> "## What was done
- Created Grid class with sparse storage
- Added unit tests
- Files: src/grid.gd, test/test_grid.gd

## Left undone / deferred
- Edge case: negative coordinates (deferred to M2)

## Gotchas
- GDScript truncates negative division unexpectedly"
```

### Step 5: CLOSE
```bash
bd close <id> --reason "Implemented and tested"
```

### Step 6: COMMIT
```bash
git commit -m "feat: proj-xyz - Grid data structure"
SHA=$(git rev-parse HEAD)
bd comments add <id> "Commit: $SHA"
```

**Why this order matters:**
- Comment before close — ensures closure isn't accidental
- Commit with ID — enables traceability
- Back-link SHA — bidirectional audit trail

---

## Component 3: Enforcement Hooks

Hooks make quality **mandatory**, not optional.

### Pre-Task Hook (`pre-task.sh`)

Runs before the agent starts work. Ensures context is gathered.

```bash
#!/usr/bin/env bash
TICKET_ID="$1"

echo "═══════════════════════════════════════════"
echo "  PRE-TASK: $TICKET_ID"
echo "═══════════════════════════════════════════"

# Show ticket details
bd show "$TICKET_ID"

# Extract keywords from title for related ticket search
TITLE=$(bd show "$TICKET_ID" --json | jq -r '.title')
KEYWORDS=$(echo "$TITLE" | tr ' ' '\n' | grep -v '^[a-z]\{1,3\}$')

echo ""
echo "RELATED TICKETS:"
for kw in $KEYWORDS; do
    bd search "$kw" --status open 2>/dev/null | head -5
done

# Suggest documentation based on labels
LABELS=$(bd show "$TICKET_ID" --json | jq -r '.labels[]?' 2>/dev/null)
echo ""
echo "SUGGESTED DOCUMENTATION:"
for label in $LABELS; do
    case "$label" in
        *grid*|*block*) echo "  - documentation/game-design/blocks/" ;;
        *ui*|*hud*)     echo "  - documentation/ui/views.md" ;;
        *camera*)       echo "  - documentation/technical/camera.md" ;;
    esac
done

echo ""
echo "CHECKLIST before starting:"
echo "  [ ] Read suggested documentation"
echo "  [ ] Check ticket comments for context"
echo "  [ ] Identify any unclear requirements"
echo "  [ ] Scan related tickets for prior work"
```

### Post-Task Hook (`post-task.sh`)

Runs before closure is allowed. Enforces Definition of Done.

```bash
#!/usr/bin/env bash
TICKET_ID="$1"
ERRORS=0

echo "═══════════════════════════════════════════"
echo "  POST-TASK: $TICKET_ID"
echo "═══════════════════════════════════════════"

# Check 1: Completion comment exists
COMMENTS=$(bd show "$TICKET_ID" --json | jq '.comments | length')
if [ "$COMMENTS" -eq 0 ]; then
    echo "ERROR: No comments found on $TICKET_ID"
    echo ""
    echo "You MUST add a completion comment before closing."
    echo "Format:"
    echo "  bd comments add $TICKET_ID \"## What was done"
    echo "  - [changes]"
    echo "  ## Left undone"
    echo "  - [or None]"
    echo "  ## Gotchas"
    echo "  - [surprises]\""
    ERRORS=$((ERRORS + 1))
fi

# Check 2: At least one commit references the ticket
COMMITS=$(git log --oneline --all --grep="$TICKET_ID" | wc -l)
if [ "$COMMITS" -eq 0 ]; then
    echo "ERROR: No commits found referencing $TICKET_ID"
    echo ""
    echo "You MUST create a commit with the ticket ID in the message."
    echo "Format: git commit -m \"feat: $TICKET_ID - Description\""
    ERRORS=$((ERRORS + 1))
fi

# Check 3: Uncommitted changes (warning only)
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "WARNING: You have uncommitted changes!"
    git status --short
fi

# Summary
echo ""
if [ $ERRORS -gt 0 ]; then
    echo "BLOCKED: $ERRORS error(s) must be fixed before closing."
    exit 1
else
    echo "READY TO CLOSE"
    echo "  Comment count: $COMMENTS ✓"
    echo "  Commit count: $COMMITS ✓"
    exit 0
fi
```

**Critical insight:** Hooks exit non-zero to block. The agent learns to satisfy them.

---

## Component 4: Agent Instructions (CLAUDE.md)

Two levels of instructions:

### Project-Level (`/CLAUDE.md`)
Universal guidance for any agent:
- Code conventions
- Architecture overview
- Key concepts
- When to ask for help

### Agent-Specific (`/scripts/ralph/CLAUDE.md`)
Workflow instructions for Ralph:

```markdown
# Ralph Autonomous Agent Instructions

You are Ralph, an autonomous coding agent. Follow this workflow exactly.

## Workflow (15 Steps)

1. Read `progress.txt` for codebase patterns
2. Consult `documentation/` for implementation details
3. Run pre-task hook: `./scripts/hooks/pre-task.sh <id>`
4. Claim task: `bd update <id> --status in_progress`
5. Implement the task
6. Write tests (positive AND negative assertions)
7. Run quality checks (project opens, no errors)
8. **MANDATORY:** Commit with ticket ID
   ```bash
   git commit -m "feat: <id> - Description"
   ```
9. **MANDATORY:** Back-link commit SHA
   ```bash
   SHA=$(git rev-parse HEAD)
   bd comments add <id> "Commit: $SHA"
   ```
10. Run post-task hook: `./scripts/hooks/post-task.sh <id>`
11. **MANDATORY:** Add completion comment
12. Close task: `bd close <id> --reason "Implemented"`
13. Sync: `./scripts/hooks/bd-sync-rich.sh`
14. Append learnings to `progress.txt`
15. **STOP** — Let the loop assign the next task

## Completion Comment Format

```markdown
## What Was Done
- [specific change 1]
- [specific change 2]
- Files: [list]

## What Was Left Undone / Deferred
- [scope cut / edge cases / "None"]

## Gotchas / Notes
- [anything surprising]
- [patterns to document]

## Test Coverage
- [how verified]
```

## Stop Conditions

- Output `<promise>COMPLETE</promise>` when all tasks are done
- Output `<ralph>STUCK</ralph>` if blocked after 3+ attempts
```

---

## Component 5: Knowledge Accumulation

`progress.txt` is a living playbook:

```markdown
## Codebase Patterns

(Reusable patterns discovered. Future iterations read this first.)

- Use Vector3i for grid positions (Y-up coordinate system)
- CELL_SIZE = 6.0 (6m cubes), CHUNK_SIZE = 8
- Signals over polling: objects emit, systems connect
- Load balance numbers from data/balance.json

---

## Iteration Log

### Iteration 1 - 2026-01-25
Task: proj-abc - Grid data structure

Docs Consulted:
- documentation/architecture/milestone-1.md
- documentation/game-design/blocks.md

Status: PASSED

Changes:
- src/grid.gd: Created Grid class with sparse storage
- test/test_grid.gd: Added unit tests

Learnings:
- GDScript requires explicit type hints for Dictionary
- Use Vector3i consistently for grid positions

Followup Tickets Created:
- proj-def: Chunk-based rendering optimization
```

**Why this works:**
- Future iterations read patterns first → avoid re-discovery
- Iteration log shows what was tried → don't repeat failures
- Followup tickets maintain the chain

---

## Component 6: Rich Git Sync

`bd-sync-rich.sh` creates informative sync commits:

```bash
#!/usr/bin/env bash

# Find last sync commit
LAST_SYNC=$(git log --oneline --all --grep="bd sync:" -1 --format="%H %ci" 2>/dev/null)
SINCE_TIMESTAMP=$(echo "$LAST_SYNC" | cut -d' ' -f2-)

# Get activity since last sync
ACTIVITY=$(bd activity --since "$SINCE_TIMESTAMP" --json 2>/dev/null || echo "[]")

# Build changelog
CHANGELOG=""
while read -r line; do
    ID=$(echo "$line" | jq -r '.issue_id')
    ACTION=$(echo "$line" | jq -r '.action')
    TITLE=$(echo "$line" | jq -r '.title // ""')

    case "$ACTION" in
        created)     CHANGELOG+="+ $ID created — $TITLE\n" ;;
        in_progress) CHANGELOG+="→ $ID in_progress\n" ;;
        closed)      CHANGELOG+="✓ $ID closed\n" ;;
        deleted)     CHANGELOG+="⊘ $ID deleted\n" ;;
        *)           CHANGELOG+="~ $ID $ACTION\n" ;;
    esac
done < <(echo "$ACTIVITY" | jq -c '.[]')

# Sync and commit
bd sync
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
git add .beads/issues.jsonl
git commit -m "bd sync: $TIMESTAMP

Changes:
$CHANGELOG"
```

**Result:**
```
commit abc123
Author: Ralph <ralph@example.com>
Date:   2026-01-25 14:35:00

    bd sync: 2026-01-25 14:35:00

    Changes:
    + proj-abc created — Grid data structure
    → proj-abc in_progress
    ✓ proj-abc closed
```

---

## Component 7: Epic Reviews

When all children of an epic close, trigger a review:

```bash
# In ralph-beads.sh, after task completion:
EPIC_ID=$(echo "$TASK_ID" | sed 's/\.[0-9]*$//')
EPIC_STATUS=$(bd epic status "$EPIC_ID" --json)
ALL_CLOSED=$(echo "$EPIC_STATUS" | jq '.all_children_closed')

if [ "$ALL_CLOSED" = "true" ]; then
    # Gather context
    REVIEW_CONTEXT=$(./scripts/hooks/check-epic-completion.sh "$TASK_ID")

    # Pass to agent for review
    REVIEW_PROMPT="## EPIC REVIEW

$REVIEW_CONTEXT

Review dimensions:
1. SUCCESS CRITERIA — Do all acceptance criteria pass?
2. IMPLEMENTATION GAPS — Any missing features?
3. DEFERRED WORK — Aggregate 'left undone' items
4. INTEGRATION — Do the pieces fit together?
5. BEST PRACTICES — Tests? Docs? Conventions?

Either:
- CLOSE the epic with a summary comment
- CREATE follow-up tickets for gaps"

    echo "$REVIEW_PROMPT" | claude --print
fi
```

---

## Why This Design Works

### 1. Traceability
```
commit → ticket ID → completion comment → related tickets → docs
```

Any bug can be traced back to the original design decision.

### 2. Quality Enforcement
Hooks **block** closure without:
- Completion comment
- At least one commit referencing ticket ID

This makes it impossible to close a task without an audit trail.

### 3. Knowledge Accumulation
`progress.txt` acts as institutional memory:
- Patterns are tested and proven
- Mistakes aren't repeated
- New iterations start smarter

### 4. Autonomy Without Chaos
The agent can work without human supervision because:
- Pre-task hook enforces documentation reading
- Post-task hook enforces Definition of Done
- Task dependencies prevent blocked work
- Stop signals handle completion and failures

### 5. Zero External Dependencies
After Beads is installed, everything runs locally:
- Beads uses SQLite (local)
- Hooks are bash scripts (no services)
- Git is the sync mechanism

---

## Quick Reference: bd Commands

| Action | Command |
|--------|---------|
| List ready tasks | `bd ready` |
| Search | `bd search "keyword"` |
| Show task | `bd show <id>` |
| Claim | `bd update <id> --status in_progress` |
| Create | `bd create "Title" -t task -p 2` |
| Create linked | `bd create "Title" --deps "discovered-from:<id>"` |
| Comment | `bd comments add <id> "text"` |
| Close | `bd close <id> --reason "Done"` |
| Sync to git | `bd sync` |
| Epic status | `bd epic status <id>` |

---

## Failure Modes & Recovery

### Agent Gets Stuck
```bash
bd comments add <id> "Blocked: <reason>"
# Output: <ralph>STUCK</ralph>
```
Loop exits with code 2. Human reviews comments, simplifies task, restarts.

### Commit Without Ticket ID
Post-task hook blocks closure. Agent must amend commit or create new one.

### No Completion Comment
Post-task hook blocks closure. Agent must add comment.

### Uncommitted Changes
Warning only (doesn't block), but if no commits reference ticket, it becomes an error.

---

## Setting Up Your Own System

1. **Install Beads**
   ```bash
   # See beads documentation for installation
   bd init
   ```

2. **Create hooks directory**
   ```bash
   mkdir -p scripts/hooks
   # Copy pre-task.sh and post-task.sh from above
   chmod +x scripts/hooks/*.sh
   ```

3. **Create agent instructions**
   ```bash
   # Project-level CLAUDE.md at root
   # Agent-specific CLAUDE.md in scripts/ralph/
   ```

4. **Create the loop script**
   ```bash
   mkdir -p scripts/ralph
   # Copy ralph-beads.sh from above
   chmod +x scripts/ralph/ralph-beads.sh
   ```

5. **Initialize progress tracking**
   ```bash
   touch scripts/ralph/progress.txt
   ```

6. **Create initial tasks**
   ```bash
   bd create "Epic: Milestone 1" -t epic
   bd create "First task" -t task -p 1 --description "..."
   ```

7. **Run**
   ```bash
   ./scripts/ralph/ralph-beads.sh 10
   ```

---

## Key Insights

1. **Enforcement > Documentation** — Hooks that block are more effective than instructions that suggest.

2. **Bidirectional Links** — Commit → ticket AND ticket → commit. Both directions matter for traceability.

3. **Completion Comments Are Non-Negotiable** — The "What was done / Left undone / Gotchas" format captures exactly what future developers need.

4. **Progress Files Beat Memory** — Agents forget between sessions. A progress file makes knowledge persistent.

5. **Atomic Tasks** — Keep tasks small (10-30 min). Easier to trace, easier to recover from failures.

6. **Stop Signals** — Explicit `COMPLETE` and `STUCK` signals let the loop know when to exit gracefully.

7. **Epic Reviews** — Don't auto-close epics. Require a holistic review when all children complete.

---

## Credits

Developed for the Arcology project. Ralph is an autonomous coding agent; Beads is a git-native ticket tracker.

---

*Last updated: 2026-02-01*
