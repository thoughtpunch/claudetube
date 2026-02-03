#!/usr/bin/env bash
#
# ralph-beads.sh - Autonomous agent loop
#
# Usage: ./scripts/ralph/ralph-beads.sh [options] [max_iterations]
#
# Options:
#   --debug, -d     Enable debug output (show Claude's full response as it streams)
#   --verbose, -v   Show additional loop state info
#   --dry-run       Show what would be done without running Claude
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# ── Parse arguments ────────────────────────────────────────────────
DEBUG=false
VERBOSE=false
DRY_RUN=false
MAX_ITERATIONS=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug|-d)
            DEBUG=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Usage: $0 [--debug|-d] [--verbose|-v] [--dry-run] [max_iterations]"
            exit 1
            ;;
        *)
            MAX_ITERATIONS=$1
            shift
            ;;
    esac
done

LOG_FILE="ralph.log"
PROGRESS_FILE="scripts/ralph/progress.txt"

# ── Logging helpers ────────────────────────────────────────────────
log() {
    echo "$@" | tee -a "$LOG_FILE"
}

debug() {
    if $DEBUG; then
        echo "[DEBUG] $@" | tee -a "$LOG_FILE"
    fi
}

verbose() {
    if $VERBOSE || $DEBUG; then
        echo "[INFO] $@" | tee -a "$LOG_FILE"
    fi
}

# ── Run Claude with streaming or capture ───────────────────────────
run_claude() {
    local prompt="$1"
    local output_var="$2"
    local temp_file
    temp_file=$(mktemp)

    if $DRY_RUN; then
        log "[DRY-RUN] Would run Claude with prompt (${#prompt} chars)"
        echo "DRY_RUN_OUTPUT" > "$temp_file"
    elif $DEBUG; then
        # Stream output in real-time AND capture it
        log "─── Claude output (streaming) ───"
        echo "$prompt" | claude --dangerously-skip-permissions --print 2>&1 | tee -a "$LOG_FILE" "$temp_file"
        log "─── End Claude output ───"
    else
        # Capture output, show progress dots
        log -n "Running Claude"
        echo "$prompt" | claude --dangerously-skip-permissions --print > "$temp_file" 2>&1 &
        local pid=$!
        while kill -0 "$pid" 2>/dev/null; do
            echo -n "." | tee -a "$LOG_FILE"
            sleep 5
        done
        wait "$pid" || true
        log ""  # newline after dots

        # Show summary of output
        local lines
        lines=$(wc -l < "$temp_file" | tr -d ' ')
        verbose "Claude returned $lines lines"
    fi

    # Read output into variable (bash 3 compatible)
    eval "$output_var"'=$(cat "$temp_file")'
    rm -f "$temp_file"
}

# ── Main loop ──────────────────────────────────────────────────────
log ""
log "══════════════════════════════════════════"
log "  RALPH LOOP - claudetube"
log "  Max: $MAX_ITERATIONS iterations"
log "  Debug: $DEBUG | Verbose: $VERBOSE | Dry-run: $DRY_RUN"
log "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
log "══════════════════════════════════════════"

iteration=0
no_task_streak=0
MAX_NO_TASK_STREAK=3

while [ $iteration -lt $MAX_ITERATIONS ]; do
    ((++iteration))
    log ""
    log "━━━ Iteration $iteration / $MAX_ITERATIONS ━━━"

    # ── Phase 1: Epic Review ──────────────────────────────────────
    verbose "Phase 1: Checking for epics needing review..."
    REVIEW_EPIC=""
    OPEN_EPICS=$(bd list --status open --json 2>/dev/null \
        | jq -r '[.[] | select(.issue_type == "epic") | .id] | .[]' 2>/dev/null || true)

    debug "Open epics: ${OPEN_EPICS:-none}"

    for epic_id in $OPEN_EPICS; do
        ALL_CLOSED=$(bd show "$epic_id" --json 2>/dev/null | jq -r '
            .[0].dependents // []
            | [.[] | select(.dependency_type == "parent-child")]
            | if length == 0 then "no_children"
              elif all(.status == "closed") then "all_closed"
              else "has_open"
              end
        ' 2>/dev/null || echo "error")
        debug "Epic $epic_id children status: $ALL_CLOSED"
        if [ "$ALL_CLOSED" = "all_closed" ]; then
            REVIEW_EPIC="$epic_id"
            break
        fi
    done

    if [ -n "$REVIEW_EPIC" ]; then
        log "Epic review: $REVIEW_EPIC (all children closed)"

        EPIC_DETAILS=$(bd show "$REVIEW_EPIC" 2>/dev/null)
        RALPH_CLAUDE=$(cat scripts/ralph/CLAUDE.md 2>/dev/null || echo "")
        PROJECT_CLAUDE=$(cat CLAUDE.md 2>/dev/null || echo "")

        PROMPT="## AUTONOMOUS MODE

$PROJECT_CLAUDE

---

$RALPH_CLAUDE

---

## EPIC REVIEW: $REVIEW_EPIC

All children of this epic are now closed. Perform a holistic review.

$EPIC_DETAILS

---

## Review Dimensions

1. **SUCCESS CRITERIA** — Do all acceptance criteria pass?
2. **IMPLEMENTATION GAPS** — Any missing features or edge cases?
3. **DEFERRED WORK** — Aggregate 'left undone' items from child tickets
4. **INTEGRATION** — Do the pieces fit together? Any conflicts?
5. **BEST PRACTICES** — Tests? Docs? Conventions followed?

## Actions

After reviewing, do ONE of:
- **CLOSE** the epic with a summary comment covering all 5 dimensions
- **CREATE** follow-up tickets for gaps, then close the epic

Use: bd close $REVIEW_EPIC --reason \"Epic review complete\"

Begin. Review thoroughly."

        run_claude "$PROMPT" OUTPUT

        if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
            log "All tasks complete!"
            exit 0
        fi
        if echo "$OUTPUT" | grep -q "<ralph>STUCK</ralph>"; then
            log "Agent stuck on epic review. Human needed."
            exit 2
        fi

        bd sync 2>/dev/null || true
        no_task_streak=0
        sleep 2
        continue
    fi

    # ── Phase 2: Find actionable task ─────────────────────────────
    verbose "Phase 2: Finding actionable task..."
    READY_JSON=$(bd ready --json --limit 50 2>/dev/null || echo "[]")
    debug "Ready JSON: $(echo "$READY_JSON" | jq -c '.[].id' 2>/dev/null || echo 'parse error')"

    # Try non-epic ready tasks first, sorted by priority
    TASK=$(echo "$READY_JSON" | jq -r '
        [.[] | select(.issue_type == "epic" | not)
             | select((.title | startswith("Epic:")) | not)]
        | sort_by(.priority // 99)
        | .[0] // empty
    ')

    # If no non-epic ready tasks, try open children of ready epics
    if [ -z "$TASK" ] || [ "$TASK" = "null" ]; then
        verbose "No direct tasks, checking epic children..."
        EPIC_IDS=$(echo "$READY_JSON" | jq -r '
            [.[] | select(.issue_type == "epic") | .id] | .[]
        ' 2>/dev/null || true)
        for eid in $EPIC_IDS; do
            debug "Checking children of epic $eid"
            TASK=$(bd show "$eid" --json 2>/dev/null | jq -r '
                .[0].dependents // []
                | [.[] | select(.dependency_type == "parent-child")
                       | select(.status == "open")
                       | select(.issue_type == "epic" | not)]
                | sort_by(.priority // 99)
                | .[0] // empty
            ' 2>/dev/null || true)
            if [ -n "$TASK" ] && [ "$TASK" != "null" ]; then
                break
            fi
        done
    fi

    # ── Phase 3: Handle no-task state ─────────────────────────────
    if [ -z "$TASK" ] || [ "$TASK" = "null" ]; then
        ((++no_task_streak))
        TOTAL=$(bd list --status open --json 2>/dev/null | jq 'length' || echo "0")
        READY_COUNT=$(echo "$READY_JSON" | jq 'length' 2>/dev/null || echo "0")

        if [ "$TOTAL" -eq 0 ]; then
            log "All tasks closed."
            log "<promise>COMPLETE</promise>"
            exit 0
        fi

        log "No actionable tasks (${READY_COUNT} ready epics, ${TOTAL} total open, streak ${no_task_streak}/${MAX_NO_TASK_STREAK})."

        if [ "$no_task_streak" -ge "$MAX_NO_TASK_STREAK" ]; then
            log "Stuck: no actionable tasks after ${MAX_NO_TASK_STREAK} checks."
            log "Remaining work may be blocked or needs epic review."
            bd blocked 2>/dev/null || true
            log "<ralph>STUCK</ralph>"
            exit 2
        fi

        sleep 2
        continue
    fi

    # ── Phase 4: Execute task ─────────────────────────────────────
    no_task_streak=0

    TASK_ID=$(echo "$TASK" | jq -r '.id')
    TITLE=$(echo "$TASK" | jq -r '.title')
    PRIORITY=$(echo "$TASK" | jq -r '.priority // "?"')
    TYPE=$(echo "$TASK" | jq -r '.issue_type // "task"')

    log "Task: $TASK_ID - $TITLE"
    verbose "Priority: P$PRIORITY | Type: $TYPE"

    # Log to progress file
    echo "- $(date '+%Y-%m-%d %H:%M') | $TASK_ID [P$PRIORITY] | $TITLE" >> "$PROGRESS_FILE"

    # Build prompt
    TASK_DETAILS=$(bd show "$TASK_ID" 2>/dev/null)
    RALPH_CLAUDE=$(cat scripts/ralph/CLAUDE.md 2>/dev/null || echo "")
    PROJECT_CLAUDE=$(cat CLAUDE.md 2>/dev/null || echo "")

    debug "Prompt size: PROJECT_CLAUDE=${#PROJECT_CLAUDE}, RALPH_CLAUDE=${#RALPH_CLAUDE}, TASK_DETAILS=${#TASK_DETAILS}"

    PROMPT="## AUTONOMOUS MODE

$PROJECT_CLAUDE

---

$RALPH_CLAUDE

---

## TASK: $TASK_ID

$TASK_DETAILS

---

Begin. Follow the workflow."

    # Run agent
    run_claude "$PROMPT" OUTPUT

    # Check exit status from output content
    if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
        log "All tasks complete!"
        exit 0
    fi
    if echo "$OUTPUT" | grep -q "<ralph>STUCK</ralph>"; then
        log "Agent stuck. Human needed."
        exit 2
    fi

    # Check for common failure patterns
    if echo "$OUTPUT" | grep -qi "error\|failed\|exception"; then
        verbose "Warning: Output contains error-like text"
    fi

    bd sync 2>/dev/null || true
    sleep 2
done

log "Max iterations reached."
exit 1
