#!/usr/bin/env bash
#
# ralph-beads.sh - Autonomous agent loop
#
# Usage: ./scripts/ralph/ralph-beads.sh [max_iterations]
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MAX_ITERATIONS=${1:-10}
LOG_FILE="ralph.log"
PROGRESS_FILE="scripts/ralph/progress.txt"

exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo "══════════════════════════════════════════"
echo "  RALPH LOOP - claudetube"
echo "  Max: $MAX_ITERATIONS iterations"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════"

iteration=0
no_task_streak=0
MAX_NO_TASK_STREAK=3
while [ $iteration -lt $MAX_ITERATIONS ]; do
    ((++iteration))
    echo ""
    echo "━━━ Iteration $iteration / $MAX_ITERATIONS ━━━"

    # ── Phase 1: Epic Review ──────────────────────────────────────
    # Check for epics whose children are all closed. These need a
    # holistic review before closing — the agent decides whether to
    # close or create follow-up tickets.
    REVIEW_EPIC=""
    OPEN_EPICS=$(bd list --status open --json 2>/dev/null \
        | jq -r '[.[] | select(.issue_type == "epic") | .id] | .[]' 2>/dev/null || true)
    for epic_id in $OPEN_EPICS; do
        ALL_CLOSED=$(bd show "$epic_id" --json 2>/dev/null | jq -r '
            .[0].dependents // []
            | [.[] | select(.dependency_type == "parent-child")]
            | if length == 0 then "no_children"
              elif all(.status == "closed") then "all_closed"
              else "has_open"
              end
        ' 2>/dev/null || echo "error")
        if [ "$ALL_CLOSED" = "all_closed" ]; then
            REVIEW_EPIC="$epic_id"
            break
        fi
    done

    if [ -n "$REVIEW_EPIC" ]; then
        echo "Epic review: $REVIEW_EPIC (all children closed)"

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

        OUTPUT=$(echo "$PROMPT" | claude --dangerously-skip-permissions --print 2>&1) || true
        echo "$OUTPUT"

        if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
            echo "All tasks complete!"
            exit 0
        fi
        if echo "$OUTPUT" | grep -q "<ralph>STUCK</ralph>"; then
            echo "Agent stuck on epic review. Human needed."
            exit 2
        fi

        bd sync 2>/dev/null || true
        no_task_streak=0
        sleep 2
        continue
    fi

    # ── Phase 2: Find actionable task ─────────────────────────────
    READY_JSON=$(bd ready --json --limit 50 2>/dev/null || echo "[]")

    # Try non-epic ready tasks first, sorted by priority
    TASK=$(echo "$READY_JSON" | jq -r '
        [.[] | select(.issue_type == "epic" | not)
             | select((.title | startswith("Epic:")) | not)]
        | sort_by(.priority // 99)
        | .[0] // empty
    ')

    # If no non-epic ready tasks, try open children of ready epics
    if [ -z "$TASK" ] || [ "$TASK" = "null" ]; then
        EPIC_IDS=$(echo "$READY_JSON" | jq -r '
            [.[] | select(.issue_type == "epic") | .id] | .[]
        ' 2>/dev/null || true)
        for eid in $EPIC_IDS; do
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
            echo "All tasks closed."
            echo "<promise>COMPLETE</promise>"
            exit 0
        fi

        echo "No actionable tasks (${READY_COUNT} ready epics, ${TOTAL} total open, streak ${no_task_streak}/${MAX_NO_TASK_STREAK})."

        if [ "$no_task_streak" -ge "$MAX_NO_TASK_STREAK" ]; then
            echo "Stuck: no actionable tasks after ${MAX_NO_TASK_STREAK} checks."
            echo "Remaining work may be blocked or needs epic review."
            bd blocked 2>/dev/null || true
            echo "<ralph>STUCK</ralph>"
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

    echo "Task: $TASK_ID - $TITLE"

    # Log to progress file
    echo "- $(date '+%Y-%m-%d %H:%M') | $TASK_ID [P$PRIORITY] | $TITLE" >> "$PROGRESS_FILE"

    # Build prompt
    TASK_DETAILS=$(bd show "$TASK_ID" 2>/dev/null)
    RALPH_CLAUDE=$(cat scripts/ralph/CLAUDE.md 2>/dev/null || echo "")
    PROJECT_CLAUDE=$(cat CLAUDE.md 2>/dev/null || echo "")

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
    OUTPUT=$(echo "$PROMPT" | claude --dangerously-skip-permissions --print 2>&1) || true
    echo "$OUTPUT"

    # Check stop signals
    if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
        echo "All tasks complete!"
        exit 0
    fi
    if echo "$OUTPUT" | grep -q "<ralph>STUCK</ralph>"; then
        echo "Agent stuck. Human needed."
        exit 2
    fi

    bd sync 2>/dev/null || true
    sleep 2
done

echo "Max iterations reached."
