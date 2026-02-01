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

exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo "══════════════════════════════════════════"
echo "  RALPH LOOP - claudetube"
echo "  Max: $MAX_ITERATIONS iterations"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════"

iteration=0
while [ $iteration -lt $MAX_ITERATIONS ]; do
    ((++iteration))
    echo ""
    echo "━━━ Iteration $iteration / $MAX_ITERATIONS ━━━"

    # Get ready tasks, filter for non-epics, sort by priority
    TASK=$(bd ready --json --limit 50 2>/dev/null | jq -r '
        [.[] | select(.issue_type != "epic")
             | select((.title | startswith("Epic:") | not))]
        | sort_by(.priority // 99)
        | .[0] // empty
    ')

    if [ -z "$TASK" ] || [ "$TASK" = "null" ]; then
        echo "No ready tasks."
        TOTAL=$(bd list --status open --json 2>/dev/null | jq 'length' || echo "0")
        if [ "$TOTAL" -eq 0 ]; then
            echo "<promise>COMPLETE</promise>"
            exit 0
        fi
        sleep 2
        continue
    fi

    TASK_ID=$(echo "$TASK" | jq -r '.id')
    TITLE=$(echo "$TASK" | jq -r '.title')

    echo "Task: $TASK_ID - $TITLE"

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
