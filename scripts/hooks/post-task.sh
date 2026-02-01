#!/usr/bin/env bash
# Post-task hook: Checks Definition of Done before closure
set -euo pipefail

TICKET_ID="${1:-}"

if [ -z "$TICKET_ID" ]; then
    echo "Usage: $0 <ticket-id>"
    exit 1
fi

ERRORS=0

echo "═══════════════════════════════════════════"
echo "  POST-TASK: $TICKET_ID"
echo "═══════════════════════════════════════════"

# Check 1: Completion comment
COMMENTS=$(bd show "$TICKET_ID" --json | jq '.comments | length')
if [ "$COMMENTS" -eq 0 ]; then
    echo "ERROR: No completion comment"
    echo "  Run: bd comments add $TICKET_ID \"## What was done ...\""
    ERRORS=$((ERRORS + 1))
else
    echo "OK: $COMMENTS comment(s)"
fi

# Check 2: Commit references ticket
COMMITS=$(git log --oneline --all --grep="$TICKET_ID" 2>/dev/null | wc -l | tr -d ' ')
if [ "$COMMITS" -eq 0 ]; then
    echo "ERROR: No commit references $TICKET_ID"
    echo "  Run: git commit -m \"feat: $TICKET_ID - Description\""
    ERRORS=$((ERRORS + 1))
else
    echo "OK: $COMMITS commit(s)"
fi

# Check 3: Uncommitted changes (warning)
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    echo "WARN: Uncommitted changes"
fi

echo ""
if [ $ERRORS -gt 0 ]; then
    echo "BLOCKED: Fix $ERRORS error(s) before closing"
    exit 1
else
    echo "READY TO CLOSE"
    exit 0
fi
