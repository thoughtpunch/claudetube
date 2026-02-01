#!/usr/bin/env bash
# Pre-task hook: Shows ticket details and related context
set -euo pipefail

TICKET_ID="${1:-}"

if [ -z "$TICKET_ID" ]; then
    echo "Usage: $0 <ticket-id>"
    exit 1
fi

echo "═══════════════════════════════════════════"
echo "  PRE-TASK: $TICKET_ID"
echo "═══════════════════════════════════════════"
echo ""

# Show ticket details
bd show "$TICKET_ID"

# Search for related open tickets
TITLE=$(bd show "$TICKET_ID" --json | jq -r '.title')
KEYWORDS=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '\n' | grep -E '^.{4,}$' | grep -vE '^(this|that|with|from|have|will|epic|task|feature)$' | head -3)

echo ""
echo "RELATED TICKETS:"
for kw in $KEYWORDS; do
    RESULTS=$(bd search "$kw" --status open 2>/dev/null | grep -v "^$TICKET_ID" | head -3)
    if [ -n "$RESULTS" ]; then
        echo "  [$kw]: $RESULTS"
    fi
done

echo ""
echo "KEY FILES:"
echo "  src/claudetube/core.py      - Main processing logic"
echo "  src/claudetube/mcp_server.py - MCP tool definitions"
echo "  src/claudetube/urls.py      - URL/video ID handling"
echo "  CLAUDE.md                   - Project conventions"
echo ""
