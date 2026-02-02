#!/usr/bin/env bash
# extract_followups.sh - Extract "Left undone" and "Gotchas" from closed beads tickets
# Usage: ./scripts/extract_followups.sh [--json]
#
# Scans all closed tickets for completion comments containing
# "## Left undone" and "## Gotchas" sections, then outputs a report
# of items that may need follow-up tickets.

set -euo pipefail

JSON_MODE=false
if [[ "${1:-}" == "--json" ]]; then
    JSON_MODE=true
fi

# Get all closed ticket IDs
TICKET_IDS=$(bd list --status=closed --json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for d in data:
    print(d['id'] + '\t' + d['title'])
")

if [[ -z "$TICKET_IDS" ]]; then
    echo "No closed tickets found."
    exit 0
fi

# Collect all comments from closed tickets and extract undone/gotchas
python3 -c "
import subprocess
import json
import re
import sys

json_mode = $( $JSON_MODE && echo 'True' || echo 'False' )

ticket_lines = '''$TICKET_IDS'''.strip().split('\n')
tickets = []
for line in ticket_lines:
    parts = line.split('\t', 1)
    tickets.append({'id': parts[0], 'title': parts[1] if len(parts) > 1 else ''})

undone_items = []
gotcha_items = []

for ticket in tickets:
    tid = ticket['id']
    title = ticket['title']

    # Get comments as JSON
    try:
        result = subprocess.run(
            ['bd', 'comments', tid, '--json'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            continue
        comments = json.loads(result.stdout)
    except Exception:
        continue

    for comment in comments:
        text = comment.get('text', '')

        # Extract 'Left undone' section
        undone_match = re.search(
            r'## Left undone\s*\n(.*?)(?=\n## |\Z)',
            text, re.DOTALL
        )
        if undone_match:
            content = undone_match.group(1).strip()
            # Skip if it's just 'None' or 'N/A'
            if content.lower() not in ('none', 'n/a', '- none', '- n/a', 'none.', '- none.'):
                items = [line.strip().lstrip('- ') for line in content.split('\n') if line.strip() and line.strip() != '-']
                if items:
                    undone_items.append({
                        'ticket_id': tid,
                        'ticket_title': title,
                        'items': items
                    })

        # Extract 'Gotchas' section
        gotcha_match = re.search(
            r'## Gotchas\s*\n(.*?)(?=\n## |\Z)',
            text, re.DOTALL
        )
        if gotcha_match:
            content = gotcha_match.group(1).strip()
            if content.lower() not in ('none', 'n/a', '- none', '- n/a', 'none.', '- none.'):
                items = [line.strip().lstrip('- ') for line in content.split('\n') if line.strip() and line.strip() != '-']
                if items:
                    gotcha_items.append({
                        'ticket_id': tid,
                        'ticket_title': title,
                        'items': items
                    })

if json_mode:
    output = {
        'total_closed_tickets': len(tickets),
        'tickets_with_undone': len(undone_items),
        'tickets_with_gotchas': len(gotcha_items),
        'undone': undone_items,
        'gotchas': gotcha_items
    }
    print(json.dumps(output, indent=2))
else:
    print('=' * 70)
    print('FOLLOW-UP REPORT FROM CLOSED BEADS TICKETS')
    print('=' * 70)
    print(f'Total closed tickets scanned: {len(tickets)}')
    print(f'Tickets with undone items:    {len(undone_items)}')
    print(f'Tickets with gotchas:         {len(gotcha_items)}')
    print()

    print('=' * 70)
    print('LEFT UNDONE (deferred work that may need follow-up tickets)')
    print('=' * 70)
    if not undone_items:
        print('  (none found)')
    for entry in undone_items:
        print(f'\n  [{entry[\"ticket_id\"]}] {entry[\"ticket_title\"]}')
        for item in entry['items']:
            print(f'    • {item}')

    print()
    print('=' * 70)
    print('GOTCHAS (surprises, edge cases, patterns to be aware of)')
    print('=' * 70)
    if not gotcha_items:
        print('  (none found)')
    for entry in gotcha_items:
        print(f'\n  [{entry[\"ticket_id\"]}] {entry[\"ticket_title\"]}')
        for item in entry['items']:
            print(f'    • {item}')

    print()
    print('=' * 70)
    print('END OF REPORT')
    print('=' * 70)
"
