#!/usr/bin/env bash
# Sync beads and commit if changed
set -euo pipefail

bd sync 2>/dev/null || true

if git diff --quiet .beads/issues.jsonl 2>/dev/null; then
    echo "No changes to sync."
    exit 0
fi

git add .beads/issues.jsonl
git commit -m "bd sync: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Synced."
