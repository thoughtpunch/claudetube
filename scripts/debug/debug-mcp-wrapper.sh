#!/bin/bash
# Debug wrapper for claudetube MCP server

LOG="/Users/danielbarrett/Library/Logs/Claude/mcp-wrapper-debug.log"

echo "=== WRAPPER STARTING ===" >> "$LOG"
echo "Date: $(date -Iseconds)" >> "$LOG"
echo "PID: $$" >> "$LOG"
echo "PPID: $PPID" >> "$LOG"
echo "PWD: $PWD" >> "$LOG"
echo "TTY: $(tty 2>&1)" >> "$LOG"
echo "Stdin fd 0 info: $(ls -la /dev/fd/0 2>&1)" >> "$LOG"
echo "Stdout fd 1 info: $(ls -la /dev/fd/1 2>&1)" >> "$LOG"
echo "Stderr fd 2 info: $(ls -la /dev/fd/2 2>&1)" >> "$LOG"
echo "Environment:" >> "$LOG"
env | sort >> "$LOG"
echo "=== CALLING PYTHON ===" >> "$LOG"

# Exec the real server (replaces this shell process)
exec /Users/danielbarrett/sites/claudetube/.venv/bin/claudetube-mcp "$@"
