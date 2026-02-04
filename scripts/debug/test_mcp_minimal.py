#!/usr/bin/env python3
"""Minimal MCP server for debugging."""
import sys
from datetime import datetime
from pathlib import Path

LOG = Path.home() / "Library/Logs/Claude/test-mcp-minimal.log"

def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{datetime.now().isoformat()} {msg}\n")

log("=== MINIMAL SERVER STARTING ===")
log(f"Python: {sys.executable}")

# Log all stdin until EOF
log("Reading stdin...")
try:
    line_count = 0
    while True:
        line = sys.stdin.readline()
        if not line:
            log("EOF reached (empty line)")
            break
        line_count += 1
        log(f"Line {line_count}: {line.strip()[:200]}")
except Exception as e:
    log(f"Error reading stdin: {e}")

log(f"Total lines read: {line_count}")
log("=== SERVER EXITING ===")
