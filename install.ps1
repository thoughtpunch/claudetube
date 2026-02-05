# claudetube installer (Windows)
# Installs Python package into a stable venv + registers MCP server

$ErrorActionPreference = "Stop"

$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallDir = Join-Path $HOME ".claudetube"
$VenvDir = Join-Path $InstallDir "venv"

Write-Host "Installing claudetube..." -ForegroundColor Cyan
Write-Host ""

# --- Check required system dependencies ---
$Missing = $false

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: python is not installed." -ForegroundColor Red
    Write-Host "  Install from: https://python.org/downloads/"
    Write-Host "  Or: winget install Python.Python.3.12"
    $Missing = $true
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: ffmpeg is not installed (required for frame extraction)." -ForegroundColor Red
    Write-Host "  Install with: winget install Gyan.FFmpeg"
    Write-Host "  Or: choco install ffmpeg"
    $Missing = $true
}

if ($Missing) {
    Write-Host ""
    Write-Host "Please install the missing dependencies above and re-run this script." -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# 1. Create stable venv at ~/.claudetube/venv/
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating Python venv at $VenvDir..."
    python -m venv $VenvDir
}

$PipExe = Join-Path $VenvDir "Scripts" "pip.exe"
$McpExe = Join-Path $VenvDir "Scripts" "claudetube-mcp.exe"

# 2. Install package with MCP support into the venv
Write-Host "Installing Python package (with MCP support)..."
& $PipExe install --upgrade pip -q
& $PipExe install "$RepoDir[mcp]" -q

# 3. Install faster-whisper
Write-Host "Installing faster-whisper..."
& $PipExe install faster-whisper -q

Write-Host ""
Write-Host "=== Installation complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Installed to: $InstallDir"
Write-Host "Python venv:  $VenvDir"
Write-Host ""
Write-Host "=== Register MCP Server ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Add claudetube to Claude Code:"
Write-Host ""
Write-Host "  claude mcp add claudetube $McpExe"
Write-Host ""
Write-Host "Then restart Claude Code. All 40+ tools will be available."
Write-Host ""
Write-Host "Example usage (just talk to Claude):"
Write-Host "  'Summarize this video: https://youtube.com/watch?v=...'"
Write-Host "  'What happens at minute 5 in video abc123?'"
Write-Host ""
