# claudetube installer (Windows)
# Installs Python package into a stable venv + Claude Code slash commands + MCP server

$ErrorActionPreference = "Stop"

$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallDir = Join-Path $HOME ".claudetube"
$VenvDir = Join-Path $InstallDir "venv"
$ClaudeCommandsDir = Join-Path $HOME ".claude" "commands"

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

# 4. Install Claude Code commands
Write-Host "Installing Claude Code slash commands..."
New-Item -ItemType Directory -Force -Path $ClaudeCommandsDir | Out-Null

$YtCmd = Join-Path $RepoDir "commands" "yt.md"
if (Test-Path $YtCmd) {
    Copy-Item $YtCmd (Join-Path $ClaudeCommandsDir "yt.md") -Force
    Write-Host "  Installed /yt"
}

$YtSubDir = Join-Path $RepoDir "commands" "yt"
if (Test-Path $YtSubDir) {
    $DestYtDir = Join-Path $ClaudeCommandsDir "yt"
    if (Test-Path $DestYtDir) { Remove-Item -Recurse -Force $DestYtDir }
    Copy-Item -Recurse $YtSubDir $DestYtDir
    Get-ChildItem (Join-Path $DestYtDir "*.md") | ForEach-Object {
        $name = $_.BaseName
        Write-Host "  Installed /yt:$name"
    }
}

Write-Host ""
Write-Host "=== Installation complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Installed to: $InstallDir"
Write-Host "Python venv:  $VenvDir"
Write-Host ""
Write-Host "Slash commands (in ANY Claude Code session):"
Write-Host "  /yt <url> [question]          - Analyze a YouTube video"
Write-Host "  /yt:see <id> <timestamp>      - Extract frames at timestamp"
Write-Host "  /yt:hq <id> <timestamp>       - Extract high-quality frames"
Write-Host "  /yt:transcript <id>           - Show cached transcript"
Write-Host "  /yt:list                      - List cached videos"
Write-Host ""
Write-Host "=== MCP Server Setup (optional) ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To add claudetube as an MCP server for Claude Code:"
Write-Host ""
Write-Host "  claude mcp add claudetube $McpExe"
Write-Host ""
Write-Host "Or start it manually:"
Write-Host ""
Write-Host "  $McpExe"
Write-Host ""
