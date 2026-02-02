[← Documentation](../README.md)

# YouTube Authentication Guide

YouTube increasingly requires authentication for video downloads. claudetube works out of the box for most public videos, but you may need additional configuration if you encounter errors.

## What Changed and Why

YouTube has been tightening third-party access to video streams through two mechanisms:

1. **PO (Proof of Origin) tokens** -- cryptographic tokens that prove a request comes from a real YouTube client, not a scraper. Without them, YouTube returns 403 errors or withholds certain formats.

2. **SABR (Server-Based Adaptive Bit Rate)** -- a new streaming protocol that replaces direct download URLs with a proprietary binary format. yt-dlp skips SABR-only formats and falls back to clients that still provide direct URLs.

3. **JavaScript challenges** -- YouTube now requires solving JS-based signature and n-parameter challenges. yt-dlp delegates this to an external JS runtime (deno or node).

**The bottom line:** yt-dlp handles the complexity. Your job is to make sure the right tools and credentials are available on your system so yt-dlp can do its work.

For deep technical details, see the [yt-dlp PO Token Guide](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide).

---

## Do I Need to Do Anything?

**If your videos are downloading fine, no action needed.** claudetube uses yt-dlp's default client selection, which currently works for most public videos without any configuration.

You'll likely hit problems if:
- You see `HTTP Error 403: Forbidden` on YouTube videos
- You need age-restricted or private/unlisted content
- You're downloading many videos in rapid succession
- YouTube has blocked the `android_vr` fallback client (may happen in the future)

The `android_vr` client works today without PO tokens, but YouTube has been systematically disabling clients that don't require authentication. It's worth setting up at least Level 1 (deno) proactively.

---

## Quick Summary

| Setup Level | What You Need | Coverage |
|-------------|--------------|----------|
| **Level 0** (default) | Nothing | Most public videos via `android_vr` client |
| **Level 1** | Install [deno](https://deno.land) | Better JS challenge solving |
| **Level 2** | Deno + browser cookies | Premium users get full access |
| **Level 3** | Deno + cookies + manual PO token | All content (token expires ~12hr) |
| **Level 4** | Deno + cookies + bgutil server | All content (automated token refresh) |

Most users won't need more than Level 1 or 2. Start here only if you're seeing errors.

---

## Prerequisites

### deno (Recommended)

yt-dlp (>= 2026.01.29) uses deno to solve YouTube's JavaScript challenges for signature and n-parameter extraction. Without it, some clients may not work.

```bash
# macOS
brew install deno

# Linux
curl -fsSL https://deno.land/install.sh | sh

# Windows
irm https://deno.land/install.ps1 | iex
```

Verify: `deno --version`

See [yt-dlp #15012](https://github.com/yt-dlp/yt-dlp/issues/15012) for context on why an external JS runtime is now required.

### Node.js (For bgutil server only)

Only needed if you want automated PO token generation (Level 4).

```bash
# macOS
brew install node

# Or use nvm
nvm install 18
```

Requires Node.js >= 18.

### Docker (Optional)

An alternative way to run the bgutil server without installing Node.js locally. See [Level 4](#level-4-bgutil-po-token-provider-automated) below.

---

## Level 2: Browser Cookies (Easiest Auth)

Cookies let yt-dlp authenticate as your YouTube account. For YouTube Premium subscribers, this is usually sufficient -- no PO tokens needed.

### Option A: Export cookies from browser

1. Open a **private/incognito** browser window
2. Log into YouTube
3. Export cookies using one of:
   - [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) (Chrome)
   - [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/) (Firefox)
4. Save the file (e.g., `~/.config/claudetube/youtube-cookies.txt`)
5. **Close the private window** -- YouTube rotates cookies on open tabs

Add to `.claudetube/config.yaml` or `~/.config/claudetube/config.yaml`:

```yaml
youtube:
  cookies_file: "~/.config/claudetube/youtube-cookies.txt"
```

### Option B: Extract cookies directly from browser

yt-dlp can read cookies directly from your browser's cookie store. The browser must NOT be running when you use this option.

```yaml
youtube:
  cookies_from_browser: "firefox"
```

Supported browsers: `firefox`, `chrome`, `chromium`, `brave`, `edge`, `safari`, `opera`, `vivaldi`

### Important Caveats

- **Export from a private window** -- prevents cookie rotation from interfering
- **Close the browser after export** -- YouTube rotates session cookies on open tabs
- **Premium users:** cookies alone are generally sufficient (no PO tokens needed)
- **Free users:** cookies help but you may still need PO tokens for some content
- **Priority:** `cookies_from_browser` takes precedence over `cookies_file` if both are set

For more details, see [yt-dlp: Exporting YouTube Cookies](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies).

---

## Level 3: Manual PO Token (Advanced)

If cookies alone aren't enough (common for free-tier accounts), you can provide a manually-generated PO token.

### Generate a PO token

1. Open [YouTube Music](https://music.youtube.com) in your browser
2. Open DevTools (F12) → **Network** tab → filter for `v1/player`
3. Play any video
4. Find `poToken` in the request payload
5. Copy the token value

The token format is `CLIENT.TYPE+TOKEN_VALUE`, e.g., `mweb.gvs+AbCdEf...`

### Configure

```yaml
youtube:
  po_token: "mweb.gvs+YOUR_TOKEN_VALUE_HERE"
  cookies_file: "~/.config/claudetube/youtube-cookies.txt"
```

> **Warning:** GVS tokens expire after approximately 12 hours. You'll need to regenerate them periodically. For automated token generation, see Level 4.

> **Important:** Cookies and PO tokens must match -- a token generated in one session cannot be used with cookies from a different session.

For the full extraction procedure, see the [yt-dlp PO Token Guide](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide).

---

## Level 4: bgutil PO Token Provider (Recommended for Full Access)

The [bgutil-ytdlp-pot-provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider) plugin automates PO token generation. It runs a Node.js server that generates tokens on demand, so you never need to manually extract them.

### Step 1: Install the yt-dlp plugin

```bash
pip install "claudetube[youtube-pot]"
```

This installs `bgutil-ytdlp-pot-provider` which yt-dlp auto-discovers via its plugin system.

### Step 2: Start the bgutil server

**Option A: Docker (recommended)**

```bash
docker run -d --name bgutil-pot \
  -p 4416:4416 \
  brainicism/bgutil-ytdlp-pot-provider
```

**Option B: Native Node.js**

```bash
git clone https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git
cd bgutil-ytdlp-pot-provider/server
npm install
npx tsc
node build/main.js
```

The server listens on `http://127.0.0.1:4416` by default.

### Step 3: Verify

Run yt-dlp with verbose output to confirm the plugin is loaded:

```bash
yt-dlp -v --skip-download "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 2>&1 | grep "PO Token"
```

**Plugin loaded:**
```
[debug] [youtube] [pot] PO Token Providers: bgutil:http-1.2.2 (external), bgutil:script-1.2.2 (external)
```

**Plugin NOT loaded:**
```
[pot] PO Token Providers: none
```

### Step 4: Configure (optional)

If the server runs on the default port, no configuration is needed -- the plugin finds it automatically.

For a non-default URL:

```yaml
youtube:
  pot_server_url: "http://127.0.0.1:4416"
```

For script mode (no server, slower):

```yaml
youtube:
  pot_script_path: "~/bgutil-ytdlp-pot-provider/server/build/generate_once.js"
```

---

## Troubleshooting

### Error: `HTTP Error 403: Forbidden`

YouTube is blocking the request. This usually means a PO token is needed or has expired.

**What to try:**
1. Make sure `deno` is installed: `deno --version`
2. Add browser cookies (Level 2)
3. If using manual PO token, regenerate it (they expire after ~12 hours)
4. Set up bgutil server for automated tokens (Level 4)

### Error: `Requested format is not available`

The YouTube client being used is blocked or doesn't have the requested format. claudetube automatically retries with alternative clients.

**If this persists:** Add cookies (Level 2) to enable additional clients.

### Warning: `YouTube is forcing SABR streaming for this client`

This is **informational, not an error**. yt-dlp is telling you that the `web` or `web_safari` client returned SABR-only formats, which it skipped. Downloads proceed via fallback clients (`android_vr`, etc.).

**No action needed** unless the download actually fails.

### Warning: `GVS PO Token ... was not provided`

yt-dlp is noting that some higher-quality formats require a PO token. Downloads will still work via fallback clients that don't require tokens. To access all formats, set up Level 3 or 4.

### `PO Token Providers: none` in verbose output

The bgutil plugin is not installed or the server is not running.

**Check the plugin is installed:**
```bash
pip show bgutil-ytdlp-pot-provider
```

**Check yt-dlp version** (must be >= 2025.05.22):
```bash
yt-dlp --version
```

**Ensure the plugin is in the same Python environment as yt-dlp.** If you're using a virtual environment, both must be installed there.

### bgutil server connection refused

The server is not running or not reachable.

**Test the server directly:**
```bash
curl http://127.0.0.1:4416/get_pot -X POST -H "Content-Type: application/json" -d '{}'
```

**For Docker:** Check container status and logs:
```bash
docker ps | grep bgutil
docker logs bgutil-pot
```

### Token expired (manual PO token)

Manual GVS tokens expire after ~12 hours. Regenerate using the DevTools method in Level 3, or switch to bgutil (Level 4) for automated refresh.

---

## Config Reference

All YouTube options in `.claudetube/config.yaml` or `~/.config/claudetube/config.yaml`:

```yaml
youtube:
  # --- Cookies (choose one) ---

  # Option A: Extract cookies directly from browser (browser must not be running)
  # Supported: firefox, chrome, chromium, brave, edge, safari, opera, vivaldi
  cookies_from_browser: "firefox"

  # Option B: Netscape-format cookie file exported from your browser
  cookies_file: "~/.config/claudetube/youtube-cookies.txt"

  # --- PO Token ---

  # Manual PO token (format: CLIENT.TYPE+TOKEN)
  # See: https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide
  po_token: "mweb.gvs+TOKEN_VALUE"

  # --- bgutil PO Token Provider ---

  # bgutil HTTP server URL (default: http://127.0.0.1:4416)
  # Only needed for non-default host/port
  pot_server_url: "http://127.0.0.1:4416"

  # bgutil script path (alternative to HTTP server, slower)
  pot_script_path: "~/bgutil-ytdlp-pot-provider/server/build/generate_once.js"
```

All fields are optional. With nothing configured, claudetube uses yt-dlp's default client selection.

**Priority order:** `cookies_from_browser` > `cookies_file` > `po_token`

---

## Further Reading

- [yt-dlp PO Token Guide](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide) -- Canonical reference for PO token types and manual extraction
- [yt-dlp Extractors Wiki: YouTube](https://github.com/yt-dlp/yt-dlp/wiki/Extractors) -- Client table, cookie export guide
- [yt-dlp: Exporting YouTube Cookies](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies) -- Cookie export best practices
- [bgutil-ytdlp-pot-provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider) -- Automated PO token plugin
- [yt-dlp YouTube Extractor Args](https://github.com/yt-dlp/yt-dlp#youtube) -- CLI syntax reference
- [yt-dlp #12482](https://github.com/yt-dlp/yt-dlp/issues/12482) -- SABR streaming tracking issue
- [yt-dlp #15012](https://github.com/yt-dlp/yt-dlp/issues/15012) -- External JS runtime requirement announcement
