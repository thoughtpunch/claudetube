# YouTube Authentication Guide

YouTube increasingly requires authentication for video downloads. claudetube works out of the box for most public videos, but you may need additional configuration if you encounter 403 errors.

## Quick Summary

| Setup Level | What You Need | Coverage |
|-------------|--------------|----------|
| **Level 0** (default) | Nothing | Most public videos via `android_vr` client |
| **Level 1** | Install [deno](https://deno.land) | Better JS challenge solving |
| **Level 2** | Deno + browser cookies | Premium users get full access |
| **Level 3** | Deno + cookies + manual PO token | All content (token expires ~12hr) |
| **Level 4** | Deno + cookies + bgutil server | All content (automated token refresh) |

Most users won't need any configuration. Start here only if you're seeing errors.

---

## Prerequisites

### deno (Recommended)

yt-dlp uses deno to solve YouTube's JavaScript challenges (signature and n-parameter extraction).

```bash
# macOS
brew install deno

# Linux
curl -fsSL https://deno.land/install.sh | sh

# Windows
irm https://deno.land/install.ps1 | iex
```

Verify: `deno --version`

### Node.js (For bgutil server only)

Only needed if you want automated PO token generation (Level 4).

```bash
# macOS
brew install node

# Or use nvm
nvm install 18
```

Requires Node.js >= 18.

---

## Level 2: Browser Cookies

Cookies let yt-dlp authenticate as your YouTube account.

### Export cookies from your browser

1. Open a **private/incognito** browser window
2. Log into YouTube
3. Export cookies using one of:
   - [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) (Chrome)
   - [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/) (Firefox)
4. Save the file (e.g., `~/.config/claudetube/youtube-cookies.txt`)
5. **Close the private window** (YouTube rotates cookies on open tabs)

### Configure

Add to `.claudetube/config.yaml` or `~/.config/claudetube/config.yaml`:

```yaml
youtube:
  cookies_file: "~/.config/claudetube/youtube-cookies.txt"
```

> **Note:** YouTube Premium subscribers with cookies generally don't need PO tokens.

---

## Level 3: Manual PO Token

If cookies alone aren't enough, you can provide a manually-generated PO token.

### Generate a PO token

Follow the [yt-dlp PO Token Guide](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide) to generate a token using browser DevTools or BgUtils.

### Configure

```yaml
youtube:
  po_token: "mweb.gvs+YOUR_TOKEN_VALUE_HERE"
  cookies_file: "~/.config/claudetube/youtube-cookies.txt"
```

> **Warning:** GVS tokens expire after approximately 12 hours. You'll need to regenerate them periodically.

---

## Level 4: bgutil PO Token Provider (Automated)

The [bgutil-ytdlp-pot-provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider) plugin automates PO token generation. It runs a Node.js server that generates tokens on demand.

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

### Step 3: Configure (optional)

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

### Step 4: Verify

The plugin is loaded when yt-dlp verbose output shows:

```
[debug] [youtube] [pot] PO Token Providers: bgutil:http-1.2.2 (external), bgutil:script-1.2.2 (external)
```

Without the plugin:

```
[pot] PO Token Providers: none
```

---

## Troubleshooting

### Error: `HTTP Error 403: Forbidden`

YouTube is blocking the request. Work through the levels above starting from your current setup.

### Error: `Requested format is not available`

claudetube automatically retries with alternative clients. If this persists, try adding cookies (Level 2).

### Warning: `YouTube is forcing SABR streaming for this client`

This is informational. yt-dlp skips SABR-only formats and falls back to compatible clients. No action needed unless downloads fail.

### Warning: `GVS PO Token ... was not provided`

yt-dlp is noting that some higher-quality formats require a PO token. Downloads will still work via fallback clients. To access all formats, set up Level 3 or 4.

### bgutil plugin not detected

Verify the plugin is installed in the same Python environment as yt-dlp:

```bash
pip show bgutil-ytdlp-pot-provider
```

If installed but not detected, check that yt-dlp version is >= 2025.05.22:

```bash
yt-dlp --version
```

### bgutil server connection refused

Check the server is running:

```bash
curl http://127.0.0.1:4416/get_pot -X POST -H "Content-Type: application/json" -d '{}'
```

For Docker: `docker logs bgutil-pot`

---

## Config Reference

All YouTube options in `.claudetube/config.yaml`:

```yaml
youtube:
  # Netscape-format cookie file exported from your browser
  cookies_file: "~/.config/claudetube/youtube-cookies.txt"

  # Manual PO token (format: CLIENT.TYPE+TOKEN)
  # See: https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide
  po_token: "mweb.gvs+TOKEN_VALUE"

  # bgutil HTTP server URL (default: http://127.0.0.1:4416)
  # Only needed for non-default host/port
  pot_server_url: "http://127.0.0.1:4416"

  # bgutil script path (alternative to HTTP server, slower)
  pot_script_path: "~/bgutil-ytdlp-pot-provider/server/build/generate_once.js"
```

All fields are optional. With nothing configured, claudetube uses yt-dlp's default client selection.

---

## Further Reading

- [yt-dlp PO Token Guide](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide)
- [yt-dlp Extractors Wiki: YouTube](https://github.com/yt-dlp/yt-dlp/wiki/Extractors)
- [bgutil-ytdlp-pot-provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider)
- [yt-dlp YouTube Extractor Args](https://github.com/yt-dlp/yt-dlp#youtube)
