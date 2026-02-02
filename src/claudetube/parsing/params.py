"""
URL query parameter extraction and timestamp parsing.

Extracts well-known query parameters from video URLs (timestamps, playlist IDs,
channel hints) and parses various timestamp formats to float seconds.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

# Mapping: provider name -> {url_param: canonical_field_name}
PROVIDER_PARAMS: dict[str, dict[str, str]] = {
    "youtube": {
        "list": "playlist",
        "t": "start_time",
        "start": "start_time",
        "index": "playlist_position",
        "ab_channel": "channel_hint",
        "feature": "referral_source",
    },
    "vimeo": {
        "h": "private_hash",
        "time": "start_time",
    },
    "twitch": {
        "t": "start_time",
    },
}

# Generic params applied to all providers
_GENERIC_PARAMS: dict[str, str] = {
    "utm_source": "referral_source",
    "utm_medium": "referral_medium",
}

# Regex for XhYmZs compact timestamp format (e.g. "1h2m3s", "2m30s", "45s")
_COMPACT_TS_RE = re.compile(
    r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$",
    re.IGNORECASE,
)


def extract_query_params(url: str, provider_name: str) -> dict[str, str]:
    """Extract known query parameters from a video URL.

    Parses the URL query string and maps known parameters to canonical field
    names based on the provider. Also extracts generic UTM parameters for all
    providers.

    Args:
        url: Full video URL string.
        provider_name: Lowercase provider name (e.g. "youtube", "vimeo", "twitch").

    Returns:
        Dict mapping canonical field names to their string values.
        Empty dict if no known parameters are found.
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    result: dict[str, str] = {}

    # Provider-specific params (first match wins when multiple params map to same field)
    provider_map = PROVIDER_PARAMS.get(provider_name.lower(), {})
    for param, field in provider_map.items():
        if field not in result:
            values = qs.get(param)
            if values:
                # parse_qs returns lists; take the first value
                result[field] = values[0]

    # Generic params (all providers) -- don't overwrite provider-specific values
    for param, field in _GENERIC_PARAMS.items():
        if field not in result:
            values = qs.get(param)
            if values:
                result[field] = values[0]

    return result


def parse_timestamp(value: str) -> float | None:
    """Parse various timestamp formats to seconds.

    Supports:
      - Pure seconds: "120", "300"
      - Compact duration: "2m30s", "1h2m3s", "45s"
      - Colon format: "1:30:45" (h:m:s), "2:30" (m:s)

    Args:
        value: Timestamp string to parse.

    Returns:
        Float seconds, or None if the value cannot be parsed.
    """
    value = value.strip()
    if not value:
        return None

    # Try pure seconds (integer or float)
    try:
        result = float(value)
        if result < 0:
            return None
        return result
    except ValueError:
        pass

    # Try compact format: XhYmZs
    m = _COMPACT_TS_RE.match(value)
    if m and any(m.groups()):
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
        seconds = int(m.group(3) or 0)
        return float(hours * 3600 + minutes * 60 + seconds)

    # Try colon format: H:M:S or M:S
    parts = value.split(":")
    if len(parts) in (2, 3):
        try:
            int_parts = [int(p) for p in parts]
            if any(p < 0 for p in int_parts):
                return None
            if len(int_parts) == 3:
                return float(int_parts[0] * 3600 + int_parts[1] * 60 + int_parts[2])
            return float(int_parts[0] * 60 + int_parts[1])
        except ValueError:
            pass

    return None
