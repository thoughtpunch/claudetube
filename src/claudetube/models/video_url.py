"""
VideoURL Pydantic model for URL parsing and validation.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, ClassVar
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from claudetube.config.providers import VIDEO_PROVIDERS

if TYPE_CHECKING:
    from claudetube.models.video_path import VideoPath

# Build domain lookup for fast matching
_DOMAIN_TO_PROVIDER: dict[str, dict] = {}
for _provider in VIDEO_PROVIDERS:
    for _domain in _provider["domains"]:
        _DOMAIN_TO_PROVIDER[_domain] = _provider

# Priority order for extracting the primary video ID from regex groups
_VIDEO_ID_PRIORITY = [
    "video_id",
    "clip_id",
    "clip_slug",
    "short_id",
    "embed_id",
    "av_id",
    "vredd_id",
    "channel_video_id",
    "player_video_id",
    "story_id",
    "owner_id",
    "channel",
    "user",
]


def _merge_query_params(url: str, provider_name: str, regex_data: dict) -> dict:
    """Merge query params into regex-extracted data. Regex captures take priority."""
    from claudetube.parsing.params import extract_query_params

    query_params = extract_query_params(url, provider_name)
    # Query params fill gaps -- don't overwrite regex captures
    merged = dict(regex_data)
    for key, value in query_params.items():
        if key not in merged:
            merged[key] = value
    return merged


class VideoURL(BaseModel):
    """Parsed and validated video URL with extracted metadata."""

    url: str = Field(..., description="Original URL (normalized)")
    video_id: str = Field(..., description="Extracted video ID for caching/lookup")
    provider: str | None = Field(None, description="Detected provider name")
    provider_data: dict = Field(
        default_factory=dict,
        description="Extra extracted fields (channel, clip_id, etc.)",
    )

    # Class-level compiled patterns for performance
    _compiled_patterns: ClassVar[dict[str, re.Pattern]] = {}

    @field_validator("url", mode="before")
    @classmethod
    def normalize_url(cls, v: str) -> str:
        """Normalize URL - strip whitespace, ensure scheme."""
        v = v.strip()
        if not v:
            raise ValueError("URL cannot be empty")
        if not v.startswith(("http://", "https://")):
            v = "https://" + v
        return v

    @model_validator(mode="after")
    def extract_video_id(self) -> VideoURL:
        """Extract video_id and provider from URL after validation."""
        parsed = urlparse(self.url)
        host = parsed.netloc.lower()

        # Strip www. and m. prefixes for matching
        host_clean = host.removeprefix("www.").removeprefix("m.")

        # Try to find matching provider
        provider = _DOMAIN_TO_PROVIDER.get(host) or _DOMAIN_TO_PROVIDER.get(host_clean)

        if provider:
            # Use site-specific pattern
            pattern_key = provider["name"]
            if pattern_key not in self._compiled_patterns:
                self._compiled_patterns[pattern_key] = re.compile(
                    provider["pattern"], re.IGNORECASE
                )

            pattern = self._compiled_patterns[pattern_key]
            match = pattern.search(self.url)

            if match:
                groups = match.groupdict()
                # Find the primary video_id (first non-None in priority order)
                vid = None
                for key in _VIDEO_ID_PRIORITY:
                    if groups.get(key):
                        vid = groups[key]
                        break

                if vid:
                    self.video_id = vid
                    self.provider = provider["name"]
                    # Start with regex captures
                    regex_data = {
                        k: v for k, v in groups.items() if v is not None
                    }
                    # Merge query params (regex captures take priority)
                    self.provider_data = _merge_query_params(
                        self.url, provider["name"], regex_data
                    )
                    return self

        # Fallback: generic extraction
        self.video_id = self._extract_generic_id(parsed)
        self.provider = None
        return self

    def _extract_generic_id(self, parsed) -> str:
        """Generic video ID extraction for unknown sites."""
        query = parse_qs(parsed.query)
        path_parts = [p for p in parsed.path.split("/") if p]

        # Check common query params
        for key in ["v", "video_id", "id", "vid", "videoId", "viewkey"]:
            if key in query:
                return query[key][0]

        # Look for ID-like path segments
        candidates: list[str] = []
        skip_segments = {
            "video",
            "videos",
            "watch",
            "embed",
            "shorts",
            "reel",
            "p",
            "status",
            "clips",
            "details",
            "channel",
            "user",
            "playlist",
            "live",
            "stories",
            "comments",
            "r",
            "w",
            "share",
            "post",
            "posts",
            "gag",
            "view",
            "media",
        }

        for part in path_parts:
            part_lower = part.lower()
            if part_lower in skip_segments:
                continue
            if part.startswith("@"):
                continue
            if "." in part and part.rsplit(".", 1)[-1] in [
                "html",
                "php",
                "aspx",
                "mp4",
                "gifv",
            ]:
                part = part.rsplit(".", 1)[0]
                if "-" in part:
                    prefix = part.split("-")[0]
                    if re.match(r"^[a-zA-Z]?\d+[a-zA-Z0-9]*$", prefix):
                        candidates.insert(0, prefix)
                        continue

            if re.match(r"^[a-zA-Z0-9_-]{4,50}$", part):
                candidates.append(part)

        if candidates:

            def id_score(s):
                length_score = len(s)
                has_numbers = any(c.isdigit() for c in s)
                return (0 if has_numbers else 100) + length_score

            candidates.sort(key=id_score)
            return candidates[0]

        # Ultimate fallback: hash the URL
        clean = re.sub(r"^https?://", "", self.url)
        clean = re.sub(r"[^\w.-]", "_", clean)
        if len(clean) > 50:
            return hashlib.sha256(self.url.encode()).hexdigest()[:16]
        return clean

    @classmethod
    def parse(cls, url: str) -> VideoURL:
        """Parse a URL and extract video information.

        Args:
            url: Video URL string

        Returns:
            VideoURL with extracted video_id and provider info

        Raises:
            ValueError: If URL is invalid or cannot be parsed
        """
        return cls(url=url, video_id="__pending__")  # type: ignore[call-arg]

    @classmethod
    def try_parse(cls, url: str) -> VideoURL | None:
        """Try to parse a URL, returning None on failure instead of raising."""
        try:
            return cls.parse(url)
        except Exception:
            return None

    @property
    def is_known_provider(self) -> bool:
        """Check if URL is from a known/supported provider."""
        return self.provider is not None

    @property
    def cache_key(self) -> str:
        """Get a filesystem-safe cache key for this video."""
        if len(self.video_id) <= 50 and re.match(r"^[\w-]+$", self.video_id):
            return self.video_id
        return hashlib.sha256(self.video_id.encode()).hexdigest()[:16]

    @property
    def video_path(self) -> VideoPath:
        """Construct a VideoPath from this parsed URL.

        Extracts domain from the URL hostname, and uses provider_data
        for channel/playlist if available.
        """
        from claudetube.models.video_path import (
            VideoPath,
            _sanitize_path_component,
            sanitize_domain,
        )

        parsed = urlparse(self.url)
        domain = sanitize_domain(parsed.netloc)

        channel = self.provider_data.get("channel")
        playlist = self.provider_data.get("playlist")

        if channel:
            channel = _sanitize_path_component(channel)
        if playlist:
            playlist = _sanitize_path_component(playlist)

        return VideoPath(
            domain=domain,
            channel=channel or None,
            playlist=playlist or None,
            video_id=self.video_id,
        )

    def __str__(self) -> str:
        if self.provider:
            return f"{self.provider}:{self.video_id}"
        return self.video_id

    def __repr__(self) -> str:
        return f"VideoURL(url={self.url!r}, video_id={self.video_id!r}, provider={self.provider!r})"
