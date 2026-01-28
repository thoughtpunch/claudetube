"""
URL parsing and video ID extraction for claudetube.

Supports 70+ video providers with site-specific regex patterns,
plus a generic fallback for unknown sites.
"""

import hashlib
import re
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

# Video provider patterns - ranked by popularity/traffic
# Each pattern extracts video_id (and optionally other fields like channel, clip_id, etc.)
# Note: Python regex doesn't allow duplicate named groups, so we use unique names
# (video_id, video_id_alt, short_id, etc.) and resolve to primary ID in extraction
VIDEO_PROVIDERS = [
    {
        "rank": 1,
        "name": "YouTube",
        "domains": ["youtube.com", "youtu.be", "m.youtube.com", "www.youtube.com", "music.youtube.com"],
        "pattern": r"(?:youtube\.com/(?:watch\?v=|embed/|v/|shorts/|live/)|youtu\.be/)(?P<video_id>[a-zA-Z0-9_-]{11})",
    },
    {
        "rank": 2,
        "name": "Facebook",
        "domains": ["facebook.com", "fb.watch", "www.facebook.com", "m.facebook.com"],
        "pattern": r"(?:facebook\.com/(?:watch/\?v=|[^/]+/videos/|stories/|story\.php\?story_fbid=)|fb\.watch/)(?P<video_id>[a-zA-Z0-9_-]+)",
    },
    {
        "rank": 3,
        "name": "Instagram",
        "domains": ["instagram.com", "www.instagram.com"],
        "pattern": r"instagram\.com/(?:reel|p)/(?P<video_id>[A-Za-z0-9_-]+)|instagram\.com/stories/(?P<user>[^/]+)/(?P<story_id>\d+)",
    },
    {
        "rank": 4,
        "name": "TikTok",
        "domains": ["tiktok.com", "www.tiktok.com", "vm.tiktok.com"],
        "pattern": r"tiktok\.com/(?:@[\w.-]+/video/(?P<video_id>\d+)|t/(?P<short_id>[A-Za-z0-9]+))",
    },
    {
        "rank": 5,
        "name": "Pornhub",
        "domains": ["pornhub.com", "www.pornhub.com"],
        "pattern": r"pornhub\.com/(?:view_video\.php\?viewkey=|embed/)(?P<video_id>[a-z0-9]+)",
    },
    {
        "rank": 6,
        "name": "Twitch",
        "domains": ["twitch.tv", "www.twitch.tv", "clips.twitch.tv"],
        "pattern": r"(?:twitch\.tv/(?:videos/(?P<video_id>\d+)|(?P<channel>[a-zA-Z0-9_]+)/clip/(?P<clip_id>[a-zA-Z0-9_-]+))|clips\.twitch\.tv/(?P<clip_slug>[a-zA-Z0-9_-]+))",
    },
    {
        "rank": 7,
        "name": "Vimeo",
        "domains": ["vimeo.com", "player.vimeo.com"],
        "pattern": r"(?:vimeo\.com/(?:video/)?(?P<video_id>\d+)(?:/(?P<hash>[a-f0-9]+))?|vimeo\.com/(?:channels|groups)/[\w-]+/(?:videos/)?(?P<channel_video_id>\d+)|player\.vimeo\.com/video/(?P<player_video_id>\d+))",
    },
    {
        "rank": 8,
        "name": "Xvideos",
        "domains": ["xvideos.com", "www.xvideos.com"],
        "pattern": r"xvideos\.com/(?:video|embedframe)/(?P<video_id>[a-z0-9]+)",
    },
    {
        "rank": 9,
        "name": "Twitter/X",
        "domains": ["twitter.com", "x.com", "www.twitter.com", "www.x.com", "mobile.twitter.com"],
        "pattern": r"(?:twitter|x)\.com/[^/]+/status/(?P<video_id>\d+)",
    },
    {
        "rank": 10,
        "name": "Dailymotion",
        "domains": ["dailymotion.com", "dai.ly", "www.dailymotion.com"],
        "pattern": r"(?:dailymotion\.com/(?:video|embed/video)/|dai\.ly/)(?P<video_id>[a-z0-9]+)",
    },
    {
        "rank": 11,
        "name": "xHamster",
        "domains": ["xhamster.com", "www.xhamster.com"],
        "pattern": r"xhamster\.com/(?:videos|embed)/(?P<video_id>[a-z0-9-]+)",
    },
    {
        "rank": 12,
        "name": "Redtube",
        "domains": ["redtube.com", "www.redtube.com"],
        "pattern": r"redtube\.com/(?:(?P<video_id>[0-9]+)|embed\?id=(?P<embed_id>[0-9]+))",
    },
    {
        "rank": 13,
        "name": "Douyin",
        "domains": ["douyin.com", "www.douyin.com"],
        "pattern": r"douyin\.com/(?:video|user)/(?P<video_id>[a-zA-Z0-9]+)",
    },
    {
        "rank": 14,
        "name": "Kuaishou",
        "domains": ["kuaishou.com", "www.kuaishou.com"],
        "pattern": r"kuaishou\.com/(?:short-video|profile)/(?P<video_id>[a-zA-Z0-9]+)",
    },
    {
        "rank": 15,
        "name": "Rumble",
        "domains": ["rumble.com", "www.rumble.com"],
        "pattern": r"rumble\.com/(?:(?P<video_id>[a-z0-9]+)-[\w-]+\.html|embed/(?P<embed_id>[a-z0-9]+))",
    },
    {
        "rank": 16,
        "name": "Reddit",
        "domains": ["reddit.com", "www.reddit.com", "old.reddit.com", "v.redd.it"],
        "pattern": r"(?:reddit\.com/r/[^/]+/comments/(?P<video_id>[a-z0-9]+)|v\.redd\.it/(?P<vredd_id>[a-z0-9]+))",
    },
    {
        "rank": 17,
        "name": "iQiyi",
        "domains": ["iqiyi.com", "iq.com", "www.iqiyi.com"],
        "pattern": r"(?:iqiyi|iq)\.com/(?:v_|w_|a_)(?P<video_id>[a-z0-9]+)",
    },
    {
        "rank": 18,
        "name": "Bilibili",
        "domains": ["bilibili.com", "www.bilibili.com", "b23.tv"],
        "pattern": r"bilibili\.com/video/(?:av(?P<av_id>\d+)|(?P<video_id>BV[a-zA-Z0-9]+))",
    },
    {
        "rank": 19,
        "name": "Streamable",
        "domains": ["streamable.com"],
        "pattern": r"streamable\.com/(?:e/|o/)?(?P<video_id>[a-z0-9]+)",
    },
    {
        "rank": 20,
        "name": "Imgur",
        "domains": ["imgur.com", "i.imgur.com"],
        "pattern": r"(?:imgur\.com/(?:gallery/|a/)?|i\.imgur\.com/)(?P<video_id>[a-zA-Z0-9]+)(?:\.(?:mp4|gifv))?",
    },
    {
        "rank": 21,
        "name": "Loom",
        "domains": ["loom.com", "www.loom.com"],
        "pattern": r"loom\.com/(?:share|embed)/(?P<video_id>[a-f0-9]{32})",
    },
    {
        "rank": 22,
        "name": "Kick",
        "domains": ["kick.com", "www.kick.com"],
        "pattern": r"kick\.com/(?P<channel>[a-zA-Z0-9_-]+)(?:/(?:clips|videos)/(?P<video_id>[a-zA-Z0-9_-]+))?",
    },
    {
        "rank": 23,
        "name": "Youku",
        "domains": ["youku.com", "v.youku.com"],
        "pattern": r"youku\.com/(?:v_show/id_|embed/)(?P<video_id>[a-zA-Z0-9=]+)",
    },
    {
        "rank": 24,
        "name": "Huya",
        "domains": ["huya.com", "www.huya.com"],
        "pattern": r"huya\.com/(?:(?P<channel>[a-zA-Z0-9_]+)|video/play/(?P<video_id>\d+))",
    },
    {
        "rank": 25,
        "name": "Douyu",
        "domains": ["douyu.com", "www.douyu.com"],
        "pattern": r"douyu\.com/(?:(?P<channel>[a-zA-Z0-9_]+)|v/(?P<video_id>\d+))",
    },
    {
        "rank": 26,
        "name": "Niconico",
        "domains": ["nicovideo.jp", "www.nicovideo.jp"],
        "pattern": r"nicovideo\.jp/(?:watch/)?(?P<video_id>(?:sm|so|nm)\d+)",
    },
    {
        "rank": 27,
        "name": "Spankbang",
        "domains": ["spankbang.com", "www.spankbang.com"],
        "pattern": r"spankbang\.com/(?P<video_id>[a-z0-9]+)/(?:video|embed)",
    },
    {
        "rank": 28,
        "name": "XNXX",
        "domains": ["xnxx.com", "www.xnxx.com"],
        "pattern": r"xnxx\.com/video-(?P<video_id>[a-z0-9]+)",
    },
    {
        "rank": 29,
        "name": "VK",
        "domains": ["vk.com", "www.vk.com"],
        "pattern": r"vk\.com/video(?P<owner_id>-?\d+)_(?P<video_id>\d+)",
    },
    {
        "rank": 30,
        "name": "Weibo",
        "domains": ["weibo.com", "www.weibo.com"],
        "pattern": r"weibo\.com/(?:tv/(?:show|v)/|\d+/)(?P<video_id>[a-zA-Z0-9]+)",
    },
    {
        "rank": 31,
        "name": "Eporner",
        "domains": ["eporner.com", "www.eporner.com"],
        "pattern": r"eporner\.com/(?:video-|embed/)(?P<video_id>[a-zA-Z0-9]+)",
    },
    {
        "rank": 32,
        "name": "Wistia",
        "domains": ["wistia.com", "wistia.net", "fast.wistia.net"],
        "pattern": r"wistia\.(?:com|net)/(?:medias|embed/iframe)/(?P<video_id>[a-z0-9]+)",
    },
    {
        "rank": 33,
        "name": "Youporn",
        "domains": ["youporn.com", "www.youporn.com"],
        "pattern": r"youporn\.com/(?:watch|embed)/(?P<video_id>\d+)",
    },
    {
        "rank": 34,
        "name": "Tube8",
        "domains": ["tube8.com", "www.tube8.com"],
        "pattern": r"tube8\.com/(?:[^/]+/[^/]+|embed)/(?P<video_id>[0-9]+)",
    },
    {
        "rank": 35,
        "name": "9GAG",
        "domains": ["9gag.com", "www.9gag.com"],
        "pattern": r"9gag\.com/gag/(?P<video_id>[a-zA-Z0-9]+)",
    },
    {
        "rank": 36,
        "name": "ThisVid",
        "domains": ["thisvid.com", "www.thisvid.com"],
        "pattern": r"thisvid\.com/(?:videos|embed)/(?P<video_id>[a-z0-9-]+)",
    },
    {
        "rank": 37,
        "name": "Likee",
        "domains": ["likee.video", "www.likee.video"],
        "pattern": r"likee\.video/(?:v|@[^/]+)/(?P<video_id>[a-zA-Z0-9]+)",
    },
    {
        "rank": 38,
        "name": "Patreon",
        "domains": ["patreon.com", "www.patreon.com"],
        "pattern": r"patreon\.com/(?:posts/(?P<video_id>[a-z0-9-]+)|(?P<user>\w+))",
    },
    {
        "rank": 39,
        "name": "Metacafe",
        "domains": ["metacafe.com", "www.metacafe.com"],
        "pattern": r"metacafe\.com/watch/(?P<video_id>\d+)",
    },
    {
        "rank": 40,
        "name": "Kaltura",
        "domains": ["kaltura.com"],
        "pattern": r"kaltura\.com.*entry_id/(?P<video_id>[a-z0-9_]+)",
    },
    {
        "rank": 41,
        "name": "Brightcove",
        "domains": ["brightcove.com", "bcove.video"],
        "pattern": r"(?:brightcove\.com|bcove\.video).*videoId=(?P<video_id>\d+)",
    },
    {
        "rank": 42,
        "name": "JW Player",
        "domains": ["jwplatform.com", "content.jwplatform.com"],
        "pattern": r"(?:jwplatform|content\.jwplatform)\.com/(?:players|videos)/(?P<video_id>[a-zA-Z0-9]+)-(?P<player_id>[a-zA-Z0-9]+)",
    },
    {
        "rank": 43,
        "name": "Archive.org",
        "domains": ["archive.org", "www.archive.org"],
        "pattern": r"archive\.org/(?:details|embed)/(?P<video_id>[a-zA-Z0-9_.-]+)",
    },
    {
        "rank": 44,
        "name": "Triller",
        "domains": ["triller.co", "www.triller.co"],
        "pattern": r"triller\.co/(?:@[^/]+/video/|v/)(?P<video_id>[a-zA-Z0-9-]+)",
    },
    {
        "rank": 45,
        "name": "Panopto",
        "domains": ["panopto.com"],
        "pattern": r"panopto\.com/Panopto/Pages/(?:Viewer|Embed)\.aspx\?id=(?P<video_id>[a-f0-9-]{36})",
    },
    {
        "rank": 46,
        "name": "ESPN",
        "domains": ["espn.com", "www.espn.com"],
        "pattern": r"espn\.com/(?:video/clip|watch)/(?:_/id/)?(?P<video_id>\d+)",
    },
    {
        "rank": 47,
        "name": "CBC",
        "domains": ["cbc.ca", "gem.cbc.ca"],
        "pattern": r"(?:cbc|gem\.cbc)\.ca/(?:player/play/|media/)(?P<video_id>[a-z0-9-]+)",
    },
    {
        "rank": 48,
        "name": "Fox",
        "domains": ["fox.com", "foxnews.com", "www.fox.com", "www.foxnews.com"],
        "pattern": r"(?:fox|foxnews)\.com/(?:video|watch)/(?P<video_id>\d+)",
    },
    {
        "rank": 49,
        "name": "Vidyard",
        "domains": ["vidyard.com", "www.vidyard.com"],
        "pattern": r"vidyard\.com/(?:watch|embed)/(?P<video_id>[A-Za-z0-9]+)",
    },
    {
        "rank": 50,
        "name": "Peertube",
        "domains": ["peertube.tv", "tube.tchncs.de", "video.ploud.fr"],  # Common instances
        "pattern": r"(?:peertube\.[^/]+|tube\.[^/]+|video\.[^/]+)/(?:videos/watch|w)/(?P<video_id>[a-f0-9-]{36})",
    },
    {
        "rank": 51,
        "name": "Odysee",
        "domains": ["odysee.com", "www.odysee.com"],
        "pattern": r"odysee\.com/@(?P<channel>[\w-]+)(?::[\da-f]+)?/(?P<video_id>[\w-]+)(?::(?P<claim_id>[\da-f]+))?",
    },
    {
        "rank": 52,
        "name": "BitChute",
        "domains": ["bitchute.com", "www.bitchute.com"],
        "pattern": r"bitchute\.com/(?:video|embed)/(?P<video_id>[a-zA-Z0-9]+)",
    },
    {
        "rank": 53,
        "name": "Nebula",
        "domains": ["nebula.tv", "watchnebula.com"],
        "pattern": r"(?:nebula\.tv|watchnebula\.com)/videos/(?P<video_id>[a-z0-9-]+)",
    },
    {
        "rank": 54,
        "name": "Floatplane",
        "domains": ["floatplane.com", "www.floatplane.com"],
        "pattern": r"floatplane\.com/post/(?P<video_id>[a-zA-Z0-9]+)",
    },
    {
        "rank": 55,
        "name": "Dropout",
        "domains": ["dropout.tv", "www.dropout.tv"],
        "pattern": r"dropout\.tv/(?:videos/|[^/]+/)(?P<video_id>[a-z0-9-]+)",
    },
    {
        "rank": 56,
        "name": "Curiositystream",
        "domains": ["curiositystream.com", "www.curiositystream.com"],
        "pattern": r"curiositystream\.com/(?:video|series)/(?P<video_id>[a-z0-9-]+)",
    },
    {
        "rank": 57,
        "name": "Ted",
        "domains": ["ted.com", "www.ted.com"],
        "pattern": r"ted\.com/talks/(?P<video_id>[a-z0-9_]+)",
    },
    {
        "rank": 58,
        "name": "Ustream/IBM Video",
        "domains": ["ustream.tv", "video.ibm.com"],
        "pattern": r"(?:ustream\.tv|video\.ibm\.com)/(?:recorded|embed)/(?P<video_id>\d+)",
    },
    {
        "rank": 59,
        "name": "Kickstarter",
        "domains": ["kickstarter.com", "www.kickstarter.com"],
        "pattern": r"kickstarter\.com/projects/(?P<creator>[^/]+)/(?P<video_id>[^/]+)",
    },
    {
        "rank": 60,
        "name": "Soundcloud",
        "domains": ["soundcloud.com", "www.soundcloud.com"],
        "pattern": r"soundcloud\.com/(?P<user>[^/]+)/(?P<video_id>[^/?]+)",
    },
    {
        "rank": 61,
        "name": "Mixcloud",
        "domains": ["mixcloud.com", "www.mixcloud.com"],
        "pattern": r"mixcloud\.com/(?P<user>[^/]+)/(?P<video_id>[^/]+)",
    },
    {
        "rank": 62,
        "name": "Gfycat",
        "domains": ["gfycat.com", "www.gfycat.com"],
        "pattern": r"gfycat\.com/(?:ifr/)?(?P<video_id>[a-zA-Z]+)",
    },
    {
        "rank": 63,
        "name": "Giphy",
        "domains": ["giphy.com", "www.giphy.com"],
        "pattern": r"giphy\.com/(?:gifs|embed)/(?:[^/]+-)?(?P<video_id>[a-zA-Z0-9]+)",
    },
    {
        "rank": 64,
        "name": "Vine",
        "domains": ["vine.co"],
        "pattern": r"vine\.co/v/(?P<video_id>[a-zA-Z0-9]+)",
    },
    {
        "rank": 65,
        "name": "Vevo",
        "domains": ["vevo.com", "www.vevo.com"],
        "pattern": r"vevo\.com/watch/(?P<artist>[A-Z0-9]+)/(?P<video_id>[A-Z0-9]+)",
    },
    {
        "rank": 66,
        "name": "Coub",
        "domains": ["coub.com", "www.coub.com"],
        "pattern": r"coub\.com/(?:view|embed)/(?P<video_id>[a-z0-9]+)",
    },
    {
        "rank": 67,
        "name": "Liveleak",
        "domains": ["liveleak.com", "www.liveleak.com"],
        "pattern": r"liveleak\.com/view\?[it]=(?P<video_id>[a-zA-Z0-9_]+)",
    },
    {
        "rank": 68,
        "name": "Snapchat",
        "domains": ["snapchat.com", "www.snapchat.com"],
        "pattern": r"snapchat\.com/(?:add|spotlight)/(?P<video_id>[a-z0-9_.-]+)",
    },
    {
        "rank": 69,
        "name": "OnlyFans",
        "domains": ["onlyfans.com", "www.onlyfans.com"],
        "pattern": r"onlyfans\.com/(?P<user_id>\d+)/(?P<video_id>[a-z0-9_-]+)",
    },
    {
        "rank": 70,
        "name": "DTube",
        "domains": ["d.tube"],
        "pattern": r"d\.tube/#!/v/(?P<user>[^/]+)/(?P<video_id>[a-z0-9]+)",
    },
]

# Build domain lookup for fast matching
_DOMAIN_TO_PROVIDER: dict[str, dict] = {}
for provider in VIDEO_PROVIDERS:
    for domain in provider["domains"]:
        _DOMAIN_TO_PROVIDER[domain] = provider

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
    "owner_id",  # VK uses owner_id + video_id combo
    "channel",   # Some sites use channel as identifier
    "user",      # Fallback to user if nothing else
]


class VideoURL(BaseModel):
    """Parsed and validated video URL with extracted metadata."""

    url: str = Field(..., description="Original URL (normalized)")
    video_id: str = Field(..., description="Extracted video ID for caching/lookup")
    provider: str | None = Field(None, description="Detected provider name")
    provider_data: dict = Field(default_factory=dict, description="Extra extracted fields (channel, clip_id, etc.)")

    # Class-level compiled patterns for performance
    _compiled_patterns: ClassVar[dict[str, re.Pattern]] = {}

    @field_validator("url", mode="before")
    @classmethod
    def normalize_url(cls, v: str) -> str:
        """Normalize URL - strip whitespace, ensure scheme."""
        v = v.strip()
        if not v:
            raise ValueError("URL cannot be empty")
        # Add https if no scheme
        if not v.startswith(("http://", "https://")):
            v = "https://" + v
        return v

    @model_validator(mode="after")
    def extract_video_id(self) -> "VideoURL":
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
                self._compiled_patterns[pattern_key] = re.compile(provider["pattern"], re.IGNORECASE)

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
                    # Store all extracted groups as provider_data
                    self.provider_data = {k: v for k, v in groups.items() if v is not None}
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

        # Look for ID-like path segments (prefer short alphanumeric)
        candidates = []
        skip_segments = {
            "video", "videos", "watch", "embed", "shorts", "reel",
            "p", "status", "clips", "details", "channel", "user",
            "playlist", "live", "stories", "comments", "r", "w",
            "share", "post", "posts", "gag", "view", "media",
        }

        for part in path_parts:
            part_lower = part.lower()
            # Skip known non-ID segments
            if part_lower in skip_segments:
                continue
            # Skip usernames
            if part.startswith("@"):
                continue
            # Skip file extensions
            if "." in part and part.rsplit(".", 1)[-1] in ["html", "php", "aspx", "mp4", "gifv"]:
                part = part.rsplit(".", 1)[0]
                # For patterns like "v4p5yd5-title-here", extract the ID prefix
                if "-" in part:
                    prefix = part.split("-")[0]
                    if re.match(r"^[a-zA-Z]?\d+[a-zA-Z0-9]*$", prefix):
                        candidates.insert(0, prefix)  # Prioritize
                        continue

            # Good candidate: alphanumeric 4-50 chars
            if re.match(r"^[a-zA-Z0-9_-]{4,50}$", part):
                candidates.append(part)

        if candidates:
            # Prefer shorter candidates (IDs are usually compact)
            # But also prefer ones that look more "ID-like" (mixed case, numbers)
            def id_score(s):
                # Lower is better
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
    def parse(cls, url: str) -> "VideoURL":
        """Parse a URL and extract video information.

        Args:
            url: Video URL string

        Returns:
            VideoURL with extracted video_id and provider info

        Raises:
            ValueError: If URL is invalid or cannot be parsed
        """
        # Initial placeholder video_id (will be replaced by validator)
        return cls(url=url, video_id="__pending__")

    @classmethod
    def try_parse(cls, url: str) -> "VideoURL | None":
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
        # Use video_id if short enough and safe
        if len(self.video_id) <= 50 and re.match(r"^[\w-]+$", self.video_id):
            return self.video_id
        # Fallback to hash for weird IDs
        return hashlib.sha256(self.video_id.encode()).hexdigest()[:16]

    def __str__(self) -> str:
        if self.provider:
            return f"{self.provider}:{self.video_id}"
        return self.video_id

    def __repr__(self) -> str:
        return f"VideoURL(url={self.url!r}, video_id={self.video_id!r}, provider={self.provider!r})"


# Convenience functions for backwards compatibility
def extract_video_id(url: str) -> str:
    """Extract video ID from URL (backwards-compatible function)."""
    try:
        return VideoURL.parse(url).video_id
    except Exception:
        # Absolute fallback
        clean = re.sub(r"^https?://", "", url)
        clean = re.sub(r"[^\w.-]", "_", clean)
        return clean[:50] if len(clean) > 50 else clean


def extract_playlist_id(url: str) -> str | None:
    """Extract YouTube playlist ID from URL (list= parameter)."""
    match = re.search(r"[?&]list=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def extract_url_context(url: str) -> dict:
    """Extract video ID, playlist ID, and other context from URL."""
    parsed = VideoURL.try_parse(url)
    if parsed:
        return {
            "video_id": parsed.video_id,
            "playlist_id": extract_playlist_id(url),
            "original_url": url,
            "clean_url": re.sub(r"[&?]list=[^&]+", "", url),
            "provider": parsed.provider,
            "provider_data": parsed.provider_data,
        }
    # Fallback
    return {
        "video_id": extract_video_id(url),
        "playlist_id": extract_playlist_id(url),
        "original_url": url,
        "clean_url": re.sub(r"[&?]list=[^&]+", "", url),
        "provider": None,
        "provider_data": {},
    }


def get_provider_for_url(url: str) -> str | None:
    """Get the provider name for a URL, or None if unknown."""
    parsed = VideoURL.try_parse(url)
    return parsed.provider if parsed else None


def list_supported_providers() -> list[str]:
    """List all supported provider names."""
    return [p["name"] for p in VIDEO_PROVIDERS]


def get_provider_count() -> int:
    """Return the number of supported providers."""
    return len(VIDEO_PROVIDERS)
