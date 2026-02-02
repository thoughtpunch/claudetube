"""Smoke tests for claudetube-2a6.4 success criteria."""

from claudetube.models.video_url import VideoURL
from claudetube.models.state import VideoState


def test_query_params_merged_into_provider_data():
    """VideoURL.parse includes playlist and start_time from query params."""
    v = VideoURL.parse("https://youtube.com/watch?v=dQw4w9WgXcQ&list=PLxxx&t=120")
    assert v.provider_data.get("playlist") == "PLxxx"
    assert v.provider_data.get("start_time") == "120"


def test_video_path_from_url():
    """VideoURL.video_path returns valid VideoPath with domain from URL."""
    v = VideoURL.parse("https://youtube.com/watch?v=dQw4w9WgXcQ&list=PLxxx&t=120")
    vp = v.video_path
    assert vp.domain == "youtube"
    assert vp.playlist == "PLxxx"
    assert vp.video_id == "dQw4w9WgXcQ"


def test_video_path_without_extras():
    """VideoURL.video_path works for URLs without channel/playlist."""
    v = VideoURL.parse("https://vimeo.com/347119375")
    vp = v.video_path
    assert vp.domain == "vimeo"
    assert vp.channel is None
    assert vp.playlist is None
    assert vp.video_id == "347119375"


def test_video_path_with_channel():
    """VideoURL.video_path includes channel when available."""
    v = VideoURL.parse("https://www.tiktok.com/@celiatoks/video/7454538602173697313")
    vp = v.video_path
    assert vp.domain == "tiktok"
    assert vp.channel == "celiatoks"
    assert vp.video_id == "7454538602173697313"


def test_video_state_new_fields_serialize():
    """VideoState with domain/channel_id/playlist_id serializes to/from JSON."""
    state = VideoState(
        video_id="test", domain="youtube", channel_id="UCxxx", playlist_id="PLyyy"
    )
    d = state.to_dict()
    restored = VideoState.from_dict(d)
    assert restored.domain == "youtube"
    assert restored.channel_id == "UCxxx"
    assert restored.playlist_id == "PLyyy"


def test_old_state_json_loads_without_new_fields():
    """Existing state.json files without new fields load correctly."""
    old_data = {"video_id": "old_video", "url": "http://example.com"}
    old_state = VideoState.from_dict(old_data)
    assert old_state.domain is None
    assert old_state.channel_id is None


def test_from_metadata_domain():
    """VideoState.from_metadata() populates domain from URL."""
    state = VideoState.from_metadata(
        "abc", "https://youtube.com/watch?v=abc", {"title": "Test"}
    )
    assert state.domain == "youtube"


def test_from_metadata_domain_from_extractor():
    """VideoState.from_metadata() falls back to extractor_key for domain."""
    state = VideoState.from_metadata(
        "abc", "", {"title": "Test", "extractor_key": "Youtube"}
    )
    assert state.domain == "youtube"


def test_from_metadata_channel_id():
    """VideoState.from_metadata() populates channel_id from channel_id."""
    state = VideoState.from_metadata(
        "abc",
        "https://youtube.com/watch?v=abc",
        {"title": "Test", "channel_id": "UC12345"},
    )
    assert state.channel_id == "UC12345"


def test_from_metadata_channel_id_from_uploader():
    """VideoState.from_metadata() falls back to uploader_id."""
    state = VideoState.from_metadata(
        "abc",
        "https://youtube.com/watch?v=abc",
        {"title": "Test", "uploader_id": "uploaderXYZ"},
    )
    assert state.channel_id == "uploaderXYZ"


def test_from_metadata_channel_id_sanitized():
    """VideoState.from_metadata() sanitizes channel name for channel_id."""
    state = VideoState.from_metadata(
        "abc",
        "https://youtube.com/watch?v=abc",
        {"title": "Test", "channel": "My Cool Channel!"},
    )
    assert state.channel_id == "My_Cool_Channel_"


def test_from_metadata_playlist_id():
    """VideoState.from_metadata() populates playlist_id from yt-dlp."""
    state = VideoState.from_metadata(
        "abc",
        "https://youtube.com/watch?v=abc",
        {"title": "Test", "playlist_id": "PLtest"},
    )
    assert state.playlist_id == "PLtest"


def test_regex_captures_take_priority_over_query_params():
    """Regex captures should take priority over query params for same field."""
    # YouTube regex captures 'playlist' from the pattern.
    # Query param extraction also maps 'list' -> 'playlist'.
    # Both should produce the same value since they read the same URL param.
    v = VideoURL.parse("https://youtube.com/watch?v=dQw4w9WgXcQ&list=PLxxx")
    assert v.provider_data["playlist"] == "PLxxx"


def test_query_params_fill_gaps_not_overwrite():
    """Query params should fill gaps in provider_data, not overwrite regex captures."""
    # start_time comes only from query params (t=120), not from regex
    v = VideoURL.parse("https://youtube.com/watch?v=dQw4w9WgXcQ&t=120")
    assert v.provider_data.get("start_time") == "120"
    # video_id comes from regex, not query params
    assert v.provider_data.get("video_id") == "dQw4w9WgXcQ"
