"""
Deep integration test for claudetube - exercises all operations against a real playlist.

Uses the 3Blue1Brown "Essence of Linear Algebra" playlist (16 videos).

NOT a pytest test (no test_* functions / Test* classes).
All logic in helper functions + main() with if __name__ == "__main__".

Usage:
    python tests/deep_integration_test.py                    # all videos, all features
    python tests/deep_integration_test.py --first 2          # first 2 videos only
    python tests/deep_integration_test.py --skip-providers   # skip operations requiring AI providers
    python tests/deep_integration_test.py --resume           # skip already-processed videos
    python tests/deep_integration_test.py -v                 # verbose output
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLAYLIST_URL = (
    "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
)
CACHE_DIR = Path(__file__).parent / "_deep_integration_cache"
RESULTS_FILE = Path(__file__).parent / "deep_integration_results.json"

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


class StepResult:
    """Result of a single test step."""

    def __init__(self, name: str):
        self.name = name
        self.status = "PENDING"  # OK, FAIL, SKIP, ERROR
        self.elapsed = 0.0
        self.detail = ""
        self.error = ""

    def ok(self, detail: str = ""):
        self.status = "OK"
        self.detail = detail

    def fail(self, error: str):
        self.status = "FAIL"
        self.error = error[:500]

    def skip(self, reason: str = ""):
        self.status = "SKIP"
        self.detail = reason

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "status": self.status,
            "elapsed": round(self.elapsed, 2),
        }
        if self.detail:
            d["detail"] = self.detail
        if self.error:
            d["error"] = self.error
        return d


class RunContext:
    """Shared context across all phases."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.verbose = args.verbose
        self.skip_providers = args.skip_providers
        self.resume = args.resume
        self.first_n = args.first

        # Populated during phases
        self.providers_available: list[str] = []
        self.providers_all: list[str] = []
        self.factory = None
        self.has_vision = False
        self.has_reasoner = False
        self.has_embedder = False
        self.has_transcriber = False
        self.has_video_analyzer = False

        self.playlist_data: dict | None = None
        self.playlist_id: str | None = None
        self.video_urls: list[str] = []
        self.video_ids: list[str] = []
        self.video_results: dict[str, dict] = {}  # video_id -> per-video results

        self.phase_results: dict[str, list[dict]] = {}
        self.start_time = time.time()

    def log(self, msg: str, verbose_only: bool = False):
        if verbose_only and not self.verbose:
            return
        print(msg)


def run_step(ctx: RunContext, step_name: str, func, *args, **kwargs) -> StepResult:
    """Run a single test step with timing and error handling."""
    result = StepResult(step_name)
    t0 = time.time()
    try:
        func(result, ctx, *args, **kwargs)
    except Exception as e:
        result.status = "ERROR"
        result.error = f"{type(e).__name__}: {str(e)[:400]}"
        if ctx.verbose:
            traceback.print_exc()
    result.elapsed = time.time() - t0

    icon = {"OK": "+", "FAIL": "-", "SKIP": "~", "ERROR": "!", "PENDING": "?"}[
        result.status
    ]
    extra = ""
    if result.elapsed > 0.1:
        extra += f" ({result.elapsed:.1f}s)"
    if result.detail:
        extra += f" {result.detail}"
    if result.error:
        extra += f" [{result.error[:80]}]"
    ctx.log(f"  [{icon}] {step_name}{extra}")
    return result


# ===========================================================================
# PHASE 0: Setup & Provider Detection
# ===========================================================================


def phase0_setup(ctx: RunContext) -> list[dict]:
    """Initialize cache dir, detect providers."""
    ctx.log(f"\n{'=' * 60}")
    ctx.log(" PHASE 0: Setup & Provider Detection")
    ctx.log(f"{'=' * 60}")
    results = []

    # Init cache dir
    def step_cache_dir(r: StepResult, c: RunContext):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        r.ok(f"cache_dir={CACHE_DIR}")

    results.append(run_step(ctx, "init_cache_dir", step_cache_dir).to_dict())

    # Detect providers
    def step_list_all(r: StepResult, c: RunContext):
        from claudetube.providers.registry import list_all

        c.providers_all = list_all()
        r.ok(f"known={len(c.providers_all)}: {', '.join(c.providers_all)}")

    results.append(run_step(ctx, "list_all_providers", step_list_all).to_dict())

    def step_list_available(r: StepResult, c: RunContext):
        from claudetube.providers.registry import list_available

        c.providers_available = list_available()
        r.ok(
            f"available={len(c.providers_available)}: {', '.join(c.providers_available)}"
        )

    results.append(
        run_step(ctx, "list_available_providers", step_list_available).to_dict()
    )

    # Initialize factory
    def step_factory(r: StepResult, c: RunContext):
        if c.skip_providers:
            r.skip("--skip-providers")
            return
        from claudetube.operations.factory import get_factory

        c.factory = get_factory()

        # Probe capabilities
        try:
            c.has_transcriber = c.factory.get_transcriber() is not None
        except Exception:
            c.has_transcriber = False
        try:
            c.has_vision = c.factory.get_vision_analyzer() is not None
        except Exception:
            c.has_vision = False
        try:
            c.has_video_analyzer = c.factory.get_video_analyzer() is not None
        except Exception:
            c.has_video_analyzer = False
        try:
            c.has_reasoner = c.factory.get_reasoner() is not None
        except Exception:
            c.has_reasoner = False

        caps = []
        if c.has_transcriber:
            caps.append("transcription")
        if c.has_vision:
            caps.append("vision")
        if c.has_video_analyzer:
            caps.append("video_analysis")
        if c.has_reasoner:
            caps.append("reasoning")
        r.ok(f"capabilities: {', '.join(caps) or 'none'}")

    results.append(run_step(ctx, "init_factory", step_factory).to_dict())

    # Print summary
    ctx.log(f"\n  Providers available: {ctx.providers_available}")
    ctx.log(f"  Skip providers: {ctx.skip_providers}")
    ctx.log(f"  Resume mode: {ctx.resume}")
    if ctx.first_n:
        ctx.log(f"  Processing first {ctx.first_n} videos only")

    return results


# ===========================================================================
# PHASE 1: Playlist Operations
# ===========================================================================


def phase1_playlist(ctx: RunContext) -> list[dict]:
    """Exercise playlist metadata operations."""
    ctx.log(f"\n{'=' * 60}")
    ctx.log(" PHASE 1: Playlist Operations")
    ctx.log(f"{'=' * 60}")
    results = []

    # 1. Extract playlist metadata
    def step_extract(r: StepResult, c: RunContext):
        from claudetube.operations.playlist import extract_playlist_metadata

        data = extract_playlist_metadata(PLAYLIST_URL)
        c.playlist_data = data
        video_count = data.get("video_count", len(data.get("videos", [])))
        c.playlist_id = data.get("playlist_id", "unknown")
        title = data.get("title", "?")
        r.ok(f'id={c.playlist_id} title="{title[:60]}" videos={video_count}')
        assert video_count > 0, f"Expected videos, got {video_count}"

    results.append(run_step(ctx, "extract_playlist_metadata", step_extract).to_dict())

    # 2. Save playlist metadata
    def step_save(r: StepResult, c: RunContext):
        from claudetube.operations.playlist import save_playlist_metadata

        if not c.playlist_data:
            r.skip("no playlist data")
            return
        path = save_playlist_metadata(c.playlist_data, cache_base=CACHE_DIR)
        r.ok(f"saved to {path}")

    results.append(run_step(ctx, "save_playlist_metadata", step_save).to_dict())

    # 3. Load playlist metadata
    def step_load(r: StepResult, c: RunContext):
        from claudetube.operations.playlist import load_playlist_metadata

        if not c.playlist_id:
            r.skip("no playlist_id")
            return
        loaded = load_playlist_metadata(c.playlist_id, cache_base=CACHE_DIR)
        assert loaded is not None, "load_playlist_metadata returned None"
        r.ok(f"loaded, videos={len(loaded.get('videos', []))}")

    results.append(run_step(ctx, "load_playlist_metadata", step_load).to_dict())

    # 4. List cached playlists
    def step_list(r: StepResult, c: RunContext):
        from claudetube.operations.playlist import list_cached_playlists

        playlists = list_cached_playlists(cache_base=CACHE_DIR)
        r.ok(f"found {len(playlists)} cached playlist(s)")

    results.append(run_step(ctx, "list_cached_playlists", step_list).to_dict())

    # 5. Classify playlist type
    def step_classify(r: StepResult, c: RunContext):
        from claudetube.operations.playlist import classify_playlist_type

        if not c.playlist_data:
            r.skip("no playlist data")
            return
        playlist_info = {k: v for k, v in c.playlist_data.items() if k != "videos"}
        videos = c.playlist_data.get("videos", [])
        ptype = classify_playlist_type(playlist_info, videos)
        r.ok(f'type="{ptype}"')

    results.append(run_step(ctx, "classify_playlist_type", step_classify).to_dict())

    # Build video URL list for subsequent phases
    if ctx.playlist_data:
        videos = ctx.playlist_data.get("videos", [])
        for entry in videos:
            url = entry.get("url", "")
            if not url and entry.get("video_id"):
                url = f"https://www.youtube.com/watch?v={entry['video_id']}"
            if url:
                ctx.video_urls.append(url)
        if ctx.first_n:
            ctx.video_urls = ctx.video_urls[: ctx.first_n]
        ctx.log(f"\n  Videos to process: {len(ctx.video_urls)}")

    return results


# ===========================================================================
# PHASE 2: Per-Video Processing
# ===========================================================================


def phase2_per_video(ctx: RunContext) -> list[dict]:
    """Process each video through all operations."""
    ctx.log(f"\n{'=' * 60}")
    ctx.log(" PHASE 2: Per-Video Processing")
    ctx.log(f"{'=' * 60}")
    all_video_results = []

    for i, url in enumerate(ctx.video_urls):
        ctx.log(f"\n  --- Video {i + 1}/{len(ctx.video_urls)}: {url[:80]} ---")
        video_result = process_single_video(ctx, url, i)
        all_video_results.append(video_result)

        # Save intermediate results
        _save_results(ctx, all_video_results)

    return all_video_results


def process_single_video(ctx: RunContext, url: str, index: int) -> dict:
    """Process a single video through all sub-steps (2a through 2r)."""
    video_data = {"url": url, "index": index, "video_id": None, "steps": {}}

    # 2a: URL Parsing
    video_id = None

    def step_parse_url(r: StepResult, c: RunContext):
        nonlocal video_id
        from claudetube.models.video_url import VideoURL

        parsed = VideoURL.parse(url)
        video_id = parsed.video_id
        r.ok(
            f"id={parsed.video_id} provider={parsed.provider} known={parsed.is_known_provider}"
        )

    video_data["steps"]["2a_parse_url"] = run_step(
        ctx, "2a: VideoURL.parse", step_parse_url
    ).to_dict()

    def step_extract_vid(r: StepResult, c: RunContext):
        from claudetube.parsing.utils import extract_video_id

        vid = extract_video_id(url)
        r.ok(f"video_id={vid}")

    video_data["steps"]["2a_extract_video_id"] = run_step(
        ctx, "2a: extract_video_id", step_extract_vid
    ).to_dict()

    if not video_id:
        ctx.log("  [!] Could not parse video ID, skipping remaining steps")
        return video_data
    video_data["video_id"] = video_id
    ctx.video_ids.append(video_id)
    cache_dir = CACHE_DIR / video_id

    # Check resume mode
    if ctx.resume and cache_dir.exists() and (cache_dir / "state.json").exists():
        ctx.log("  [~] Resuming: video already cached, skipping process_video")
        video_data["steps"]["2b_process_video"] = {
            "name": "2b: process_video",
            "status": "SKIP",
            "elapsed": 0,
            "detail": "resume mode - already cached",
        }
    else:
        # 2b: Process Video
        def step_process(r: StepResult, c: RunContext):
            from claudetube.operations.processor import process_video

            vr = process_video(url, output_base=CACHE_DIR)
            assert vr.success, f"process_video failed: {vr.error}"
            title = vr.metadata.get("title", "?")[:60]
            r.ok(f'title="{title}" has_srt={vr.transcript_srt is not None}')

        video_data["steps"]["2b_process_video"] = run_step(
            ctx, "2b: process_video", step_process
        ).to_dict()

    # 2c: Transcript Access
    def step_transcript(r: StepResult, c: RunContext):
        txt_path = cache_dir / "audio.txt"
        srt_path = cache_dir / "audio.srt"
        # Also check for yt-dlp downloaded subtitles (e.g., {id}.en.srt)
        srt_files = list(cache_dir.glob("*.srt"))
        has_txt = txt_path.exists() and txt_path.stat().st_size > 0
        has_srt = srt_path.exists() or len(srt_files) > 0
        if has_txt:
            txt_len = len(txt_path.read_text())
            r.ok(f"txt={txt_len} chars, srt={'yes' if has_srt else 'no'}")
        elif has_srt:
            srt_file = srt_path if srt_path.exists() else srt_files[0]
            srt_len = srt_file.stat().st_size
            r.ok(f"txt=no, srt={srt_file.name} ({srt_len} bytes)")
        else:
            r.fail("no transcript files found")

    video_data["steps"]["2c_transcript"] = run_step(
        ctx, "2c: transcript_access", step_transcript
    ).to_dict()

    # 2d: Frame Extraction
    def step_frames(r: StepResult, c: RunContext):
        from claudetube.operations.extract_frames import extract_frames

        frames = extract_frames(
            video_id, start_time=10, duration=3, interval=1, output_base=CACHE_DIR
        )
        assert len(frames) > 0, "No frames extracted"
        exists = sum(1 for f in frames if f.exists())
        r.ok(f"{exists}/{len(frames)} frame files exist")

    video_data["steps"]["2d_frames"] = run_step(
        ctx, "2d: extract_frames", step_frames
    ).to_dict()

    def step_hq_frames(r: StepResult, c: RunContext):
        from claudetube.operations.extract_frames import extract_hq_frames

        hq = extract_hq_frames(
            video_id, start_time=10, duration=2, interval=1, output_base=CACHE_DIR
        )
        assert len(hq) > 0, "No HQ frames extracted"
        exists = sum(1 for f in hq if f.exists())
        r.ok(f"{exists}/{len(hq)} HQ frame files exist")

    video_data["steps"]["2d_hq_frames"] = run_step(
        ctx, "2d: extract_hq_frames", step_hq_frames
    ).to_dict()

    # 2e: Chapters
    def step_chapters(r: StepResult, c: RunContext):
        from claudetube.operations.chapters import extract_youtube_chapters

        state_path = cache_dir / "state.json"
        if not state_path.exists():
            r.skip("no state.json")
            return
        state = json.loads(state_path.read_text())
        video_info = state.get("metadata", state)
        chapters = extract_youtube_chapters(video_info)
        r.ok(f"{len(chapters)} chapters found")

    video_data["steps"]["2e_chapters"] = run_step(
        ctx, "2e: extract_youtube_chapters", step_chapters
    ).to_dict()

    # 2f: Scene Segmentation
    scenes_data = None

    def step_segment(r: StepResult, c: RunContext):
        nonlocal scenes_data
        from claudetube.operations.segmentation import segment_video_smart

        state_path = cache_dir / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        video_info = state.get("metadata", state)
        srt_path = cache_dir / "audio.srt"
        sd = segment_video_smart(
            video_id=video_id,
            video_path=None,
            transcript_segments=None,
            video_info=video_info,
            cache_dir=cache_dir,
            srt_path=str(srt_path) if srt_path.exists() else None,
        )
        scenes_data = sd
        assert len(sd.scenes) > 0, "No scenes produced"
        for s in sd.scenes:
            assert s.start_time is not None
            assert s.end_time is not None
        r.ok(f"{len(sd.scenes)} scenes, method={sd.method}")

    video_data["steps"]["2f_segmentation"] = run_step(
        ctx, "2f: segment_video_smart", step_segment
    ).to_dict()

    # 2g: Visual Transcript (requires vision)
    def step_visual_transcript(r: StepResult, c: RunContext):
        if c.skip_providers or not c.has_vision:
            r.skip("no vision provider")
            return
        from claudetube.operations.visual_transcript import generate_visual_transcript

        result = generate_visual_transcript(video_id, output_base=CACHE_DIR)
        r.ok(f"keys={list(result.keys())[:5]}")

    video_data["steps"]["2g_visual_transcript"] = run_step(
        ctx, "2g: generate_visual_transcript", step_visual_transcript
    ).to_dict()

    # 2h: Entity Extraction (requires vision/reasoner)
    def step_entities_extract(r: StepResult, c: RunContext):
        if c.skip_providers or (not c.has_vision and not c.has_reasoner):
            r.skip("no vision/reasoner provider")
            return
        from claudetube.operations.entity_extraction import extract_entities_for_video

        result = extract_entities_for_video(video_id, output_base=CACHE_DIR)
        r.ok(f"keys={list(result.keys())[:5]}")

    video_data["steps"]["2h_extract_entities"] = run_step(
        ctx, "2h: extract_entities_for_video", step_entities_extract
    ).to_dict()

    def step_entities_get(r: StepResult, c: RunContext):
        if c.skip_providers or (not c.has_vision and not c.has_reasoner):
            r.skip("no vision/reasoner provider")
            return
        from claudetube.operations.entity_extraction import get_extracted_entities

        result = get_extracted_entities(video_id, output_base=CACHE_DIR)
        r.ok(f"keys={list(result.keys())[:5]}")

    video_data["steps"]["2h_get_entities"] = run_step(
        ctx, "2h: get_extracted_entities", step_entities_get
    ).to_dict()

    # 2i: Person Tracking
    def step_people(r: StepResult, c: RunContext):
        from claudetube.operations.person_tracking import track_people

        result = track_people(video_id, output_base=CACHE_DIR)
        r.ok(f"keys={list(result.keys())[:5]}")

    video_data["steps"]["2i_person_tracking"] = run_step(
        ctx, "2i: track_people", step_people
    ).to_dict()

    # 2j: Analysis Depth
    def step_analyze_standard(r: StepResult, c: RunContext):
        from claudetube.operations.analysis_depth import AnalysisDepth, analyze_video

        result = analyze_video(
            video_id, depth=AnalysisDepth.STANDARD, output_base=CACHE_DIR
        )
        r.ok(f"scenes={len(result.scenes)} method={result.method}")

    video_data["steps"]["2j_analyze_standard"] = run_step(
        ctx, "2j: analyze_video(STANDARD)", step_analyze_standard
    ).to_dict()

    def step_analysis_status(r: StepResult, c: RunContext):
        from claudetube.operations.analysis_depth import get_analysis_status

        status = get_analysis_status(video_id, output_base=CACHE_DIR)
        r.ok(f"keys={list(status.keys())[:5]}")

    video_data["steps"]["2j_analysis_status"] = run_step(
        ctx, "2j: get_analysis_status", step_analysis_status
    ).to_dict()

    def step_analyze_deep(r: StepResult, c: RunContext):
        if c.skip_providers or not c.has_vision:
            r.skip("no vision provider for deep analysis")
            return
        from claudetube.operations.analysis_depth import AnalysisDepth, analyze_video

        result = analyze_video(
            video_id, depth=AnalysisDepth.DEEP, output_base=CACHE_DIR
        )
        r.ok(f"scenes={len(result.scenes)} method={result.method}")

    video_data["steps"]["2j_analyze_deep"] = run_step(
        ctx, "2j: analyze_video(DEEP)", step_analyze_deep
    ).to_dict()

    # 2k: Embeddings
    def step_embeddings(r: StepResult, c: RunContext):
        from claudetube.analysis.embeddings import embed_scenes
        from claudetube.cache.scenes import load_scenes_data

        sd = load_scenes_data(cache_dir)
        if not sd or not sd.scenes:
            r.skip("no scenes data for embedding")
            return
        embeddings = embed_scenes(sd.scenes, cache_dir)
        if embeddings:
            r.ok(f"{len(embeddings)} embeddings, dim={len(embeddings[0].embedding)}")
        else:
            r.ok("0 embeddings (embedder may not be available)")

    video_data["steps"]["2k_embeddings"] = run_step(
        ctx, "2k: embed_scenes", step_embeddings
    ).to_dict()

    # 2l: Change Detection
    def step_changes_detect(r: StepResult, c: RunContext):
        from claudetube.operations.change_detection import detect_scene_changes

        result = detect_scene_changes(video_id, output_base=CACHE_DIR)
        r.ok(f"keys={list(result.keys())[:5]}")

    video_data["steps"]["2l_detect_changes"] = run_step(
        ctx, "2l: detect_scene_changes", step_changes_detect
    ).to_dict()

    def step_changes_get(r: StepResult, c: RunContext):
        from claudetube.operations.change_detection import get_scene_changes

        result = get_scene_changes(video_id, output_base=CACHE_DIR)
        r.ok(f"keys={list(result.keys())[:5]}")

    video_data["steps"]["2l_get_changes"] = run_step(
        ctx, "2l: get_scene_changes", step_changes_get
    ).to_dict()

    # 2m: Narrative Structure
    def step_narrative_detect(r: StepResult, c: RunContext):
        from claudetube.operations.narrative_structure import detect_narrative_structure

        result = detect_narrative_structure(video_id, output_base=CACHE_DIR)
        assert "sections" in result or "error" not in result, (
            f"Unexpected result: {result}"
        )
        r.ok(f"keys={list(result.keys())[:5]}")

    video_data["steps"]["2m_detect_narrative"] = run_step(
        ctx, "2m: detect_narrative_structure", step_narrative_detect
    ).to_dict()

    def step_narrative_get(r: StepResult, c: RunContext):
        from claudetube.operations.narrative_structure import get_narrative_structure

        result = get_narrative_structure(video_id, output_base=CACHE_DIR)
        r.ok(f"keys={list(result.keys())[:5]}")

    video_data["steps"]["2m_get_narrative"] = run_step(
        ctx, "2m: get_narrative_structure", step_narrative_get
    ).to_dict()

    # 2n: Moment Search
    def step_search_la(r: StepResult, c: RunContext):
        from claudetube.analysis.search import find_moments

        moments = find_moments(video_id, "linear algebra", top_k=3, cache_dir=CACHE_DIR)
        r.ok(f"{len(moments)} moments found")

    video_data["steps"]["2n_search_linear_algebra"] = run_step(
        ctx, "2n: find_moments('linear algebra')", step_search_la
    ).to_dict()

    def step_search_matrix(r: StepResult, c: RunContext):
        from claudetube.analysis.search import find_moments

        moments = find_moments(video_id, "matrix", top_k=3, cache_dir=CACHE_DIR)
        r.ok(f"{len(moments)} moments found")

    video_data["steps"]["2n_search_matrix"] = run_step(
        ctx, "2n: find_moments('matrix')", step_search_matrix
    ).to_dict()

    # 2o: Watch Video (requires reasoner)
    def step_watch(r: StepResult, c: RunContext):
        if c.skip_providers or not c.has_reasoner:
            r.skip("no reasoner provider")
            return
        from claudetube.operations.watch import watch_video

        result = watch_video(
            video_id,
            "What is this video about?",
            max_iterations=5,
            output_base=CACHE_DIR,
        )
        r.ok(f"keys={list(result.keys())[:5]}")

    video_data["steps"]["2o_watch_video"] = run_step(
        ctx, "2o: watch_video", step_watch
    ).to_dict()

    # 2p: Audio Description
    def step_ad_compile(r: StepResult, c: RunContext):
        from claudetube.operations.audio_description import compile_scene_descriptions

        result = compile_scene_descriptions(video_id, output_base=CACHE_DIR)
        r.ok(f"keys={list(result.keys())[:5]}")

    video_data["steps"]["2p_compile_descriptions"] = run_step(
        ctx, "2p: compile_scene_descriptions", step_ad_compile
    ).to_dict()

    def step_ad_get(r: StepResult, c: RunContext):
        from claudetube.operations.audio_description import get_scene_descriptions

        result = get_scene_descriptions(video_id, output_base=CACHE_DIR)
        r.ok(f"keys={list(result.keys())[:5]}")

    video_data["steps"]["2p_get_descriptions"] = run_step(
        ctx, "2p: get_scene_descriptions", step_ad_get
    ).to_dict()

    # 2q: Enrichment & Learning
    def step_enrich_frame(r: StepResult, c: RunContext):
        from claudetube.cache.enrichment import record_frame_examination

        result = record_frame_examination(
            video_id, cache_dir, start_time=10, duration=3, quality="standard"
        )
        r.ok(f"result={type(result).__name__}")

    video_data["steps"]["2q_record_frame_exam"] = run_step(
        ctx, "2q: record_frame_examination", step_enrich_frame
    ).to_dict()

    def step_enrich_qa(r: StepResult, c: RunContext):
        from claudetube.cache.enrichment import record_qa_interaction

        result = record_qa_interaction(
            video_id, cache_dir, "What is this about?", "Linear algebra concepts"
        )
        r.ok(f"keys={list(result.keys())[:5]}")

    video_data["steps"]["2q_record_qa"] = run_step(
        ctx, "2q: record_qa_interaction", step_enrich_qa
    ).to_dict()

    def step_search_qa(r: StepResult, c: RunContext):
        from claudetube.cache.enrichment import search_cached_qa

        results_list = search_cached_qa(video_id, cache_dir, "linear algebra")
        r.ok(f"{len(results_list)} cached QA results")

    video_data["steps"]["2q_search_qa"] = run_step(
        ctx, "2q: search_cached_qa", step_search_qa
    ).to_dict()

    def step_scene_context(r: StepResult, c: RunContext):
        from claudetube.cache.enrichment import get_scene_context

        result = get_scene_context(video_id, cache_dir, scene_id=0)
        r.ok(f"keys={list(result.keys())[:5]}")

    video_data["steps"]["2q_scene_context"] = run_step(
        ctx, "2q: get_scene_context", step_scene_context
    ).to_dict()

    def step_enrich_stats(r: StepResult, c: RunContext):
        from claudetube.cache.enrichment import get_enrichment_stats

        stats = get_enrichment_stats(cache_dir)
        r.ok(f"keys={list(stats.keys())[:5]}")

    video_data["steps"]["2q_enrichment_stats"] = run_step(
        ctx, "2q: get_enrichment_stats", step_enrich_stats
    ).to_dict()

    # 2r: Knowledge Graph
    def step_kg_index(r: StepResult, c: RunContext):
        from claudetube.cache.knowledge_graph import index_video_to_graph

        graph_dir = CACHE_DIR / "_knowledge_graph"
        result = index_video_to_graph(video_id, cache_dir, graph_dir=graph_dir)
        r.ok(f"keys={list(result.keys())[:5]}")

    video_data["steps"]["2r_kg_index"] = run_step(
        ctx, "2r: index_video_to_graph", step_kg_index
    ).to_dict()

    # Store per-video results
    if video_id:
        ctx.video_results[video_id] = video_data

    return video_data


# ===========================================================================
# PHASE 3: Cross-Video Operations
# ===========================================================================


def phase3_cross_video(ctx: RunContext) -> list[dict]:
    """Operations that span multiple videos."""
    ctx.log(f"\n{'=' * 60}")
    ctx.log(" PHASE 3: Cross-Video Operations")
    ctx.log(f"{'=' * 60}")
    results = []

    # Knowledge graph
    def step_kg_get(r: StepResult, c: RunContext):
        from claudetube.cache.knowledge_graph import get_knowledge_graph

        graph_dir = CACHE_DIR / "_knowledge_graph"
        graph = get_knowledge_graph(graph_dir=graph_dir)
        stats = graph.get_stats()
        r.ok(
            f"videos={stats.get('video_count', 0)} entities={stats.get('entity_count', 0)} concepts={stats.get('concept_count', 0)}"
        )

    results.append(
        run_step(ctx, "get_knowledge_graph + get_stats", step_kg_get).to_dict()
    )

    def step_kg_search(r: StepResult, c: RunContext):
        from claudetube.cache.knowledge_graph import get_knowledge_graph

        graph_dir = CACHE_DIR / "_knowledge_graph"
        graph = get_knowledge_graph(graph_dir=graph_dir)
        matches = graph.find_related_videos("linear algebra")
        r.ok(f"{len(matches)} matches for 'linear algebra'")

    results.append(
        run_step(ctx, "find_related_videos('linear algebra')", step_kg_search).to_dict()
    )

    def step_kg_connections(r: StepResult, c: RunContext):
        from claudetube.cache.knowledge_graph import get_knowledge_graph

        graph_dir = CACHE_DIR / "_knowledge_graph"
        graph = get_knowledge_graph(graph_dir=graph_dir)
        if c.video_ids:
            connections = graph.get_video_connections(c.video_ids[0])
            r.ok(f"{len(connections)} connections for {c.video_ids[0]}")
        else:
            r.skip("no video IDs available")

    results.append(
        run_step(ctx, "get_video_connections", step_kg_connections).to_dict()
    )

    # CacheManager
    def step_cache_list(r: StepResult, c: RunContext):
        from claudetube.cache.manager import CacheManager

        cm = CacheManager(CACHE_DIR)
        videos = cm.list_cached_videos()
        r.ok(f"{len(videos)} cached videos")

    results.append(
        run_step(ctx, "CacheManager.list_cached_videos", step_cache_list).to_dict()
    )

    # List cached playlists
    def step_playlists(r: StepResult, c: RunContext):
        from claudetube.operations.playlist import list_cached_playlists

        playlists = list_cached_playlists(cache_base=CACHE_DIR)
        r.ok(f"{len(playlists)} cached playlists")

    results.append(
        run_step(ctx, "list_cached_playlists (cross-video)", step_playlists).to_dict()
    )

    # Cross-video connections check
    def step_cross_connections(r: StepResult, c: RunContext):
        from claudetube.cache.knowledge_graph import get_knowledge_graph

        graph_dir = CACHE_DIR / "_knowledge_graph"
        graph = get_knowledge_graph(graph_dir=graph_dir)
        total_connections = 0
        for vid in c.video_ids:
            conns = graph.get_video_connections(vid)
            total_connections += len(conns)
        r.ok(
            f"{total_connections} total cross-video connections across {len(c.video_ids)} videos"
        )

    results.append(
        run_step(ctx, "cross-video connections audit", step_cross_connections).to_dict()
    )

    return results


# ===========================================================================
# PHASE 4: Factory & Provider Tests
# ===========================================================================


def phase4_factory(ctx: RunContext) -> list[dict]:
    """Test OperationFactory methods."""
    ctx.log(f"\n{'=' * 60}")
    ctx.log(" PHASE 4: Factory & Provider Tests")
    ctx.log(f"{'=' * 60}")
    results = []

    if ctx.skip_providers:
        ctx.log("  [~] Skipping provider tests (--skip-providers)")
        return [{"name": "phase4", "status": "SKIP", "detail": "--skip-providers"}]

    def step_factory_init(r: StepResult, c: RunContext):
        from claudetube.operations.factory import OperationFactory

        OperationFactory()
        r.ok("factory created")

    results.append(run_step(ctx, "OperationFactory()", step_factory_init).to_dict())

    def step_get_transcriber(r: StepResult, c: RunContext):
        if not c.factory:
            r.skip("no factory")
            return
        t = c.factory.get_transcriber()
        r.ok(f"transcriber={'available' if t else 'None'}")

    results.append(
        run_step(ctx, "factory.get_transcriber()", step_get_transcriber).to_dict()
    )

    def step_get_vision(r: StepResult, c: RunContext):
        if not c.factory:
            r.skip("no factory")
            return
        v = c.factory.get_vision_analyzer()
        r.ok(f"vision={'available' if v else 'None'}")

    results.append(
        run_step(ctx, "factory.get_vision_analyzer()", step_get_vision).to_dict()
    )

    def step_get_video_analyzer(r: StepResult, c: RunContext):
        if not c.factory:
            r.skip("no factory")
            return
        va = c.factory.get_video_analyzer()
        r.ok(f"video_analyzer={'available' if va else 'None'}")

    results.append(
        run_step(ctx, "factory.get_video_analyzer()", step_get_video_analyzer).to_dict()
    )

    def step_get_reasoner(r: StepResult, c: RunContext):
        if not c.factory:
            r.skip("no factory")
            return
        reasoner = c.factory.get_reasoner()
        r.ok(f"reasoner={'available' if reasoner else 'None'}")

    results.append(run_step(ctx, "factory.get_reasoner()", step_get_reasoner).to_dict())

    def step_clear_cache(r: StepResult, c: RunContext):
        if not c.factory:
            r.skip("no factory")
            return
        c.factory.clear_cache()
        r.ok("cache cleared")

    results.append(run_step(ctx, "factory.clear_cache()", step_clear_cache).to_dict())

    return results


# ===========================================================================
# Results & Reporting
# ===========================================================================


def _save_results(ctx: RunContext, phase2_results: list | None = None):
    """Save current results to JSON file."""
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_total": round(time.time() - ctx.start_time, 1),
        "config": {
            "playlist_url": PLAYLIST_URL,
            "cache_dir": str(CACHE_DIR),
            "first_n": ctx.first_n,
            "skip_providers": ctx.skip_providers,
            "resume": ctx.resume,
            "providers_available": ctx.providers_available,
        },
        "phases": ctx.phase_results,
    }
    if phase2_results:
        output["phases"]["phase2_videos"] = phase2_results
    RESULTS_FILE.write_text(json.dumps(output, indent=2, default=str))


def print_summary(ctx: RunContext):
    """Print final summary table."""
    total_elapsed = time.time() - ctx.start_time
    ctx.log(f"\n{'=' * 60}")
    ctx.log(f" SUMMARY ({total_elapsed:.1f}s total)")
    ctx.log(f"{'=' * 60}")

    grand_ok = 0
    grand_fail = 0
    grand_skip = 0
    grand_error = 0

    for phase_name, steps in ctx.phase_results.items():
        if phase_name == "phase2_videos":
            # Phase 2 is a list of video dicts
            for vid_data in steps:
                vid_ok = vid_fail = vid_skip = vid_err = 0
                for step_data in vid_data.get("steps", {}).values():
                    s = step_data.get("status", "?")
                    if s == "OK":
                        vid_ok += 1
                    elif s == "FAIL":
                        vid_fail += 1
                    elif s == "SKIP":
                        vid_skip += 1
                    else:
                        vid_err += 1
                vid_id = vid_data.get("video_id", "?")[:20]
                icon = "+" if vid_fail == 0 and vid_err == 0 else "-"
                ctx.log(
                    f"  [{icon}] video {vid_id:20s}  OK={vid_ok}  FAIL={vid_fail}  SKIP={vid_skip}  ERR={vid_err}"
                )
                grand_ok += vid_ok
                grand_fail += vid_fail
                grand_skip += vid_skip
                grand_error += vid_err
        else:
            ok = fail = skip = err = 0
            for step_data in steps:
                s = step_data.get("status", "?")
                if s == "OK":
                    ok += 1
                elif s == "FAIL":
                    fail += 1
                elif s == "SKIP":
                    skip += 1
                else:
                    err += 1
            icon = "+" if fail == 0 and err == 0 else "-"
            ctx.log(
                f"  [{icon}] {phase_name:24s}  OK={ok}  FAIL={fail}  SKIP={skip}  ERR={err}"
            )
            grand_ok += ok
            grand_fail += fail
            grand_skip += skip
            grand_error += err

    ctx.log(
        f"\n  TOTAL: OK={grand_ok}  FAIL={grand_fail}  SKIP={grand_skip}  ERR={grand_error}"
    )
    ctx.log(f"\n  Results saved to: {RESULTS_FILE}")

    # Cache size
    if CACHE_DIR.exists():
        size = sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file())
        ctx.log(f"  Cache size: {size / 1024 / 1024:.1f}MB at {CACHE_DIR}")


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="Deep integration test for claudetube")
    parser.add_argument(
        "--first", type=int, default=None, help="Process only the first N videos"
    )
    parser.add_argument(
        "--skip-providers",
        action="store_true",
        help="Skip operations requiring AI providers",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip already-processed videos"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    ctx = RunContext(args)

    ctx.log(f"\n{'=' * 60}")
    ctx.log(" CLAUDETUBE DEEP INTEGRATION TEST")
    ctx.log(f"{'=' * 60}")
    ctx.log(f" Playlist: {PLAYLIST_URL}")
    ctx.log(f" Cache:    {CACHE_DIR}")
    ctx.log(f" First N:  {args.first or 'all'}")
    ctx.log(f" Skip AI:  {args.skip_providers}")
    ctx.log(f" Resume:   {args.resume}")

    # Phase 0
    ctx.phase_results["phase0_setup"] = phase0_setup(ctx)

    # Phase 1
    ctx.phase_results["phase1_playlist"] = phase1_playlist(ctx)

    # Phase 2
    phase2_results = phase2_per_video(ctx)
    ctx.phase_results["phase2_videos"] = phase2_results

    # Phase 3
    ctx.phase_results["phase3_cross_video"] = phase3_cross_video(ctx)

    # Phase 4
    ctx.phase_results["phase4_factory"] = phase4_factory(ctx)

    # Final save & summary
    _save_results(ctx, phase2_results)
    print_summary(ctx)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Pytest entry point — skipped unless: pytest --run-deep-integration
# ---------------------------------------------------------------------------
try:
    import pytest

    @pytest.mark.deep_integration
    def test_deep_integration_first1():
        """Run the deep integration test against a single video.

        Skipped by default. Enable with: pytest --run-deep-integration
        """
        args = argparse.Namespace(
            first=1, skip_providers=False, resume=True, verbose=True
        )
        ctx = RunContext(args)
        ctx.phase_results["phase0_setup"] = phase0_setup(ctx)
        ctx.phase_results["phase1_playlist"] = phase1_playlist(ctx)
        phase2_results = phase2_per_video(ctx)
        ctx.phase_results["phase2_videos"] = phase2_results
        ctx.phase_results["phase3_cross_video"] = phase3_cross_video(ctx)
        ctx.phase_results["phase4_factory"] = phase4_factory(ctx)
        _save_results(ctx, phase2_results)
        print_summary(ctx)

        # Collect failures
        failures = []
        for phase_name, steps in ctx.phase_results.items():
            if phase_name == "phase2_videos":
                for vid_data in steps:
                    for step_name, step_data in vid_data.get("steps", {}).items():
                        if step_data.get("status") in ("FAIL", "ERROR"):
                            failures.append(
                                f"{step_name}: {step_data.get('error', 'unknown')}"
                            )
            else:
                for step_data in steps:
                    if step_data.get("status") in ("FAIL", "ERROR"):
                        failures.append(
                            f"{step_data['name']}: {step_data.get('error', 'unknown')}"
                        )

        assert not failures, f"{len(failures)} step(s) failed:\n" + "\n".join(failures)

except ImportError:
    pass
