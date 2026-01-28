"""
End-to-end integration test for claudetube across multiple sites.

Modes:
  --metadata-only  Fast test: URL parsing + metadata fetch only (~2-5s per video)
  --full           Full pipeline: metadata + transcript + frames (slow)
  (default)        Same as --metadata-only

Runs the full pipeline for each URL:
  1. extract_video_id (URL parsing)
  2. _get_metadata (yt-dlp metadata fetch)
  3. process_video (metadata + transcript) [--full only]
  4. get_frames (extract frames) [--full only]
  5. get_hq_frames (extract HQ frames) [--full only]

Usage:
    python tests/integration_test.py                  # metadata-only test, all sites
    python tests/integration_test.py --first          # first URL per site only
    python tests/integration_test.py youtube rumble   # specific sites
    python tests/integration_test.py --full           # full pipeline (slow!)
    python tests/integration_test.py --full --first   # full pipeline, first URL only
"""

import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from claudetube.core import (  # noqa: E402
    _get_metadata,
    get_frames_at,
    get_hq_frames_at,
    process_video,
)
from claudetube.urls import VideoURL, extract_video_id  # noqa: E402

TEST_CACHE = Path(__file__).parent / "_integration_cache"
RESULTS_FILE = Path(__file__).parent / "integration_results.json"


def load_test_urls() -> dict:
    with open(Path(__file__).parent / "test_videos.json") as f:
        return json.load(f)


def run_metadata_test(url: str, site: str) -> dict:
    """Fast test: URL parsing + metadata fetch only (~2-5s per video)."""
    result = {
        "url": url,
        "site": site,
        "video_id": None,
        "steps": {},
    }

    # Step 1: URL parsing with VideoURL
    try:
        t0 = time.time()
        parsed = VideoURL.parse(url)
        elapsed = time.time() - t0
        result["video_id"] = parsed.video_id
        result["steps"]["parse_url"] = {
            "status": "OK",
            "elapsed": round(elapsed, 3),
            "video_id": parsed.video_id,
            "provider": parsed.provider,
            "is_known": parsed.is_known_provider,
        }
    except Exception as e:
        result["steps"]["parse_url"] = {"status": "FAIL", "error": str(e)[:200]}
        return result

    # Step 2: Metadata fetch via yt-dlp
    try:
        t0 = time.time()
        meta = _get_metadata(url)
        elapsed = time.time() - t0

        if meta and "_error" not in meta:
            result["steps"]["get_metadata"] = {
                "status": "OK",
                "elapsed": round(elapsed, 1),
                "title": meta.get("title", "?")[:80],
                "duration": meta.get("duration"),
                "uploader": meta.get("uploader", meta.get("channel", "?"))[:40],
                "has_subtitles": bool(
                    meta.get("subtitles") or meta.get("automatic_captions")
                ),
                "has_audio_only": any(
                    f.get("vcodec") == "none"
                    or f.get("acodec") != "none"
                    and not f.get("vcodec")
                    for f in meta.get("formats", [])
                    if f.get("acodec") != "none"
                ),
            }
        else:
            error_msg = (
                meta.get("_error", "Unknown error") if meta else "No metadata returned"
            )
            result["steps"]["get_metadata"] = {
                "status": "FAIL",
                "elapsed": round(elapsed, 1),
                "error": error_msg[:200],
            }
    except Exception as e:
        result["steps"]["get_metadata"] = {"status": "ERROR", "error": str(e)[:200]}

    return result


def run_full_pipeline_test(url: str, site: str) -> dict:
    """Full pipeline test: metadata + transcript + frames (slow!)."""
    result = {
        "url": url,
        "site": site,
        "video_id": None,
        "steps": {},
    }

    # Step 0: extract_video_id
    try:
        vid = extract_video_id(url)
        result["video_id"] = vid
        result["steps"]["extract_id"] = {"status": "OK", "video_id": vid}
    except Exception as e:
        result["steps"]["extract_id"] = {"status": "FAIL", "error": str(e)}
        return result

    # Step 1: process_video (metadata + transcript)
    try:
        t0 = time.time()
        vr = process_video(url, output_base=TEST_CACHE)
        elapsed = time.time() - t0
        if vr.success:
            result["steps"]["process_video"] = {
                "status": "OK",
                "elapsed": round(elapsed, 1),
                "title": vr.metadata.get("title", "?")[:80],
                "duration": vr.metadata.get("duration"),
                "transcript_source": vr.metadata.get("transcript_source"),
                "has_srt": vr.transcript_srt is not None and vr.transcript_srt.exists(),
                "has_txt": vr.transcript_txt is not None and vr.transcript_txt.exists(),
                "has_thumbnail": vr.thumbnail is not None,
            }
        else:
            result["steps"]["process_video"] = {
                "status": "FAIL",
                "elapsed": round(elapsed, 1),
                "error": vr.error[:200] if vr.error else "Unknown",
            }
            return result
    except Exception as e:
        result["steps"]["process_video"] = {"status": "ERROR", "error": str(e)[:200]}
        return result

    # Step 2: get_frames (low quality, 2 frames at t=5)
    try:
        t0 = time.time()
        frames = get_frames_at(
            vid, start_time=5, duration=2, interval=1, output_base=TEST_CACHE
        )
        elapsed = time.time() - t0
        result["steps"]["get_frames"] = {
            "status": "OK" if frames else "NO_FRAMES",
            "elapsed": round(elapsed, 1),
            "frame_count": len(frames),
        }
    except Exception as e:
        result["steps"]["get_frames"] = {"status": "ERROR", "error": str(e)[:200]}

    # Step 3: get_hq_frames (high quality, 1 frame at t=5)
    try:
        t0 = time.time()
        hq_frames = get_hq_frames_at(
            vid, start_time=5, duration=1, interval=1, output_base=TEST_CACHE
        )
        elapsed = time.time() - t0
        result["steps"]["get_hq_frames"] = {
            "status": "OK" if hq_frames else "NO_FRAMES",
            "elapsed": round(elapsed, 1),
            "frame_count": len(hq_frames),
        }
    except Exception as e:
        result["steps"]["get_hq_frames"] = {"status": "ERROR", "error": str(e)[:200]}

    return result


def print_result(r: dict, verbose: bool = False):
    """Print a single test result."""
    vid = r["video_id"] or "?"
    print(f"\n  URL: {r['url']}")
    print(f"  ID:  {vid}")
    for step, info in r["steps"].items():
        status = info["status"]
        icon = "+" if status == "OK" else "-" if status == "FAIL" else "~"
        extra = ""
        if "elapsed" in info:
            extra += f" ({info['elapsed']}s)"
        if "title" in info:
            extra += f" \"{info['title']}\""
        if "provider" in info and info["provider"]:
            extra += f" [{info['provider']}]"
        if "error" in info:
            extra += f" [{info['error'][:80]}]"
        if "frame_count" in info:
            extra += f" [{info['frame_count']} frames]"
        if "transcript_source" in info:
            extra += f" [source: {info['transcript_source']}]"
        if verbose:
            if "has_subtitles" in info:
                extra += f" [subs: {info['has_subtitles']}]"
            if "has_audio_only" in info:
                extra += f" [audio-only: {info['has_audio_only']}]"
        print(f"  [{icon}] {step}: {status}{extra}")


def main():
    args = sys.argv[1:]

    # Parse flags
    first_only = "--first" in args
    full_mode = "--full" in args
    verbose = "--verbose" in args or "-v" in args

    # Remove flags from args
    args = [a for a in args if not a.startswith("-")]

    all_urls = load_test_urls()

    # Filter sites if specified
    if args:
        sites = {k: v for k, v in all_urls.items() if k in args}
    else:
        sites = all_urls

    # Print mode
    mode_name = "FULL PIPELINE" if full_mode else "METADATA-ONLY (fast)"
    print(f"\n{'='*60}")
    print(f" MODE: {mode_name}")
    print(f"{'='*60}")

    all_results = []
    summary = {}
    total_time = time.time()

    for site, urls in sites.items():
        if first_only:
            urls = urls[:1]

        print(f"\n{'='*60}")
        print(f" {site.upper()} ({len(urls)} URLs)")
        print(f"{'='*60}")

        site_pass = 0
        site_total = len(urls)

        for url in urls:
            if full_mode:
                r = run_full_pipeline_test(url, site)
                # Count as pass if process_video succeeded
                passed = r["steps"].get("process_video", {}).get("status") == "OK"
            else:
                r = run_metadata_test(url, site)
                # Count as pass if metadata fetch succeeded
                passed = r["steps"].get("get_metadata", {}).get("status") == "OK"

            all_results.append(r)
            print_result(r, verbose=verbose)

            if passed:
                site_pass += 1

        summary[site] = {"passed": site_pass, "total": site_total}

    # Save results
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))

    # Print summary
    elapsed = time.time() - total_time
    print(f"\n{'='*60}")
    print(f" SUMMARY ({elapsed:.1f}s total)")
    print(f"{'='*60}")
    total_pass = 0
    total_all = 0
    for site, s in summary.items():
        total_pass += s["passed"]
        total_all += s["total"]
        icon = "+" if s["passed"] == s["total"] else "-" if s["passed"] == 0 else "~"
        print(f"  [{icon}] {site:20s} {s['passed']}/{s['total']}")
    print(f"\n  TOTAL: {total_pass}/{total_all}")
    print(f"\n  Results saved to: {RESULTS_FILE}")

    # Cache info (only relevant for full mode)
    if full_mode and TEST_CACHE.exists():
        size = sum(f.stat().st_size for f in TEST_CACHE.rglob("*") if f.is_file())
        print(f"  Cache size: {size / 1024 / 1024:.1f}MB at {TEST_CACHE}")


if __name__ == "__main__":
    main()
