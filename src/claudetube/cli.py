#!/usr/bin/env python3
"""
claudetube CLI - Let Claude watch YouTube videos.

Usage:
    claudetube "https://youtube.com/watch?v=VIDEO_ID"
    claudetube "https://youtube.com/watch?v=VIDEO_ID" --frames
"""

import argparse
from pathlib import Path

from claudetube.fast import process_video


def main():
    parser = argparse.ArgumentParser(
        description="Let Claude watch YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s "https://youtube.com/watch?v=VIDEO_ID"
    %(prog)s "https://youtube.com/watch?v=VIDEO_ID" --frames
    %(prog)s "https://youtube.com/watch?v=VIDEO_ID" --model base
        """
    )

    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--frames", action="store_true", help="Extract frames")
    parser.add_argument("--model", default="tiny", help="Whisper model (default: tiny)")
    parser.add_argument("--interval", type=int, default=30, help="Frame interval in seconds")
    parser.add_argument("-o", "--output", type=Path, help="Output directory")

    args = parser.parse_args()

    output_base = args.output or (Path.home() / ".claude" / "video_cache")

    result = process_video(
        args.url,
        output_base=output_base,
        whisper_model=args.model,
        extract_frames=args.frames,
        frame_interval=args.interval,
    )

    if result.success:
        print(f"\n=== RESULT ===")
        print(f"Video ID: {result.video_id}")
        print(f"Title: {result.metadata.get('title')}")
        print(f"Transcript: {result.transcript_srt}")
        if result.frames:
            print(f"Frames: {len(result.frames)}")
    else:
        print(f"ERROR: {result.error}")


if __name__ == "__main__":
    main()
