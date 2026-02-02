#!/usr/bin/env python3
"""
claudetube CLI - Let Claude watch YouTube videos.

Usage:
    claudetube "https://youtube.com/watch?v=VIDEO_ID"
    claudetube "https://youtube.com/watch?v=VIDEO_ID" --frames
    claudetube validate-config
    claudetube extract-entities VIDEO_ID
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from claudetube.operations.processor import process_video


def _cmd_process(args):
    """Handle the default process-video command."""
    output_base = args.output or (Path.home() / ".claude" / "video_cache")

    result = process_video(
        args.url,
        output_base=output_base,
        whisper_model=args.model,
        extract_frames=args.frames,
        frame_interval=args.interval,
    )

    if result.success:
        print("\n=== RESULT ===")
        print(f"Video ID: {result.video_id}")
        print(f"Title: {result.metadata.get('title')}")
        print(f"Transcript: {result.transcript_srt}")
        if result.frames:
            print(f"Frames: {len(result.frames)}")
    else:
        print(f"ERROR: {result.error}")


def _cmd_validate_config(args):
    """Handle the validate-config subcommand."""
    from claudetube.config.loader import (
        _find_project_config,
        _get_user_config_path,
        _load_yaml_config,
    )
    from claudetube.providers.config import validate_providers_config
    from claudetube.providers.registry import list_all, list_available

    # Find and load config file
    config_path = None
    yaml_config = None

    project_path = _find_project_config()
    if project_path:
        config_path = project_path
        yaml_config = _load_yaml_config(project_path)

    if yaml_config is None:
        user_path = _get_user_config_path()
        if user_path.exists():
            config_path = user_path
            yaml_config = _load_yaml_config(user_path)

    if config_path:
        print(f"Config file: {config_path}")
    else:
        print("No config file found.")
        print("  Searched: .claudetube/config.yaml (project)")
        print(f"  Searched: {_get_user_config_path()} (user)")
        print("\nUsing defaults (no validation needed).")
        sys.exit(0)

    if yaml_config is None:
        print("  Failed to parse config file.")
        sys.exit(1)

    # Run validation
    result = validate_providers_config(yaml_config)

    # Print errors
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  ✗ {error}")

    # Print warnings
    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  ! {warning}")

    # Check provider availability
    if not args.skip_availability:
        print("\nProvider availability:")
        all_providers = list_all()
        available = list_available()
        for name in all_providers:
            status = "available" if name in available else "not available"
            marker = "+" if name in available else "-"
            print(f"  {marker} {name}: {status}")

    # Summary
    if result.is_valid and not result.warnings:
        print("\nConfig is valid.")
    elif result.is_valid:
        print(f"\nConfig is valid with {len(result.warnings)} warning(s).")
    else:
        print(f"\nConfig is invalid: {len(result.errors)} error(s), {len(result.warnings)} warning(s).")

    sys.exit(0 if result.is_valid else 1)


def _cmd_extract_entities(args):
    """Handle the extract-entities subcommand."""
    from claudetube.operations.entity_extraction import extract_entities_for_video
    from claudetube.parsing.utils import extract_video_id

    video_id = extract_video_id(args.video_id)
    output_base = args.output or (Path.home() / ".claude" / "video_cache")

    vision_analyzer = None
    reasoner = None
    if args.provider:
        from claudetube.providers import get_provider
        from claudetube.providers.base import Reasoner, VisionAnalyzer

        p = get_provider(args.provider)
        if isinstance(p, VisionAnalyzer):
            vision_analyzer = p
        if isinstance(p, Reasoner):
            reasoner = p

    result = extract_entities_for_video(
        video_id,
        scene_id=args.scene_id,
        force=args.force,
        generate_visual=not args.no_visual,
        output_base=output_base,
        vision_analyzer=vision_analyzer,
        reasoner=reasoner,
    )

    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="Let Claude watch YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s "https://youtube.com/watch?v=VIDEO_ID"
    %(prog)s "https://youtube.com/watch?v=VIDEO_ID" --frames
    %(prog)s "https://youtube.com/watch?v=VIDEO_ID" --model base
    %(prog)s validate-config
    %(prog)s validate-config --skip-availability
    %(prog)s extract-entities VIDEO_ID
    %(prog)s extract-entities VIDEO_ID --scene-id 0 --force
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # validate-config subcommand
    vc_parser = subparsers.add_parser(
        "validate-config",
        help="Validate provider configuration",
    )
    vc_parser.add_argument(
        "--skip-availability", action="store_true",
        help="Skip checking provider availability (faster)",
    )

    # extract-entities subcommand
    ee_parser = subparsers.add_parser(
        "extract-entities",
        help="Extract entities from video scenes using AI providers",
    )
    ee_parser.add_argument("video_id", help="Video ID or URL")
    ee_parser.add_argument(
        "--scene-id", type=int, default=None,
        help="Specific scene ID to extract (default: all scenes)",
    )
    ee_parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if cached",
    )
    ee_parser.add_argument(
        "--no-visual", action="store_true",
        help="Skip generating visual.json from entities",
    )
    ee_parser.add_argument(
        "--provider", default=None,
        help="Override AI provider (e.g., anthropic, openai, claude-code)",
    )
    ee_parser.add_argument("-o", "--output", type=Path, help="Output directory")

    # Default process-video arguments (when no subcommand)
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    parser.add_argument("--frames", action="store_true", help="Extract frames")
    parser.add_argument("--model", default="tiny", help="Whisper model (default: tiny)")
    parser.add_argument(
        "--interval", type=int, default=30, help="Frame interval in seconds"
    )
    parser.add_argument("-o", "--output", type=Path, help="Output directory")

    args = parser.parse_args()

    if args.command == "validate-config":
        _cmd_validate_config(args)
    elif args.command == "extract-entities":
        _cmd_extract_entities(args)
    elif args.url:
        _cmd_process(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
