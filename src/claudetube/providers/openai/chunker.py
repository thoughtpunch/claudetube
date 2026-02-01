"""
claudetube.providers.openai.chunker - Audio chunking for Whisper API.

The OpenAI Whisper API has a 25MB file size limit. This module handles
splitting larger audio files into chunks using ffmpeg, tracking time
offsets so segment timestamps can be corrected after transcription.

Example:
    >>> chunks = await chunk_audio_if_needed(Path("long_audio.mp3"), max_size_mb=25)
    >>> for chunk in chunks:
    ...     print(f"Chunk at {chunk.offset}s: {chunk.path}")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# 10 minutes per chunk - well under 25MB at 64kbps (~4.8MB)
DEFAULT_CHUNK_DURATION = 600


@dataclass
class AudioChunk:
    """A chunk of audio with its time offset.

    Attributes:
        path: Path to the chunk audio file.
        offset: Start time of this chunk in the original file (seconds).
    """

    path: Path
    offset: float


async def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe.

    Args:
        audio_path: Path to audio file.

    Returns:
        Duration in seconds.

    Raises:
        RuntimeError: If ffprobe fails to read the file.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {audio_path}: {stderr.decode().strip()}"
        )

    return float(stdout.decode().strip())


async def chunk_audio_if_needed(
    audio_path: Path,
    max_size_mb: float = 25,
    chunk_duration: int = DEFAULT_CHUNK_DURATION,
) -> list[AudioChunk]:
    """Split audio file if it exceeds the size limit.

    If the file is under max_size_mb, returns a single chunk with offset 0.
    Otherwise, splits into segments of chunk_duration seconds each.

    Args:
        audio_path: Path to audio file.
        max_size_mb: Maximum file size in MB before chunking.
        chunk_duration: Duration of each chunk in seconds.

    Returns:
        List of AudioChunk with paths and time offsets.
    """
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)

    if file_size_mb <= max_size_mb:
        return [AudioChunk(path=audio_path, offset=0.0)]

    logger.info(
        f"Audio file {file_size_mb:.1f}MB exceeds {max_size_mb}MB limit, chunking"
    )

    duration = await get_audio_duration(audio_path)
    chunk_dir = audio_path.parent / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    chunks = []
    offset = 0.0
    chunk_num = 0

    while offset < duration:
        chunk_path = chunk_dir / f"chunk_{chunk_num:03d}.mp3"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            str(offset),
            "-t",
            str(chunk_duration),
            "-acodec",
            "libmp3lame",
            "-ab",
            "64k",
            str(chunk_path),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()

        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            chunks.append(AudioChunk(path=chunk_path, offset=offset))
            logger.debug(f"Created chunk {chunk_num} at offset {offset}s")
        else:
            logger.warning(f"Failed to create chunk {chunk_num} at offset {offset}s")

        offset += chunk_duration
        chunk_num += 1

    logger.info(f"Split audio into {len(chunks)} chunks")
    return chunks
