"""Small ffmpeg-backed helpers for reproducible media audio access."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from . import clocked


def probe(path):
    """Return duration and first-audio-stream metadata for a local media file."""
    path = Path(path)
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "format=duration:stream=codec_name,sample_rate,channels",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError(f"no audio stream in {path}")
    stream = data["streams"][0]
    return {
        "path": str(path),
        "duration_seconds": float(data["format"]["duration"]),
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
    }


def read_clip(path, start_seconds=0.0, duration_seconds=None,
              sample_rate=clocked.SR):
    """Decode one media interval to mono float32 audio at ``sample_rate``."""
    if start_seconds < 0:
        raise ValueError("start_seconds must be nonnegative")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    command = ["ffmpeg", "-v", "error", "-ss", f"{start_seconds:.6f}",
               "-i", str(path)]
    if duration_seconds is not None:
        command += ["-t", f"{duration_seconds:.6f}"]
    command += ["-vn", "-ac", "1", "-ar", str(sample_rate),
                "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, check=True, capture_output=True)
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def stream(path, chunk_samples, sample_rate=clocked.SR):
    """Yield mono float32 chunks while ffmpeg decodes a complete media file."""
    if chunk_samples < 1:
        raise ValueError("chunk_samples must be positive")
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
               "-ar", str(sample_rate), "-f", "f32le", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    size = chunk_samples * 4
    try:
        while True:
            data = process.stdout.read(size)
            if not data:
                break
            yield np.frombuffer(data, dtype="<f4").copy()
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command, stderr=stderr)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
