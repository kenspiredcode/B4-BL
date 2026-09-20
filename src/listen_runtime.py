#!/usr/bin/env python3
"""Listen for compact droid packets on a desktop microphone.

This is an integration harness, not a background service. It prints one JSON
line per runtime event and can save the optional spoken response as a WAV file.
It never plays audio automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from b4bl import capture, clocked
from b4bl.runtime_receiver import RuntimeConfig, StreamingPacketReceiver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-device", required=True)
    parser.add_argument("--duration", type=float,
                        help="stop after this many seconds; otherwise Ctrl-C stops")
    parser.add_argument("--chunk-ms", type=float, default=100.0)
    parser.add_argument("--confirm-success", action="store_true")
    parser.add_argument("--no-repeat", action="store_true")
    parser.add_argument("--min-marker-snr-db", type=float,
                        help="optional device-specific amplitude gate in dB")
    parser.add_argument("--reply-directory",
                        help="save suggested ACK/repeat audio here")
    parser.add_argument("--play-replies", action="store_true",
                        help="play suggested replies (explicit opt-in; can be loud)")
    parser.add_argument("--output-device",
                        help="required with --play-replies; explicit device name or index")
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.chunk_ms <= 0:
        parser.error("--chunk-ms must be positive")
    if args.play_replies and not args.output_device:
        parser.error("--play-replies requires --output-device")

    model = joblib.load(args.model)
    if hasattr(model.get("model"), "n_jobs"):
        model["model"].n_jobs = 1
    config = RuntimeConfig(
        analysis_block_ms=args.chunk_ms,
        confirm_success=args.confirm_success,
        request_repeat=not args.no_repeat,
        min_marker_snr_db=args.min_marker_snr_db)
    receiver = StreamingPacketReceiver(model, config)
    block = max(1, round(args.chunk_ms / 1000 * clocked.SR))
    device = capture._resolve_device(args.input_device, "input")
    output_device = (capture._resolve_device(args.output_device, "output")
                     if args.play_replies else None)
    reply_dir = Path(args.reply_directory) if args.reply_directory else None
    if reply_dir:
        reply_dir.mkdir(parents=True, exist_ok=True)

    sd = capture._sd()
    started = time.monotonic()
    event_index = 0
    print(json.dumps({
        "status": "listening", "sample_rate": clocked.SR,
        "chunk_ms": args.chunk_ms, "input_device": device,
        "automatic_playback": args.play_replies,
        "output_device": output_device
    }), flush=True)

    def handle(event):
        nonlocal event_index
        item = event.summary()
        if reply_dir and event.reply_audio is not None:
            path = reply_dir / f"reply_{event_index:05d}_{event.kind}.wav"
            wavfile.write(path, clocked.SR,
                          np.int16(np.clip(event.reply_audio, -1, 1) * 32767))
            item["reply_file"] = str(path)
        print(json.dumps(item), flush=True)
        event_index += 1
        if args.play_replies and event.reply_audio is not None:
            receiver.suppress_for(len(event.reply_audio) / clocked.SR + .25)
            sd.play(event.reply_audio, samplerate=clocked.SR,
                    device=output_device, blocking=True)
    try:
        with sd.InputStream(samplerate=clocked.SR, channels=1,
                            dtype="float32", blocksize=block, device=device) as stream:
            while args.duration is None or time.monotonic() - started < args.duration:
                chunk, overflowed = stream.read(block)
                if overflowed:
                    print(json.dumps({"warning": "input overflow"}), flush=True)
                for event in receiver.feed(np.asarray(chunk[:, 0], dtype=np.float32)):
                    handle(event)
    except KeyboardInterrupt:
        pass
    finally:
        for event in receiver.flush():
            handle(event)
        print(json.dumps({"status": "stopped", "events": event_index,
                          "audio_seconds": receiver.sample_cursor / clocked.SR}), flush=True)


if __name__ == "__main__":
    main()
