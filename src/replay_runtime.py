#!/usr/bin/env python3
"""Replay recordings through the streaming receiver without opening devices."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from b4bl import clocked, clocked_receiver, compact_clocked
from b4bl.compact_packet_experiment import select_attempts
from b4bl.runtime_receiver import RuntimeConfig, StreamingPacketReceiver


def _safe_name(path):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(path).stem)


def _run_file(path, model, config, chunk_samples):
    audio = clocked_receiver.read_audio(path)
    receiver = StreamingPacketReceiver(model, config)
    events = []
    began = time.monotonic()
    for start in range(0, len(audio), chunk_samples):
        events.extend(receiver.feed(audio[start:start + chunk_samples]))
    events.extend(receiver.flush())
    wall = time.monotonic() - began
    return audio, events, wall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--channel", help="replay raw attempts from recordings/attempts.jsonl")
    source.add_argument("--audio", nargs="+", help="one or more mono 44.1-kHz WAV files")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recordings", default="recordings")
    parser.add_argument("--chunk-ms", type=float, default=100.0)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--confirm-success", action="store_true")
    parser.add_argument("--no-repeat", action="store_true")
    parser.add_argument("--min-marker-snr-db", type=float, default=3.0)
    parser.add_argument("--render-replies", action="store_true")
    args = parser.parse_args()
    if args.chunk_ms <= 0:
        parser.error("--chunk-ms must be positive")
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be positive")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model_path = Path(args.model)
    model = joblib.load(model_path)
    if hasattr(model.get("model"), "n_jobs"):
        model["model"].n_jobs = 1
    config = RuntimeConfig(
        analysis_block_ms=args.chunk_ms,
        confirm_success=args.confirm_success,
        request_repeat=not args.no_repeat,
        min_marker_snr_db=args.min_marker_snr_db)
    chunk_samples = max(1, round(args.chunk_ms / 1000 * clocked.SR))

    recordings = Path(args.recordings)
    labeled = args.channel is not None
    if labeled:
        rows, source_rows = select_attempts(
            (recordings / "attempts.jsonl").read_text().splitlines(),
            recordings, args.channel)
        inputs = [(recordings / r["file"].replace(".wav", ".raw.wav"), r) for r in rows]
    else:
        source_rows = None
        inputs = [(Path(p), None) for p in args.audio]
    if args.max_files is not None:
        inputs = inputs[:args.max_files]

    counts = Counter(files=len(inputs))
    latencies = []
    latencies_by_kind = defaultdict(list)
    processing = []
    audio_seconds = 0.0
    wall_seconds = 0.0
    reply_dir = output / "replies"
    if args.render_replies:
        reply_dir.mkdir()

    with (output / "events.jsonl").open("w") as log:
        for file_index, (path, row) in enumerate(inputs):
            if not path.exists():
                counts["missing"] += 1
                log.write(json.dumps({"file": str(path), "missing": True}) + "\n")
                continue
            audio, events, wall = _run_file(path, model, config, chunk_samples)
            counts["available"] += 1
            audio_seconds += len(audio) / clocked.SR
            wall_seconds += wall
            accepted = [event for event in events if event.accepted]
            counts["accepted_events"] += len(accepted)
            counts["rejected_events"] += sum(event.kind == "rejected" for event in events)
            counts["incomplete_events"] += sum(event.kind == "incomplete" for event in events)
            counts["false_trigger_events"] += sum(event.kind == "false_trigger" for event in events)
            if row is not None:
                expected = list(row["concepts"])
                exact = any(event.result.acoustic.words == expected for event in accepted)
                wrong = any(event.result.acoustic.words != expected for event in accepted)
                counts["verified_exact"] += int(exact)
                counts["accepted_wrong"] += int(wrong)
                counts["no_exact_delivery"] += int(not exact)
            for event_index, event in enumerate(events):
                latencies.append(event.decision_latency_seconds)
                latencies_by_kind[event.kind].append(event.decision_latency_seconds)
                processing.append(event.processing_seconds)
                item = {"file": str(path), "event": event.summary()}
                if row is not None:
                    item["expected"] = list(row["concepts"])
                if args.render_replies and event.reply_audio is not None:
                    reply_path = reply_dir / f"{file_index:05d}_{event_index}_{_safe_name(path)}.wav"
                    pcm = np.int16(np.clip(event.reply_audio, -1, 1) * 32767)
                    wavfile.write(reply_path, clocked.SR, pcm)
                    item["reply_file"] = str(reply_path)
                log.write(json.dumps(item) + "\n")
            if file_index and file_index % 25 == 0:
                print(f"replayed {file_index}/{len(inputs)}", flush=True)

    summary = {
        "profile": "streaming-runtime-v1",
        "channel": args.channel,
        "source_attempt_rows": source_rows,
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "config": config.__dict__,
        "counts": dict(counts),
        "audio_seconds": audio_seconds,
        "wall_seconds": wall_seconds,
        "processing_realtime_factor": wall_seconds / audio_seconds if audio_seconds else None,
        "decision_latency_seconds": {
            "mean": float(np.mean(latencies)) if latencies else None,
            "max": max(latencies) if latencies else None,
            "by_kind": {
                kind: {"count": len(values), "mean": float(np.mean(values)),
                       "max": max(values)}
                for kind, values in sorted(latencies_by_kind.items())
            },
        },
        "decode_processing_seconds": {
            "mean": float(np.mean(processing)) if processing else None,
            "max": max(processing) if processing else None,
        },
        "notes": [
            "Decision latency is audio-time from the closing marker through the chunked detector.",
            "Processing time is measured on this computer and is not a robot-hardware benchmark.",
            "False-trigger rates require long representative ambient recordings, not packet corpora.",
            "Runtime settings tuned on a corpus are development results, not a fresh validation on that corpus."
        ]
    }
    if labeled and inputs:
        summary["verified_delivery_rate"] = counts["verified_exact"] / len(inputs)
        summary["accepted_wrong_rate"] = counts["accepted_wrong"] / len(inputs)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
