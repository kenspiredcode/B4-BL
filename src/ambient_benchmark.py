#!/usr/bin/env python3
"""Build and run the reproducible B4-BL ambient audio benchmark."""
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from b4bl import ambient_benchmark as benchmark
from b4bl import clocked, clocked_receiver, compact_clocked, media_audio
from b4bl.compact_packet_experiment import select_attempts
from b4bl.runtime_receiver import RuntimeConfig, StreamingPacketReceiver


DEFAULT_MODEL = "experiments/music-interference-model-v2-20260918/real_classifier.joblib"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def load_model(path):
    model = joblib.load(path)
    if hasattr(model.get("model"), "n_jobs"):
        model["model"].n_jobs = 1
    return model


def runtime_config(args):
    return RuntimeConfig(
        analysis_block_ms=args.chunk_ms,
        min_marker_snr_db=(None if args.no_marker_snr_gate else
                           args.min_marker_snr_db),
        marker_threshold=args.marker_threshold,
        confirm_success=False, request_repeat=True)


def inventory(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in benchmark.media_files(args.ambient_dir):
        item = media_audio.probe(path)
        item["file"] = path.name
        item["split"] = benchmark.source_split(path.name)
        item.pop("path")
        rows.append(item)
    counts = Counter(row["split"] for row in rows)
    manifest = {
        "profile": benchmark.PROFILE,
        "split_seed": benchmark.SPLIT_SEED,
        "validation_fraction_rule": benchmark.VALIDATION_FRACTION,
        "counts": dict(counts),
        "duration_hours": sum(r["duration_seconds"] for r in rows) / 3600,
        "sources": rows,
        "notes": [
            "The split is by complete source file; no source contributes to both partitions.",
            "Validation source membership is frozen before detector calibration or mixture evaluation.",
            "Raw media remains local and is decoded on demand with ffmpeg.",
        ],
    }
    write_json(output / "sources.json", manifest)
    print(json.dumps(manifest["counts"], indent=2))


def characterize(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    paths = benchmark.media_files(args.ambient_dir)
    for index, path in enumerate(paths):
        metadata = media_audio.probe(path)
        for clip_index, start in enumerate(benchmark.clip_starts(
                metadata["duration_seconds"], args.clips_per_source, args.clip_seconds)):
            audio = media_audio.read_clip(path, start, args.clip_seconds)
            rows.append({
                "file": path.name, "split": benchmark.source_split(path.name),
                "clip_index": clip_index, "start_seconds": start,
                "duration_seconds": len(audio) / clocked.SR,
                "features": benchmark.audio_features(audio),
            })
        print(f"characterized {index + 1}/{len(paths)}", flush=True)
    values = defaultdict(list)
    for row in rows:
        for key, value in row["features"].items():
            values[key].append(value)
    summary = {
        key: {"min": float(np.min(v)), "median": float(np.median(v)),
              "max": float(np.max(v)), "p10": float(np.percentile(v, 10)),
              "p90": float(np.percentile(v, 90))}
        for key, v in values.items()
    }
    write_json(output / "clips.json", {
        "profile": benchmark.PROFILE, "clip_seconds": args.clip_seconds,
        "clips_per_source": args.clips_per_source, "summary": summary, "clips": rows})
    print(json.dumps(summary, indent=2))


def negative(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model_path = Path(args.model)
    model = load_model(model_path)
    config = runtime_config(args)
    paths = [p for p in benchmark.media_files(args.ambient_dir)
             if benchmark.source_split(p.name) == args.split]
    if args.max_sources:
        paths = paths[:args.max_sources]
    counts = Counter(sources=len(paths))
    audio_seconds = wall_seconds = 0.0
    with (output / "events.jsonl").open("w") as event_log:
        for index, path in enumerate(paths):
            receiver = StreamingPacketReceiver(model, config)
            began = time.monotonic()
            events = []
            source_samples = 0
            for chunk in media_audio.stream(path, round(args.chunk_ms / 1000 * clocked.SR)):
                source_samples += len(chunk)
                events.extend(receiver.feed(chunk))
            events.extend(receiver.flush())
            elapsed = time.monotonic() - began
            audio_seconds += source_samples / clocked.SR
            wall_seconds += elapsed
            for event in events:
                counts[f"{event.kind}_events"] += 1
                counts["spoken_repeat_events"] += int(bool(event.reply_concepts))
                counts["accepted_events"] += int(event.accepted)
                event_log.write(json.dumps({"file": path.name,
                                            "event": event.summary()}) + "\n")
            print(f"negative replay {index + 1}/{len(paths)}: {path.name}", flush=True)
    summary = {
        "profile": benchmark.PROFILE, "mode": "negative-continuous",
        "split": args.split, "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "runtime_config": config.__dict__, "counts": dict(counts),
        "audio_hours": audio_seconds / 3600, "wall_seconds": wall_seconds,
        "processing_realtime_factor": wall_seconds / audio_seconds if audio_seconds else None,
        "spoken_repeats_per_hour": counts["spoken_repeat_events"] / (audio_seconds / 3600) if audio_seconds else None,
        "accepted_false_packets_per_hour": counts["accepted_events"] / (audio_seconds / 3600) if audio_seconds else None,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def mixed(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model_path = Path(args.model)
    model = load_model(model_path)
    config = runtime_config(args)
    ambient_paths = [p for p in benchmark.media_files(args.ambient_dir)
                     if benchmark.source_split(p.name) == args.split]
    recordings = Path(args.recordings)
    rows, source_rows = select_attempts(
        (recordings / "attempts.jsonl").read_text().splitlines(), recordings, args.channel)
    rows = [r for r in rows if (recordings / r["file"].replace(".wav", ".raw.wav")).exists()]
    if args.packet_offset:
        rows = rows[args.packet_offset:] + rows[:args.packet_offset]
    if not rows:
        raise SystemExit(f"no available packet recordings for channel {args.channel!r}")
    snrs = [float(x) for x in args.snrs.split(",")]
    rng = np.random.default_rng(args.seed)
    plan = []
    if args.factorial and args.mixtures % len(snrs):
        raise SystemExit("--factorial requires --mixtures divisible by the SNR count")
    case_count = args.mixtures // len(snrs) if args.factorial else args.mixtures
    cases = []
    for case_index in range(case_count):
        packet_row = rows[case_index % len(rows)]
        ambient_path = ambient_paths[case_index % len(ambient_paths)]
        metadata = media_audio.probe(ambient_path)
        packet_path = recordings / packet_row["file"].replace(".wav", ".raw.wav")
        packet = clocked_receiver.read_audio(packet_path)
        maximum = max(0.0, metadata["duration_seconds"] - len(packet) / clocked.SR - 1)
        start = float(rng.uniform(min(5.0, maximum), maximum)) if maximum else 0.0
        cases.append((packet_row, packet, ambient_path, start))
    if args.factorial:
        for packet_row, packet, ambient_path, start in cases:
            for snr in snrs:
                plan.append((packet_row, packet, ambient_path, start, snr))
    else:
        for index, case in enumerate(cases):
            plan.append((*case, snrs[index % len(snrs)]))
    counts = Counter(mixtures=len(plan), source_attempt_rows=source_rows)
    by_snr = defaultdict(Counter)
    with (output / "results.jsonl").open("w") as log:
        for index, (row, packet, ambient_path, start, snr) in enumerate(plan):
            ambient = media_audio.read_clip(ambient_path, start, len(packet) / clocked.SR)
            length = min(len(packet), len(ambient))
            mixed_audio, mix_info = benchmark.mix_at_snr(packet[:length], ambient[:length], snr)
            receiver = StreamingPacketReceiver(model, config)
            events = []
            chunk = round(args.chunk_ms / 1000 * clocked.SR)
            # Give the adaptive floor real ambient context before the packet mixture.
            lead_start = max(0.0, start - args.context_seconds)
            lead = media_audio.read_clip(ambient_path, lead_start, start - lead_start)
            for begin in range(0, len(lead), chunk):
                events.extend(receiver.feed(lead[begin:begin + chunk]))
            for begin in range(0, len(mixed_audio), chunk):
                events.extend(receiver.feed(mixed_audio[begin:begin + chunk]))
            events.extend(receiver.flush())
            expected = list(row["concepts"])
            accepted = [e for e in events if e.accepted]
            exact = any(e.result.acoustic.words == expected for e in accepted)
            wrong = any(e.result.acoustic.words != expected for e in accepted)
            counts["verified_exact"] += int(exact)
            counts["accepted_wrong"] += int(wrong)
            counts["no_exact_delivery"] += int(not exact)
            by_snr[str(snr)]["attempts"] += 1
            by_snr[str(snr)]["verified_exact"] += int(exact)
            by_snr[str(snr)]["accepted_wrong"] += int(wrong)
            log.write(json.dumps({
                "index": index, "packet_file": row["file"], "expected": expected,
                "ambient_file": ambient_path.name, "ambient_start_seconds": start,
                "mix": mix_info, "events": [e.summary() for e in events],
                "verified_exact": exact, "accepted_wrong": wrong,
            }) + "\n")
            if (index + 1) % 25 == 0:
                print(f"mixed replay {index + 1}/{len(plan)}", flush=True)
    summary = {
        "profile": benchmark.PROFILE, "mode": "controlled-snr-mixtures",
        "split": args.split, "packet_channel": args.channel,
        "packet_offset": args.packet_offset,
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "runtime_config": config.__dict__, "seed": args.seed, "snrs_db": snrs,
        "design": "same-case-factorial" if args.factorial else "rotating-snr",
        "counts": dict(counts),
        "verified_delivery_rate": counts["verified_exact"] / len(plan),
        "accepted_wrong_rate": counts["accepted_wrong"] / len(plan),
        "by_snr": {key: dict(value) for key, value in by_snr.items()},
        "caveat": "Digital mixtures test interference robustness; they do not reproduce room transfer, speaker nonlinearities, or microphone AGC.",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ambient-dir", default="ambientNoises")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inventory")
    p.add_argument("--output", required=True)
    p.set_defaults(function=inventory)
    p = sub.add_parser("characterize")
    p.add_argument("--output", required=True)
    p.add_argument("--clips-per-source", type=int, default=5)
    p.add_argument("--clip-seconds", type=float, default=12.0)
    p.set_defaults(function=characterize)
    for name, function in [("negative", negative), ("mixed", mixed)]:
        p = sub.add_parser(name)
        p.add_argument("--output", required=True)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--split", choices=["development", "validation"], default="development")
        p.add_argument("--chunk-ms", type=float, default=100.0)
        p.add_argument("--min-marker-snr-db", type=float, default=3.0)
        p.add_argument("--no-marker-snr-gate", action="store_true",
                       help="rely on normalized marker shape, SYNC cadence, and CRC")
        p.add_argument("--marker-threshold", type=float, default=.55)
        p.set_defaults(function=function)
    p = sub.choices["negative"]
    p.add_argument("--max-sources", type=int)
    p = sub.choices["mixed"]
    p.add_argument("--channel", required=True)
    p.add_argument("--recordings", default="recordings")
    p.add_argument("--mixtures", type=int, default=500)
    p.add_argument("--snrs", default="12,8,4,0,-4")
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--context-seconds", type=float, default=3.0)
    p.add_argument("--packet-offset", type=int, default=0,
                   help="rotate packet rows so validation can exclude development cases")
    p.add_argument("--factorial", action="store_true",
                   help="replay every packet/ambient/start case at every SNR")
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
