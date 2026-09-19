#!/usr/bin/env python3
"""Compare marker detection against clean-packet marker locations in mixtures."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from b4bl import (ambient_benchmark, clocked, clocked_receiver, media_audio,
                  robust_markers)
from b4bl.compact_packet_experiment import select_attempts


def matched(expected, found, tolerance=.06 * clocked.SR):
    remaining = list(found)
    hits = 0
    for target in expected:
        if not remaining:
            break
        index = int(np.argmin(np.abs(np.asarray(remaining) - target)))
        if abs(remaining[index] - target) <= tolerance:
            hits += 1
            remaining.pop(index)
    return hits, len(remaining)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--snrs", default="30,24,18,12")
    parser.add_argument("--channel", default="compact_room5_v2")
    parser.add_argument("--ambient-dir", default="ambientNoises")
    parser.add_argument("--recordings", default="recordings")
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    snrs = [float(value) for value in args.snrs.split(",")]
    recordings = Path(args.recordings)
    rows, _ = select_attempts((recordings / "attempts.jsonl").read_text().splitlines(),
                              recordings, args.channel)
    rows = [row for row in rows if
            (recordings / row["file"].replace(".wav", ".raw.wav")).exists()]
    sources = [path for path in ambient_benchmark.media_files(args.ambient_dir)
               if ambient_benchmark.source_split(path.name) == "development"]
    detectors = {
        "spectral055": lambda audio: clocked_receiver.marker_positions(audio, .55),
        "whitened030": lambda audio: robust_markers.marker_positions(audio, "whitened", .30),
        "whitened025": lambda audio: robust_markers.marker_positions(audio, "whitened", .25),
        "lattice020": lambda audio: robust_markers.lattice(audio, "whitened", .20),
        "lattice015": lambda audio: robust_markers.lattice(audio, "whitened", .15),
    }
    totals = defaultdict(Counter)
    by_snr = defaultdict(lambda: defaultdict(Counter))
    rng = np.random.default_rng(20260918)
    for case in range(args.cases):
        row, source = rows[case], sources[case]
        packet = clocked_receiver.read_audio(
            recordings / row["file"].replace(".wav", ".raw.wav"))
        clean = clocked_receiver.preprocess(packet)
        expected = clocked_receiver.marker_positions(clean)
        metadata = media_audio.probe(source)
        maximum = max(0., metadata["duration_seconds"] - len(packet) / clocked.SR - 1)
        start = float(rng.uniform(min(5., maximum), maximum)) if maximum else 0.
        ambient = media_audio.read_clip(source, start, len(packet) / clocked.SR)
        length = min(len(packet), len(ambient))
        for snr in snrs:
            mixed, _ = ambient_benchmark.mix_at_snr(packet[:length], ambient[:length], snr)
            processed = clocked_receiver.preprocess(mixed)
            for name, detector in detectors.items():
                found = detector(processed)
                hits, extras = matched(expected, found)
                target = by_snr[name][str(snr)]
                target["expected"] += len(expected); target["hits"] += hits
                target["extras"] += extras; target["exact_sequences"] += int(found == expected)
                totals[name]["expected"] += len(expected); totals[name]["hits"] += hits
                totals[name]["extras"] += extras; totals[name]["exact_sequences"] += int(found == expected)
        print(f"marker case {case + 1}/{args.cases}", flush=True)
    summary = {
        "profile": "marker-development-benchmark-v1", "cases": args.cases,
        "snrs_db": snrs, "totals": {k: dict(v) for k, v in totals.items()},
        "by_snr": {name: {snr: dict(counts) for snr, counts in values.items()}
                   for name, values in by_snr.items()},
        "validation_sources_opened": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
