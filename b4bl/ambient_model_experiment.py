"""Train an experimental clocked classifier with real ambient augmentation.

This is development tooling. Ambient sources are restricted to the development
partition; the source-level validation partition must remain untouched until a
model and decoder configuration are frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import ambient_benchmark, clocked, clocked_receiver, media_audio, prosody
from .compact_packet_experiment import select_attempts
from .real_clocked_experiment import select_rows, training_windows


def synthetic_windows(rng, min_periodicity):
    X, y = [], []
    names = sorted({name for sequence in clocked.vocabulary().values()
                    for name in sequence})
    shorts = [name for name in names if clocked.ticks(name) == 1]
    for name in names:
        for index in range(60):
            expression = prosody.Prosody(float(rng.uniform(.15, 1)),
                                         float(rng.uniform(0, 1)))
            audio = clocked.body(name, expression)
            audio = clocked_receiver.preprocess(
                audio + rng.normal(0, .005, len(audio)))
            X.append(clocked.window_features(audio, min_periodicity))
            y.append(name)
            if index % 3 == 0:
                first, second = rng.choice(shorts, size=2)
                bad = np.concatenate([
                    clocked.body(first, expression), np.zeros(clocked.GUARD),
                    clocked.body(second, expression)])
                X.append(clocked.window_features(
                    clocked_receiver.preprocess(bad), min_periodicity))
                y.append("__invalid__")
    return X, y


def compact_rows(attempts_path, recordings, channel, stop):
    rows, _ = select_attempts(Path(attempts_path).read_text().splitlines(),
                              recordings, channel)
    return [{"raw_file": row["file"].replace(".wav", ".raw.wav"),
             "concepts": row["concepts"], "origin": channel}
            for row in rows[:stop]]


def load_training_rows(split_path, attempts_path, recordings):
    split_rows = json.loads(Path(split_path).read_text())
    base, _ = select_rows(split_rows, include_room3_training=True)
    rows = [{"raw_file": row["raw_file"], "concepts": row["concepts"],
             "origin": "base-real"} for row in base]
    rows += compact_rows(attempts_path, recordings,
                         "compact_aukey_room6_musiclow_v1", 98)
    rows += compact_rows(attempts_path, recordings,
                         "compact_aukey_room6_music_validation_v2", 150)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="experiments/real-clocked-20260917/split.json")
    parser.add_argument("--attempts", default="recordings/attempts.jsonl")
    parser.add_argument("--recordings", default="recordings")
    parser.add_argument("--ambient-dir", default="ambientNoises")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ambient-recordings", type=int, default=600,
                        help="deterministic training-row prefix receiving one ambient copy")
    parser.add_argument("--snr-low", type=float, default=24.0)
    parser.add_argument("--snr-high", type=float, default=36.0)
    parser.add_argument("--ambient-weight", type=float, default=1.0,
                        help="classifier sample weight for ambient-augmented windows")
    parser.add_argument("--min-periodicity", type=float, default=.5)
    args = parser.parse_args()
    if not 0 <= args.min_periodicity < 1:
        parser.error("--min-periodicity must be in [0, 1)")
    if (args.ambient_recordings < 1 or args.snr_low >= args.snr_high or
            args.ambient_weight <= 0):
        parser.error("invalid ambient augmentation settings")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    recordings = Path(args.recordings)
    rows = load_training_rows(args.split, args.attempts, recordings)
    sources = [p for p in ambient_benchmark.media_files(args.ambient_dir)
               if ambient_benchmark.source_split(p.name) == "development"]
    if not sources:
        raise SystemExit("no development ambient sources")

    X, y, weights = [], [], []
    stats = Counter(attempted=len(rows))
    aligned = []
    for index, row in enumerate(rows):
        path = recordings / row["raw_file"]
        if not path.exists():
            stats["missing"] += 1
            continue
        audio = clocked_receiver.read_audio(path)
        windows = training_windows(audio, row["concepts"], args.min_periodicity)
        if windows is None:
            stats["alignment_rejected"] += 1
            continue
        xx, yy = windows
        X.extend(xx); y.extend(yy); weights.extend([1.0] * len(yy))
        aligned.append((row, audio))
        stats[f"original_{row['origin']}"] += 1
        if index and index % 200 == 0:
            print(f"base alignment {index}/{len(rows)}", flush=True)

    rng = np.random.default_rng(20260918)
    selected = np.linspace(0, len(aligned) - 1,
                           min(args.ambient_recordings, len(aligned)), dtype=int)
    augmentation_manifest = []
    for aug_index, row_index in enumerate(selected):
        row, audio = aligned[row_index]
        source = sources[aug_index % len(sources)]
        metadata = media_audio.probe(source)
        duration = len(audio) / clocked.SR
        maximum = max(0.0, metadata["duration_seconds"] - duration - 1)
        start = float(rng.uniform(min(5.0, maximum), maximum)) if maximum else 0.0
        ambient = media_audio.read_clip(source, start, duration)
        length = min(len(audio), len(ambient))
        snr = float(rng.uniform(args.snr_low, args.snr_high))
        mixed, info = ambient_benchmark.mix_at_snr(audio[:length], ambient[:length], snr)
        windows = training_windows(mixed, row["concepts"], args.min_periodicity)
        accepted = windows is not None
        if accepted:
            xx, yy = windows
            X.extend(xx); y.extend(yy)
            weights.extend([args.ambient_weight] * len(yy))
            stats["ambient_augmented_recordings"] += 1
        else:
            stats["ambient_alignment_rejected"] += 1
        augmentation_manifest.append({
            "raw_file": row["raw_file"], "ambient_file": source.name,
            "ambient_start_seconds": start, "mix": info,
            "alignment_accepted": accepted,
        })
        if (aug_index + 1) % 50 == 0:
            print(f"ambient augmentation {aug_index + 1}/{len(selected)}", flush=True)

    synth_X, synth_y = synthetic_windows(np.random.default_rng(170926),
                                         args.min_periodicity)
    X.extend(synth_X); y.extend(synth_y); weights.extend([1.0] * len(synth_y))
    stats["synthetic_windows"] = len(synth_y)
    model = RandomForestClassifier(n_estimators=200, max_depth=22, n_jobs=-1,
                                   random_state=170926)
    model.fit(np.asarray(X), y, sample_weight=np.asarray(weights))
    model.n_jobs = 1
    bundle = {
        "profile": clocked.PROFILE,
        "features": f"{clocked.CONFIDENT_FEATURE_PREFIX}{args.min_periodicity:g}",
        "min_periodicity": args.min_periodicity,
        "frontend": clocked_receiver.PROFILE,
        "model": model,
        "seed": 170926,
        "ambient_profile": ambient_benchmark.PROFILE,
        "ambient_split": "development",
        "ambient_snr_db": [args.snr_low, args.snr_high],
        "ambient_sample_weight": args.ambient_weight,
        "training": "base real+synthetic, prior music, and source-disjoint ambient augmentation",
    }
    model_path = output / "real_classifier.joblib"
    joblib.dump(bundle, model_path)
    (output / "augmentation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in augmentation_manifest))
    summary = {
        "profile": "ambient-augmented-classifier-development-v1",
        "status": "development model; validation ambient sources remain sealed",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "ambient_source_count": len(sources),
        "ambient_source_split": "development",
        "snr_db_range": [args.snr_low, args.snr_high],
        "ambient_sample_weight": args.ambient_weight,
        "stats": dict(stats), "training_windows": len(y),
        "next": "Compare on development mixtures and clean/music regression corpora before freezing; then evaluate once on validation ambient sources.",
    }
    (output / "training-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
