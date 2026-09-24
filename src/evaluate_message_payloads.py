#!/usr/bin/env python3
"""Retrospective Messages evaluation on retained, labeled Full captures.

Extract payload intervals using detected word markers; no physical recording is
made. This is post-hoc analysis, never prospective Messages validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from b4bl import clocked, clocked_receiver, compact_clocked, decode_message
from b4bl.compact_packet_experiment import select_attempts
from b4bl.models import MODEL_SHA256, load_model


CHANNEL = "compact_aukey_room6_music_validation_v3"
HEADER_WORDS = 7
TRAILER_WORDS = 8


def payload_crop(audio, words):
    """Return a measured payload interval, or an explicit extraction failure."""
    starts = clocked_receiver.marker_positions(audio)
    if len(starts) != len(words) + 1:
        return None, f"marker_count_{len(starts)}_expected_{len(words) + 1}"
    first = HEADER_WORDS
    closing = len(words) - TRAILER_WORDS
    if closing <= first:
        return None, "empty_payload"
    pad = round(.10 * clocked.SR)
    begin = max(0, starts[first] - pad)
    end = min(len(audio), starts[closing] + len(clocked.MARKER) + pad)
    return audio[begin:end], ""


def evaluate(recordings, model, limit=None):
    recordings = Path(recordings)
    lines = (recordings / "attempts.jsonl").read_text().splitlines()
    rows, source_count = select_attempts(lines, recordings, CHANNEL)
    rows = sorted(rows, key=lambda row: row["file"])
    if limit is not None:
        rows = rows[:limit]
    counts = Counter()
    reasons = Counter()
    predictions = []
    manifest = []
    for row in rows:
        words = list(row["concepts"])
        expected = words[HEADER_WORDS:-TRAILER_WORDS]
        raw = recordings / row["file"].replace(".wav", ".raw.wav")
        item = {"file": raw.name, "expected": expected}
        if not raw.exists():
            counts["missing_raw"] += 1
            item.update(status="missing_raw")
        else:
            manifest.append({"file": raw.name, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                             "expected": expected})
            audio = clocked_receiver.read_audio(raw)
            crop, failure = payload_crop(audio, words)
            if crop is None:
                counts["extraction_failed"] += 1
                reasons[failure] += 1
                item.update(status="extraction_failed", reason=failure)
            else:
                result = decode_message(crop, model)
                counts["evaluated"] += 1
                counts["accepted_exact"] += int(result.accepted and result.words == expected)
                counts["accepted_wrong"] += int(result.accepted and result.words != expected)
                counts["rejected"] += int(not result.accepted)
                reasons[result.reason or "accepted"] += 1
                item.update(status="accepted" if result.accepted else "rejected",
                            reason=result.reason, heard=result.words,
                            hypothesis=result.hypothesis,
                            word_margins=result.word_margins,
                            marker_count=result.marker_count,
                            crop_samples=len(crop))
        predictions.append(item)
    attempts = len(rows)
    summary = {
        "status": "retrospective_offline_extracted_payloads",
        "channel": CHANNEL,
        "model_sha256": MODEL_SHA256,
        "model_profile": model["profile"],
        "wire_profile_source": compact_clocked.PROFILE,
        "message_profile": clocked.PROFILE,
        "source_attempt_rows": source_count,
        "selected_attempts": attempts,
        "selection": "all unique labeled channel attempts sorted by filename" if limit is None
                     else f"first {limit} unique labeled channel attempts sorted by filename",
        "counts": dict(counts),
        "reasons": dict(reasons),
        "exact_rate_over_all_selected": counts["accepted_exact"] / attempts if attempts else None,
        "exact_rate_over_extracted": counts["accepted_exact"] / counts["evaluated"]
                                     if counts["evaluated"] else None,
        "limits": [
            "Payloads are cut from recordings of Full packets; no standalone Messages capture exists.",
            "Full packet markers provide the extraction boundary; extraction failures remain in the all-selected denominator.",
            "The model was developed before this Full validation corpus, but this Messages analysis was designed after data collection.",
            "No CRC or integrity verification exists for Messages; accepted-wrong means an acoustic hypothesis was returned for the wrong words.",
            "These figures do not measure always-on ambient detection or new hardware and rooms.",
        ],
    }
    return summary, predictions, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=Path("recordings"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="development smoke only")
    args = parser.parse_args()
    model = load_model(args.model)
    if hasattr(model["model"], "n_jobs"):
        model["model"].n_jobs = 1
    summary, predictions, manifest = evaluate(args.recordings, model, args.limit)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in predictions))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
