"""Evaluate real compact-clocked-v2 capture attempts with a frozen model.

This command is read-only with respect to recordings and never opens an audio
device. Missing and collector-rejected attempts remain in the denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import joblib

from . import clocked_receiver as receiver, compact_clocked


def select_attempts(lines, recordings, channel):
    """Choose one attempt per labeled packet, preferring an available raw retry."""
    selected = {}
    source_rows = 0
    recordings = Path(recordings)
    for line in lines:
        row = json.loads(line) if isinstance(line, str) else line
        if (row.get('channel') != channel or
                row.get('encoding') != compact_clocked.PROFILE):
            continue
        source_rows += 1
        key = (tuple(row['concepts']), row['prosody'])
        raw_exists = (recordings/row['file'].replace('.wav', '.raw.wav')).exists()
        previous = selected.get(key)
        previous_raw = (previous is not None and
                        (recordings/previous['file'].replace('.wav', '.raw.wav')).exists())
        if previous is None or raw_exists or not previous_raw:
            selected[key] = row
    return list(selected.values()), source_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--recordings', default='recordings')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--beam-width', type=int, default=10000)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 5:
        parser.error('--top-k must be in [1, 5]')
    if args.beam_width < 1:
        parser.error('--beam-width must be positive')

    recordings = Path(args.recordings)
    attempts_path = recordings/'attempts.jsonl'
    rows, source_rows = select_attempts(
        attempts_path.read_text().splitlines(), recordings, args.channel)
    if not rows:
        parser.error(f'no {compact_clocked.PROFILE} attempts for channel {args.channel!r}')

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model_path = Path(args.model)
    model = joblib.load(model_path)
    if hasattr(model.get('model'), 'n_jobs'):
        model['model'].n_jobs = 1

    counts = Counter(attempts=len(rows))
    reasons = Counter()
    checked = []
    with (output/'predictions.jsonl').open('w') as predictions:
        for index, row in enumerate(rows):
            expected = list(row['concepts'])
            expected_parse = compact_clocked.parse_concepts(expected)
            if not expected_parse.ok:
                raise ValueError(f"attempt {row.get('file')} has an invalid expected packet")
            raw_path = recordings/row['file'].replace('.wav', '.raw.wav')
            result = None
            if raw_path.exists():
                counts['available'] += 1
                audio = receiver.read_audio(raw_path)
                result = compact_clocked.decode(
                    audio, model, acoustic_decoder=receiver.decode,
                    top_k=args.top_k, beam_width=args.beam_width)
                counts['marker_count_ok'] += int(
                    result.acoustic.marker_count == len(expected) + 1)
                counts['top_path_exact'] += int(result.acoustic.hypothesis == expected)
                counts['verified_exact'] += int(
                    result.accepted and result.acoustic.words == expected)
                counts['accepted_wrong'] += int(
                    result.accepted and result.acoustic.words != expected)
                counts['rejected'] += int(not result.accepted)
                counts['selected_by_validation'] += int(result.selected_by_validation)
                reasons[result.parsed.error or 'accepted'] += 1
                checked.append(result.candidate_paths_checked)
            else:
                counts['missing'] += 1
            predictions.write(json.dumps({
                'file': row.get('file'), 'returncode': row.get('returncode'),
                'expected': expected, 'expected_frame': asdict(expected_parse.frame),
                'result': asdict(result) if result else None,
            }) + '\n')
            if index and index % 50 == 0:
                print(f'evaluated {index}/{len(rows)}', flush=True)

    summary = {
        'profile': compact_clocked.PROFILE,
        'channel': args.channel,
        'source_attempt_rows': source_rows,
        'unique_labeled_packets': len(rows),
        'model': str(model_path),
        'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'top_k': args.top_k,
        'beam_width': args.beam_width,
        'counts': dict(counts),
        'verified_delivery_rate': counts['verified_exact'] / len(rows),
        'accepted_wrong_rate': counts['accepted_wrong'] / len(rows),
        'parse_reasons': dict(reasons),
        'candidate_paths_checked': {
            'mean': sum(checked) / len(checked) if checked else None,
            'max': max(checked) if checked else None,
        },
        'limitations': [
            'This channel is a measured environment, not universal room generalization.',
            'CRC32 detects accidental corruption; it is not authentication.',
            'The same packet corpus must not be used for both model tuning and final claims.',
        ],
    }
    (output/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
