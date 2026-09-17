"""Offline training/evaluation from a frozen attempt-level split manifest.

Run with --split PATH --output DIR. No capture imports or device access.
Evaluation retains missing recordings and framing failures in its denominator.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from . import clocked, clocked_receiver as receiver, prosody


def select_rows(rows, include_room3_training=False, room3_train_limit=None):
    base = [r for r in rows if r['partition'] == 'train']
    room = [r for r in rows if r['partition'] == 'room3_test'
            and not r['composition_holdout']]
    if room3_train_limit is not None:
        room = room[:room3_train_limit]
    training = base + (room if include_room3_training else [])
    evaluation = [r for r in rows if r['partition'] == 'composition_test' or
                  (r['partition'] == 'room3_test' and
                   (not include_room3_training or r['composition_holdout']))]
    return training, evaluation


def augment_channel(audio, rng):
    """Distort a complete training stream, retaining its sample clock.

    Early echoes and frequency response alter contours/envelopes without changing
    their labels. Noise is scaled once per utterance, not per phoneme. Do not call
    this on evaluation recordings or fit its parameters to held-out waveforms.
    """
    audio = np.asarray(audio, dtype=float)
    output = audio.copy()
    for _ in range(int(rng.integers(2, 7))):
        delay = int(rng.uniform(.004, .15)*clocked.SR)
        if delay < len(output):
            output[delay:] += rng.uniform(-.3, .45)*audio[:-delay]
    # Room/speaker coloration via a smooth FIR response, including spectral tilt.
    freq = np.array([0, 200, 500, 1000, 2000, 4000, 8000, clocked.SR/2])
    gains = 10 ** (rng.uniform(-8, 8, len(freq))/20)
    fir = signal.firwin2(65, freq, gains, fs=clocked.SR)
    output = signal.convolve(output, fir, mode='same')
    rms = np.sqrt(np.mean(output*output))
    noise = rng.normal(0, rms / 10**(rng.uniform(16, 32)/20), len(output))
    return (output + noise).astype(np.float32)


def training_windows(audio, words, min_periodicity=0.0):
    """Labels are allowed ONLY during training alignment, never inference.

    Check count AND every word's expected duration before extracting labels.
    This checks framing consistency, not an independently measured acoustic
    ground truth for internal boundaries.
    """
    audio = receiver.preprocess(audio)
    starts = receiver.marker_positions(audio)
    if len(starts) != len(words)+1:
        return None
    vocab = clocked.vocabulary()
    expected = np.array([clocked.PREFIX+sum(map(clocked.ticks, vocab[w]))*clocked.TICK
                         for w in words])
    scale = float(np.median(np.diff(starts)/expected))
    if not .96 < scale < 1.04 or np.max(abs(np.diff(starts)-scale*expected)) > .025*clocked.SR:
        return None
    X, y = [], []
    for index, word in enumerate(words):
        seq = vocab[word]
        begin = starts[index] + round(clocked.PREFIX*scale)
        pos = 0
        for j, name in enumerate(seq):
            width = clocked.ticks(name)
            lo = begin+round(pos*clocked.TICK*scale)
            hi = lo+round((width*clocked.TICK-clocked.GUARD)*scale)
            if hi > len(audio):
                return None
            X.append(clocked.window_features(audio[lo:hi], min_periodicity)); y.append(name)
            if width == 1 and j+1 < len(seq) and clocked.ticks(seq[j+1]) == 1:
                end = lo+round((2*clocked.TICK-clocked.GUARD)*scale)
                X.append(clocked.window_features(audio[lo:end], min_periodicity)); y.append('__invalid__')
            pos += width
    return X, y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--recordings', default='recordings')
    ap.add_argument('--baseline', default='experiments/clocked-balanced-final-20260916/clocked_classifier.joblib')
    ap.add_argument('--augment', type=int, default=0, help='additional distorted copies per training recording')
    ap.add_argument('--trained-only', action='store_true', help='evaluate only new model; prior ablations remain in earlier runs')
    ap.add_argument('--min-periodicity', type=float, default=0.0,
                    help='experimental voiced-frame confidence gate; requires retraining')
    ap.add_argument('--include-room3-training', action='store_true',
                    help='train on room3 non-holdout compositions; evaluates represented-room interpolation')
    ap.add_argument('--room3-train-limit', type=int,
                    help='deterministic prefix of non-holdout room3 attempts to train on')
    ap.add_argument('--estimator', choices=('random_forest', 'extra_trees'), default='random_forest')
    args = ap.parse_args()
    if args.augment < 0:
        ap.error('--augment must be nonnegative')
    if not 0 <= args.min_periodicity < 1:
        ap.error('--min-periodicity must be in [0, 1)')
    if args.room3_train_limit is not None and (not args.include_room3_training or args.room3_train_limit < 1):
        ap.error('--room3-train-limit requires --include-room3-training and must be positive')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    rows = json.loads(Path(args.split).read_text())
    training, evaluation = select_rows(rows, args.include_room3_training,
                                       args.room3_train_limit)
    # Composition holdouts cannot appear in training in any room or prosody.
    train_keys = {tuple(r['concepts']) for r in training}
    evaluation_keys = {tuple(r['concepts']) for r in evaluation}
    assert not train_keys.intersection(evaluation_keys)
    X, y, used = [], [], []
    stats = Counter()
    augmentation_rng = np.random.default_rng(170927)
    for i, row in enumerate(training):
        path = Path(args.recordings)/row['raw_file']
        if not path.exists():
            stats['missing'] += 1; continue
        audio = receiver.read_audio(path)
        windows = training_windows(audio, row['concepts'], args.min_periodicity)
        if windows is None:
            stats['alignment_rejected'] += 1; continue
        xx, yy = windows; X.extend(xx); y.extend(yy); used.append(row['raw_file'])
        for _ in range(args.augment):
            augmented = training_windows(augment_channel(audio, augmentation_rng), row['concepts'], args.min_periodicity)
            if augmented is None:
                stats['augmentation_alignment_rejected'] += 1
                continue
            xx, yy = augmented; X.extend(xx); y.extend(yy)
            stats['augmented_recordings'] += 1
        if i % 100 == 0:
            print(f'training alignment {i}/{len(training)}', flush=True)
    real_counts = dict(Counter(y))
    # Supply rare/absent classes and wrong-duration negatives from known synthesis.
    rng = np.random.default_rng(170926)
    names = sorted({n for seq in clocked.vocabulary().values() for n in seq})
    shorts = [n for n in names if clocked.ticks(n) == 1]
    for name in names:
        for i in range(60):
            pr = prosody.Prosody(float(rng.uniform(.15, 1)), float(rng.uniform(0, 1)))
            a = clocked.body(name, pr)
            a = receiver.preprocess(a+rng.normal(0, .005, len(a)))
            X.append(clocked.window_features(a, args.min_periodicity)); y.append(name)
            if i % 3 == 0:
                aa, bb = rng.choice(shorts, size=2)
                bad = np.concatenate([clocked.body(aa, pr), np.zeros(clocked.GUARD), clocked.body(bb, pr)])
                X.append(clocked.window_features(receiver.preprocess(bad), args.min_periodicity)); y.append('__invalid__')
    estimator = RandomForestClassifier if args.estimator == 'random_forest' else ExtraTreesClassifier
    model = estimator(n_estimators=200, max_depth=22, n_jobs=-1, random_state=170926)
    model.fit(np.asarray(X), y); model.n_jobs = 1
    bundle = dict(profile=clocked.PROFILE, features=(
                  f'{clocked.CONFIDENT_FEATURE_PREFIX}{args.min_periodicity:g}'
                  if args.min_periodicity else clocked.FEATURE_PROFILE),
                  min_periodicity=args.min_periodicity,
                  frontend=receiver.PROFILE, model=model, seed=170926,
                  split_sha256=hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
                  augmentation_copies=args.augment,
                  estimator=args.estimator,
                  training='real train partitions plus 60 synthetic examples per phoneme')
    joblib.dump(bundle, out/'real_classifier.joblib')
    baseline = joblib.load(args.baseline); baseline['model'].n_jobs = 1
    report = dict(evaluation_note='Repeated runs on this frozen split are development comparisons, not fresh unseen tests.',
                  feature_profile=bundle['features'], augmentation_copies=args.augment,
                  estimator=args.estimator,
                  include_room3_training=args.include_room3_training,
                  room3_train_limit=args.room3_train_limit,
                  training=dict(used=len(used), attempted=len(training), exclusions=dict(stats),
                               real_label_counts=real_counts, total_windows=len(y)),
                  split_sha256=bundle['split_sha256'], results={})
    (out/'training.json').write_text(json.dumps(report, indent=2))
    metrics = defaultdict(Counter)
    with (out/'predictions.jsonl').open('w') as log:
        for i, row in enumerate(evaluation):
            path = Path(args.recordings)/row['raw_file']
            audio = receiver.read_audio(path) if path.exists() else None
            for variant in (('real_trained',) if args.trained_only else ('original', 'frontend_only', 'real_trained')):
                result = None
                if audio is not None:
                    result = (clocked.decode(audio, baseline) if variant == 'original' else
                              receiver.decode(audio, baseline if variant == 'frontend_only' else bundle))
                for length in ('all', 'multi' if len(row['concepts']) > 1 else 'single'):
                    # Report each channel separately; room3 contains an untouched
                    # full run AND earlier development pilots, never pool silently.
                    key = f"{row['partition']}|{row['channel']}|{length}|{variant}"
                    s = metrics[key]; s['attempts'] += 1
                    if result is None:
                        s['missing'] += 1; continue
                    s['markers_count_ok'] += int(result.marker_count == len(row['concepts'])+1)
                    s['exact'] += int(result.accepted and result.words == row['concepts'])
                    s['best_exact'] += int(result.hypothesis == row['concepts'])
                    s['accepted_wrong'] += int(result.accepted and result.words != row['concepts'])
                    s['rejected'] += int(not result.accepted)
                log.write(json.dumps(dict(file=row['raw_file'], variant=variant,
                                          expected=row['concepts'], result=asdict(result) if result else None))+'\n')
            if i % 50 == 0:
                print(f'evaluated {i}/{len(evaluation)}', flush=True)
    report['results'] = dict(metrics)
    (out/'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
