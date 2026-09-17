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
from sklearn.ensemble import RandomForestClassifier
from . import clocked, clocked_receiver as receiver, prosody


def training_windows(audio, words):
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
            X.append(clocked.window_features(audio[lo:hi])); y.append(name)
            if width == 1 and j+1 < len(seq) and clocked.ticks(seq[j+1]) == 1:
                end = lo+round((2*clocked.TICK-clocked.GUARD)*scale)
                X.append(clocked.window_features(audio[lo:end])); y.append('__invalid__')
            pos += width
    return X, y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--recordings', default='recordings')
    ap.add_argument('--baseline', default='experiments/clocked-balanced-final-20260916/clocked_classifier.joblib')
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    rows = json.loads(Path(args.split).read_text())
    training = [r for r in rows if r['partition'] == 'train']
    # Composition holdouts cannot appear in training in any room or prosody.
    train_keys = {tuple(r['concepts']) for r in training}
    assert not any(tuple(r['concepts']) in train_keys for r in rows
                   if r['partition'] == 'composition_test')
    X, y, used = [], [], []
    stats = Counter()
    for i, row in enumerate(training):
        path = Path(args.recordings)/row['raw_file']
        if not path.exists():
            stats['missing'] += 1; continue
        windows = training_windows(receiver.read_audio(path), row['concepts'])
        if windows is None:
            stats['alignment_rejected'] += 1; continue
        xx, yy = windows; X.extend(xx); y.extend(yy); used.append(row['raw_file'])
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
            X.append(clocked.window_features(a)); y.append(name)
            if i % 3 == 0:
                aa, bb = rng.choice(shorts, size=2)
                bad = np.concatenate([clocked.body(aa, pr), np.zeros(clocked.GUARD), clocked.body(bb, pr)])
                X.append(clocked.window_features(receiver.preprocess(bad))); y.append('__invalid__')
    model = RandomForestClassifier(n_estimators=200, max_depth=22, n_jobs=-1, random_state=170926)
    model.fit(np.asarray(X), y); model.n_jobs = 1
    bundle = dict(profile=clocked.PROFILE, features=clocked.FEATURE_PROFILE,
                  frontend=receiver.PROFILE, model=model, seed=170926,
                  split_sha256=hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
                  training='real train partitions plus 60 synthetic examples per phoneme')
    joblib.dump(bundle, out/'real_classifier.joblib')
    baseline = joblib.load(args.baseline); baseline['model'].n_jobs = 1
    report = dict(training=dict(used=len(used), attempted=len(training), exclusions=dict(stats),
                               real_label_counts=real_counts, total_windows=len(y)),
                  split_sha256=bundle['split_sha256'], results={})
    (out/'training.json').write_text(json.dumps(report, indent=2))
    metrics = defaultdict(Counter)
    evaluation = [r for r in rows if r['partition'] in ('composition_test', 'room3_test')]
    with (out/'predictions.jsonl').open('w') as log:
        for i, row in enumerate(evaluation):
            path = Path(args.recordings)/row['raw_file']
            audio = receiver.read_audio(path) if path.exists() else None
            for variant in ('original', 'frontend_only', 'real_trained'):
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
