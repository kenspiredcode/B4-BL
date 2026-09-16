"""Silent, reproducible evaluation of existing slotted recordings.

Run: python -m b4bl.benchmark --output experiments/offline
No playback or capture imports. Existing-model scores are explicitly retrospective.
Fresh-model training excludes the noisy capture round and a deterministic 20% of
concept sequences. Historical channels are session PROXIES, not true session IDs.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import wavfile

from . import classifier, codec, features, lexicon, phonology, slot_decoder

CHANNELS = ('bt_aukey_slots', 'bt_aukey_slots_r2')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sequence_holdout(concepts):
    key = json.dumps(concepts, separators=(',', ':')).encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16) % 5 == 0


def edit_counts(reference, hypothesis):
    """Levenshtein alignment, with explicit substitutions/deletions/insertions."""
    dp = [[(0, 0, j) for j in range(len(hypothesis) + 1)]]
    for i, ref in enumerate(reference, 1):
        row = [(0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            s, d, ins = dp[i-1][j-1]
            diagonal = (s + (ref != hyp), d, ins)
            s, d, ins = dp[i-1][j]
            deletion = (s, d+1, ins)
            s, d, ins = row[j-1]
            insertion = (s, d, ins+1)
            row.append(min((diagonal, deletion, insertion), key=sum))
        dp.append(row)
    return dp[-1][-1]


def metrics(rows):
    n = len(rows)
    exact = sum(r['expected'] == r['decoded'] for r in rows)
    errors = np.sum([edit_counts(r['expected'], r['decoded']) for r in rows], axis=0) if n else [0]*3
    words = sum(len(r['expected']) for r in rows)
    p = exact / n if n else 0
    z = 1.96
    denom = 1 + z*z/n if n else 1
    mid = (p + z*z/(2*n))/denom if n else 0
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom if n else 0
    return dict(n=n, exact=exact, exact_rate=p, exact_95ci=[mid-half, mid+half],
                reference_words=words, substitutions=int(errors[0]),
                deletions=int(errors[1]), insertions=int(errors[2]),
                word_error_rate=float(sum(errors)/words) if words else None,
                empty_outputs=sum(not r['decoded'] for r in rows),
                accepted=(sum(r['accepted'] for r in rows) if rows and all('accepted' in r for r in rows) else None),
                incorrect_accepted=(sum(r['accepted'] and r['decoded'] != r['expected'] for r in rows)
                                    if rows and all('accepted' in r for r in rows) else None),
                correct_phoneme_count=(sum(r['segments'] == r['expected_segments'] for r in rows)
                                       if rows and all(r['segments'] is not None for r in rows) else None))


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups['all'].append(r)
        groups['multiword' if len(r['expected']) > 1 else 'singleword'].append(r)
        groups[f"length/{len(r['expected'])}"].append(r)
        groups[f"channel/{r['channel']}/length/{len(r['expected'])}"].append(r)
        groups[f"channel/{r['channel']}/{'multi' if len(r['expected']) > 1 else 'single'}"].append(r)
        groups[f"prosody/{r['prosody']}"].append(r)
        groups[f"composition/{'unseen' if r['sequence_holdout'] else 'seen'}"].append(r)
    return {k: metrics(v) for k, v in sorted(groups.items())}


def inventory_audit():
    codes = {c: tuple(phonology.canonical(n) for n in lexicon.concept_to_phonemes(c))
             for c in lexicon.MORPHEMES if c not in lexicon.ALIASES}
    reverse = {v: k for k, v in codes.items()}
    collisions = [[a, b, reverse[sa+sb]] for a, sa in codes.items()
                  for b, sb in codes.items() if sa+sb in reverse]
    return {'words': len(codes), 'lengths': dict(Counter(map(len, codes.values()))),
            'one_vs_two_word_collisions': collisions,
            'note': 'Flat phoneme ambiguity, not identical waveforms: timing/markers may differ.'}


def load_audio(path):
    sr, data = wavfile.read(path)
    if sr != codec.gen.SR or data.ndim != 1:
        raise ValueError(f'{path}: expected mono {codec.gen.SR} Hz, got {data.shape}, {sr}')
    if np.issubdtype(data.dtype, np.integer):
        if data.dtype == np.uint8:
            data = (data.astype(np.float32)-128)/128
        else:
            data = data.astype(np.float32) / max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    data = data.astype(np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f'{path}: non-finite samples')
    return data / max(float(np.max(np.abs(data), initial=0)), 1e-12)


def extract(manifest, output):
    source = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    source = [r for r in source if r['channel'] in CHANNELS]
    if not all(any(r['channel'] == c for r in source) for c in CHANNELS):
        raise ValueError('both historical slotted capture rounds are required')
    rows, matrix = [], []
    for i, r in enumerate(source):
        path = Path(manifest).parent / r['file']
        audio = load_audio(path)
        segments = [(seg, repeat) for word in slot_decoder._segment_words(audio) for seg, repeat in word]
        expected = [phonology.canonical(p) for w in codec.concepts_to_phoneme_words(r['concepts']) for p in w]
        row = dict(r, file=str(path.resolve()), sha256=digest(path), expected=r['concepts'],
                   sequence_holdout=sequence_holdout(r['concepts']),
                   offset=len(matrix), segments=len(segments), expected_segments=len(expected),
                   phonemes=expected, repeats=[bool(rep) for _, rep in segments])
        matrix.extend(features.extract(seg) for seg, _ in segments)
        rows.append(row)
        if (i+1) % 250 == 0:
            print(f'features {i+1}/{len(source)}', flush=True)
    X = np.asarray(matrix, dtype=np.float32)
    np.savez_compressed(output/'segment_features.npz', X=X)
    (output/'recordings.json').write_text(json.dumps(rows, indent=2)+'\n')
    return rows, X


def decode_rows(rows, X, raw_aliases=False, perfect_phones=False):
    candidates = None if perfect_phones else classifier.candidates_from_features(X)
    old_index = slot_decoder._MORPH_INDEX
    if raw_aliases:
        slot_decoder._MORPH_INDEX = [(c, tuple(lexicon.concept_to_phonemes(c)))
                                    for c in lexicon.MORPHEMES if c not in lexicon.ALIASES]
    results = []
    try:
        for r in rows:
            if perfect_phones:
                flat = [[(name, 1.0)] for name in r['phonemes']]
            else:
                flat = []
                for j, rep in enumerate(r['repeats']):
                    flat.append(flat[-1] if rep and flat else candidates[r['offset']+j])
            decoded = slot_decoder.decode_candidates(flat)
            results.append(dict(r, decoded=decoded))
    finally:
        slot_decoder._MORPH_INDEX = old_index
    return results


def train_fresh(rows, X, output, samples):
    from sklearn.ensemble import RandomForestClassifier
    from . import train_classifier as trainer
    import joblib
    train_rows = [r for r in rows if r['channel'] == CHANNELS[0] and not r['sequence_holdout']]
    aligned = [r for r in train_rows if r['segments'] == r['expected_segments']]
    indices, labels = [], []
    for r in aligned:
        indices.extend(range(r['offset'], r['offset']+r['segments']))
        labels.extend(phonology.BY_NAME[p].features for p in r['phonemes'])
    print(f'training on {len(aligned)}/{len(train_rows)} count-matched recordings; {len(indices)} segments', flush=True)
    trainer._rng = np.random.default_rng(20260916)
    codec.gen._rng = np.random.default_rng(20260916)
    synth = trainer.build_dataset(samples_per=samples, slots=True)
    train_X = np.vstack([X[indices], synth[0]])
    labels = np.asarray(labels)
    models = {}
    for j, name in enumerate(('band', 'contour', 'dur', 'cls')):
        model = RandomForestClassifier(n_estimators=80, max_depth=16, n_jobs=-1, random_state=1)
        model.fit(train_X, np.concatenate([labels[:, j], synth[j+1]]))
        models[name] = model
    provenance = dict(train_files=[r['sha256'] for r in train_rows],
                      training_channel=CHANNELS[0], test_channel=CHANNELS[1],
                      excluded_sequence_rule='sha256(JSON compact concepts) first 8 hex modulo 5 == 0',
                      eligible_recordings=len(train_rows), count_matched_recordings=len(aligned),
                      warning='Count matching is a weak label heuristic, not verified alignment.',
                      synthetic_samples_per_phoneme=samples)
    result = dict(models=models, feature_len=X.shape[1], provenance=provenance)
    joblib.dump(result, output/'heldout_classifier.joblib')
    classifier._MODEL = result
    test_rows = [r for r in rows if r['channel'] == CHANNELS[1] or r['sequence_holdout']]
    if set(r['sha256'] for r in test_rows) & set(provenance['train_files']):
        raise ValueError('identical recording content crosses train/test boundary')
    return test_rows, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='recordings/manifest.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=120)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows, X = extract(args.manifest, output)
    import joblib
    existing = classifier.load()
    if existing is None:
        raise RuntimeError('existing model required for retrospective baseline')
    joblib.dump(existing, output/'baseline_classifier.joblib')
    reports = {'manifest_sha256': digest(args.manifest), 'inventory': inventory_audit(),
               'limitations': ['Existing-model results are retrospective, not held out.',
                              'Channel tags proxy capture sessions; historical files lack session IDs.',
                              'Existing audio is already message-trimmed: excludes acquisition failures.',
                              'No real negative/background-only set; false acceptance not measured.',
                              'No independently verified real phoneme boundaries: no acoustic oracle score.']}
    for name, kwargs in [('retrospective_before_alias_fix', {'raw_aliases': True}),
                         ('retrospective_after_alias_fix', {}),
                         ('perfect_phonemes_flat_decoder', {'perfect_phones': True})]:
        predictions = decode_rows(rows, X, **kwargs)
        reports[name] = summarize(predictions)
        (output/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
        print(name, reports[name]['multiword'], flush=True)
    test_rows, provenance = train_fresh(rows, X, output, args.samples)
    predictions = decode_rows(test_rows, X)
    reports['fresh_model_provenance'] = provenance
    reports['heldout_after_alias_fix'] = summarize(predictions)
    (output/'heldout_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    reports['elapsed_seconds'] = time.monotonic()-started
    (output/'summary.json').write_text(json.dumps(reports, indent=2)+'\n')
    print('heldout multiword', reports['heldout_after_alias_fix']['multiword'], flush=True)
    print(f'Wrote {output}/summary.json', flush=True)


if __name__ == '__main__':
    main()
