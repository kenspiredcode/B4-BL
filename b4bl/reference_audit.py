"""Reference-assisted boundary diagnostic for historical slotted recordings.

Reconstruct the OLD transmitted waveform (including its contour truncation), align
by waveform correlation, and measure classification at source-derived boundaries.
Labels and timing come from the transmitter, not the energy segmenter. Alignment
is still estimated; this is NOT an independently verified acoustic oracle.
No playback/capture. The current decoder is not given labels in ordinary operation.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy import signal
import joblib
from . import benchmark, classifier, codec, features, generators as gen, lexicon, phonology, prosody


def historical_reference(concepts, pr):
    clips, spans, cursor = [], [], 0
    previous_rng = gen._rng
    gen._rng = np.random.default_rng(11)  # historical subprocess initialization
    try:
        for wi, concept in enumerate(concepts):
            if isinstance(lexicon.morpheme_body(concept), lexicon.Rep):
                raise ValueError('rhythm-defined word has no historical slot grid')
            names = lexicon.concept_to_phonemes(concept)
            for j, name in enumerate(names):
                p = phonology.BY_NAME[name]
                raw = p.render(pr)
                width = int(gen.SR*codec._slot_width(p.dur))
                room = width-int(gen.SR*codec.SLOT_PHONE_GAP)
                body = raw[:room].copy()
                if len(raw) > room:
                    fade = min(len(body), int(.02*gen.SR))
                    body[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
                cell = np.zeros(width, np.float32); cell[:len(body)] = body
                spans.append((cursor, cursor+len(body), wi, p.name))
                clips.append(cell); cursor += len(cell)
                if j+1 < len(names) and names[j+1] == name:
                    twin = np.concatenate([codec._silence(.1), codec._render_geminate(), codec._silence(.1)])
                    clips.append(twin); cursor += len(twin)
            if wi+1 < len(concepts):
                gap = codec._silence(codec.SLOT_WORD_GAP)
                clips.append(gap); cursor += len(gap)
    finally:
        gen._rng = previous_rng
    return np.concatenate(clips), spans


def align(audio, reference):
    corr = signal.correlate(audio, reference, mode='full', method='fft')
    lags = signal.correlation_lags(len(audio), len(reference))
    allowed = np.where(np.abs(lags) <= int(.15*gen.SR))[0]
    best = allowed[np.argmax(np.abs(corr[allowed]))]
    lag = int(lags[best])
    a0, r0 = max(lag, 0), max(-lag, 0)
    n = min(len(audio)-a0, len(reference)-r0)
    a, r = audio[a0:a0+n], reference[r0:r0+n]
    coherence = abs(float(np.dot(a, r)))/max(float(np.linalg.norm(a)*np.linalg.norm(r)), 1e-12)
    return lag, coherence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, help='existing benchmark output directory')
    parser.add_argument('--per-stratum', type=int, default=2)
    args = parser.parse_args()
    root = Path(args.cache)
    rows = json.loads((root/'recordings.json').read_text())
    # Only unseen channel/sequence rows, selected deterministically before decode.
    selected = defaultdict(list)
    for r in rows:
        if r['channel'] != benchmark.CHANNELS[1] and not r['sequence_holdout']:
            continue
        if len(r['expected']) < 2:
            continue
        if any(isinstance(lexicon.morpheme_body(c), lexicon.Rep) for c in r['expected']):
            continue
        selected[(r['channel'], len(r['expected']), r['prosody'])].append(r)
    subset = [r for group in selected.values() for r in sorted(group, key=lambda x:x['sha256'])[:args.per_stratum]]
    model = joblib.load(root/'heldout_classifier.joblib')
    classifier._MODEL = model
    presets = dict(neutral=prosody.NEUTRAL, uncertain=prosody.UNCERTAIN,
                   urgent=prosody.URGENT, calm=prosody.CALM)
    matrix, labels, results = [], [], []
    for r in subset:
        audio = benchmark.load_audio(r['file'])
        reference, spans = historical_reference(r['expected'], presets[r['prosody']])
        lag, coherence = align(audio, reference)
        intervals = []
        for lo, hi, wi, name in spans:
            start, end = max(0, lo+lag), min(len(audio), hi+lag)
            if end-start < 64:
                intervals = []; break
            intervals.append(dict(start=start, end=end, word=wi, phoneme=name))
        result = dict(file=r['file'], expected=r['expected'], channel=r['channel'],
                      lag_samples=lag, correlation=coherence, spans=intervals,
                      offset=len(matrix), sequence_holdout=r['sequence_holdout'],
                      prosody=r['prosody'], segments=len(intervals), expected_segments=len(spans))
        for iv in intervals:
            matrix.append(features.extract(audio[iv['start']:iv['end']]))
            labels.append(phonology.BY_NAME[iv['phoneme']].features)
        results.append(result)
    X = np.asarray(matrix)
    cands = classifier.candidates_from_features(X)
    for r in results:
        words = defaultdict(list)
        for j, iv in enumerate(r['spans']):
            words[iv['word']].append(cands[r['offset']+j])
        r['decoded'] = codec.candidate_words_to_concepts(list(words.values())) if words else []
        r['joint_phone_correct'] = sum(cands[r['offset']+j][0][0] == iv['phoneme'] for j, iv in enumerate(r['spans']))
    axes = {}
    for j, name in enumerate(('band','contour','dur','cls')):
        axes[name] = float(np.mean(model['models'][name].predict(X) == np.asarray(labels)[:, j]))
    report = dict(status='reference-assisted diagnostic, NOT verified oracle boundaries',
                  selection='up to N per channel/length/prosody; held-out rows only; Rep words excluded',
                  n=len(results), per_axis_accuracy=axes,
                  joint_phoneme_accuracy=sum(r['joint_phone_correct'] for r in results)/len(X),
                  all_alignments=benchmark.summarize(results),
                  correlation_quantiles=np.quantile([r['correlation'] for r in results], [0,.25,.5,.75,1]).tolist(),
                  limitations=['Cross-correlation chooses strongest path; echo can shift boundaries.',
                               'Whole-body windows differ from energy-trimmed classifier training.',
                               'No sample-clock drift fit; source reconstruction assumes historical current palette.',
                               'All selected alignments reported, including weak correlations; no cherry-picking.'])
    (root/'reference_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    (root/'reference_boundaries.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
