"""Silent synthetic stress test. Does not claim speaker/microphone accuracy.
Run: python -m b4bl.clocked_benchmark --output experiments/clocked --messages 80
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy import signal
from . import benchmark, clocked, generators, prosody


def channel(audio, rng, condition):
    x = np.asarray(audio, dtype=np.float32)
    if condition == 'echo_noise_15db':
        y = np.pad(x, (0, int(.12*clocked.SR)))
        for delay, gain in ((.027, .3), (.071, .18), (.113, .09)):
            d = int(delay*clocked.SR)
            y[d:d+len(x)] += gain*x
        x = y
        noise = np.sqrt(np.mean(x*x)/10**1.5)
        x = x + rng.normal(0, noise, len(x))
    elif condition == 'echo_noise_5db':
        y = np.pad(x, (0, int(.18*clocked.SR)))
        for delay, gain in ((.045, .5), (.09, .3), (.16, .15)):
            d = int(delay*clocked.SR)
            y[d:d+len(x)] += gain*x
        x = y + rng.normal(0, np.sqrt(np.mean(y*y)/10**.5), len(y))
    elif condition == 'clock_skew_0.1pct':
        x = signal.resample_poly(x, 1001, 1000)
    elif condition != 'clean':
        raise ValueError(condition)
    # Unannounced leading offset/gain: decoder must find its own markers.
    return np.pad(x*float(rng.uniform(.2, .9)), (int(rng.uniform(.05,.4)*clocked.SR), int(.2*clocked.SR))).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--messages', type=int, default=80)
    parser.add_argument('--samples', type=int, default=240)
    parser.add_argument('--model', help='reuse a frozen clocked model, without retraining')
    parser.add_argument('--seed', type=int, default=731203)
    parser.add_argument('--repetition', type=int, choices=(1, 3), default=1)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    import joblib
    started = time.monotonic()
    print('loading frozen model' if args.model else 'training clock-window classifier (silent)', flush=True)
    model = joblib.load(args.model) if args.model else clocked.train(samples_per=args.samples)
    joblib.dump(model, out/'clocked_classifier.joblib')
    rng = np.random.default_rng(args.seed)
    names = list(clocked.vocabulary())
    cases = [(['SELF', 'SELF'], 'neutral'), (['GIVE', 'SELF'], 'neutral'),
             (['DENY', 'IT'], 'uncertain'), (['SEARCH', 'IT'], 'uncertain')]
    presets = dict(neutral=prosody.NEUTRAL, uncertain=prosody.UNCERTAIN,
                   urgent=prosody.URGENT, calm=prosody.CALM)
    for i in range(args.messages):
        cases.append((rng.choice(names, size=2+i%4, replace=True).tolist(), list(presets)[(i//4)%4]))
    rows = []
    for condition in ('clean', 'echo_noise_15db', 'echo_noise_5db', 'clock_skew_0.1pct'):
        for words, preset in cases:
            audio = clocked.encode(words, presets[preset], repetition=args.repetition)
            result = clocked.decode(channel(audio, rng, condition), model, repetition=args.repetition)
            rows.append(dict(expected=words, decoded=result.words if result.accepted else [],
                             accepted=result.accepted, hypothesis=result.hypothesis,
                             word_margins=result.word_margins, reason=result.reason, channel=condition,
                             prosody=preset, segments=None, expected_segments=None, sequence_holdout=True,
                             duration_seconds=len(audio)/clocked.SR))
        subset = [r for r in rows if r['channel'] == condition]
        print(condition, benchmark.metrics(subset), flush=True)
    # Negative controls do not substitute for recordings of real music/motors.
    negatives = []
    for i in range(30):
        x = rng.normal(0, .15, 3*clocked.SR).astype(np.float32) if i%2 else np.zeros(3*clocked.SR, np.float32)
        negatives.append(clocked.decode(x, model, repetition=args.repetition).accepted)
    report = dict(profile=clocked.PROFILE, evaluation='synthetic only; no playback or capture',
                  seed=args.seed, training_seed=model['seed'], repetition=args.repetition,
                  design='2-5 words crossed with all four prosodies; plus four boundary adversaries',
                  model_sha256=benchmark.digest(out/'clocked_classifier.joblib'),
                  summary=benchmark.summarize(rows),
                  hypothesis_exact_by_channel={c: sum(r['hypothesis'] == r['expected'] for r in rows if r['channel'] == c)
                      for c in sorted({r['channel'] for r in rows})},
                  negative_controls=dict(n=len(negatives), accepted=sum(negatives)),
                  mean_duration_seconds=float(np.mean([r['duration_seconds'] for r in rows])),
                  elapsed_seconds=time.monotonic()-started,
                  limitations=['No music, motor, AGC, codec or microphone validation.',
                               'Fixed tempo; rhythm morphemes and spelling excluded.',
                               'Acoustic acceptance is not CRC-verified delivery.',
                               'Nominal input SNR, not calibrated room SNR.'])
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


if __name__ == '__main__':
    main()
