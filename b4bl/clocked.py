"""Experimental synchronized transport, separate from historical slots=True.

A distinctive pure-tonal marker brackets every word. Within each word, SHORT
occupies one 300 ms tick; LONG occupies two. Complete gestures use 200/500 ms,
leaving a 100 ms guard. Recognition tests whole word candidates on this grid;
it never segments payload by energy gaps. Word markers reset the clock.

This profile fixes tempo, and supports ordinary morphemes only. Rhythm-defined
Rep words and spelling are rejected explicitly, not silently reinterpreted.
A separate model must be trained on complete clock windows. No device I/O.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
import math

import numpy as np
from scipy import signal

from . import codec, features, generators as gen, lexicon as lex, phonology as ph
from .prosody import NEUTRAL, Prosody

SR = gen.SR
TICK = int(0.30 * SR)
GUARD = int(0.10 * SR)
MARK_GAP = int(0.04 * SR)
PAD = int(0.10 * SR)
# A two-part sweep, outside the lexical contour set; still the tonal generator.
MARKER = gen.tonal(gen.Gesture([(0, 1400), (.35, 3100), (.55, 1800), (1, 2800)],
                              dur=.12, env='even', vib=None))
PREFIX = len(MARKER) + MARK_GAP
PROFILE = 'clocked-v1-fixed-300ms'
FEATURE_PROFILE = 'pitch-envelope-v1'


def window_features(audio):
    """Pitch features plus normalized envelope shape, preserving evidence for a
    missing guard interval when a proposed LONG window actually spans two SHORTs.
    """
    chunks = np.array_split(np.asarray(audio, dtype=float), 12)
    rms = np.asarray([np.sqrt(np.mean(c*c)) if len(c) else 0 for c in chunks])
    rms /= max(float(rms.max()), 1e-9)
    return np.concatenate([features.extract(audio), rms]).astype(np.float32)


def vocabulary():
    return {c: tuple(ph.canonical(n) for n in lex.concept_to_phonemes(c))
            for c in lex.MORPHEMES if c not in lex.ALIASES
            and not isinstance(lex.morpheme_body(c), lex.Rep)}


def ticks(name):
    return 2 if ph.BY_NAME[name].dur == ph.Dur.LONG else 1


def body(name, prosody):
    duration = (ticks(name)*TICK-GUARD)/SR
    return ph.BY_NAME[name].render(prosody, duration_sec=duration)


def encode(concepts, prosody=NEUTRAL, repetition=1, return_spans=False):
    """Encode bracketing markers + fixed clock. repetition=3 enables majority FEC.
    Repetition is a configured profile, not negotiated by this prototype.
    return_spans exports transmitter timing for future independent alignment QA."""
    if repetition not in (1, 3):
        raise ValueError('repetition must be 1 or 3')
    if not concepts:
        raise ValueError('empty utterance')
    vocab = vocabulary()
    unsupported = [c for c in concepts if c not in vocab]
    if unsupported:
        raise ValueError(f'clocked-v1 does not support these words: {unsupported}')
    clips = [np.zeros(PAD, np.float32)]
    spans, cursor = [], PAD
    for word_index, concept in enumerate(concepts):
        for repeat_index in range(repetition):
            clips.extend([MARKER, np.zeros(MARK_GAP, np.float32)])
            cursor += PREFIX
            for name in vocab[concept]:
                cell = np.zeros(ticks(name)*TICK, np.float32)
                sig = body(name, prosody)
                cell[:len(sig)] = sig
                clips.append(cell)
                spans.append(dict(word_index=word_index, repetition_index=repeat_index,
                                  phoneme=name, start=cursor, end=cursor+len(sig),
                                  slot_end=cursor+len(cell)))
                cursor += len(cell)
    clips.extend([MARKER, np.zeros(PAD, np.float32)])
    audio = np.concatenate(clips)
    return (audio, spans) if return_spans else audio


def marker_positions(audio, threshold=.45):
    """Matched-filter peaks with local energy normalization. Markers delimit
    words; they do not depend on finding silence. Delayed echo peaks are suppressed.
    Threshold is fixed by the experimental profile, not tuned on real holdout data."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError('expected finite mono audio')
    m = MARKER.astype(np.float64)
    if len(a) < len(m):
        return []
    corr = signal.correlate(a, m, mode='valid', method='fft')
    sums = np.concatenate([[0.0], np.cumsum(a*a)])
    energy = np.maximum(sums[len(m):]-sums[:-len(m)], 0)
    scores = np.abs(corr) / np.sqrt(np.maximum(energy*np.dot(m, m), 1e-20))
    # Include first/last possible positions; scipy's peak finder excludes edges.
    peaks, _ = signal.find_peaks(np.pad(scores, (1, 1)), height=threshold,
                                distance=int(.25*SR))
    return [int(p-1) for p in peaks]


@dataclass
class DecodeResult:
    words: list[str] = field(default_factory=list)
    accepted: bool = False
    reason: str = ''
    marker_count: int = 0
    word_margins: list[float] = field(default_factory=list)
    hypothesis: list[str] = field(default_factory=list)
    # This acceptance means acoustic framing passed, NOT checksum verification.


def decode(audio, model, repetition=1, min_margin=.5):
    if repetition not in (1, 3):
        raise ValueError('repetition must be 1 or 3')
    if model.get('profile') != PROFILE or model.get('features') != FEATURE_PROFILE:
        raise ValueError('requires a clocked-v1 model, not the historical segment model')
    a = np.asarray(audio, dtype=np.float32)
    starts = marker_positions(a)
    result = DecodeResult(marker_count=len(starts))
    if len(starts) < 2:
        result.reason = 'no complete marker pair'
        return result
    vocab = vocabulary()
    # Build exactly bounded windows for the timing paths of candidate WORDS.
    # Never use a whole-phoneme classifier on arbitrary sliding fragments.
    windows, keys, regions = [], {}, []
    for wi, (left, right) in enumerate(zip(starts, starts[1:])):
        begin = left + PREFIX
        span = right - begin
        count = round(span/TICK)
        if count < 1 or abs(span-count*TICK) > int(.025*SR):
            result.reason = 'word marker is off the symbol grid'
            return result
        candidates = {c: seq for c, seq in vocab.items() if sum(map(ticks, seq)) == count}
        if not candidates:
            result.reason = 'no word matches clock length'
            return result
        for seq in candidates.values():
            pos = 0
            for name in seq:
                width = ticks(name)
                key = (wi, pos, width)
                if key not in keys:
                    lo = begin + pos*TICK
                    hi = lo + width*TICK-GUARD
                    if hi > len(a):
                        result.reason = 'truncated phoneme'
                        return result
                    keys[key] = len(windows)
                    windows.append(window_features(a[lo:hi]))
                pos += width
        regions.append(candidates)
    X = np.asarray(windows)
    probs = model['model'].predict_proba(X)
    labels = {str(c): i for i, c in enumerate(model['model'].classes_)}
    raw_words = []
    for wi, candidates in enumerate(regions):
        scored = []
        for concept, seq in candidates.items():
            pos, score = 0, 0.0
            for name in seq:
                index = keys[(wi, pos, ticks(name))]
                score += math.log(max(float(probs[index, labels[name]]), 1e-9))
                pos += ticks(name)
            # All candidates consume the same clock interval. Full-phoneme RF
            # probabilities are not a calibrated sequence likelihood; margin is
            # a heuristic, measured separately from CRC-verified acceptance.
            scored.append((score, concept))
        scored.sort(reverse=True)
        margin = scored[0][0]-scored[1][0] if len(scored) > 1 else 100.0
        result.word_margins.append(float(margin))
        result.hypothesis.append(scored[0][1])
        raw_words.append(scored[0][1] if margin >= min_margin else None)
    if len(raw_words) % repetition:
        result.reason = 'incomplete repetition block'
        return result
    if repetition == 3:
        result.hypothesis = [Counter(result.hypothesis[i:i+3]).most_common(1)[0][0]
                             for i in range(0, len(result.hypothesis), 3)]
    for i in range(0, len(raw_words), repetition):
        votes = Counter(w for w in raw_words[i:i+repetition] if w is not None)
        if not votes or votes.most_common(1)[0][1] < repetition//2+1:
            result.reason = 'ambiguous word or no repetition majority'
            return result
        result.words.append(votes.most_common(1)[0][0])
    result.accepted = True
    return result


def train(samples_per=160, seed=20260916):
    """Synthetic training for clock-window recognition only. Explicit profile tag
    prevents accidentally using legacy segment features at inference."""
    from sklearn.ensemble import RandomForestClassifier
    from . import train_classifier as trainer
    rng = np.random.default_rng(seed)
    old_rng, old_gen = trainer._rng, gen._rng
    trainer._rng = np.random.default_rng(seed)
    gen._rng = np.random.default_rng(seed)
    names = sorted({n for seq in vocabulary().values() for n in seq})
    X, y = [], []
    try:
        for name in names:
            for i in range(samples_per):
                pr = Prosody(float(rng.uniform(.15, 1)), float(rng.uniform(0, 1)))
                sig = body(name, pr)
                # Augment a complete stream before extracting its window, so
                # preceding-symbol reverb, background noise, and timing error
                # affect training in the same order as they affect reception.
                def observed(target):
                    if i % 5 < 2:
                        return target
                    context_name = str(rng.choice(names))
                    preceding = body(context_name, pr)
                    pad = int(.2*SR)
                    offset = len(preceding)+GUARD+pad
                    stream = np.concatenate([np.zeros(pad), preceding,
                                             np.zeros(GUARD), target, np.zeros(pad)])
                    echoed = stream.copy()
                    for _ in range(int(rng.integers(1, 5))):
                        delay = int(rng.uniform(.008, .19)*SR)
                        echoed[delay:] += float(rng.uniform(.03, .45))*stream[:-delay]
                    # Noise is room-level, not scaled to each phoneme's energy.
                    echoed += rng.normal(0, float(rng.uniform(.003, .10)), len(stream))
                    jitter = int(rng.uniform(-.003, .003)*SR)
                    return echoed[offset+jitter:offset+jitter+len(target)].astype(np.float32)
                X.append(window_features(observed(sig))); y.append(name)
                if i % 3 == 0:
                    # Teach explicit rejection of the wrong duration path. A long
                    # candidate can otherwise absorb two valid short symbols.
                    shorts = [n for n in names if ticks(n) == 1]
                    short1, short2 = rng.choice(shorts, size=2)
                    false_long = np.concatenate([body(short1, pr), np.zeros(GUARD), body(short2, pr)])
                    X.append(window_features(observed(false_long))); y.append('__invalid__')
    finally:
        trainer._rng, gen._rng = old_rng, old_gen
    model = RandomForestClassifier(n_estimators=100, max_depth=18, random_state=seed, n_jobs=-1)
    model.fit(np.asarray(X), y)
    return {'profile': PROFILE, 'features': FEATURE_PROFILE, 'model': model, 'seed': seed, 'samples_per': samples_per,
            'training': 'synthetic complete clock windows; no real microphone validation'}
