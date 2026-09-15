"""B4-BL — grid (symbol-clock) decoder: read phonemes by POSITION, not by silence.

STATUS: EXPERIMENT, NEGATIVE RESULT (2026-09-15). This underperforms the vocab
decoder on hard audio (round-2: 1/20 exact vs slot_decoder.decode_vocab's 7/20) and
is NOT the default. Kept as a documented experiment. Why it failed: the segment
classifier was trained on cleanly-bounded WHOLE phonemes, so scoring it over
arbitrary grid-aligned windows (partial phonemes, wrong widths) yields noisy
probabilities and the vocab DP picks wrong morphemes; it is also slow (~8s/recording
even after batching the RF predicts). Making this work needs a classifier trained to
score FRAMES, not clean segments (a small frame model) — see the decoder-approaches
ledger in the vault. Prefer cheaper root-cause fixes first (wider encoder gap,
higher playback SNR).

The slot decoder finds phonemes as silence-delimited voiced runs. That works on a
clean channel but breaks on a reverberant + noisy one: reverb tails create phantom
runs and background noise masks the gaps, so only ~42% of noisy recordings get the
phoneme count right (see docs/symbol-clock-reset.md).

This decoder cashes in the actual promise of the symbol clock: every phoneme body
has a known duration CLASS (SHORT vs LONG, nominal ~0.16s vs ~0.5s), so we never
look for silence. We slide the segment classifier over the utterance, then align a
phoneme sequence to the frames with a Viterbi DP where each phoneme occupies a
duration-consistent run of frames. The vocabulary constrains labels to valid
morphemes.

Tempo is NOT fixed: prosody scales every phoneme's duration by the SAME factor
within an utterance (~0.5x when urgent, ~1.8x when hesitant — see prosody.py), so
the grid is uniformly stretched but its SHORT:LONG ratio is preserved. We therefore
let each phoneme consume a RANGE of frames around its nominal duration and let the
DP pick the best span. This absorbs the global tempo and per-phoneme jitter without
having to estimate the tempo up front, while still forbidding a SHORT phoneme from
occupying a LONG-sized span (the ratio, not the absolute length, is what's lexical).

Pipeline:
  1. frame scores: slide the classifier over SHORT- and LONG-width windows, giving
     for each frame the best phoneme (and prob) at each duration class.
  2. build a phoneme lattice: at each frame, the top-k (name, dur, score) reachable.
  3. vocab DP over the lattice: find the morpheme sequence whose phonemes align to
     consecutive duration-consistent spans with the best total score.

No energy-gap segmentation, no 280MB frame model — just the 24MB segment
classifier slid over the timeline.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np

from . import generators as gen
from . import classifier as _clf
from . import phonology as ph
from . import lexicon as lex
from . import codec

SR = gen.SR

# frame hop for the sliding classifier. Coarse enough to keep the number of
# classifier invocations tractable (each is a full feature extraction + 4 RF
# predicts), fine enough to place phoneme boundaries within a gap. 0.08s ≈ half a
# SHORT phoneme; the flexible spans still cover the tempo range.
HOP = 0.08                      # seconds between frame starts
# the two body durations the encoder actually renders (phonology.DUR_SEC)
SHORT_SEC = ph.DUR_SEC[ph.Dur.SHORT]
LONG_SEC = ph.DUR_SEC[ph.Dur.LONG]
# prosody scales duration ~0.5x (urgent) .. ~1.8x (hesitant); allow a bit of margin
# so real-channel jitter on top of tempo still fits.
TEMPO_MIN = 0.45
TEMPO_MAX = 2.0
NEG_INF = -1e30


def _energy_extent(a: np.ndarray) -> Tuple[int, int]:
    """First and last sample above a low energy floor — bounds the search so we
    don't align over leading/trailing silence."""
    win = int(0.02 * SR)
    if len(a) < win:
        return 0, len(a)
    rms = np.array([np.sqrt(np.mean(a[i:i + win] ** 2))
                    for i in range(0, len(a) - win, win)])
    thr = 0.06 * (rms.max() or 1.0)
    voiced = np.where(rms > thr)[0]
    if len(voiced) == 0:
        return 0, len(a)
    return int(voiced[0] * win), int((voiced[-1] + 1) * win)


def _frame_grid(a: np.ndarray):
    """Frame-start sample indices spaced HOP apart across the voiced extent."""
    lo, hi = _energy_extent(a)
    hop = int(HOP * SR)
    starts = list(range(lo, max(lo + 1, hi), hop))
    return starts, lo, hi


def _span_frames(dur_sec: float):
    """(min, max) number of HOP frames a phoneme of this nominal body duration may
    occupy, allowing for prosody's global tempo scaling. Uses the body duration only
    (the trailing gap is absorbed as slack at the span's end)."""
    n_nom = (dur_sec) / HOP
    lo = max(1, int(round(n_nom * TEMPO_MIN)))
    hi = max(lo, int(round(n_nom * TEMPO_MAX)))
    return lo, hi


_SPANS = None


def _spans():
    global _SPANS
    if _SPANS is None:
        _SPANS = {ph.Dur.SHORT.value: _span_frames(SHORT_SEC),
                  ph.Dur.LONG.value: _span_frames(LONG_SEC)}
    return _SPANS


def _phoneme_dur_class(name: str) -> str:
    p = ph.BY_NAME.get(name)
    return p.dur.value if p else ph.Dur.SHORT.value


def _precompute_windows(a, starts, k):
    """Classify every candidate phoneme window ONCE, BATCHED. A window is (i, j)
    where the span j-i lies in some phoneme's allowed frame range. Returns
    win[(i,j)] = dict {phoneme_name: prob}.

    The RF predict_proba calls dominate cost (~56ms each), so we extract all window
    features first and run ONE batched predict per feature-dimension model over the
    whole matrix — far cheaper per window than per-window calls. This is what makes
    dense grid scoring tractable with the segment classifier."""
    from . import features as _F
    F = len(starts)
    short_lo, short_hi = _spans()[ph.Dur.SHORT.value]
    long_lo, long_hi = _spans()[ph.Dur.LONG.value]
    spans = sorted(set(range(short_lo, short_hi + 1)) | set(range(long_lo, long_hi + 1)))
    keys, feats = [], []
    for i in range(F):
        for span in spans:
            j = i + span
            if j > F:
                continue
            s_samp = starts[i]
            e_samp = starts[j] if j < F else len(a)
            seg = a[s_samp:e_samp]
            if len(seg) < 64:
                continue
            keys.append((i, j))
            feats.append(_F.extract(seg))
    if not feats:
        return {}
    X = np.asarray(feats, dtype=np.float32)
    m = _clf.load()
    if m is None:
        # fall back to per-window (no model) — rare; keeps behavior defined
        return {key: dict(_clf.phoneme_candidates(a[starts[i]:(starts[j] if j < F else len(a))], k=k))
                for (i, j) in keys}
    models = m["models"]
    # batched per-dimension probabilities
    bandP = models["band"].predict_proba(X);   bandC = models["band"].classes_
    contP = models["contour"].predict_proba(X); contC = models["contour"].classes_
    durP = models["dur"].predict_proba(X);     durC = models["dur"].classes_
    clsP = models["cls"].predict_proba(X);     clsC = models["cls"].classes_
    win = {}
    for r, key in enumerate(keys):
        # top-k per dimension, combine into phoneme candidates (mirrors
        # classifier.phoneme_candidates but from the batched rows)
        bi = np.argsort(bandP[r])[::-1][:k]
        ci = np.argsort(contP[r])[::-1][:k]
        di = np.argsort(durP[r])[::-1][:1]
        li = np.argsort(clsP[r])[::-1][:1]
        cands = {}
        for b in bi:
            for c in ci:
                for d in di:
                    for l in li:
                        name = _clf._nearest_by_features(bandC[b], contC[c], durC[d], clsC[l])
                        score = bandP[r][b] * contP[r][c] * durP[r][d] * clsP[r][l]
                        if name not in cands or score > cands[name]:
                            cands[name] = float(score)
        win[key] = cands
    return win


def _morph_index():
    """Valid morphemes -> phoneme sequence (skip aliases / empty / Rep-only)."""
    out = []
    for c in lex.MORPHEMES:
        if c in lex.ALIASES:
            continue
        seq = tuple(lex.concept_to_phonemes(c))
        if seq:
            out.append((c, seq))
    return out


_INDEX = None


def _index():
    global _INDEX
    if _INDEX is None:
        _INDEX = _morph_index()
    return _INDEX


def _phoneme_spans(win, i, name, dur_class):
    """All (end_frame, log_score) for phoneme `name` starting at frame i, over its
    duration class's allowed spans, from the precomputed window table."""
    import math
    lo_n, hi_n = _spans()[dur_class]
    out = []
    for span in range(lo_n, hi_n + 1):
        j = i + span
        d = win.get((i, j))
        if d is None:
            continue
        p = d.get(name)
        if p:
            out.append((j, math.log(max(p, 1e-6))))
    return out


def decode(audio: np.ndarray, k: int = 3, word_penalty: float = 2.0) -> List[str]:
    """Grid decode: align valid morphemes to the frame timeline by phoneme duration
    CLASS, with flexible per-phoneme spans (absorbing prosody tempo) and no reliance
    on silence gaps. Returns the concept list."""
    a = audio.astype(np.float32)
    if np.max(np.abs(a)) > 0:
        a = a / np.max(np.abs(a))
    starts, lo, hi = _frame_grid(a)
    F = len(starts)
    if F == 0:
        return []
    win = _precompute_windows(a, starts, k)
    index = _index()
    NEG = NEG_INF

    # phoneme-level DP: pdp[f] = best score to reach frame f having placed a whole
    # number of PHONEMES; but we need MORPHEME boundaries, so we DP at the morpheme
    # level, using an inner walk over each morpheme's phoneme spans.
    dp = [NEG] * (F + 1)
    dp[0] = 0.0
    back = [(-1, None)] * (F + 1)

    for f in range(F):
        if dp[f] <= NEG:
            continue
        for concept, seq in index:
            # inner: place this morpheme's phonemes as flexible spans from frame f.
            # frontier maps end_frame -> best cumulative score for this morpheme.
            frontier = {f: 0.0}
            for name in seq:
                dur_class = _phoneme_dur_class(name)
                nxt = {}
                for fi, acc in frontier.items():
                    if fi >= F:
                        continue
                    for ef, sc in _phoneme_spans(win, fi, name, dur_class):
                        v = acc + sc
                        if ef not in nxt or v > nxt[ef]:
                            nxt[ef] = v
                frontier = nxt
                if not frontier:
                    break
            for fi, total in frontier.items():
                cand = dp[f] + total - word_penalty
                if cand > dp[fi]:
                    dp[fi] = cand
                    back[fi] = (f, concept)

    # Endpoint: the alignment must consume (nearly) the whole voiced extent, or a
    # short high-scoring prefix wins and the rest of the message is dropped. Only
    # consider endpoints within the last few frames of the timeline; among those,
    # take the best score.
    tail = max(1, int(round(0.20 / HOP)))     # allow ending within ~0.20s of the end
    best_f, best_s = -1, NEG
    for f in range(F, max(0, F - tail) - 1, -1):
        if 0 <= f <= F and dp[f] > best_s:
            best_s, best_f = dp[f], f
    if best_f < 0:
        # nothing reached the tail; fall back to the best full alignment anywhere
        for f in range(F, 0, -1):
            if dp[f] > best_s:
                best_s, best_f = dp[f], f
    if best_f < 0:
        return []
    words, f = [], best_f
    while f > 0 and back[f][0] >= 0:
        pf, c = back[f]
        words.append(c)
        f = pf
    return list(reversed(words))
