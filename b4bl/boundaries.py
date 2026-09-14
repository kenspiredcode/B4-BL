"""
B4-BL — phoneme boundary detection.

Root-cause finding: when phoneme boundaries are positioned correctly, the
classifier recovers band/contour/duration/class at 93-99% on real audio. The whole
bottleneck is FINDING those boundaries. Silence-gap segmentation only gets them
right ~45% of the time because phonemes within a morpheme often run together with
no gap, while noise creates false gaps.

A boundary is a CHANGE POINT in the signal, not necessarily a silence:
  - a pitch jump (adjacent phonemes sit in different pitch bands / contours),
  - a contour-direction reversal,
  - a spectral/timbre shift (tone <-> gargle <-> rasp),
  - a real silence gap (still a boundary when present).

This module computes a frame-wise "novelty" curve from those cues and picks peaks
as boundaries. It returns segment spans for the classifier. It does NOT need to
know the phoneme count; but a variant can be constrained to a known count (for
force-alignment / training-label recovery).
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from scipy import signal as sps

from . import generators as gen
from . import decoder as _dec

SR = gen.SR
FRAME = 1024
HOP = 128            # finer hop than the decoder's pitch track, for boundary precision


def _frame_feats(x):
    """Per-frame cues for change detection: log-pitch (autocorr), spectral centroid,
    log-energy, spectral flatness. Returns arrays over frames + frame-start samples."""
    n = len(x)
    starts = list(range(0, max(1, n - FRAME), HOP))
    lp, cen, en, flat = [], [], [], []
    lag_min, lag_max = int(SR / 3200), int(SR / 120)
    for i in starts:
        fr = x[i:i + FRAME].astype(float)
        e = np.sqrt(np.mean(fr ** 2)) + 1e-9
        en.append(np.log(e))
        w = fr * np.hanning(len(fr))
        mag = np.abs(np.fft.rfft(w)) + 1e-9
        freqs = np.fft.rfftfreq(len(fr), 1 / SR)
        cen.append(np.sum(freqs * mag) / np.sum(mag))
        gm = np.exp(np.mean(np.log(mag))); flat.append(gm / np.mean(mag))
        # pitch via autocorr first-peak
        f = fr - fr.mean()
        if np.sqrt(np.mean(f ** 2)) < 1e-4:
            lp.append(0.0)
        else:
            ac = np.correlate(f, f, "full")[len(f) - 1:]
            ac = ac / (ac[0] or 1.0)
            seg = ac[lag_min:lag_max]
            pk = _dec._first_peak_lag(seg) if len(seg) else None
            lp.append(np.log(SR / (pk + lag_min)) if pk else 0.0)
    return (np.array(lp), np.array(cen), np.array(en), np.array(flat),
            np.array(starts))


def _novelty(lp, cen, en, flat):
    """Frame-to-frame change magnitude, combining normalized cue deltas."""
    def d(a):
        a = a.astype(float)
        rng = (np.percentile(a, 95) - np.percentile(a, 5)) or 1.0
        return np.abs(np.diff(a, prepend=a[:1])) / rng
    nov = (1.4 * d(lp)      # pitch change (band/contour) — weighted highest
           + 0.8 * d(cen)   # timbre/centroid
           + 0.6 * d(en)    # energy (onsets)
           + 0.8 * d(flat)) # tone<->texture
    # smooth a little so single-frame jitter doesn't peak
    k = 3
    nov = np.convolve(nov, np.ones(k) / k, "same")
    return nov


def detect_boundaries(audio: np.ndarray, count: Optional[int] = None) -> List[Tuple[int, int]]:
    """Return phoneme segment spans [(start,end), ...] in samples.

    Peaks in the novelty curve mark boundaries; the voiced extent bounds the ends.
    If `count` is given, take the `count-1` strongest interior peaks (used when the
    phoneme count is known, e.g. training-label recovery). Otherwise threshold."""
    a = audio.astype(np.float32)
    if np.max(np.abs(a)) > 0:
        a = a / np.max(np.abs(a))
    # voiced extent (trim leading/trailing silence)
    segs = _dec._segments(a)
    if not segs:
        return []
    lo, hi = segs[0][0], segs[-1][1]
    lp, cen, en, flat, starts = _frame_feats(a)
    nov = _novelty(lp, cen, en, flat)
    # candidate boundary frames = novelty peaks within the voiced extent
    voiced = (starts >= lo) & (starts <= hi)
    # also treat internal silences (energy dips) as strong boundary candidates
    peaks, props = sps.find_peaks(nov, distance=int(0.05 * SR / HOP))
    peaks = [p for p in peaks if voiced[p]]
    if count is not None and count >= 1:
        # want count-1 interior boundaries: take strongest peaks
        peaks = sorted(peaks, key=lambda p: nov[p], reverse=True)[:max(0, count - 1)]
        peaks = sorted(peaks)
    else:
        thr = np.percentile(nov[voiced], 80) if voiced.any() else 0
        peaks = [p for p in peaks if nov[p] >= thr]
    bnds = [lo] + [int(starts[p]) for p in peaks] + [hi]
    bnds = sorted(set(bnds))
    spans = [(bnds[i], bnds[i + 1]) for i in range(len(bnds) - 1)
             if bnds[i + 1] - bnds[i] >= int(0.04 * SR)]
    return spans


def detect_words(audio: np.ndarray) -> List[Tuple[int, int]]:
    """Word-level spans: split on LONG silence gaps only (reliable word boundaries),
    ignoring the short within-word phoneme gaps. Returns [(start,end),...] per word."""
    a = audio.astype(np.float32)
    if np.max(np.abs(a)) > 0:
        a = a / np.max(np.abs(a))
    segs = _dec._segments(a)
    if not segs:
        return []
    WORD_GAP = 0.11
    words = [[segs[0][0], segs[0][1]]]
    for s, e, _g in segs[1:]:
        if (s - words[-1][1]) / SR >= WORD_GAP:
            words.append([s, e])
        else:
            words[-1][1] = e
    return [(s, e) for s, e in words]


def detect_phoneme_spans(audio: np.ndarray, word_counts=None) -> List[List[Tuple[int, int]]]:
    """Two-level: split into WORDS on long silence, then phoneme boundaries WITHIN
    each word. Returns a list of words, each a list of phoneme spans. If
    word_counts is given (per-word phoneme count), constrain each word's split to
    that count (used when the message's word structure is known)."""
    a = audio.astype(np.float32)
    if np.max(np.abs(a)) > 0:
        a = a / np.max(np.abs(a))
    words = detect_words(a)
    out = []
    for wi, (ws, we) in enumerate(words):
        cnt = word_counts[wi] if (word_counts and wi < len(word_counts)) else None
        spans = detect_boundaries(a[ws:we], count=cnt)
        # shift back to absolute sample indices
        out.append([(ws + s, ws + e) for s, e in spans])
    return out
