"""
B4-BL acoustic decoder — audio -> phoneme names.

This is the Phase 2 signal-processing front end. v1 target (per the spec) is
FILE / LOOPBACK decode: recover the phoneme sequence from our own synthesized
audio. It validates the whole feature-tuple approach through real DSP —
segmentation + per-segment feature estimation + nearest-match — without yet
fighting room reverb and noise (Phase 4).

Pipeline
--------
1. segment on silence:
     - gaps >= WORD_GAP_MIN  -> word boundary
     - gaps >= PHONE_GAP_MIN -> phoneme boundary within a word
2. per voiced segment, estimate the invariant feature tuple that IS the phoneme's
   identity (see phonology.Phoneme.features):
     band     : from median pitch -> nearest BAND_CENTER
     contour  : from the pitch track's shape (start/mid/end comparison)
     duration : from segment length vs DUR threshold
     class    : tone vs texture (gargle/rasp) from spectral flatness / AM depth
3. nearest phoneme in the inventory by feature tuple.

Pitch is tracked by autocorrelation per frame. Good enough for clean synthesized
input; the same feature targets carry forward to the noisy over-the-air version.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from . import generators as gen
from . import phonology as ph

SR = gen.SR
FRAME = 1024
HOP = 256

# silence segmentation thresholds (seconds). Chosen to sit between the codec's
# PHONE_GAP (0.04) and WORD_GAP (0.14).
PHONE_GAP_MIN = 0.025
WORD_GAP_MIN = 0.09
SILENCE_RMS = 0.02          # below this (of peak) is silence
DUR_LONG_MIN = 0.30         # segment >= this -> Dur.LONG


# ---------------------------------------------------------------------------
# framing / energy
# ---------------------------------------------------------------------------
def _rms_envelope(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (times, rms) per hop."""
    n = len(x)
    idx = np.arange(0, max(1, n - FRAME), HOP)
    rms = np.array([np.sqrt(np.mean(x[i:i + FRAME] ** 2)) for i in idx])
    times = idx / SR
    return times, rms


def _segments(x: np.ndarray):
    """Yield (start_sample, end_sample, gap_before_sec) for each voiced run."""
    x = x / (np.max(np.abs(x)) or 1.0)
    times, rms = _rms_envelope(x)
    voiced = rms > SILENCE_RMS
    segs = []
    i = 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            start = int(times[i] * SR)
            end = int(times[min(j, len(times) - 1)] * SR)
            segs.append([start, end])
            i = j
        else:
            i += 1
    # compute gaps before each segment
    out = []
    prev_end = 0
    for (s, e) in segs:
        gap = (s - prev_end) / SR
        out.append((s, e, gap))
        prev_end = e
    return out


# ---------------------------------------------------------------------------
# pitch tracking (autocorrelation)
# ---------------------------------------------------------------------------
def _pitch_track(seg: np.ndarray) -> np.ndarray:
    fmin, fmax = 120, 3200
    lag_min, lag_max = int(SR / fmax), int(SR / fmin)
    pitches = []
    for i in range(0, max(1, len(seg) - FRAME), HOP):
        frame = seg[i:i + FRAME].astype(float)
        frame = frame - frame.mean()
        if np.sqrt(np.mean(frame ** 2)) < 1e-4:
            continue
        ac = np.correlate(frame, frame, "full")[len(frame) - 1:]
        ac0 = ac[0] if ac[0] != 0 else 1.0
        ac = ac / ac0                       # normalize so ac[0] == 1
        seg_ac = ac[lag_min:lag_max]
        if len(seg_ac) == 0:
            continue
        # find the FIRST strong local peak (the fundamental), not the global max —
        # avoids picking a short-lag harmonic and reporting an octave too high.
        f0_lag = _first_peak_lag(seg_ac)
        if f0_lag is None:
            continue
        lag = f0_lag + lag_min
        if lag > 0:
            pitches.append(SR / lag)
    return np.array(pitches) if pitches else np.array([0.0])


def _first_peak_lag(ac: np.ndarray, thresh: float = 0.5):
    """Lag of the first local maximum that exceeds `thresh` of the global peak
    within the search window. Falls back to global argmax."""
    peak = np.max(ac)
    if peak <= 0:
        return None
    floor = thresh * peak
    for i in range(1, len(ac) - 1):
        if ac[i] >= floor and ac[i] > ac[i - 1] and ac[i] >= ac[i + 1]:
            return i
    return int(np.argmax(ac))


def _spectral_flatness(seg: np.ndarray) -> float:
    """High for noise/rasp, low for tonal. Used to separate tone vs texture."""
    w = seg * np.hanning(len(seg))
    mag = np.abs(np.fft.rfft(w)) + 1e-9
    gm = np.exp(np.mean(np.log(mag)))
    am = np.mean(mag)
    return float(gm / am)


def _am_depth(seg: np.ndarray) -> float:
    """Flutter periodicity — high for gargle (its amplitude is gated at a steady
    ~20-50 Hz), low for an ordinary tone (whose envelope has one attack/decay, no
    periodic flutter). Measured as the strength of the envelope's spectrum in the
    20-60 Hz flutter band relative to its total, over the sustained middle."""
    n = len(seg)
    if n < 200:
        return 0.0
    mid = seg[int(0.2 * n): int(0.8 * n)].astype(float)
    env = np.abs(mid)
    env = env - env.mean()
    if np.sqrt(np.mean(env ** 2)) < 1e-6:
        return 0.0
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    freqs = np.fft.rfftfreq(len(env), 1 / SR)
    band = (freqs >= 18) & (freqs <= 60)
    total = np.sum(spec) + 1e-9
    return float(np.sum(spec[band]) / total)


# ---------------------------------------------------------------------------
# feature estimation
# ---------------------------------------------------------------------------
def _nearest_band(hz: float) -> ph.Band:
    best, bestd = None, 1e18
    for band, c in ph.BAND_CENTER.items():
        d = abs(np.log2((hz + 1e-6) / c))
        if d < bestd:
            best, bestd = band, d
    return best


# normalized contour templates (start, mid, end-ish) sampled at 5 points, in
# semitone-ish (log) space, zero-mean. Matched by correlation against the track.
_CONTOUR_TEMPLATES = {
    ph.Contour.FLAT:   np.array([0, 0, 0, 0, 0.0]),
    ph.Contour.RISE:   np.array([-1, -0.5, 0, 0.5, 1.0]),
    ph.Contour.FALL:   np.array([1, 0.5, 0, -0.5, -1.0]),
    ph.Contour.ARCH:   np.array([-1, 0.2, 1, 0.2, -0.4]),
    ph.Contour.DIP:    np.array([1, -0.2, -1, -0.2, 0.4]),
    ph.Contour.SCOOP:  np.array([0.2, -0.6, -0.3, 0.4, 1.0]),
    ph.Contour.DOUBLE: np.array([-0.8, 0.8, -0.5, 0.8, 0.0]),
}


def _classify_contour(track: np.ndarray) -> ph.Contour:
    t = track[track > 0]
    if len(t) < 2:
        return ph.Contour.FLAT
    # trim edge frames — the pitch tracker drifts at onset/offset, which can fake
    # a small slope on an otherwise flat tone.
    if len(t) >= 7:
        t = t[1:-1]
    logt = np.log(t)
    span = logt.max() - logt.min()
    # a flat tone shows ~0.13 of tracker jitter; a real sweep (span ~0.33 center)
    # is much larger. Gate between them.
    if span < 0.18:
        return ph.Contour.FLAT
    # resample the track to 5 points, zero-mean, unit-scale
    xs = np.linspace(0, 1, len(logt))
    samp = np.interp(np.linspace(0, 1, 5), xs, logt)
    samp = samp - samp.mean()
    if np.max(np.abs(samp)) > 0:
        samp = samp / np.max(np.abs(samp))
    # best-correlating template (FLAT already handled by the span gate above and
    # excluded here — its zero template can't be correlated).
    best, bestscore = ph.Contour.RISE, -1e9
    for c, tmpl in _CONTOUR_TEMPLATES.items():
        if c == ph.Contour.FLAT:
            continue
        tm = tmpl - tmpl.mean()
        denom = np.linalg.norm(samp) * np.linalg.norm(tm) + 1e-9
        score = float(np.dot(samp, tm) / denom)
        if score > bestscore:
            best, bestscore = c, score
    return best


def _classify_sound(seg: np.ndarray) -> ph.SoundClass:
    flat = _spectral_flatness(seg)
    am = _am_depth(seg)
    if flat > 0.15:               # broadband noise -> rasp
        return ph.SoundClass.RASP
    if am > 0.08:                 # periodic flutter -> gargle
        return ph.SoundClass.GARGLE
    return ph.SoundClass.TONE


def _band_anchor(track: np.ndarray) -> float:
    """A band estimate robust to contour sweeps. Our contours are symmetric around
    the band center (rise = center-span..center+span, fall the reverse, arch/dip
    return to center), so the GEOMETRIC MEAN of the pitch track approximates the
    center regardless of contour direction — unlike the median of a sweep."""
    t = track[track > 0]
    if len(t) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(t))))


def classify_segment(seg: np.ndarray) -> Tuple[str, tuple]:
    """Return (best phoneme name, estimated feature tuple)."""
    track = _pitch_track(seg)
    anchor = _band_anchor(track)
    band = _nearest_band(anchor) if anchor > 0 else ph.Band.MID
    contour = _classify_contour(track)
    dur = ph.Dur.LONG if len(seg) / SR >= DUR_LONG_MIN else ph.Dur.SHORT
    cls = _classify_sound(seg)
    feats = (band.value, contour.value, dur.value, cls.value)
    name = _nearest_phoneme(band, contour, dur, cls)
    return name, feats


def _nearest_phoneme(band, contour, dur, cls) -> str:
    # exact feature match first
    for p in ph.INVENTORY:
        if (p.band, p.contour, p.dur, p.cls) == (band, contour, dur, cls):
            return p.name
    # relax duration, then contour, keeping band+class
    for p in ph.INVENTORY:
        if (p.band, p.contour, p.cls) == (band, contour, cls):
            return p.name
    for p in ph.INVENTORY:
        if (p.band, p.cls) == (band, cls):
            return p.name
    # last resort: same band
    for p in ph.INVENTORY:
        if p.band == band:
            return p.name
    return ph.INVENTORY[0].name


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------
def audio_to_phoneme_words(audio: np.ndarray) -> List[List[str]]:
    """Segment audio, group phonemes into words by gap size, classify each."""
    segs = _segments(audio)
    words: List[List[str]] = []
    current: List[str] = []
    for k, (s, e, gap) in enumerate(segs):
        seg = audio[s:e]
        if len(seg) < int(0.02 * SR):
            continue
        name, _feats = classify_segment(seg)
        if k > 0 and gap >= WORD_GAP_MIN and current:
            words.append(current)
            current = []
        current.append(name)
    if current:
        words.append(current)
    return words
