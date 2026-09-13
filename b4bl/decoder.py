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


# segment post-processing thresholds (seconds)
MIN_SEG_DUR = 0.05      # drop fragments shorter than this (noise blips)
MERGE_GAP = 0.05        # merge two segments separated by a gap shorter than this
                        # (a phoneme's brief internal dip / reverb notch, not a
                        #  real boundary)


def _raw_segments(voiced, times):
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
    return segs


def _segments(x: np.ndarray):
    """Segment into voiced runs, then CLEAN UP for real audio:
      1. merge runs separated by a gap < MERGE_GAP (spurious internal dips), and
      2. drop runs shorter than MIN_SEG_DUR (noise blips / reverb fragments).
    Real recordings otherwise over-segment (noise creates false gaps and tiny
    fragments) or split a phoneme on a brief internal amplitude notch."""
    x = x / (np.max(np.abs(x)) or 1.0)
    times, rms = _rms_envelope(x)
    voiced = rms > SILENCE_RMS
    segs = _raw_segments(voiced, times)

    # Only DROP tiny fragments (noise blips) — do NOT merge on short gaps, because
    # real within-morpheme phoneme gaps (~30ms) are the same size as noise gaps,
    # so gap-based merging collapses distinct phonemes. Boundary detection needs a
    # different signal than energy gaps (see _refine_boundaries / pitch-based).
    kept = [list(seg) for seg in segs if (seg[1] - seg[0]) / SR >= MIN_SEG_DUR]
    if not kept:                       # if all dropped, keep the loudest raw run
        kept = [list(max(segs, key=lambda z: z[1] - z[0]))] if segs else []

    out = []
    prev_end = 0
    for s, e in kept:
        out.append((s, e, (s - prev_end) / SR))
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
        # VHIGH is reserved/rare; bias against it so a bright HIGH sound doesn't
        # get pulled up into it (a common error with neutral prosody's lift).
        if band == ph.Band.VHIGH:
            d += 0.10
        if d < bestd:
            best, bestd = band, d
    return best


# normalized contour templates at 8 points in log-pitch space, zero-mean.
# Calibrated against the actual rendered pitch tracks (synth and recognizer share
# the model) so scoop/dip/arch/double don't collapse together.
_CONTOUR_TEMPLATES = {
    ph.Contour.RISE:   np.array([-1.0, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.0]),
    ph.Contour.FALL:   np.array([1.0, 0.7, 0.4, 0.1, -0.2, -0.5, -0.8, -1.0]),
    ph.Contour.ARCH:   np.array([-1.0, -0.2, 0.5, 1.0, 0.9, 0.5, 0.0, -0.4]),
    ph.Contour.DIP:    np.array([1.0, 0.3, -0.5, -1.0, -0.8, -0.3, 0.1, 0.4]),
    ph.Contour.SCOOP:  np.array([0.0, -0.5, -0.7, -0.5, -0.2, 0.3, 0.7, 1.0]),
    ph.Contour.DOUBLE: np.array([-0.7, 0.7, 0.7, -0.5, -1.0, -0.2, 0.55, 0.35]),
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
    if span < 0.18:
        return ph.Contour.FLAT
    xs = np.linspace(0, 1, len(logt))
    samp = np.interp(np.linspace(0, 1, 8), xs, logt)
    samp = samp - samp.mean()
    if np.max(np.abs(samp)) > 0:
        samp = samp / np.max(np.abs(samp))
    best, bestscore = ph.Contour.RISE, -1e9
    for c, tmpl in _CONTOUR_TEMPLATES.items():
        tm = tmpl - tmpl.mean()
        denom = np.linalg.norm(samp) * np.linalg.norm(tm) + 1e-9
        score = float(np.dot(samp, tm) / denom)
        if score > bestscore:
            best, bestscore = c, score
    return best


def _pitch_clarity(seg: np.ndarray) -> float:
    """Strength of the autocorrelation fundamental peak (0..1). High for clean
    pitched tones/gargle, low for broadband rasp noise."""
    fmin, fmax = 120, 3200
    lag_min, lag_max = int(SR / fmax), int(SR / fmin)
    n = len(seg)
    mid = seg[int(0.2 * n):int(0.8 * n)].astype(float)
    if len(mid) < FRAME:
        mid = seg.astype(float)
    mid = mid - mid.mean()
    if np.sqrt(np.mean(mid ** 2)) < 1e-6:
        return 0.0
    ac = np.correlate(mid, mid, "full")[len(mid) - 1:]
    ac = ac / (ac[0] or 1.0)
    window = ac[lag_min:lag_max]
    return float(np.max(window)) if len(window) else 0.0


def _classify_sound(seg: np.ndarray) -> ph.SoundClass:
    """TONE / GARGLE / RASP — the coarse talking-vs-texture meaning axis, keyed on
    spectral flatness, which separates them with wide margins:
      TONE   ~0.00  (pure sine)
      RASP   ~0.06  (filtered buzz)
      GARGLE ~0.28  (hard AM gating spreads energy -> high flatness)
    (whistle/trill collapse into TONE — they aren't decodable classes.)"""
    flat = _spectral_flatness(seg)
    if flat > 0.15:
        return ph.SoundClass.GARGLE
    if flat > 0.03:
        return ph.SoundClass.RASP
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


def _norm_shape(track: np.ndarray, k: int = 8) -> np.ndarray:
    """Resample a pitch track to k points in log space, zero-mean, unit-scale —
    a contour 'shape' fingerprint independent of absolute pitch and span."""
    t = track[track > 0]
    if len(t) < 2:
        return np.zeros(k)
    if len(t) >= 7:
        t = t[1:-1]
    logt = np.log(t)
    samp = np.interp(np.linspace(0, 1, k), np.linspace(0, 1, len(logt)), logt)
    samp = samp - samp.mean()
    m = np.max(np.abs(samp))
    return samp / m if m > 0 else samp


# reference contour shapes, generated ONCE from the inventory's own rendering —
# synth and recognizer share the model, so we match a segment against the real
# rendered shapes rather than hand-written templates.
_REF_SHAPES = {}


def _ref_shapes():
    if not _REF_SHAPES:
        for p in ph.INVENTORY:
            if p.cls == ph.SoundClass.TONE:
                _REF_SHAPES[p.contour] = _norm_shape(_pitch_track(p.render()))
    return _REF_SHAPES


def _classify_contour_ref(track: np.ndarray) -> ph.Contour:
    t = track[track > 0]
    if len(t) < 2:
        return ph.Contour.FLAT
    logt = np.log(t if len(t) < 7 else t[1:-1])
    if logt.max() - logt.min() < 0.18:
        return ph.Contour.FLAT
    shape = _norm_shape(track)
    best, bestscore = ph.Contour.FLAT, -1e9
    for contour, ref in _ref_shapes().items():
        if contour == ph.Contour.FLAT:
            continue
        denom = np.linalg.norm(shape) * np.linalg.norm(ref) + 1e-9
        score = float(np.dot(shape, ref) / denom)
        if score > bestscore:
            best, bestscore = contour, score
    return best


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
    """Segment audio, group phonemes into words by gap size, classify each
    (top-1 per phoneme). Uses the trained classifier when present, else thresholds.
    For lexicon-corrected decoding to concepts, use audio_to_concepts()."""
    try:
        from . import classifier as _clf
        seg_classify = _clf.classify_segment if _clf.available() else classify_segment
    except Exception:
        seg_classify = classify_segment
    segs = _segments(audio)
    words: List[List[str]] = []
    current: List[str] = []
    for k, (s, e, gap) in enumerate(segs):
        seg = audio[s:e]
        if len(seg) < int(0.02 * SR):
            continue
        name, _feats = seg_classify(seg)
        if k > 0 and gap >= WORD_GAP_MIN and current:
            words.append(current)
            current = []
        current.append(name)
    if current:
        words.append(current)
    return words


def audio_to_candidate_words(audio: np.ndarray, k: int = 2):
    """Like audio_to_phoneme_words, but each phoneme is a ranked candidate list
    [(name, score), ...] rather than a single name. Feeds lexicon-constrained
    decoding. Requires the trained classifier (falls back to top-1 otherwise)."""
    from . import classifier as _clf
    have = _clf.available()
    segs = _segments(audio)
    words = []
    current = []
    for idx, (s, e, gap) in enumerate(segs):
        seg = audio[s:e]
        if len(seg) < int(0.02 * SR):
            continue
        cands = _clf.phoneme_candidates(seg, k=k) if have else [(classify_segment(seg)[0], 1.0)]
        if idx > 0 and gap >= WORD_GAP_MIN and current:
            words.append(current)
            current = []
        current.append(cands)
    if current:
        words.append(current)
    return words
