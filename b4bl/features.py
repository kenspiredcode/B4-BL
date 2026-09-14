"""
B4-BL feature extraction — audio segment -> fixed-length numeric vector.

Shared by the trained classifier (training + inference). The vector is designed
to expose exactly what distinguishes phonemes:

  - pitch-track SHAPE (resampled, normalized) -> contour
  - pitch-track CENTER (log Hz)               -> band
  - segment duration                          -> duration class
  - spectral flatness + AM flutter            -> sound class (tone/gargle/rasp)
  - a few spectral-shape summaries            -> extra separation

Everything is derived from the same DSP as the threshold decoder, so the trained
model learns on the same signal the runtime sees.
"""

from __future__ import annotations
import numpy as np
from . import generators as gen
from . import decoder as _dec   # reuse pitch track / flatness / am
from . import mfcc as _mfcc

SR = gen.SR
SHAPE_POINTS = 12   # resampled pitch-track shape length


def _pitch_shape_and_center(seg):
    track = _dec._pitch_track(seg)
    t = track[track > 0]
    if len(t) < 2:
        return np.zeros(SHAPE_POINTS), 0.0, 0.0
    if len(t) >= 7:
        t = t[1:-1]
    logt = np.log(t)
    center = float(np.exp(np.mean(logt)))            # geometric-mean pitch (band anchor)
    span = float(logt.max() - logt.min())
    xs = np.linspace(0, 1, len(logt))
    shape = np.interp(np.linspace(0, 1, SHAPE_POINTS), xs, logt)
    shape = shape - shape.mean()
    m = np.max(np.abs(shape))
    if m > 0:
        shape = shape / m
    return shape, center, span


def _spectral_summary(seg):
    w = seg * np.hanning(len(seg))
    mag = np.abs(np.fft.rfft(w)) + 1e-9
    freqs = np.fft.rfftfreq(len(seg), 1 / SR)
    total = np.sum(mag)
    # energy in a few coarse bands (spectral centroid-ish shape)
    edges = [0, 500, 1000, 2000, 4000, SR / 2]
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        bands.append(float(np.sum(mag[m]) / total))
    centroid = float(np.sum(freqs * mag) / total)
    return bands, centroid


def extract(seg: np.ndarray) -> np.ndarray:
    """Return the feature vector for one audio segment.

    NOTE: appending MFCCs here was tried and REVERTED — it hurt end-to-end decode
    (real test single 54%->42%). Over a short (~90ms) frame, per-frame MFCCs are
    noisy and their 14 dims diluted the pitch-shape/contour features that carry the
    actual meaning axes (band + contour), so the forest lost the real signal among
    them. Kept the pitch/contour/spectral-cue set that worked. (MFCCs may still
    help computed over a WHOLE phoneme rather than a frame — future work.)"""
    seg = seg.astype(float)
    if len(seg) < 64:
        seg = np.pad(seg, (0, 64 - len(seg)))
    shape, center, span = _pitch_shape_and_center(seg)
    flat = _dec._spectral_flatness(seg)
    am = _dec._am_depth(seg)
    dur = len(seg) / SR
    bands, centroid = _spectral_summary(seg)
    feats = np.concatenate([
        shape,                                  # SHAPE_POINTS  (contour)
        [np.log(center + 1e-6), span, dur],     # pitch center / span / duration (band)
        [flat, am],                             # sound-class cues
        bands,                                  # 5 spectral-band energies
        [np.log(centroid + 1e-6)],              # spectral centroid
    ])
    return feats.astype(np.float32)


FEATURE_LEN = SHAPE_POINTS + 3 + 2 + 5 + 1
