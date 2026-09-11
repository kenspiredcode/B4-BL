#!/usr/bin/env python3
"""
B4-BL Phase 0 — "whistle" direction (Style C+).

Finding from the first listen: the pure sine/whistle tone reads as R2; the ring-mod
grit (Style A) and stacked harmonics (Style B) sound like a kazoo / SNES SFX.

So this engine keeps a NEARLY PURE tone and puts ALL the character into the pitch
gesture and phrasing:
  - smooth multi-point pitch contours (not just start->end)
  - vibrato/warble done as PITCH movement, never added harmonics
  - grace-note chirps, scoops, and tails
  - varied amplitude envelopes (stab / swell / decay)

No information is encoded yet. These are timbre + phrasing sketches.
"""

import numpy as np
from scipy.io import wavfile
import os

SR = 44100
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "audio_samples")


def phase_from_freq(freq_curve):
    return 2 * np.pi * np.cumsum(freq_curve) / SR


def smooth_contour(dur, points):
    """Cosine-interpolated pitch contour through [(frac, hz), ...] -> smooth glides."""
    n = int(SR * dur)
    fracs = [p[0] for p in points]
    hz = [p[1] for p in points]
    out = np.empty(n)
    seg_bounds = [int(f * n) for f in fracs]
    for i in range(len(points) - 1):
        a, b = seg_bounds[i], seg_bounds[i + 1]
        if b <= a:
            continue
        x = np.linspace(0, 1, b - a, endpoint=False)
        # smoothstep for natural, non-linear glides
        s = x * x * (3 - 2 * x)
        out[a:b] = hz[i] + (hz[i + 1] - hz[i]) * s
    out[seg_bounds[-1]:] = hz[-1]
    out[:seg_bounds[0]] = hz[0]
    return out


def vibrato(freq_curve, rate_hz, depth_frac, onset_frac=0.3):
    """Multiply frequency by a slowly-onset vibrato. Depth is a fraction of freq."""
    n = len(freq_curve)
    t = np.arange(n) / SR
    onset = np.clip(np.linspace(0, 1 / max(onset_frac, 1e-6), n), 0, 1)
    return freq_curve * (1 + depth_frac * onset * np.sin(2 * np.pi * rate_hz * t))


def env(n, kind="decay"):
    a = np.arange(n) / n
    if kind == "stab":
        e = np.exp(-4 * a)
    elif kind == "swell":
        e = np.sin(np.pi * a) ** 0.7
    elif kind == "decay":
        e = (1 - a) ** 0.6
    elif kind == "even":
        e = np.ones(n)
        ramp = int(0.01 * SR)
        e[:ramp] = np.linspace(0, 1, ramp)
        e[-ramp:] = np.linspace(1, 0, ramp)
    else:
        e = np.ones(n)
    # always de-click the very edges
    ramp = int(0.004 * SR)
    e[:ramp] *= np.linspace(0, 1, ramp)
    e[-ramp:] *= np.linspace(1, 0, ramp)
    return e


def tone(dur, points, vib=None, envkind="decay", purity=1.0):
    """
    A single whistle tone.
    purity=1.0 -> pure sine. Slightly below 1.0 adds a whisper of 2nd harmonic
    for a touch of body without the kazoo effect.
    """
    freq = smooth_contour(dur, points)
    if vib:
        freq = vibrato(freq, *vib)
    ph = phase_from_freq(freq)
    sig = np.sin(ph)
    if purity < 1.0:
        sig = purity * sig + (1 - purity) * 0.15 * np.sin(2 * ph)
    sig *= env(len(sig), envkind)
    return sig


def glue(*clips, gap=0.05):
    g = np.zeros(int(SR * gap))
    parts = []
    for c in clips:
        parts.append(c)
        parts.append(g)
    return np.concatenate(parts)


def normalize(x, peak=0.9):
    m = np.max(np.abs(x)) or 1.0
    return x / m * peak


def save(name, audio):
    os.makedirs(OUTDIR, exist_ok=True)
    audio = normalize(audio)
    path = os.path.join(OUTDIR, name)
    wavfile.write(path, SR, (audio * 32767).astype(np.int16))
    return path


def build():
    # A short "conversation" of pure-whistle astromech phrases.
    p1 = tone(0.45, [(0, 700), (0.4, 1650), (1.0, 1200)],
              vib=(7, 0.03), envkind="even")                       # inquisitive rise + settle
    p2 = tone(0.30, [(0, 1500), (0.6, 900), (1.0, 520)],
              envkind="decay")                                      # descending answer
    p3 = tone(0.16, [(0, 1100), (0.5, 2100), (1.0, 1300)],
              envkind="stab")                                       # quick chirp
    p4 = tone(0.7, [(0, 520), (0.25, 780), (0.5, 560), (0.75, 820), (1.0, 600)],
              vib=(5, 0.05, 0.1), envkind="swell")                  # worried warble (pitch-only)
    p5 = tone(0.14, [(0, 1700), (1.0, 2500)], envkind="stab")       # surprised blip up
    p6 = tone(0.55, [(0, 640), (0.3, 1250), (0.55, 950), (0.8, 1500), (1.0, 1150)],
              vib=(8, 0.025), envkind="even")                       # chatter
    # a scoop + tail flourish
    p7 = tone(0.5, [(0, 900), (0.15, 700), (0.5, 1800), (1.0, 1600)],
              vib=(6, 0.02), envkind="decay")                       # scooped exclaim

    montage = glue(p1, p2, p3, p4, p5, p6, p7)
    made = [save("whistle_montage.wav", montage)]

    # A/B micro-test: same phrase, pure vs. a hint of 2nd harmonic, to confirm
    # the pure tone is what you want.
    pure = tone(0.5, [(0, 700), (0.4, 1650), (1.0, 1200)], vib=(7, 0.03), envkind="even", purity=1.0)
    tinybody = tone(0.5, [(0, 700), (0.4, 1650), (1.0, 1200)], vib=(7, 0.03), envkind="even", purity=0.9)
    made.append(save("whistle_pure_vs_body.wav", glue(pure, tinybody, gap=0.25)))
    return made


if __name__ == "__main__":
    for p in build():
        print("wrote", os.path.relpath(p))
