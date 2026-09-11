"""
B4-BL sound generators — the locked Phase 0 palette.

Two generator classes, chosen by ear over three rounds of listening tests:

  TONAL  (pure sine + pitch gesture, no added harmonics)
    whistle, chirp, warble (pitch-vibrato), groan
  ROUGH  (noise / buzz, used sparingly and only where grit belongs)
    raspberry, static, gargle (amplitude flutter)

Findings that shaped this (Phase 0 listening tests):
  - Pure tone reads as R2. Added harmonics / ring-mod read as "kazoo / SNES SFX".
  - Sample-and-hold random burble reads as "low-budget sci-fi computer" -> dropped.
  - FM "scream" sounded bad -> dropped.
  - Character lives in the PITCH GESTURE, not the timbre.
  - Grit is per-family: wrong on whistles, right on raspberries.

All generators return float32 mono at SR. They take a `Gesture` (pitch contour +
timing + envelope) so the same gesture can later be re-rendered with different
prosody. No information is encoded here — this is the acoustic substrate.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from scipy import signal as sps

SR = 44100
_rng = np.random.default_rng(11)


# ---------------------------------------------------------------------------
# gesture description
# ---------------------------------------------------------------------------
@dataclass
class Gesture:
    """A pitch/timing/shape description that a generator turns into audio.

    points : [(fraction 0..1, hz), ...] smooth pitch contour.
    dur     : seconds.
    env     : 'even' | 'stab' | 'swell' | 'decay'.
    vib     : (rate_hz, depth_frac) optional pitch vibrato, or None.
    """
    points: List[Tuple[float, float]]
    dur: float
    env: str = "even"
    vib: Optional[Tuple[float, float]] = None


# ---------------------------------------------------------------------------
# low-level dsp
# ---------------------------------------------------------------------------
def _phase(freq_curve: np.ndarray) -> np.ndarray:
    return 2 * np.pi * np.cumsum(freq_curve) / SR


def smooth_contour(dur: float, points: List[Tuple[float, float]]) -> np.ndarray:
    """Smoothstep-interpolated pitch contour -> natural glides, not linear ramps."""
    n = int(SR * dur)
    if n <= 0:
        return np.zeros(0)
    out = np.empty(n)
    b = [int(p[0] * n) for p in points]
    hz = [p[1] for p in points]
    for i in range(len(points) - 1):
        lo, hi = b[i], b[i + 1]
        if hi <= lo:
            continue
        x = np.linspace(0, 1, hi - lo, endpoint=False)
        s = x * x * (3 - 2 * x)
        out[lo:hi] = hz[i] + (hz[i + 1] - hz[i]) * s
    out[b[-1]:] = hz[-1]
    out[: b[0]] = hz[0]
    return out


def envelope(n: int, kind: str = "even") -> np.ndarray:
    a = np.arange(n) / max(n, 1)
    if kind == "stab":
        e = np.exp(-5 * a)
    elif kind == "swell":
        e = np.sin(np.pi * a) ** 0.7
    elif kind == "decay":
        e = (1 - a) ** 0.6
    else:
        e = np.ones(n)
    ramp = max(1, int(0.004 * SR))
    e[:ramp] *= np.linspace(0, 1, ramp)
    e[-ramp:] *= np.linspace(1, 0, ramp)
    return e


def _apply_vib(freq: np.ndarray, vib) -> np.ndarray:
    if not vib:
        return freq
    rate, depth = vib
    t = np.arange(len(freq)) / SR
    return freq * (1 + depth * np.sin(2 * np.pi * rate * t))


# ---------------------------------------------------------------------------
# TONAL family — pure sine, character in the pitch gesture
# ---------------------------------------------------------------------------
def tonal(g: Gesture) -> np.ndarray:
    freq = _apply_vib(smooth_contour(g.dur, g.points), g.vib)
    sig = np.sin(_phase(freq))
    return (sig * envelope(len(sig), g.env)).astype(np.float32)


def whistle(dur=0.5, lo=700, peak=1650, end=1200) -> np.ndarray:
    return tonal(Gesture([(0, lo), (0.4, peak), (1, end)], dur, "even", vib=(7, 0.03)))


def chirp(bips=None) -> np.ndarray:
    """Short punchy bips, each with INTERNAL pitch movement (scoop/bend)."""
    if bips is None:
        bips = [
            Gesture([(0, 900), (0.35, 1700), (1, 1450)], 0.12, "stab"),
            Gesture([(0, 1600), (0.5, 1150), (1, 1350)], 0.10, "stab"),
            Gesture([(0, 1200), (0.4, 2100), (1, 1600)], 0.13, "stab"),
        ]
    return sequence([tonal(b) for b in bips], gap=0.045)


def warble(dur=0.6, lo=620, mid=900, end=700, rate=11, depth=0.06) -> np.ndarray:
    return tonal(Gesture([(0, lo), (0.5, mid), (1, end)], dur, "swell", vib=(rate, depth)))


def groan(dur=0.7, hi=900, mid=600, low=300) -> np.ndarray:
    return tonal(Gesture([(0, hi), (0.4, mid), (1, low)], dur, "decay"))


# ---------------------------------------------------------------------------
# ROUGH family — grit where grit belongs, used sparingly
# ---------------------------------------------------------------------------
def raspberry(dur=0.32, center=340) -> np.ndarray:
    n = int(SR * dur)
    f = np.linspace(center, center * 0.7, n)
    buzz = sps.sawtooth(_phase(f), width=0.5)
    noise = _rng.uniform(-1, 1, n)
    b, a = sps.butter(2, 2600 / (SR / 2), btype="low")
    sig = sps.lfilter(b, a, 0.8 * buzz + 0.35 * noise)
    t = np.arange(n) / SR
    sig *= 0.55 + 0.45 * np.sin(2 * np.pi * 42 * t)
    return (sig * envelope(n, "decay") * 1.6).astype(np.float32)


def static(dur=0.35, center=1400) -> np.ndarray:
    """Deliberate rhythmic mechanical chatter (hard-gated band noise). Use sparingly."""
    n = int(SR * dur)
    noise = _rng.uniform(-1, 1, n)
    w = center / (SR / 2)
    b, a = sps.butter(4, [max(0.02, w * 0.6), min(0.98, w * 1.6)], btype="band")
    sig = sps.lfilter(b, a, noise)
    t = np.arange(n) / SR
    gate = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 30 * t))
    return (sig * gate * envelope(n, "even")).astype(np.float32)


def gargle(dur=0.4, center=800, tremolo_hz=45, depth=0.9) -> np.ndarray:
    """Amplitude-flutter roll ('brrrp'). Intentional character, sits between tonal
    and rough."""
    f = smooth_contour(dur, [(0, center), (1, center * 0.95)])
    sig = np.sin(_phase(f))
    t = np.arange(len(sig)) / SR
    trem = (1 - depth) + depth * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * tremolo_hz * t)))
    return (sig * trem * envelope(len(sig), "even")).astype(np.float32)


# ---------------------------------------------------------------------------
# assembly + io
# ---------------------------------------------------------------------------
def sequence(clips: List[np.ndarray], gap: float = 0.05) -> np.ndarray:
    g = np.zeros(int(SR * gap), dtype=np.float32)
    out = []
    for c in clips:
        out.append(c.astype(np.float32))
        out.append(g)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) or 1.0
    return (x / m * peak).astype(np.float32)


def write_wav(path: str, audio: np.ndarray):
    from scipy.io import wavfile
    wavfile.write(path, SR, (normalize(audio) * 32767).astype(np.int16))
