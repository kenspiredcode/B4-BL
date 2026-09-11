#!/usr/bin/env python3
"""
B4-BL Phase 0 — sound-direction comparison.

Generates the SAME set of astromech "utterances" through three different
synthesis styles so they can be judged by ear against each other. No information
is encoded yet; these are timbre/phrasing sketches only.

Styles
------
A  ARP-2600 signal path      : osc + resonant filter + sample&hold + ring mod + envelopes.
                              The plan's recommended procedural foundation.
B  mcp-muse-style presets    : per-"emotion" parameter sets (carrier band, contour,
                              modulation depth, formant-ish shift). Affect owns the sound.
C  simple sine + pitch bend  : lightweight embedded baseline (ESP32-style).

Each style renders the same 6 phrases so differences are attributable to the
synthesis approach, not the content.

Output: mono 44.1kHz WAV files in ../audio_samples/, plus a combined montage per style.
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal as sps
import os

SR = 44100
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "audio_samples")
rng = np.random.default_rng(7)


# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------
def t_arr(dur):
    return np.linspace(0, dur, int(SR * dur), endpoint=False)


def adsr(n, a=0.01, d=0.05, s=0.7, r=0.05):
    """Simple ADSR envelope of length n samples."""
    a_n, d_n, r_n = int(a * SR), int(d * SR), int(r * SR)
    s_n = max(0, n - a_n - d_n - r_n)
    env = np.concatenate([
        np.linspace(0, 1, a_n, endpoint=False) if a_n else np.array([]),
        np.linspace(1, s, d_n, endpoint=False) if d_n else np.array([]),
        np.full(s_n, s),
        np.linspace(s, 0, r_n, endpoint=False) if r_n else np.array([]),
    ])
    if len(env) < n:
        env = np.concatenate([env, np.zeros(n - len(env))])
    return env[:n]


def phase_from_freq(freq_curve):
    """Integrate an instantaneous-frequency curve into phase."""
    return 2 * np.pi * np.cumsum(freq_curve) / SR


def contour(dur, points):
    """Interpolate a pitch contour given as [(frac, freq_hz), ...] over dur seconds."""
    n = int(SR * dur)
    xs = np.array([p[0] for p in points]) * n
    ys = np.array([p[1] for p in points])
    idx = np.arange(n)
    return np.interp(idx, xs, ys)


def normalize(x, peak=0.9):
    m = np.max(np.abs(x)) or 1.0
    return x / m * peak


def save(name, audio):
    audio = normalize(audio)
    path = os.path.join(OUTDIR, name)
    wavfile.write(path, SR, (audio * 32767).astype(np.int16))
    return path


def glue(*clips, gap=0.06):
    g = np.zeros(int(SR * gap))
    out = []
    for c in clips:
        out.append(c)
        out.append(g)
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# STYLE A — ARP-2600 signal path
# oscillator(s) -> ring mod -> resonant lowpass driven by S&H -> VCA(ADSR)
# ----------------------------------------------------------------------------
def sample_and_hold(dur, rate_hz, lo, hi):
    """Stepped random control signal, like the ARP S&H feeding the filter."""
    n = int(SR * dur)
    steps = max(1, int(dur * rate_hz))
    vals = rng.uniform(lo, hi, steps)
    return np.repeat(vals, int(np.ceil(n / steps)))[:n]


def resonant_lp(x, cutoff_curve, q=6.0):
    """Time-varying resonant lowpass, applied in short blocks (cheap TVF)."""
    n = len(x)
    block = 256
    out = np.zeros(n)
    zi = None
    for start in range(0, n, block):
        end = min(start + block, n)
        fc = float(np.clip(np.mean(cutoff_curve[start:end]), 80, SR / 2 - 500))
        w0 = fc / (SR / 2)
        b, a = sps.iirpeak(w0, q) if False else sps.butter(2, w0, btype="low")
        # add resonance by mixing a bandpass peak
        bb, ab = sps.iirpeak(w0, q)
        seg = x[start:end]
        lp = sps.lfilter(b, a, seg)
        bp = sps.lfilter(bb, ab, seg)
        out[start:end] = lp + 0.6 * bp
    return out


def style_a(dur, base_contour, sh_rate=14, ring_ratio=1.5, reson=7.0,
            env=(0.008, 0.04, 0.75, 0.05)):
    freq = contour(dur, base_contour)
    car = np.sin(phase_from_freq(freq))
    # slight second oscillator for body
    car += 0.4 * np.sin(phase_from_freq(freq * 2.001))
    # ring modulation -> metallic astromech edge
    ring = np.sin(phase_from_freq(freq * ring_ratio))
    sig = car * (0.6 + 0.4 * ring)
    # S&H sweeps the resonant filter -> the characteristic "bubbling"
    sh = sample_and_hold(dur, sh_rate, 400, 3800)
    sig = resonant_lp(sig, sh, q=reson)
    sig *= adsr(len(sig), *env)
    return sig


# ----------------------------------------------------------------------------
# STYLE B — mcp-muse-style parametric "emotion" presets
# affect owns every dimension: freq band, contour shape, modulation depth, timbre
# ----------------------------------------------------------------------------
PRESETS_B = {
    "curious":   dict(band=(500, 1600), shape="rise", moddepth=0.18, modrate=9,  harm=0.5, dur=0.55),
    "happy":     dict(band=(700, 2200), shape="arch", moddepth=0.12, modrate=12, harm=0.6, dur=0.5),
    "worried":   dict(band=(300, 900),  shape="wander", moddepth=0.35, modrate=6, harm=0.3, dur=0.7),
    "negative":  dict(band=(200, 500),  shape="fall", moddepth=0.10, modrate=5,  harm=0.25, dur=0.6),
    "surprised": dict(band=(600, 2600), shape="jump", moddepth=0.15, modrate=14, harm=0.7, dur=0.4),
    "neutral":   dict(band=(450, 1200), shape="flat", moddepth=0.08, modrate=8,  harm=0.4, dur=0.5),
}


def _shape_curve(shape, dur, lo, hi):
    if shape == "rise":
        return [(0.0, lo), (1.0, hi)]
    if shape == "fall":
        return [(0.0, hi), (1.0, lo)]
    if shape == "arch":
        return [(0.0, lo), (0.5, hi), (1.0, (lo + hi) / 2)]
    if shape == "jump":
        return [(0.0, lo), (0.15, hi), (1.0, hi * 0.9)]
    if shape == "flat":
        mid = (lo + hi) / 2
        return [(0.0, mid), (1.0, mid)]
    if shape == "wander":
        return [(0.0, (lo + hi) / 2), (0.3, hi), (0.55, lo), (0.8, hi * 0.9), (1.0, (lo + hi) / 2)]
    return [(0.0, lo), (1.0, hi)]


def style_b(preset_name):
    p = PRESETS_B[preset_name]
    dur = p["dur"]
    lo, hi = p["band"]
    freq = contour(dur, _shape_curve(p["shape"], dur, lo, hi))
    # vibrato/warble modulation, depth is the affect knob
    mod = 1 + p["moddepth"] * np.sin(2 * np.pi * p["modrate"] * t_arr(dur))
    freq = freq * mod
    sig = np.sin(phase_from_freq(freq))
    sig += p["harm"] * np.sin(phase_from_freq(freq * 2))     # harmonic richness knob
    sig += 0.25 * p["harm"] * np.sin(phase_from_freq(freq * 3))
    sig *= adsr(len(sig), a=0.01, d=0.06, s=0.7, r=0.06)
    return sig


# ----------------------------------------------------------------------------
# STYLE C — simple sine + linear pitch bend (embedded baseline)
# ----------------------------------------------------------------------------
def style_c(dur, f_start, f_end, glide="lin"):
    if glide == "lin":
        freq = np.linspace(f_start, f_end, int(SR * dur))
    else:  # exponential
        freq = np.geomspace(f_start, f_end, int(SR * dur))
    sig = np.sin(phase_from_freq(freq))
    sig *= adsr(len(sig), a=0.005, d=0.02, s=0.85, r=0.03)
    return sig


# ----------------------------------------------------------------------------
# The six shared "phrases" — same intent across all three styles
# (no meaning encoded; these are just recognizable astromech gestures)
# ----------------------------------------------------------------------------
def render_all():
    os.makedirs(OUTDIR, exist_ok=True)
    made = []

    # ---- Style A phrases ----
    a1 = style_a(0.45, [(0, 700), (0.4, 1600), (1.0, 1100)], sh_rate=16)      # inquisitive rise
    a2 = style_a(0.30, [(0, 1400), (1.0, 500)], sh_rate=20, reson=9)          # descending grumble
    a3 = style_a(0.22, [(0, 900), (0.5, 2000), (1.0, 900)], sh_rate=10)       # quick chirp-arch
    a4 = style_a(0.6,  [(0, 400), (0.5, 700), (1.0, 350)], sh_rate=8, reson=10)  # worried wander
    a5 = style_a(0.18, [(0, 1800), (1.0, 2400)], sh_rate=24)                  # surprised blip
    a6 = style_a(0.5,  [(0, 600), (0.3, 1200), (0.6, 900), (1.0, 1500)], sh_rate=14)  # chatter
    made.append(save("styleA_montage.wav", glue(a1, a2, a3, a4, a5, a6)))

    # ---- Style B phrases ----
    b_clips = [style_b(name) for name in
               ["curious", "negative", "happy", "worried", "surprised", "neutral"]]
    made.append(save("styleB_montage.wav", glue(*b_clips)))

    # ---- Style C phrases ----
    c1 = style_c(0.4, 700, 1600, "exp")     # rise
    c2 = style_c(0.3, 1500, 500, "exp")     # fall
    c3 = style_c(0.2, 900, 2000, "lin")     # blip up
    c4 = style_c(0.5, 500, 800, "lin")      # low waver-ish
    c5 = style_c(0.15, 1800, 2500, "exp")   # surprise
    c6 = style_c(0.45, 600, 1400, "exp")    # chatter
    made.append(save("styleC_montage.wav", glue(c1, c2, c3, c4, c5, c6)))

    # ---- one grand montage: A, then B, then C, with labels-by-gap ----
    big = glue(
        glue(a1, a2, a3, a4, a5, a6),
        glue(*b_clips),
        glue(c1, c2, c3, c4, c5, c6),
        gap=0.4,
    )
    made.append(save("ALL_styles_montage.wav", big))
    return made


if __name__ == "__main__":
    paths = render_all()
    for p in paths:
        print("wrote", os.path.relpath(p))
