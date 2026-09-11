#!/usr/bin/env python3
"""
B4-BL Phase 0 — full astromech PALETTE demo.

The whistle engine nails the sung/whistled family but astromech speech is a whole
palette. This renders each sound FAMILY separately so we can judge which generator
each one wants. Families are grouped by production mechanism, because that dictates
the synth path:

  TONAL generator (pure sine + pitch gesture):
    1. whistle / sweep
    2. chirp / bleep (short, punchy, discrete)
    3. pitch-warble / trill (frequency wobble, stays clean)
    4. sample&hold run (stepped random "computer thinking")
    5. groan (falling + slowing)

  NOISE / ROUGH generator (filtered noise + FM roughness):
    6. scream / alarm (harsh, wide, fast)
    7. raspberry / rude blat (buzzy noise)
    8. static / servo click (non-pitched texture)

Key lesson from the last listen: grit is WRONG on whistles but RIGHT on
screams/raspberries. So roughness is per-family, not global.

No information encoded yet.
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal as sps
import os

SR = 44100
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "audio_samples")
rng = np.random.default_rng(11)


# ---------- shared ----------
def phase_from_freq(f):
    return 2 * np.pi * np.cumsum(f) / SR


def smooth_contour(dur, points):
    n = int(SR * dur)
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
    out[:b[0]] = hz[0]
    return out


def env(n, kind="decay"):
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


# ---------- TONAL generator ----------
def tonal(dur, points, vib=None, envkind="decay"):
    f = smooth_contour(dur, points)
    if vib:
        rate, depth = vib
        t = np.arange(len(f)) / SR
        f = f * (1 + depth * np.sin(2 * np.pi * rate * t))
    sig = np.sin(phase_from_freq(f))
    return sig * env(len(sig), envkind)


def sample_hold_run(dur, step_hz, lo, hi):
    """Stepped random pitches -> the 'computer thinking' burble. Clean tone."""
    n = int(SR * dur)
    steps = max(1, int(dur * step_hz))
    pitches = rng.uniform(lo, hi, steps)
    f = np.repeat(pitches, int(np.ceil(n / steps)))[:n]
    sig = np.sin(phase_from_freq(f))
    # tiny gaps between steps for a staccato machine feel
    seglen = n // steps
    for k in range(steps):
        s = k * seglen
        g = int(0.15 * seglen)
        sig[s:s + g] *= np.linspace(0, 1, g) if g else 1
    return sig * env(n, "even")


def gargle_trill(dur, center, tremolo_hz=45, depth=0.9):
    """Fast AMPLITUDE flutter -> the 'brrrp' roll. Rough by design, on purpose."""
    f = smooth_contour(dur, [(0, center), (1, center * 0.95)])
    sig = np.sin(phase_from_freq(f))
    t = np.arange(len(sig)) / SR
    trem = (1 - depth) + depth * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * tremolo_hz * t)))
    return sig * trem * env(len(sig), "even")


# ---------- NOISE / ROUGH generator ----------
def fm_rough(dur, points, mod_ratio=3.7, mod_index=6.0, envkind="even"):
    """FM with heavy index -> harsh, inharmonic. For screams/alarms."""
    carrier = smooth_contour(dur, points)
    n = len(carrier)
    modf = carrier * mod_ratio
    modph = phase_from_freq(modf)
    ph = 2 * np.pi * np.cumsum(carrier) / SR + mod_index * np.sin(modph)
    sig = np.sin(ph)
    return sig * env(n, envkind)


def raspberry(dur, center=340):
    """Buzzy blat -> rude noise. Raised in pitch + level so it cuts through a montage.
    A gentle downward bend gives it the 'pbbbt' shape."""
    n = int(SR * dur)
    # pitch bends down over the blat
    f = np.linspace(center, center * 0.7, n)
    buzz = sps.sawtooth(phase_from_freq(f), width=0.5)
    noise = rng.uniform(-1, 1, n)
    b, a = sps.butter(2, 2600 / (SR / 2), btype="low")   # brighter than before
    sig = sps.lfilter(b, a, 0.8 * buzz + 0.35 * noise)
    # fast flutter for the buzzy tongue-roll character
    t = np.arange(n) / SR
    sig *= (0.55 + 0.45 * np.sin(2 * np.pi * 42 * t))
    return sig * env(n, "decay") * 1.6   # boost; normalize() caps the montage anyway


def servo_static(dur, center=1400):
    """Intentional mechanical chatter -> like the gargle but noise-based.
    Pitched band of noise with a hard rhythmic gate so it reads as deliberate,
    not as accidental hiss."""
    n = int(SR * dur)
    noise = rng.uniform(-1, 1, n)
    # narrower, higher band -> a definite 'voice' rather than broadband hiss
    w = center / (SR / 2)
    b, a = sps.butter(4, [max(0.02, w * 0.6), min(0.98, w * 1.6)], btype="band")
    sig = sps.lfilter(b, a, noise)
    # hard square gate -> rhythmic mechanical stutter (intentional)
    t = np.arange(n) / SR
    gate = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 30 * t))
    sig *= gate
    return sig * env(n, "even")


def glue(*clips, gap=0.08):
    g = np.zeros(int(SR * gap))
    out = []
    for c in clips:
        out += [c, g]
    return np.concatenate(out)


def normalize(x, peak=0.9):
    m = np.max(np.abs(x)) or 1.0
    return x / m * peak


def save(name, audio):
    os.makedirs(OUTDIR, exist_ok=True)
    wavfile.write(os.path.join(OUTDIR, name), SR, (normalize(audio) * 32767).astype(np.int16))
    return os.path.join(OUTDIR, name)


def build():
    fam = {}
    # 1 whistle
    fam["1_whistle"] = tonal(0.5, [(0, 700), (0.4, 1650), (1, 1200)], vib=(7, 0.03), envkind="even")
    # 2 chirp/bleep — each bip now has fast INTERNAL pitch movement (scoop/bend),
    #   which is what separates R2 chirps from generic flat sci-fi bleeps.
    fam["2_chirp"] = glue(
        tonal(0.12, [(0, 900), (0.35, 1700), (1, 1450)], envkind="stab"),   # scoop up + settle
        tonal(0.10, [(0, 1600), (0.5, 1150), (1, 1350)], envkind="stab"),   # dip
        tonal(0.13, [(0, 1200), (0.4, 2100), (1, 1600)], envkind="stab"),   # bigger scoop
        gap=0.045)
    # 3 pitch-warble/trill (clean)
    fam["3_warble"] = tonal(0.6, [(0, 620), (0.5, 900), (1, 700)], vib=(11, 0.06), envkind="swell")
    # 5 groan (falling + slowing)
    fam["5_groan"] = tonal(0.7, [(0, 900), (0.4, 600), (1, 300)], envkind="decay")
    # 7 raspberry — higher + louder so it reads amongst the others
    fam["7_raspberry"] = raspberry(0.32, center=340)
    # 8 static/servo — now a deliberate rhythmic mechanical chatter (use sparingly)
    fam["8_static"] = servo_static(0.35, center=1400)
    # 9 gargle trill (amplitude flutter, intentional character)
    fam["9_gargle"] = gargle_trill(0.4, 800)

    # DROPPED after listen: 4 (S&H run — generic low-budget sci-fi) and
    #                       6 (FM scream — sounds bad). Kept the functions in the
    #   file as reference but they are no longer built.

    made = []
    for k, v in fam.items():
        made.append(save(f"fam_{k}.wav", v))

    # montage of the surviving palette
    order = ["1_whistle", "2_chirp", "3_warble", "5_groan",
             "9_gargle", "7_raspberry", "8_static"]
    montage = glue(*[fam[k] for k in order], gap=0.18)
    made.append(save("palette_montage.wav", montage))

    # a "sentence" mixing families like real R2 chatter — no dropped families,
    # gargle stands in for the 'busy' feel that the S&H run used to provide.
    sentence = glue(
        fam["2_chirp"], fam["1_whistle"], fam["9_gargle"],
        fam["3_warble"], fam["5_groan"], fam["1_whistle"],
        fam["7_raspberry"],
        gap=0.05)
    made.append(save("palette_sentence.wav", sentence))
    return made


if __name__ == "__main__":
    for p in build():
        print("wrote", os.path.relpath(p, start=os.path.join(os.path.dirname(__file__), "..")))
