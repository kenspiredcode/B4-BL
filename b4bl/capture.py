"""
B4-BL real-world capture loop (Option A: Mac plays AND records).

Produces LABELED real-world audio: for each known message we play it through the
speaker and record the mic simultaneously, tagging the recording with the exact
concept sequence that produced it. That labeled set both (a) tests the current
model against reality and (b) becomes the training set for a hardened V2.

Sync: a distinctive PREAMBLE CHIRP is prepended to every emission. The recorder
finds the chirp and takes what follows as the labeled message — robust to the
record stream starting before/after playback.

AEC caveat (verified, not assumed): macOS echo-cancellation / noise-suppression
only applies when an app opens the mic in VOICE-PROCESSING mode (FaceTime/Zoom).
A plain PortAudio input stream (what sounddevice opens) gets the RAW signal. We do
not trust this — `self_test()` plays a tone and confirms the mic actually captured
it at level before any data is collected, and fails loudly otherwise.
"""

from __future__ import annotations
import os
import json
import time
from typing import List, Optional
import numpy as np

from . import generators as gen
from . import codec, prosody

SR = gen.SR
REC_DIR = os.path.join(os.path.dirname(__file__), "..", "recordings")

# sync chirp: a fast up-sweep, distinctive and easy to detect by correlation.
def _sync_chirp(dur=0.18):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    f = np.linspace(600, 4000, len(t))
    x = 0.9 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    # short fade to avoid clicks
    r = int(0.005 * SR)
    x[:r] *= np.linspace(0, 1, r); x[-r:] *= np.linspace(1, 0, r)
    return x.astype(np.float32)


SYNC = _sync_chirp()
LEAD_SIL = 0.25       # silence before the chirp
POST_SIL = 0.35       # silence after the message (marks its end)


def _sd():
    import sounddevice as sd
    return sd


def play_and_record(audio: np.ndarray, tail: float = 0.6) -> np.ndarray:
    """Play `audio` through the speaker while recording the mic. Returns the mic
    recording (mono float32). Uses a raw input stream (no voice-processing)."""
    sd = _sd()
    emission = np.concatenate([
        np.zeros(int(LEAD_SIL * SR), dtype=np.float32),
        SYNC,
        np.zeros(int(0.08 * SR), dtype=np.float32),
        audio.astype(np.float32),
        np.zeros(int((POST_SIL + tail) * SR), dtype=np.float32),
    ])
    rec = sd.playrec(emission, samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return rec[:, 0]


def find_message(rec: np.ndarray) -> Optional[np.ndarray]:
    """Locate the sync chirp by cross-correlation and return the audio after it
    (trimmed to the trailing silence)."""
    # normalized cross-correlation with the known chirp
    r = rec - np.mean(rec)
    corr = np.correlate(r, SYNC, mode="valid")
    if len(corr) == 0:
        return None
    peak = int(np.argmax(np.abs(corr)))
    start = peak + len(SYNC) + int(0.08 * SR)
    if start >= len(rec):
        return None
    seg = rec[start:]
    # trim trailing region: find the end of the message by locating the long
    # POST_SIL gap (>= ~0.3s of quiet), so room reverb/noise after the message
    # isn't captured as phantom phonemes.
    win = int(0.03 * SR)
    env = np.array([np.sqrt(np.mean(seg[i:i + win] ** 2))
                    for i in range(0, len(seg) - win, win)])
    if len(env) == 0:
        return None
    thr = 0.06 * (np.max(env) or 1.0)
    # walk forward; stop at the first run of >= POST_SIL*0.7 quiet windows
    quiet_needed = int((POST_SIL * 0.7) * SR / win)
    end_win = len(env)
    quiet = 0
    for i, e in enumerate(env):
        if e < thr:
            quiet += 1
            if quiet >= quiet_needed:
                end_win = i - quiet + 1
                break
        else:
            quiet = 0
    end = max(win, end_win * win + win)
    return seg[:end]


def self_test() -> bool:
    """Play a tone, record it, confirm the mic captured it (i.e. AEC is NOT
    eating our own output). Returns True if the channel is live."""
    sd = _sd()
    print("self-test: playing a tone and listening for it...")
    tone = 0.6 * np.sin(2 * np.pi * 1000 * np.arange(int(0.4 * SR)) / SR).astype(np.float32)
    rec = play_and_record(tone)
    # energy around 1kHz in the recording
    w = rec * np.hanning(len(rec))
    mag = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(rec), 1 / SR)
    band = (freqs > 850) & (freqs < 1150)
    ratio = float(np.sum(mag[band]) / (np.sum(mag) + 1e-9))
    rms = float(np.sqrt(np.mean(rec ** 2)))
    print(f"  captured rms={rms:.4f}, 1kHz energy ratio={ratio:.3f}")
    ok = rms > 0.002 and ratio > 0.05
    print("  -> channel LIVE" if ok else
          "  -> FAIL: mic didn't capture the tone (AEC active? wrong device? volume?)")
    return ok


# ---------------------------------------------------------------------------
# dataset collection
# ---------------------------------------------------------------------------
def collect(messages: List[List[str]], out_dir: str = REC_DIR,
            reps: int = 1, pr: Optional[prosody.Prosody] = None) -> str:
    """Play+record each message `reps` times, saving wavs + a manifest.
    `messages` is a list of concept lists. Returns the manifest path."""
    from scipy.io import wavfile
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    idx = 0
    for msg in messages:
        for _ in range(reps):
            audio = codec.encode(msg, pr or prosody.NEUTRAL)
            rec = play_and_record(audio)
            seg = find_message(rec)
            if seg is None:
                print(f"  [skip] no sync found for {msg}")
                continue
            fn = f"{idx:05d}.wav"
            wavfile.write(os.path.join(out_dir, fn), SR,
                          (np.clip(seg, -1, 1) * 32767).astype(np.int16))
            manifest.append({"file": fn, "concepts": msg})
            idx += 1
            time.sleep(0.1)
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"collected {len(manifest)} recordings -> {path}")
    return path
