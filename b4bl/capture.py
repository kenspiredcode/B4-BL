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


# Recording MARGIN (seconds) added on both ends of the emission. We do NOT try to
# time playback precisely against variable output latency (AirPlay's buffering
# drifts run to run); instead we record a generously long window that always
# contains the message with margin, and let the sync-chirp find + clip it. Set
# MARGIN comfortably above the worst-case output latency.
REC_MARGIN = 0.5
INPUT_GAIN = 1.0        # software boost for quiet channels (e.g. across-room AirPlay)
INPUT_DEVICE = None     # explicit mic device (name substring or index). CRITICAL for
                        # cross-device: pairing a BT/AirPlay SPEAKER often makes macOS
                        # switch the INPUT to that device's own mic too, so we'd record
                        # from the speaker's tinny mic instead of the Mac mic. Pin the
                        # real mic here.


OUTPUT_DEVICE = None    # explicit playback device; lets a run pick the speaker
                        # (e.g. MacBook Speakers) without the user changing system
                        # settings — essential for unattended overnight runs.


def _resolve_device(spec, kind):
    """Resolve a device name-substring or index to a sounddevice index. kind is
    'input' or 'output'."""
    if spec is None:
        return None
    sd = _sd()
    if isinstance(spec, int):
        return spec
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if d[key] > 0 and spec.lower() in d["name"].lower():
            return i
    return None


def _resolve_input_device(spec):
    return _resolve_device(spec, "input")


def set_channel_profile(latency: float = 0.0, gain: float = 1.0,
                        input_device=None, output_device=None):
    """Tune capture. `latency` sets the recording margin (record long, clip later);
    `gain` boosts quiet channels; `input_device` pins the recording mic so a paired
    speaker doesn't hijack the input; `output_device` pins the playback speaker so
    an unattended run can choose it without changing system settings."""
    global REC_MARGIN, INPUT_GAIN, INPUT_DEVICE, OUTPUT_DEVICE
    REC_MARGIN = max(0.5, latency + 1.0)   # always a full second past worst-case
    INPUT_GAIN = gain
    INPUT_DEVICE = _resolve_device(input_device, "input")
    OUTPUT_DEVICE = _resolve_device(output_device, "output")


def play_and_record(audio: np.ndarray, tail: float = 0.6) -> np.ndarray:
    """Play `audio` and record the mic over a GENEROUSLY long window, then let the
    sync-chirp locate the message (find_message clips it). Recording long and
    clipping afterward is robust to variable/buffered output latency (AirPlay),
    which precise timing is not. Applies INPUT_GAIN for quiet channels."""
    sd = _sd()
    emission = np.concatenate([
        np.zeros(int((LEAD_SIL + REC_MARGIN) * SR), dtype=np.float32),
        SYNC,
        np.zeros(int(0.08 * SR), dtype=np.float32),
        audio.astype(np.float32),
        np.zeros(int((POST_SIL + tail + REC_MARGIN) * SR), dtype=np.float32),
    ])
    # Use playrec (single reliable duplex call). To record from a DIFFERENT device
    # than playback (pinned mic + separate speaker), pass device=(input, output);
    # playrec then splits the duplex across the two devices.
    def _default(idx):
        dd = sd.default.device
        return dd[idx] if isinstance(dd, (list, tuple)) else dd
    if INPUT_DEVICE is not None or OUTPUT_DEVICE is not None:
        in_dev = INPUT_DEVICE if INPUT_DEVICE is not None else _default(0)
        out_dev = OUTPUT_DEVICE if OUTPUT_DEVICE is not None else _default(1)
        rec = sd.playrec(emission, samplerate=SR, channels=1, dtype="float32",
                         device=(in_dev, out_dev))
    else:
        rec = sd.playrec(emission, samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    out = rec[:, 0]
    if INPUT_GAIN != 1.0:
        out = np.clip(out * INPUT_GAIN, -1.0, 1.0)
    return out


def play_and_record_safe(audio: np.ndarray, tail: float = 0.6,
                         timeout: Optional[float] = None) -> Optional[np.ndarray]:
    """play_and_record with a WATCHDOG. sounddevice's playrec has no timeout, so a
    dropped output (AirPlay disconnect) blocks forever. Run it in a thread; if it
    doesn't finish within `timeout`, abort the audio subsystem and return None so
    the caller can skip and continue instead of hanging the whole run.

    Default timeout is generous relative to the emission length + margins."""
    import threading
    sd = _sd()
    if timeout is None:
        emission_len = (LEAD_SIL + REC_MARGIN + 0.26 + len(audio) / SR
                        + POST_SIL + tail + REC_MARGIN)
        timeout = emission_len + 5.0        # emission time + slack
    result = {}

    def worker():
        try:
            result["out"] = play_and_record(audio, tail=tail)
        except Exception as e:  # device error -> treat as a miss
            result["err"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        # the audio call hung (stream dropped). Force-reset PortAudio so the next
        # emission can start fresh; the hung thread is a daemon and won't block exit.
        try:
            sd.stop()
        except Exception:
            pass
        try:
            sd._terminate(); sd._initialize()
        except Exception:
            pass
        return None
    return result.get("out")


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
    # Find the LOUDEST 0.3s window (where the tone landed — the rest is silence /
    # latency padding, which would dilute a whole-recording measurement) and
    # measure level + tonality THERE. Channel-agnostic: works for built-in and for
    # buffered/quiet AirPlay alike.
    w = int(0.3 * SR)
    step = int(0.05 * SR)
    best_i, best_e = 0, 0.0
    for i in range(0, max(1, len(rec) - w), step):
        e = float(np.sqrt(np.mean(rec[i:i + w] ** 2)))
        if e > best_e:
            best_e, best_i = e, i
    chunk = rec[best_i:best_i + w]
    mag = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
    freqs = np.fft.rfftfreq(len(chunk), 1 / SR)
    band = (freqs > 850) & (freqs < 1150)
    ratio = float(np.sum(mag[band]) / (np.sum(mag) + 1e-9))
    noise = float(np.percentile(
        [np.sqrt(np.mean(rec[i:i + w] ** 2)) for i in range(0, max(1, len(rec) - w), step)], 30))
    snr_db = 20 * np.log10((best_e + 1e-9) / (noise + 1e-9))
    print(f"  loudest-window rms={best_e:.4f}, 1kHz ratio={ratio:.3f}, SNR~{snr_db:.0f}dB")
    # Level + SNR are the real health checks. Tonality (ratio) is only a weak
    # sanity bound — AirPlay/codec paths smear a pure tone spectrally, so a low
    # ratio at healthy level+SNR is the channel's character, not a dead mic.
    ok = best_e > 0.003 and snr_db > 12 and ratio > 0.015
    print("  -> channel LIVE" if ok else
          "  -> FAIL: tone not clearly captured (volume? wrong device? AEC?)")
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
