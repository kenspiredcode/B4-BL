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
# End-of-message silence must be clearly LONGER than any gap WITHIN the message,
# or the trailing-silence trimmer ends the message at the first internal gap. The
# slotted encoding's inter-word gap is ~0.20s, but phoneme decay tails drop below
# the trim threshold early, so an internal gap READS as ~0.36s of quiet. POST_SIL
# is set well above that (quiet_needed = POST_SIL*0.7 ≈ 0.63s of quiet to end).
POST_SIL = 0.90       # silence after the message (marks its end)


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
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = sd.query_devices()
    # argparse supplies numeric IDs as strings. An explicit but invalid selector
    # must never silently switch the capture to a system-default microphone.
    selector = str(spec).strip()
    if selector.lstrip("+-").isdigit():
        index = int(selector)
        if 0 <= index < len(devices) and devices[index][key] > 0:
            return index
        raise ValueError(f"Invalid {kind} audio device index: {spec!r}")
    matches = [i for i, d in enumerate(devices)
               if d[key] > 0 and selector and selector.lower() in d["name"].lower()]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(f"{i}: {d['name']}" for i, d in enumerate(devices) if d[key] > 0)
    problem = "Ambiguous" if matches else "Unknown"
    raise ValueError(f"{problem} {kind} audio device {spec!r}. "
                     f"Use an exact name or index, with straight shell quotes. "
                     f"Available: {available or 'none'}")


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
    # walk forward; stop at the first run of >= POST_SIL*0.7 quiet windows.
    quiet_needed = int((POST_SIL * 0.7) * SR / win)
    # Do NOT start looking for the end until the message has actually BEGUN: after
    # the sync chirp there is a leading gap (and possible sync-tail bleed), and the
    # slotted encoding starts with a phoneme preceded by silence. If we count quiet
    # from the start, the pre-message gap looks like an end and truncates the whole
    # message. So skip to the first loud window, then look for the trailing silence.
    # A brief blip (sync-chirp tail bleeding past the cut) is not the message. Require
    # a SUSTAINED onset: the first window that begins a run of >= 2 loud windows.
    onset = None
    for i in range(len(env) - 1):
        if env[i] >= thr and env[i + 1] >= thr:
            onset = i
            break
    if onset is None:
        return None
    end_win = len(env)
    quiet = 0
    for i in range(onset, len(env)):
        if env[i] < thr:
            quiet += 1
            if quiet >= quiet_needed:
                end_win = i - quiet + 1
                break
        else:
            quiet = 0
    start_samp = onset * win
    end = max(win, end_win * win + win)
    return seg[start_samp:end]


def assess_test_tone(rec):
    """Find a sustained 1 kHz tone instead of assuming the loudest sound is it.

    This is a capture health check, not a message decoder. Narrow-band power
    rejects broadband transients; off-tone windows estimate the noise floor.
    """
    rec = np.asarray(rec, dtype=float)
    w, step = int(.3 * SR), int(.05 * SR)
    if rec.ndim != 1 or len(rec) < w or not np.isfinite(rec).all():
        return dict(ok=False, tone_rms=0., concentration=0., snr_db=0., start_sec=None)
    window = np.hanning(w)
    freqs = np.fft.rfftfreq(w, 1 / SR)
    band = (freqs > 950) & (freqs < 1050)
    rows = []
    for i in range(0, len(rec) - w + 1, step):
        power = np.abs(np.fft.rfft(rec[i:i+w] * window)) ** 2
        tone_power = float(np.sum(power[band]))
        rms = np.sqrt(2 * tone_power / (w * np.sum(window ** 2)))
        concentration = tone_power / (float(np.sum(power)) + 1e-20)
        rows.append((i, float(rms), concentration))
    candidates = [row for row in rows if row[2] >= .5]
    chosen = max(candidates or rows, key=lambda row: row[1])
    noise = float(np.percentile([row[1] for row in rows], 30))
    snr_db = float(20 * np.log10((chosen[1] + 1e-9) / (noise + 1e-9)))
    return dict(ok=bool(chosen[1] > .003 and chosen[2] >= .5 and snr_db > 12),
                tone_rms=chosen[1], concentration=chosen[2], snr_db=snr_db,
                start_sec=chosen[0] / SR)


def self_test() -> bool:
    """Play a tone, record it, confirm the mic captured it (i.e. AEC is NOT
    eating our own output). Returns True if the channel is live."""
    sd = _sd()
    print("self-test: playing a tone and listening for it...")
    tone = 0.6 * np.sin(2 * np.pi * 1000 * np.arange(int(0.4 * SR)) / SR).astype(np.float32)
    rec = play_and_record(tone)
    assessment = assess_test_tone(rec)
    print(f"  tone-band rms={assessment['tone_rms']:.4f}, "
          f"concentration={assessment['concentration']:.3f}, "
          f"tone-band SNR~{assessment['snr_db']:.0f}dB")
    ok = assessment['ok']
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
