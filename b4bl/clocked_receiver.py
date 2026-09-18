"""Experimental room-tolerant frontend; no playback or recording.

The spectral marker follows the sweep shape without depending on waveform phase.
Keep the historical frontend available for controlled comparisons.
"""
from functools import lru_cache
import numpy as np
from scipy import signal
from . import clocked

PROFILE = 'spectral-marker-v2-hp300-threshold055'
DECIMATION = 3
HOP = 32
FFT_SIZE = 256
# V1 used .65. Real AUKEY captures with clearly audible packets clustered near
# .63 and were therefore rejected wholesale. A .55 threshold recovers the exact
# expected count in those 199 captures and 1,994 prior real captures, with no
# detections in the corresponding 1,994 pre-message background windows.
MARKER_THRESHOLD = .55


def read_audio(path):
    from scipy.io import wavfile
    sr, audio = wavfile.read(path)
    if sr != clocked.SR or audio.ndim != 1:
        raise ValueError('requires mono 44100 Hz capture')
    if np.issubdtype(audio.dtype, np.signedinteger):
        audio = audio.astype(np.float64) / float(2 ** (8 * audio.dtype.itemsize - 1))
    elif np.issubdtype(audio.dtype, np.unsignedinteger):
        midpoint = float(2 ** (8 * audio.dtype.itemsize - 1))
        audio = (audio.astype(np.float64) - midpoint) / midpoint
    return np.asarray(audio, dtype=np.float32)


def preprocess(audio):
    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError('requires finite mono audio')
    if len(audio) < 32:
        return audio.astype(np.float32)
    sos = signal.butter(4, 300, fs=clocked.SR, btype='highpass', output='sos')
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def _spectrum(audio):
    f, _, z = signal.stft(signal.resample_poly(audio, 1, DECIMATION),
                         fs=clocked.SR/DECIMATION, nperseg=FFT_SIZE,
                         noverlap=FFT_SIZE-HOP, boundary='zeros', padded=True)
    z = np.abs(z[(f > 1100) & (f < 3500)])
    return z / np.maximum(np.linalg.norm(z, axis=0), 1e-10)


@lru_cache(maxsize=1)
def _template():
    return _spectrum(clocked.MARKER)[:, 2:-2]


def marker_positions(audio, threshold=MARKER_THRESHOLD):
    audio = np.asarray(audio)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError('requires finite mono audio')
    if len(audio) < len(clocked.MARKER):
        return []
    observed, template = _spectrum(audio), _template()
    scores = sum(signal.correlate(row, target, mode='valid', method='fft')
                 for row, target in zip(observed, template)) / template.shape[1]
    peaks, _ = signal.find_peaks(np.pad(scores, (1, 1)), height=threshold,
                                distance=round(.25*clocked.SR/(DECIMATION*HOP)))
    # Template drops two leading STFT frames; translate to marker onset samples.
    return [int((p-3)*HOP*DECIMATION) for p in peaks if (p-3)*HOP*DECIMATION >= 0]


def decode(audio, model, **kwargs):
    return clocked.decode(preprocess(audio), model, marker_detector=marker_positions, **kwargs)
