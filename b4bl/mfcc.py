"""
B4-BL — MFCC features (compact, no heavy dependencies; scipy + numpy only).

MFCCs describe the spectral SHAPE of a sound — the standard speech feature,
much better at separating timbres (tone vs gargle vs rasp) than raw band energies.
We also compute the log-energy and can append deltas (how coefficients change
over time) elsewhere to capture contour movement.

Implementation: pre-emphasis -> power spectrum -> mel filterbank -> log -> DCT.
"""

from __future__ import annotations
import numpy as np
from . import generators as gen

SR = gen.SR
N_MFCC = 13
N_MELS = 26
FMIN, FMAX = 80.0, 8000.0


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


_FILTERBANK_CACHE = {}


def _mel_filterbank(n_fft):
    key = n_fft
    if key in _FILTERBANK_CACHE:
        return _FILTERBANK_CACHE[key]
    n_bins = n_fft // 2 + 1
    mel_pts = np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / SR).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_bins - 1)
    fb = np.zeros((N_MELS, n_bins), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        l, c, r = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if c > l:
            fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    _FILTERBANK_CACHE[key] = fb
    return fb


def _dct_matrix(n_out, n_in):
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    D = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in)) * np.sqrt(2.0 / n_in)
    D[0] *= 1 / np.sqrt(2)
    return D.astype(np.float32)


def mfcc(frame: np.ndarray) -> np.ndarray:
    """MFCC vector (N_MFCC coeffs) + log-energy for one audio frame."""
    x = frame.astype(np.float32)
    if len(x) < 8:
        return np.zeros(N_MFCC + 1, dtype=np.float32)
    # pre-emphasis
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    x = x * np.hanning(len(x))
    n_fft = 1 << int(np.ceil(np.log2(len(x))))
    spec = np.abs(np.fft.rfft(x, n_fft)) ** 2
    fb = _mel_filterbank(n_fft)
    mel_e = fb @ spec
    log_mel = np.log(mel_e + 1e-9)
    D = _dct_matrix(N_MFCC, N_MELS)
    coeffs = D @ log_mel
    log_energy = np.log(np.sum(x ** 2) + 1e-9)
    return np.concatenate([coeffs, [log_energy]]).astype(np.float32)


MFCC_LEN = N_MFCC + 1
