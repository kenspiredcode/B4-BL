"""Experimental interference-tolerant features for clocked droid phonemes."""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from . import decoder, features
from .generators import SR


PROFILES = ("ridge", "enhanced", "multipitch")
PROFILE_PREFIX = "robust-clock-features-v1:"


def _frames(audio, nperseg=512, hop=128):
    audio = np.asarray(audio, dtype=float)
    if len(audio) < nperseg:
        audio = np.pad(audio, (0, nperseg - len(audio)))
    frequency, _, spectrum = signal.stft(
        audio, fs=SR, window="hann", nperseg=nperseg,
        noverlap=nperseg - hop, boundary="zeros", padded=True)
    return frequency, spectrum


def _resample(values, count):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.zeros(count)
    if len(values) == 1:
        return np.full(count, values[0])
    return np.interp(np.linspace(0, 1, count),
                     np.linspace(0, 1, len(values)), values)


def _envelope(audio, count=12):
    chunks = np.array_split(np.asarray(audio, dtype=float), count)
    rms = np.asarray([np.sqrt(np.mean(x * x)) if len(x) else 0 for x in chunks])
    return rms / max(float(rms.max()), 1e-9)


def _ridge_track(audio):
    """Track a smooth spectral ridge across the droid's usable pitch region."""
    frequency, spectrum = _frames(audio)
    magnitude = np.abs(spectrum)
    keep = (frequency >= 150) & (frequency <= 3800)
    freq = frequency[keep]
    mag = magnitude[keep]
    if not mag.shape[1]:
        return np.zeros(1), np.zeros(1), 0.0
    # Local contrast suppresses broad speech/machinery energy while retaining a
    # narrow pure-tone ridge. The transition cost prevents jumping to unrelated
    # foreground partials from one frame to the next.
    baseline = median_filter(mag, size=(9, 1), mode="nearest") + 1e-10
    score = np.log1p(mag / baseline)
    logfreq = np.log(freq)
    paths = np.zeros_like(score, dtype=np.float32)
    back = np.zeros_like(score, dtype=np.int16)
    paths[:, 0] = score[:, 0]
    transition = 3.0 * np.abs(logfreq[:, None] - logfreq[None, :])
    for column in range(1, score.shape[1]):
        candidates = paths[:, column - 1][None, :] - transition
        back[:, column] = np.argmax(candidates, axis=1)
        paths[:, column] = score[:, column] + np.max(candidates, axis=1)
    indices = np.zeros(score.shape[1], dtype=int)
    indices[-1] = int(np.argmax(paths[:, -1]))
    for column in range(score.shape[1] - 1, 0, -1):
        indices[column - 1] = back[indices[column], column]
    confidence = score[indices, np.arange(score.shape[1])]
    return freq[indices], confidence, float(np.mean(score))


def ridge(audio):
    track, confidence, mean_score = _ridge_track(audio)
    log_track = np.log(np.maximum(track, 1))
    center = float(np.exp(np.median(log_track)))
    shape = _resample(log_track, 16)
    shape -= np.mean(shape)
    scale = max(float(np.max(np.abs(shape))), 1e-9)
    shape /= scale
    conf = _resample(confidence, 8)
    conf /= max(float(np.max(conf)), 1e-9)
    base = features.extract(np.asarray(audio), min_periodicity=0.0)
    return np.concatenate([
        shape,
        [np.log(center + 1e-6), float(np.ptp(log_track)),
         float(np.mean(confidence)), mean_score, len(audio) / SR],
        conf, base[-8:], _envelope(audio),
    ]).astype(np.float32)


def _subtract_stationary_noise(audio):
    frequency, spectrum = _frames(audio)
    magnitude = np.abs(spectrum)
    phase = np.exp(1j * np.angle(spectrum))
    # A low percentile over time is a conservative in-window noise estimate.
    noise = np.percentile(magnitude, 20, axis=1, keepdims=True)
    cleaned = np.maximum(magnitude - 1.25 * noise, .08 * magnitude) * phase
    _, restored = signal.istft(cleaned, fs=SR, window="hann", nperseg=512,
                               noverlap=384, input_onesided=True, boundary=True)
    if len(restored) < len(audio):
        restored = np.pad(restored, (0, len(audio) - len(restored)))
    return restored[:len(audio)].astype(np.float32)


def enhanced(audio):
    cleaned = _subtract_stationary_noise(audio)
    core = features.extract(cleaned, min_periodicity=.25)
    # Retain original-domain texture cues because subtraction can make rasp and
    # gargle artificially tonal.
    original = features.extract(np.asarray(audio), min_periodicity=0.0)
    return np.concatenate([core, original[-8:], _envelope(cleaned),
                           _envelope(audio)]).astype(np.float32)


def multipitch(audio):
    frequency, spectrum = _frames(audio)
    power = np.abs(spectrum) ** 2 + 1e-12
    edges = np.geomspace(150, 4000, 19)
    bands = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (frequency >= low) & (frequency < high)
        bands.append(power[selected].sum(axis=0))
    band_power = np.asarray(bands)
    # Normalize each instant to preserve simultaneous alternatives and suppress
    # overall level. Log compression prevents a single loud interferer from
    # erasing a weaker droid ridge.
    band_power /= np.maximum(band_power.sum(axis=0, keepdims=True), 1e-12)
    compressed = np.log1p(100 * band_power)
    time_grid = np.vstack([_resample(row, 8) for row in compressed]).ravel()
    # Sort, rather than select, the top alternatives at each coarse time point.
    coarse = np.vstack([_resample(row, 8) for row in band_power])
    top = np.sort(coarse, axis=0)[-3:][::-1].ravel()
    return np.concatenate([time_grid, top, _envelope(audio),
                           [len(audio) / SR, decoder._spectral_flatness(audio),
                            decoder._am_depth(audio)]]).astype(np.float32)


def extract(audio, profile):
    if profile == "ridge":
        return ridge(audio)
    if profile == "enhanced":
        return enhanced(audio)
    if profile == "multipitch":
        return multipitch(audio)
    raise ValueError(f"unknown robust feature profile {profile!r}")
