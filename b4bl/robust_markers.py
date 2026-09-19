"""Development marker scorers for interference-tolerant packet framing."""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy import signal

from . import clocked, clocked_receiver


PROFILES = ("spectral", "whitened")


def _magnitude(audio):
    frequency, _, spectrum = signal.stft(
        signal.resample_poly(np.asarray(audio, dtype=float), 1,
                             clocked_receiver.DECIMATION),
        fs=clocked.SR / clocked_receiver.DECIMATION,
        nperseg=clocked_receiver.FFT_SIZE,
        noverlap=clocked_receiver.FFT_SIZE - clocked_receiver.HOP,
        boundary="zeros", padded=True)
    keep = (frequency > 1100) & (frequency < 3500)
    return np.abs(spectrum[keep])


def _normalize_columns(magnitude):
    return magnitude / np.maximum(np.linalg.norm(magnitude, axis=0), 1e-10)


@lru_cache(maxsize=2)
def _template(profile):
    magnitude = _magnitude(clocked.MARKER)
    if profile == "whitened":
        magnitude = np.log1p(20 * magnitude / max(float(np.median(magnitude)), 1e-10))
    return _normalize_columns(magnitude)[:, 2:-2]


def scores(audio, profile="spectral"):
    """Return marker onset samples and continuous template scores."""
    if profile not in PROFILES:
        raise ValueError(f"unknown marker profile {profile!r}")
    magnitude = _magnitude(audio)
    if profile == "whitened":
        # Divide each frequency bin by its typical level over this observation.
        # Stationary music/machinery is suppressed; a moving narrow chirp remains.
        floor = np.percentile(magnitude, 35, axis=1, keepdims=True)
        magnitude = np.log1p(20 * magnitude / np.maximum(floor, 1e-10))
    observed = _normalize_columns(magnitude)
    template = _template(profile)
    if observed.shape[1] < template.shape[1]:
        return np.zeros(0, dtype=int), np.zeros(0)
    values = sum(signal.correlate(row, target, mode="valid", method="fft")
                 for row, target in zip(observed, template)) / template.shape[1]
    frame_positions = np.arange(len(values)) - 2
    samples = frame_positions * clocked_receiver.HOP * clocked_receiver.DECIMATION
    valid = samples >= 0
    return samples[valid].astype(int), values[valid]


def candidates(audio, profile="whitened", threshold=.25, distance_seconds=.20):
    positions, values = scores(audio, profile)
    if not len(values):
        return [], []
    distance = round(distance_seconds * clocked.SR /
                     (clocked_receiver.DECIMATION * clocked_receiver.HOP))
    peaks, properties = signal.find_peaks(values, height=threshold, distance=distance)
    return positions[peaks].tolist(), properties["peak_heights"].astype(float).tolist()


def marker_positions(audio, profile="whitened", threshold=.25):
    return candidates(audio, profile, threshold)[0]


def lattice(audio, profile="whitened", threshold=.15,
            interval_tolerance_seconds=.10, minimum_markers=3):
    """Select the strongest long path whose intervals match legal word clocks."""
    positions, strengths = candidates(audio, profile, threshold, .12)
    if len(positions) < minimum_markers:
        return positions
    positions = np.asarray(positions, dtype=int)
    strengths = np.asarray(strengths, dtype=float)
    legal = np.asarray(sorted({clocked.PREFIX +
                               sum(map(clocked.ticks, sequence)) * clocked.TICK
                               for sequence in clocked.vocabulary().values()}))
    tolerance = interval_tolerance_seconds * clocked.SR
    score = strengths.copy()
    length = np.ones(len(positions), dtype=int)
    previous = np.full(len(positions), -1, dtype=int)
    for end in range(len(positions)):
        intervals = positions[end] - positions[:end]
        if not len(intervals):
            continue
        residual = np.min(np.abs(intervals[:, None] - legal[None, :]), axis=1)
        valid = np.where(residual <= tolerance)[0]
        if not len(valid):
            continue
        # Reward every clock-consistent link more than a marginal peak can score.
        options = score[valid] + 1.0 - residual[valid] / max(tolerance, 1)
        best = valid[int(np.argmax(options))]
        score[end] = strengths[end] + options.max()
        length[end] = length[best] + 1
        previous[end] = best
    eligible = np.where(length >= minimum_markers)[0]
    if not len(eligible):
        return positions.tolist()
    end = eligible[np.argmax(score[eligible] + .5 * length[eligible])]
    path = []
    while end >= 0:
        path.append(int(positions[end]))
        end = previous[end]
    return path[::-1]
