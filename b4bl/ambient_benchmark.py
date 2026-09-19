"""Deterministic ambient-noise splits, characterization, and audio mixing."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from scipy import signal

from . import clocked


PROFILE = "ambient-benchmark-v1"
SPLIT_SEED = "b4bl-ambient-v1"
VALIDATION_FRACTION = .20


def source_split(name, seed=SPLIT_SEED, validation_fraction=VALIDATION_FRACTION):
    """Assign a whole source to development or validation deterministically."""
    digest = hashlib.sha256(f"{seed}\0{name}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if fraction < validation_fraction else "development"


def clip_starts(duration_seconds, clips_per_source=5, clip_seconds=12.0):
    """Return stable, spread-out clip positions excluding source edges."""
    if clips_per_source < 1 or clip_seconds <= 0:
        raise ValueError("invalid clip sampling request")
    usable = duration_seconds - clip_seconds
    if usable <= 0:
        return [0.0]
    low, high = min(5.0, usable / 2), max(min(5.0, usable / 2), usable - 5.0)
    return np.linspace(low, high, clips_per_source + 2)[1:-1].tolist()


def audio_features(audio, sample_rate=clocked.SR):
    """Summarize level and spectral character without classifying content."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not len(audio):
        raise ValueError("audio must be a nonempty vector")
    rms = float(np.sqrt(np.mean(audio * audio)))
    peak = float(np.max(np.abs(audio)))
    frequencies, _, spectrum = signal.stft(
        audio, fs=sample_rate, nperseg=1024, noverlap=512,
        boundary=None, padded=False)
    power = np.abs(spectrum) ** 2 + 1e-15
    column_sum = power.sum(axis=0)
    centroid = float(np.mean((frequencies[:, None] * power).sum(axis=0) / column_sum))
    flatness = float(np.mean(np.exp(np.mean(np.log(power), axis=0)) /
                             np.mean(power, axis=0)))
    band = (frequencies >= 1100) & (frequencies <= 3500)
    marker_band_fraction = float(power[band].sum() / power.sum())
    frames = max(1, round(.05 * sample_rate))
    trimmed = audio[:len(audio) // frames * frames]
    short_rms = np.sqrt(np.mean(trimmed.reshape(-1, frames) ** 2, axis=1)) if len(trimmed) else np.array([rms])
    p10, p90 = np.percentile(short_rms, [10, 90])
    return {
        "rms": rms,
        "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
        "peak": peak,
        "crest_db": 20 * math.log10(max(peak, 1e-12) / max(rms, 1e-12)),
        "spectral_centroid_hz": centroid,
        "spectral_flatness": flatness,
        "marker_band_fraction": marker_band_fraction,
        "short_rms_p90_p10_db": 20 * math.log10(max(p90, 1e-12) / max(p10, 1e-12)),
    }


def mix_at_snr(packet, ambient, snr_db, peak_headroom=.95):
    """Mix packet and equal-length ambient at a requested packet-to-noise SNR."""
    packet = np.asarray(packet, dtype=np.float32)
    ambient = np.asarray(ambient, dtype=np.float32)
    if packet.ndim != 1 or ambient.ndim != 1 or len(packet) != len(ambient):
        raise ValueError("packet and ambient must be equal-length vectors")
    packet_rms = float(np.sqrt(np.mean(packet.astype(float) ** 2)))
    ambient_rms = float(np.sqrt(np.mean(ambient.astype(float) ** 2)))
    if packet_rms <= 1e-12 or ambient_rms <= 1e-12:
        raise ValueError("cannot set SNR for silent audio")
    target_ambient_rms = packet_rms / (10 ** (snr_db / 20))
    scaled_ambient = ambient * (target_ambient_rms / ambient_rms)
    mixed = packet + scaled_ambient
    peak = float(np.max(np.abs(mixed)))
    headroom_scale = min(1.0, peak_headroom / peak) if peak else 1.0
    return (mixed * headroom_scale).astype(np.float32), {
        "requested_snr_db": float(snr_db),
        "packet_rms": packet_rms * headroom_scale,
        "ambient_rms": target_ambient_rms * headroom_scale,
        "headroom_scale": headroom_scale,
    }


def media_files(directory):
    extensions = {".webm", ".m4a", ".mp4", ".mov", ".mkv", ".wav"}
    return sorted(p for p in Path(directory).iterdir()
                  if p.is_file() and p.suffix.lower() in extensions)
