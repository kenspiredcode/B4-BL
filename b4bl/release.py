"""Small public surface for the two clocked acoustic profiles.

The older codec and experiment modules remain available for reproducing research,
but their waveforms are not interchangeable with these profiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import clocked, clocked_receiver, compact_clocked, protocol
from .prosody import NEUTRAL


@dataclass
class DecodeOutcome:
    profile: str
    model_revision: str | None
    model_sha256: str | None
    accepted: bool
    integrity_verified: bool
    status: str
    reason: str = ""
    words: list[str] = field(default_factory=list)
    hypothesis: list[str] = field(default_factory=list)
    word_margins: list[float] = field(default_factory=list)
    word_candidates: list[list[tuple[str, float]]] = field(default_factory=list)
    marker_count: int = 0
    frame: protocol.Frame | None = None


def supported_words() -> tuple[str, ...]:
    """Words renderable by both clocked profiles (212 in this revision)."""
    return tuple(sorted(clocked.vocabulary()))


def _validate_words(words):
    words = list(words)
    if not words:
        raise ValueError("at least one concept word is required")
    unsupported = sorted(set(words) - set(clocked.vocabulary()))
    if unsupported:
        raise ValueError(f"unsupported clocked words: {', '.join(unsupported)}")
    return words


def encode_packet(sender: int, recipient: int, msg_type: str, seq: int,
                  payload, prosody=NEUTRAL) -> np.ndarray:
    """Render the unchanged compact-clocked-v2-crc32-base32 waveform."""
    frame = protocol.Frame(sender, recipient, msg_type, seq, _validate_words(payload))
    return compact_clocked.encode(frame, prosody)


def encode_message(words, prosody=NEUTRAL) -> np.ndarray:
    """Render only concept words with clocked acoustic boundary markers."""
    return clocked.encode(_validate_words(words), prosody)


def _model(model: Any):
    if model is None:
        from .models import load_model
        return load_model()
    if not isinstance(model, dict) or model.get("profile") != clocked.PROFILE:
        raise ValueError("requires a clocked-v1 model")
    return model


def _bounded_audio(audio, max_seconds: float):
    samples = np.asarray(audio)
    if (samples.ndim != 1 or not np.issubdtype(samples.dtype, np.number) or
            not np.isfinite(samples).all()):
        raise ValueError("requires finite mono audio")
    if len(samples) == 0 or len(samples) > int(max_seconds * clocked.SR):
        raise ValueError(f"requires a bounded recording of 0–{max_seconds:g} seconds")
    return samples


def decode_packet(audio, model=None) -> DecodeOutcome:
    """Decode a bounded mono recording; accept only after packet CRC32 passes."""
    model = _model(model)
    result = compact_clocked.decode(
        _bounded_audio(audio, 60), model, acoustic_decoder=clocked_receiver.decode,
        top_k=6, beam_width=50000)
    acoustic = result.acoustic
    return DecodeOutcome(
        profile="packet", model_revision=model.get("_b4bl_revision"),
        model_sha256=model.get("_b4bl_sha256"), accepted=result.accepted,
        integrity_verified=result.accepted,
        status="verified" if result.accepted else "rejected",
        reason="" if result.accepted else result.parsed.error or acoustic.reason or "CRC validation failed",
        words=list(acoustic.words), hypothesis=list(acoustic.hypothesis),
        word_margins=list(acoustic.word_margins),
        word_candidates=list(acoustic.word_candidates),
        marker_count=acoustic.marker_count,
        frame=result.parsed.frame if result.accepted else None)


def decode_message(audio, model=None, *, min_margin: float = 0.5) -> DecodeOutcome:
    """Decode a bounded utterance; every result remains unverified without CRC."""
    model = _model(model)
    acoustic = clocked_receiver.decode(
        _bounded_audio(audio, 30), model, min_margin=min_margin)
    return DecodeOutcome(
        profile="message", model_revision=model.get("_b4bl_revision"),
        model_sha256=model.get("_b4bl_sha256"), accepted=acoustic.accepted,
        integrity_verified=False, status="unverified",
        reason=acoustic.reason, words=list(acoustic.words),
        hypothesis=list(acoustic.hypothesis),
        word_margins=list(acoustic.word_margins),
        word_candidates=list(acoustic.word_candidates),
        marker_count=acoustic.marker_count)
