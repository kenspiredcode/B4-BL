"""Streaming compact-packet receiver with no audio-device dependency.

The runtime keeps a bounded rolling buffer, detects the existing clock marker in
small audio chunks, confirms a plausible marker cadence before waking the packet
decoder, freezes one software gain for the packet, and emits an optional spoken
reply.  Device capture and playback are deliberately left to the application so
the same state machine can be exercised against files, desktop microphones, or
future robot hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable, Optional

import numpy as np

from . import clocked, clocked_receiver, compact_clocked


DecodeFunction = Callable[..., object]


@dataclass(frozen=True)
class RuntimeConfig:
    """Policy and timing for :class:`StreamingPacketReceiver`."""

    sample_rate: int = clocked.SR
    analysis_block_ms: float = 100.0
    scan_window_seconds: float = 1.2
    marker_commit_seconds: float = 0.17
    marker_edge_guard_seconds: float = 0.06
    interval_tolerance_seconds: float = 0.10
    end_timeout_seconds: float = 2.30
    # A full second of acoustic context keeps the zero-phase receiver filter and
    # STFT marker locations stable relative to batch decoding. Short 0.2-second
    # slices changed three borderline decisions in the 200-file replay.
    pre_roll_seconds: float = 1.00
    post_roll_seconds: float = 0.04
    max_packet_seconds: float = 60.0
    noise_time_constant_seconds: float = 4.0
    noise_update_gate: float = 2.5
    min_marker_snr_db: Optional[float] = 3.0
    marker_threshold: float = clocked_receiver.MARKER_THRESHOLD
    target_marker_rms: float = 0.031
    min_gain: float = 0.25
    max_gain: float = 8.0
    marker_peak_headroom: float = 0.75
    top_k: int = 6
    beam_width: int = 50000
    confirm_success: bool = False
    request_repeat: bool = True
    require_sync_start: bool = True

    def __post_init__(self):
        if self.sample_rate != clocked.SR:
            raise ValueError(f"runtime currently requires {clocked.SR} Hz audio")
        if self.analysis_block_ms <= 0 or self.scan_window_seconds <= 0:
            raise ValueError("analysis block and scan window must be positive")
        if self.end_timeout_seconds <= 0 or self.max_packet_seconds <= 0:
            raise ValueError("packet timeouts must be positive")
        if not 1 <= self.top_k <= 10 or self.beam_width < 1:
            raise ValueError("invalid list-decoder bounds")
        if not 0 < self.min_gain <= self.max_gain:
            raise ValueError("gain bounds must be positive and ordered")
        if not 0 < self.marker_threshold <= 1:
            raise ValueError("marker threshold must be in (0, 1]")


@dataclass
class RuntimeEvent:
    kind: str
    start_sample: int
    end_sample: int
    emitted_sample: int
    marker_count: int
    noise_rms: float
    marker_rms: float
    marker_snr_db: float
    gain: float
    processing_seconds: float = 0.0
    result: object | None = None
    reply_concepts: list[str] = field(default_factory=list)
    reply_audio: np.ndarray | None = None
    detail: str = ""

    @property
    def accepted(self) -> bool:
        return bool(self.result is not None and getattr(self.result, "accepted", False))

    @property
    def decision_latency_seconds(self) -> float:
        return max(0.0, (self.emitted_sample - self.end_sample) / clocked.SR)

    def summary(self) -> dict:
        parsed = getattr(self.result, "parsed", None)
        acoustic = getattr(self.result, "acoustic", None)
        return {
            "kind": self.kind,
            "accepted": self.accepted,
            "start_seconds": self.start_sample / clocked.SR,
            "end_seconds": self.end_sample / clocked.SR,
            "emitted_seconds": self.emitted_sample / clocked.SR,
            "decision_latency_seconds": self.decision_latency_seconds,
            "processing_seconds": self.processing_seconds,
            "marker_count": self.marker_count,
            "noise_rms": self.noise_rms,
            "marker_rms": self.marker_rms,
            "marker_snr_db": self.marker_snr_db,
            "gain": self.gain,
            "parse_error": getattr(parsed, "error", "") if parsed else "",
            "decoded_words": list(getattr(acoustic, "words", [])) if acoustic else [],
            "reply_concepts": list(self.reply_concepts),
            "detail": self.detail,
        }


class NoiseFloorTracker:
    """Slow idle-only RMS estimate that ignores sudden foreground energy."""

    def __init__(self, sample_rate=clocked.SR, time_constant_seconds=4.0,
                 update_gate=2.5):
        self.sample_rate = sample_rate
        self.time_constant_seconds = time_constant_seconds
        self.update_gate = update_gate
        self.rms = 0.0
        self.observations = 0

    def observe(self, audio):
        audio = np.asarray(audio, dtype=float)
        if not len(audio):
            return self.rms
        level = float(np.sqrt(np.mean(audio * audio)))
        if not math.isfinite(level):
            return self.rms
        # Do not let an initial digital-silence block pin the estimate forever;
        # once a real floor exists, reject sudden foreground energy.
        if (self.observations and self.rms > 1e-6 and
                level > self.rms * self.update_gate):
            return self.rms
        alpha = 1.0 - math.exp(-len(audio) /
                               (self.sample_rate * self.time_constant_seconds))
        if not self.observations:
            self.rms = max(level, 1e-9)
        else:
            self.rms = (1.0 - alpha) * self.rms + alpha * level
        self.observations += 1
        return self.rms


class StreamingPacketReceiver:
    """Incremental marker detector, packet buffer, decoder, and reply policy.

    ``feed`` accepts arbitrary-length mono chunks and returns zero or more
    :class:`RuntimeEvent` objects. The receiver only wakes after two markers have
    a legal clock interval. It attempts CRC-verified decoding at each possible
    closing marker, which avoids waiting for a long silence after a valid packet.
    """

    def __init__(self, model, config: RuntimeConfig | None = None,
                 decode_function: DecodeFunction = compact_clocked.decode):
        self.model = model
        self.config = config or RuntimeConfig()
        self.decode_function = decode_function
        self.noise = NoiseFloorTracker(
            self.config.sample_rate, self.config.noise_time_constant_seconds,
            self.config.noise_update_gate)
        vocab = clocked.vocabulary()
        word_ticks = [sum(map(clocked.ticks, seq)) for seq in vocab.values()]
        self._min_interval = clocked.PREFIX + min(word_ticks) * clocked.TICK
        self._max_interval = clocked.PREFIX + max(word_ticks) * clocked.TICK
        self._sync_interval = (clocked.PREFIX +
                               sum(map(clocked.ticks, vocab['SYNC'])) * clocked.TICK)
        self._block = max(1, round(self.config.analysis_block_ms / 1000 * clocked.SR))
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start = 0
        self._total = 0
        self._last_seen_marker = -10**12
        self._suppressed_until = 0
        self._state = "idle"
        self._markers: list[int] = []
        self._confirmed_sample: int | None = None
        self._gain = 1.0
        self._marker_rms = 0.0
        self._marker_snr_db = float("inf")
        self._last_decode_marker_count = 0
        self._last_result = None

    @property
    def sample_cursor(self):
        return self._total

    @property
    def state(self):
        return self._state

    def suppress_for(self, seconds: float):
        """Ignore markers during application playback to prevent self-wakes."""
        if seconds < 0:
            raise ValueError("suppression duration must be nonnegative")
        self._suppressed_until = max(
            self._suppressed_until, self._total + round(seconds * clocked.SR))
        self._reset_packet()

    def feed(self, audio) -> list[RuntimeEvent]:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("feed requires finite mono audio")
        events = []
        for start in range(0, len(audio), self._block):
            events.extend(self._feed_block(audio[start:start + self._block]))
        return events

    def flush(self) -> list[RuntimeEvent]:
        """Advance through enough silence to close a pending packet/candidate."""
        duration = self.config.end_timeout_seconds + self.config.marker_commit_seconds
        return self.feed(np.zeros(round(duration * clocked.SR), dtype=np.float32))

    def _feed_block(self, block):
        if not len(block):
            return []
        self._buffer = np.concatenate([self._buffer, block])
        self._total += len(block)
        if self._state == "idle" and self._total >= self._suppressed_until:
            self.noise.observe(block)
        events = []
        for marker in self._new_markers():
            if marker < self._suppressed_until:
                continue
            events.extend(self._on_marker(marker))
        events.extend(self._check_timeout())
        self._trim_buffer()
        return events

    def _new_markers(self):
        scan_samples = round(self.config.scan_window_seconds * clocked.SR)
        scan_start = max(self._buffer_start, self._total - scan_samples)
        local_start = scan_start - self._buffer_start
        window = self._buffer[local_start:]
        if len(window) < len(clocked.MARKER):
            return []
        processed = clocked_receiver.preprocess(window)
        found = clocked_receiver.marker_positions(
            processed, threshold=self.config.marker_threshold)
        cutoff = self._total - round(self.config.marker_commit_seconds * clocked.SR)
        edge = scan_start + round(self.config.marker_edge_guard_seconds * clocked.SR)
        spacing = round(.20 * clocked.SR)
        output = []
        for position in found:
            absolute = scan_start + position
            if absolute > cutoff or absolute < edge:
                continue
            if absolute <= self._last_seen_marker + spacing:
                continue
            self._last_seen_marker = absolute
            output.append(absolute)
        return output

    def _on_marker(self, marker):
        if self._state == "idle":
            if not self._begin_candidate(marker):
                return []
            return []

        interval = marker - self._markers[-1]
        tolerance = round(self.config.interval_tolerance_seconds * clocked.SR)
        # Compact packets always begin with SYNC. An isolated ambient chirp just
        # before a packet must not pair with its first real marker and shift the
        # entire clock sequence. Roll the candidate forward silently until the
        # first interval has the known SYNC duration.
        if (self._state == "candidate" and self.config.require_sync_start and
                abs(interval - self._sync_interval) > tolerance):
            self._begin_candidate(marker)
            return []
        if interval < self._min_interval - tolerance:
            return []
        if interval > self._max_interval + tolerance:
            event = self._abandon_event("marker interval exceeded packet clock")
            self._begin_candidate(marker)
            return [event]

        self._markers.append(marker)
        if self._state == "candidate":
            self._state = "receiving"
            self._confirmed_sample = self._total

        if len(self._markers) < compact_clocked.MIN_PACKET_WORDS + 1:
            return []
        result, elapsed = self._decode_through(marker)
        self._last_result = result
        self._last_decode_marker_count = len(self._markers)
        if getattr(result, "accepted", False):
            event = self._result_event(result, elapsed, "accepted")
            self._reset_packet()
            return [event]
        return []

    def _begin_candidate(self, marker):
        self._markers = [marker]
        self._state = "candidate"
        self._confirmed_sample = None
        self._last_decode_marker_count = 0
        self._last_result = None
        marker_audio = self._slice(marker, marker + len(clocked.MARKER))
        if len(marker_audio) < len(clocked.MARKER):
            self._reset_packet()
            return False
        self._marker_rms = float(np.sqrt(np.mean(marker_audio.astype(float) ** 2)))
        noise = max(self.noise.rms, 1e-9)
        self._marker_snr_db = 20 * math.log10(max(self._marker_rms, 1e-9) / noise)
        if (self.config.min_marker_snr_db is not None and self.noise.observations and
                self._marker_snr_db < self.config.min_marker_snr_db):
            self._reset_packet()
            return False
        desired = self.config.target_marker_rms / max(self._marker_rms, 1e-9)
        peak = float(np.max(np.abs(marker_audio)))
        if peak:
            desired = min(desired, self.config.marker_peak_headroom / peak)
        self._gain = float(np.clip(desired, self.config.min_gain, self.config.max_gain))
        return True

    def _decode_through(self, closing_marker):
        start = self._markers[0] - round(self.config.pre_roll_seconds * clocked.SR)
        end = closing_marker + len(clocked.MARKER) + round(
            self.config.post_roll_seconds * clocked.SR)
        packet = self._slice(start, min(end, self._total)) * self._gain
        began = time.monotonic()
        result = self.decode_function(
            packet, self.model, acoustic_decoder=clocked_receiver.decode,
            top_k=self.config.top_k, beam_width=self.config.beam_width)
        return result, time.monotonic() - began

    def _check_timeout(self):
        if self._state == "idle" or not self._markers:
            return []
        age = self._total - self._markers[-1]
        total_age = self._total - self._markers[0]
        if (age < round(self.config.end_timeout_seconds * clocked.SR) and
                total_age < round(self.config.max_packet_seconds * clocked.SR)):
            return []
        if len(self._markers) >= compact_clocked.MIN_PACKET_WORDS + 1:
            if self._last_decode_marker_count == len(self._markers):
                result, elapsed = self._last_result, 0.0
            else:
                result, elapsed = self._decode_through(self._markers[-1])
            event = self._result_event(result, elapsed, "rejected")
        else:
            kind = "incomplete" if len(self._markers) >= 2 else "false_trigger"
            event = self._simple_event(kind, "candidate ended before a complete packet")
        self._reset_packet()
        return [event]

    def _result_event(self, result, elapsed, kind):
        reply = compact_clocked.spoken_reply_concepts(
            result, confirm_success=self.config.confirm_success,
            request_repeat=self.config.request_repeat)
        reply_audio = compact_clocked.encode_spoken_reply(
            result, confirm_success=self.config.confirm_success,
            request_repeat=self.config.request_repeat)
        return RuntimeEvent(
            kind=kind, start_sample=self._markers[0],
            end_sample=self._markers[-1] + len(clocked.MARKER),
            emitted_sample=self._total, marker_count=len(self._markers),
            noise_rms=self.noise.rms, marker_rms=self._marker_rms,
            marker_snr_db=self._marker_snr_db, gain=self._gain,
            processing_seconds=elapsed, result=result,
            reply_concepts=reply, reply_audio=reply_audio)

    def _simple_event(self, kind, detail):
        end = self._markers[-1] + len(clocked.MARKER) if self._markers else self._total
        return RuntimeEvent(
            kind=kind, start_sample=self._markers[0] if self._markers else self._total,
            end_sample=end, emitted_sample=self._total,
            marker_count=len(self._markers), noise_rms=self.noise.rms,
            marker_rms=self._marker_rms, marker_snr_db=self._marker_snr_db,
            gain=self._gain, detail=detail)

    def _abandon_event(self, detail):
        kind = "incomplete" if len(self._markers) >= 2 else "false_trigger"
        event = self._simple_event(kind, detail)
        self._reset_packet()
        return event

    def _slice(self, start, end):
        start = max(start, self._buffer_start)
        end = min(end, self._total)
        if end <= start:
            return np.zeros(0, dtype=np.float32)
        return self._buffer[start - self._buffer_start:end - self._buffer_start]

    def _trim_buffer(self):
        if self._state == "idle":
            keep_from = self._total - round((self.config.scan_window_seconds +
                                             self.config.pre_roll_seconds) * clocked.SR)
        else:
            keep_from = self._markers[0] - round(self.config.pre_roll_seconds * clocked.SR)
        drop = max(0, keep_from - self._buffer_start)
        if drop:
            self._buffer = self._buffer[drop:]
            self._buffer_start += drop

    def _reset_packet(self):
        self._state = "idle"
        self._markers = []
        self._confirmed_sample = None
        self._gain = 1.0
        self._marker_rms = 0.0
        self._marker_snr_db = float("inf")
        self._last_decode_marker_count = 0
        self._last_result = None
