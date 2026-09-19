"""Streaming receiver tests with deterministic packet decisions."""
import numpy as np

from b4bl import clocked, compact_clocked, protocol, verified_clocked
from b4bl.runtime_receiver import NoiseFloorTracker, RuntimeConfig, StreamingPacketReceiver


def _packet():
    frame = protocol.Frame(1, 2, "MSG_TELL", 3, ["SELF"])
    words = compact_clocked.to_concepts(frame)
    return words, compact_clocked.encode(frame)


def _accepted_decoder(words):
    def decode(audio, model, **kwargs):
        acoustic = clocked.DecodeResult(
            words=list(words), accepted=True, marker_count=len(words) + 1,
            hypothesis=list(words), word_candidates=[[(w, 0.0)] for w in words])
        return verified_clocked.PacketResult(
            acoustic, compact_clocked.parse_concepts(words))
    return decode


def _rejected_decoder(words):
    def decode(audio, model, **kwargs):
        acoustic = clocked.DecodeResult(
            words=list(words), accepted=False, reason="CRC mismatch",
            marker_count=len(words) + 1,
            hypothesis=list(words), word_candidates=[[(w, 0.0)] for w in words])
        parsed = protocol.ParseResult(None, False, "CRC mismatch", False)
        return verified_clocked.PacketResult(acoustic, parsed)
    return decode


def _feed(receiver, audio, chunk=1633):
    events = []
    for start in range(0, len(audio), chunk):
        events.extend(receiver.feed(audio[start:start + chunk]))
    events.extend(receiver.flush())
    return events


def test_streaming_receiver_confirms_clock_and_emits_success_reply():
    words, packet = _packet()
    rng = np.random.default_rng(9)
    lead = rng.normal(0, .001, int(.8 * clocked.SR))
    audio = np.r_[lead, packet * .01, np.zeros(int(.5 * clocked.SR))].astype(np.float32)
    config = RuntimeConfig(confirm_success=True)
    receiver = StreamingPacketReceiver({}, config, _accepted_decoder(words))
    events = _feed(receiver, audio)
    accepted = [event for event in events if event.accepted]
    assert len(accepted) == 1
    event = accepted[0]
    assert event.marker_count == len(words) + 1
    assert event.reply_concepts == ["ACK"]
    assert event.reply_audio is not None
    assert event.gain > 1
    assert event.decision_latency_seconds < .25


def test_streaming_receiver_waits_then_requests_repeat_after_crc_reject():
    words, packet = _packet()
    audio = np.r_[np.zeros(int(.5 * clocked.SR)), packet,
                  np.zeros(int(.5 * clocked.SR))].astype(np.float32)
    receiver = StreamingPacketReceiver({}, decode_function=_rejected_decoder(words))
    events = _feed(receiver, audio)
    rejected = [event for event in events if event.kind == "rejected"]
    assert len(rejected) == 1
    assert not rejected[0].accepted
    assert rejected[0].reply_concepts == ["SORRY", "REPEAT"]


def test_streaming_receiver_ignores_stationary_tone_and_suppressed_playback():
    t = np.arange(4 * clocked.SR) / clocked.SR
    tone = (.1 * np.sin(2 * np.pi * 1800 * t)).astype(np.float32)
    receiver = StreamingPacketReceiver({})
    assert _feed(receiver, tone) == []

    words, packet = _packet()
    receiver = StreamingPacketReceiver({}, decode_function=_accepted_decoder(words))
    receiver.suppress_for(len(packet) / clocked.SR + 1)
    assert _feed(receiver, packet) == []


def test_single_marker_expires_as_false_trigger_without_spoken_reply():
    audio = np.r_[np.zeros(int(.5 * clocked.SR)), clocked.MARKER,
                  np.zeros(int(.5 * clocked.SR))].astype(np.float32)
    receiver = StreamingPacketReceiver({})
    events = _feed(receiver, audio)
    false = [event for event in events if event.kind == "false_trigger"]
    assert len(false) == 1
    assert false[0].reply_concepts == []
    assert false[0].reply_audio is None


def test_noise_floor_can_recover_after_initial_digital_silence():
    tracker = NoiseFloorTracker(time_constant_seconds=.1)
    tracker.observe(np.zeros(4410))
    tracker.observe(np.full(4410, .01))
    assert tracker.rms > .005
    before = tracker.rms
    tracker.observe(np.full(4410, .2))
    assert tracker.rms == before
