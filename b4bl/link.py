"""
B4-BL link layer — ACK / retry over the lossy acoustic channel.

IMPORTANT caveat (measured 2026-09-11): the current file/loopback decoder's
errors are LARGELY SYSTEMATIC — the same clean input decodes to the same wrong
phoneme every time. Retry/voting only recovers RANDOM errors, so against the
in-process channel this layer helps far less than the per-morpheme accuracy
(~80%) would suggest: a 17-morpheme frame rarely decodes perfectly, and repeating
it changes nothing deterministic. Two things make it work in reality:
  1. a better decoder (systematic errors -> fewer errors), and
  2. a real acoustic channel, where noise/reverb/position DO add randomness that
     retries recover, plus SHORTER frames and per-word error correction (todo).
The Sender/Receiver/ACK logic here is correct and is the right transport; it is
just gated on decoder quality. Do not read the in-process delivery rate as the
real-world one.

The acoustic decoder is imperfect. Rather than demand a perfect decoder, the link
layer makes an imperfect one usable, the way every real lossy channel does:

  sender  : speak frame (with checksum) -> retry until ACKed or max tries
  receiver: decode -> parse -> checksum ok? -> ACK(seq) : NACK(seq) (say again)

Because each retransmission is decoded independently and phoneme errors are
random, the probability that at least one of N tries decodes cleanly rises fast
(1 - p^N). A 66%-per-message decoder reaches >95% within a few tries and ~99.9%
within ~7.

This module models the loop against an in-process channel (encode->decode) so we
can measure delivery rate without real hardware. The same Sender/Receiver logic
drives real speakers/mics later — only the `channel` changes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import numpy as np

from . import codec, lexicon as lex, protocol as proto, prosody

# a Channel takes concepts -> heard concepts (lossy). Default: real synth+decode.
Channel = Callable[[List[str]], List[str]]


_JITTER = np.random.default_rng(0)


def acoustic_channel(concepts: List[str], pr: prosody.Prosody = None) -> List[str]:
    """Speak the concepts and decode what was heard — the real lossy path.

    Each call jitters prosody slightly (confidence/urgency wander a little, as a
    real droid's delivery would vary utterance to utterance). This makes each
    retransmission a genuinely independent decode — the reason ACK/retry helps.
    In the real world channel noise/reverb/position supply this variation; here we
    model it at the source so the retry logic can be measured in-process."""
    if pr is None:
        conf = float(np.clip(0.8 + _JITTER.normal(0, 0.18), 0.2, 1.0))
        urg = float(np.clip(0.3 + _JITTER.normal(0, 0.18), 0.0, 1.0))
        pr = prosody.Prosody(confidence=conf, urgency=urg)
    audio = codec.encode(concepts, pr)
    return codec.decode_to_concepts(audio)


@dataclass
class LinkStats:
    delivered: bool = False
    tries: int = 0
    nacks: int = 0


class Receiver:
    """Parses frames, verifies checksum, tracks last good sequence to dedupe."""
    def __init__(self):
        self.last_seq: Optional[int] = None
        self.inbox: List[proto.Frame] = []

    def on_frame_concepts(self, heard: List[str]) -> Tuple[bool, Optional[int]]:
        """Return (ack?, seq). ack True means checksum passed. seq may be None if
        the frame was too garbled to even read a sequence number."""
        res = proto.parse_concepts(heard)
        if not res.ok or res.frame is None:
            return False, None
        seq = res.frame.seq
        if not res.checksum_ok:
            return False, seq
        # good frame — accept once (dedupe retransmissions)
        if seq != self.last_seq:
            self.inbox.append(res.frame)
            self.last_seq = seq
        return True, seq


def send(sender_id: int, recipient_id: int, msg_type: str, seq: int,
         payload: List[str], receiver: Receiver, channel: Channel = acoustic_channel,
         max_tries: int = 8) -> LinkStats:
    """Speak a frame with ACK/retry until acknowledged or max_tries exhausted."""
    frame_concepts = proto.make_frame(sender_id, recipient_id, msg_type, seq, payload)
    stats = LinkStats()
    for attempt in range(1, max_tries + 1):
        stats.tries = attempt
        heard = channel(frame_concepts)
        ack, _seq = receiver.on_frame_concepts(heard)
        if ack:
            stats.delivered = True
            return stats
        stats.nacks += 1
    return stats


# ---------------------------------------------------------------------------
# measurement helper
# ---------------------------------------------------------------------------
def delivery_rate(payloads: List[List[str]], max_tries: int = 8,
                  channel: Channel = acoustic_channel) -> dict:
    """Run send() for each payload; report delivery rate and try-count stats."""
    delivered = 0
    tries = []
    for i, pl in enumerate(payloads):
        rx = Receiver()
        st = send(1, 2, "MSG_TELL", i % 100, pl, rx, channel=channel, max_tries=max_tries)
        delivered += st.delivered
        tries.append(st.tries)
    return {
        "n": len(payloads),
        "delivered": delivered,
        "rate": delivered / max(1, len(payloads)),
        "avg_tries": float(np.mean(tries)) if tries else 0.0,
        "max_tries_used": max(tries) if tries else 0,
    }
