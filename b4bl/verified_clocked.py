"""Experimental CRC-protected packets over clocked-v1; no acoustic ACK transport.

Two leading SYNC concepts identify this version. CRC32 covers the complete
header and payload. Ten fixed decimal checksum digits use existing morphemes.
This intentionally favors inspectability over airtime; it is not a compact modem.
"""
from dataclasses import dataclass
import json
import zlib
from . import clocked, lexicon, protocol
from .prosody import NEUTRAL


def _crc(words):
    return zlib.crc32(json.dumps(words, separators=(',', ':'), ensure_ascii=True).encode())


def to_concepts(frame):
    if not frame.payload or any(w in ('SYNC', 'CKSUM') for w in frame.payload):
        raise ValueError('payload must be nonempty and cannot contain packet markers')
    if frame.payload[0].startswith('D') and frame.payload[0][1:].isdigit():
        raise ValueError('numeric payloads must begin with NUM, not a bare digit')
    if frame.msg_type not in protocol.MSG_TYPES:
        raise ValueError('unknown message type')
    if any(not isinstance(v, int) or v < 0 for v in (frame.sender, frame.recipient, frame.seq)):
        raise ValueError('addresses and sequence must be nonnegative integers')
    if any(not lexicon.is_known(c) for c in frame.payload):
        raise ValueError('unknown concepts are unsupported in clocked packets')
    base = frame.to_concepts()
    protected = ['SYNC'] + base[:base.index('CKSUM')]
    return protected + ['CKSUM', 'NUM'] + ['D'+d for d in f'{_crc(protected):010d}']


def parse_concepts(words):
    fail = lambda reason: protocol.ParseResult(None, False, reason, False)
    if len(words) < 14 or words[:2] != ['SYNC', 'SYNC']:
        return fail('not a clocked packet v1')
    if words[-12:-10] != ['CKSUM', 'NUM'] or any(c not in [f'D{i}' for i in range(10)] for c in words[-10:]):
        return fail('invalid CRC suffix')
    expected = int(''.join(c[1:] for c in words[-10:]))
    if expected != _crc(words[:-12]):
        return fail('CRC mismatch')
    parsed = protocol.parse_concepts(words[1:])
    if not parsed.ok or parsed.frame is None:
        return fail(parsed.error)
    try:
        if to_concepts(parsed.frame) != words:
            return fail('noncanonical or ambiguous packet structure')
    except ValueError as exc:
        return fail(str(exc))
    return protocol.ParseResult(parsed.frame, True, '', True)


def encode(frame, prosody=NEUTRAL, repetition=1):
    return clocked.encode(to_concepts(frame), prosody, repetition)


@dataclass
class PacketResult:
    acoustic: clocked.DecodeResult
    parsed: protocol.ParseResult

    @property
    def accepted(self):
        return self.acoustic.accepted and self.parsed.ok and self.parsed.checksum_ok


def decode(audio, model, repetition=1):
    # CRC is the acceptance gate. Do not discard a correct whole-packet candidate
    # merely because an individual word has a low heuristic RF margin.
    acoustic = clocked.decode(audio, model, repetition, min_margin=0.0)
    parsed = (parse_concepts(acoustic.words) if acoustic.accepted else
              protocol.ParseResult(None, False, acoustic.reason, False))
    return PacketResult(acoustic, parsed)
