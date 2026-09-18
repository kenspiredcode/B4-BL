"""Experimental CRC-protected packets over clocked-v1; no acoustic ACK transport.

Two leading SYNC concepts identify this version. CRC32 covers the complete
header and payload. Ten fixed decimal checksum digits use existing morphemes.
This intentionally favors inspectability over airtime; it is not a compact modem.
"""
from dataclasses import dataclass
import json
import numpy as np
import zlib
from . import clocked, lexicon, protocol
from .prosody import NEUTRAL


def concept_crc32(words):
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
    return protected + ['CKSUM', 'NUM'] + ['D'+d for d in f'{concept_crc32(protected):010d}']


def parse_concepts(words):
    fail = lambda reason: protocol.ParseResult(None, False, reason, False)
    if len(words) < 14 or words[:2] != ['SYNC', 'SYNC']:
        return fail('not a clocked packet v1')
    if words[-12:-10] != ['CKSUM', 'NUM'] or any(c not in [f'D{i}' for i in range(10)] for c in words[-10:]):
        return fail('invalid CRC suffix')
    expected = int(''.join(c[1:] for c in words[-10:]))
    if expected != concept_crc32(words[:-12]):
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
    candidate_paths_checked: int = 1
    selected_by_validation: bool = False

    @property
    def accepted(self):
        return self.acoustic.accepted and self.parsed.ok and self.parsed.checksum_ok


def _validated_candidate(acoustic, top_k=3, beam_width=10000,
                         packet_parser=parse_concepts, candidate_filter=None):
    """Return the highest-scoring CRC-valid path from bounded word candidates.

    CRC is used only as a final integrity constraint. The beam cap prevents an
    adversarial or very long packet from causing unbounded Cartesian expansion.
    """
    if not acoustic.word_candidates:
        return None, 0
    # Keep paths and scores in separate containers. The former implementation
    # materialized and sorted up to 50,000 growing Python lists at every word.
    # Vectorized score expansion preserves the same stable descending order and
    # beam cap while constructing paths only for survivors.
    paths = [()]
    scores = np.asarray([0.0])
    total = len(acoustic.word_candidates)
    for index, candidates in enumerate(acoustic.word_candidates):
        choices = candidates[:top_k]
        if candidate_filter is not None:
            choices = [(word, score) for word, score in choices
                       if candidate_filter(index, total, word)]
        if not choices:
            return None, 0
        choice_scores = np.asarray([score for _, score in choices], dtype=float)
        expanded = (scores[:, None] + choice_scores[None, :]).ravel()
        order = np.argsort(-expanded, kind='stable')[:beam_width]
        choice_count = len(choices)
        paths = [paths[int(flat_index) // choice_count] +
                 (choices[int(flat_index) % choice_count][0],)
                 for flat_index in order]
        scores = expanded[order]
    checked = 0
    for path in paths:
        checked += 1
        words = list(path)
        parsed = packet_parser(words)
        if parsed.ok and parsed.checksum_ok:
            return (words, parsed), checked
    return None, checked


def decode(audio, model, repetition=1, acoustic_decoder=clocked.decode,
           top_k=3, beam_width=10000, packet_parser=parse_concepts,
           candidate_filter=None):
    # CRC is the acceptance gate. Do not discard a correct whole-packet candidate
    # merely because an individual word has a low heuristic RF margin.
    acoustic = acoustic_decoder(audio, model, repetition=repetition, min_margin=0.0)
    parsed = (packet_parser(acoustic.words) if acoustic.accepted else
              protocol.ParseResult(None, False, acoustic.reason, False))
    checked = 1
    selected = False
    if repetition == 1 and not (parsed.ok and parsed.checksum_ok):
        candidate, checked = _validated_candidate(
            acoustic, top_k, beam_width, packet_parser, candidate_filter)
        if candidate is not None:
            words, parsed = candidate
            selected = words != acoustic.words
            acoustic.words = words
            acoustic.accepted = True
            acoustic.reason = ''
    return PacketResult(acoustic, parsed, checked, selected)
