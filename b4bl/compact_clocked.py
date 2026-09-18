"""Compact CRC32-protected packets over the clocked acoustic transport.

Version 2 replaces spoken decimal metadata with a fixed base-32 alphabet made
from existing short morphemes. A local network has 32 addresses and 1,024
sequence values. Seven base-32 symbols carry the complete 32-bit CRC; the first
has only two significant bits. CRC32 covers the version, complete header, and
payload. No new sounds or classifier labels are introduced.

Wire format::

    SYNC D2  sender recipient msg-type seq-hi seq-lo  payload...  CKSUM crc[7]

The fixed fields and trailer make the format canonical. CRC is an integrity
gate and list-selection constraint, not error correction or authentication.
"""
from __future__ import annotations

from . import clocked, lexicon, protocol, verified_clocked
from .prosody import NEUTRAL


VERSION = ('SYNC', 'D2')
PROFILE = 'compact-clocked-v2-crc32-base32'
ADDRESS_SYMBOLS = 1
SEQUENCE_SYMBOLS = 2
CRC_SYMBOLS = 7
MIN_PACKET_WORDS = (len(VERSION) + ADDRESS_SYMBOLS * 2 + 1 +
                    SEQUENCE_SYMBOLS + 1 + 1 + CRC_SYMBOLS)
CONFIRM_REPLY = ('ACK',)
REPEAT_REPLY = ('SORRY', 'REPEAT')

# Wire constants: append-only/versioned if the vocabulary later changes. These
# are the 32 shortest clocked-v1 concepts available when v2 was defined, after
# excluding protocol structure and message types. Do not generate this list at
# runtime: its order assigns the on-air numeric values.
ALPHABET = (
    'ACK', 'ALL', 'BRING', 'CHILD_GIRL', 'COME', 'DENY', 'GO', 'MOVE',
    'NONE', 'OTHER_BOT', 'PARENT_M', 'REPEAT', 'SELF', 'STOP', 'TAKE',
    'THAT', 'YOU', 'CHILD_BOY', 'FOLLOW', 'GUEST', 'IT', 'PARENT_F',
    'THIS', 'WAIT', 'WE', 'ASK', 'CLARIFY', 'CLEAN', 'DOCK', 'ERROR',
    'FIND', 'FINISH',
)
_VALUE = {word: value for value, word in enumerate(ALPHABET)}
_BASE = len(ALPHABET)
MAX_ADDRESS = _BASE**ADDRESS_SYMBOLS - 1
MAX_SEQUENCE = _BASE**SEQUENCE_SYMBOLS - 1


def _encode_uint(value, width):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError('compact packet integers must be nonnegative integers')
    if value >= _BASE**width:
        raise ValueError(f'value {value} does not fit in {width} base-32 symbol(s)')
    out = [ALPHABET[0]] * width
    for index in range(width - 1, -1, -1):
        out[index] = ALPHABET[value % _BASE]
        value //= _BASE
    return out


def _decode_uint(words):
    value = 0
    for word in words:
        if word not in _VALUE:
            return None
        value = value * _BASE + _VALUE[word]
    return value


def _validate_frame(frame):
    if not frame.payload or any(w in ('SYNC', 'CKSUM') for w in frame.payload):
        raise ValueError('payload must be nonempty and cannot contain packet markers')
    if frame.payload[0].startswith('D') and frame.payload[0][1:].isdigit():
        raise ValueError('numeric payloads must begin with NUM, not a bare digit')
    if frame.msg_type not in protocol.MSG_TYPES:
        raise ValueError('unknown message type')
    if (not isinstance(frame.sender, int) or isinstance(frame.sender, bool) or
            not 0 <= frame.sender <= MAX_ADDRESS):
        raise ValueError(f'sender must be an integer from 0 to {MAX_ADDRESS}')
    if (not isinstance(frame.recipient, int) or isinstance(frame.recipient, bool) or
            not 0 <= frame.recipient <= MAX_ADDRESS):
        raise ValueError(f'recipient must be an integer from 0 to {MAX_ADDRESS}')
    if (not isinstance(frame.seq, int) or isinstance(frame.seq, bool) or
            not 0 <= frame.seq <= MAX_SEQUENCE):
        raise ValueError(f'sequence must be an integer from 0 to {MAX_SEQUENCE}')
    supported = clocked.vocabulary()
    if any(word not in supported for word in frame.payload):
        raise ValueError('unknown or unsupported concepts in compact packet payload')


def to_concepts(frame):
    _validate_frame(frame)
    protected = list(VERSION)
    protected += _encode_uint(frame.sender, ADDRESS_SYMBOLS)
    protected += _encode_uint(frame.recipient, ADDRESS_SYMBOLS)
    protected += [frame.msg_type]
    protected += _encode_uint(frame.seq, SEQUENCE_SYMBOLS)
    protected += list(frame.payload)
    checksum = verified_clocked.concept_crc32(protected)
    return protected + ['CKSUM'] + _encode_uint(checksum, CRC_SYMBOLS)


def parse_concepts(words):
    fail = lambda reason: protocol.ParseResult(None, False, reason, False)
    if len(words) < MIN_PACKET_WORDS or tuple(words[:len(VERSION)]) != VERSION:
        return fail('not a compact clocked packet v2')
    checksum_index = len(words) - CRC_SYMBOLS - 1
    if words[checksum_index] != 'CKSUM':
        return fail('invalid CRC suffix')
    checksum = _decode_uint(words[-CRC_SYMBOLS:])
    if checksum is None or checksum > 0xffffffff:
        return fail('invalid CRC symbols')
    protected = words[:checksum_index]
    if checksum != verified_clocked.concept_crc32(protected):
        return fail('CRC mismatch')

    index = len(VERSION)
    sender = _decode_uint(words[index:index + ADDRESS_SYMBOLS]); index += ADDRESS_SYMBOLS
    recipient = _decode_uint(words[index:index + ADDRESS_SYMBOLS]); index += ADDRESS_SYMBOLS
    msg_type = words[index] if index < checksum_index else None; index += 1
    seq = _decode_uint(words[index:index + SEQUENCE_SYMBOLS]); index += SEQUENCE_SYMBOLS
    payload = list(words[index:checksum_index])
    if sender is None or recipient is None or seq is None:
        return fail('invalid compact header')
    frame = protocol.Frame(sender, recipient, msg_type, seq, payload)
    try:
        if to_concepts(frame) != list(words):
            return fail('noncanonical or ambiguous compact packet structure')
    except ValueError as exc:
        return fail(str(exc))
    return protocol.ParseResult(frame, True, '', True)


def encode(frame, prosody=NEUTRAL, repetition=1):
    return clocked.encode(to_concepts(frame), prosody, repetition)


def _candidate_filter(index, total, word):
    """Prune impossible list paths before the bounded beam expands them."""
    checksum_index = total - CRC_SYMBOLS - 1
    if index < len(VERSION):
        return word == VERSION[index]
    if index in (2, 3, 5, 6):
        return word in _VALUE
    if index == 4:
        return word in protocol.MSG_TYPES
    if index == checksum_index:
        return word == 'CKSUM'
    if index > checksum_index:
        return word in _VALUE
    return word not in ('SYNC', 'CKSUM')


def decode(audio, model, repetition=1, acoustic_decoder=clocked.decode,
           top_k=5, beam_width=10000):
    return verified_clocked.decode(
        audio, model, repetition=repetition, acoustic_decoder=acoustic_decoder,
        top_k=top_k, beam_width=beam_width, packet_parser=parse_concepts,
        candidate_filter=_candidate_filter,
        acoustic_candidate_limit=top_k if top_k > 5 else None)


def spoken_reply_concepts(result, confirm_success=False, request_repeat=True):
    """Suggest an optional audible reply without adding a protocol handshake.

    Successful packets are silent unless the application asks to confirm them.
    A failed decode prompts for repetition only when enough word markers were
    received to identify a complete compact-packet attempt. Silence, background
    audio, and short ordinary utterances therefore do not provoke chatter.
    """
    if result.accepted:
        return list(CONFIRM_REPLY) if confirm_success else []
    packet_like = result.acoustic.marker_count >= MIN_PACKET_WORDS + 1
    return list(REPEAT_REPLY) if request_repeat and packet_like else []


def encode_spoken_reply(result, prosody=NEUTRAL, confirm_success=False,
                        request_repeat=True):
    """Render the suggested reply, or return None when the policy stays silent."""
    concepts = spoken_reply_concepts(result, confirm_success, request_repeat)
    return clocked.encode(concepts, prosody) if concepts else None
