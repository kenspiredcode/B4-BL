"""
B4-BL Layer 3 (structural) — communications protocol / framing.

Natural languages don't have addressing, sequence numbers or checksums; machines
need them. A B4-BL frame wraps a semantic payload with the structure a real
acoustic packet needs:

    PREAMBLE  SENDER  RECIPIENT  MSGTYPE  SEQ  <payload concepts>  CHECKSUM

Everything is expressed as concept morphemes (which render to phonemes to audio),
so a frame is just a longer concept list the codec already knows how to speak and
hear. The checksum lets a receiver DETECT phoneme errors from the imperfect
acoustic decoder — the reason the language stays usable without a perfect decoder.

This module builds/parses the concept-level frame. It does not do audio itself;
combine with codec.encode / codec.decode_to_concepts for over-the-air.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from . import lexicon as lex

# Frame-structure marker concepts. These are reserved morphemes added to the
# lexicon (see _ensure_protocol_morphemes) so they render like any other.
PREAMBLE = "SYNC"
MSG_TYPES = ["MSG_TELL", "MSG_ASK", "MSG_ACKF", "MSG_WARN"]
CHECKSUM_MARK = "CKSUM"
ADDR_PREFIX = "ADDR"          # addresses are ADDR + a numeral group


def _ensure_protocol_morphemes():
    """Protocol markers are defined in the lexicon's 'protocol' category and
    allocated by its ECC-aware allocator (distance >= 2 from other codes). This
    just checks they're present."""
    need = [PREAMBLE, CHECKSUM_MARK, ADDR_PREFIX] + MSG_TYPES
    missing = [m for m in need if m not in lex.MORPHEMES]
    if missing:
        raise RuntimeError(
            f"protocol markers missing from lexicon: {missing} — add them to "
            "lexicon.VOCAB_CATEGORIES['protocol']")


_ensure_protocol_morphemes()


# ---------------------------------------------------------------------------
# checksum over the concept payload
# ---------------------------------------------------------------------------
def _concept_checksum(concepts: List[str]) -> int:
    """A two-digit (0-99) checksum over the payload concepts, spoken as a numeral.
    Order- and identity-sensitive so a dropped/swapped/misheard morpheme changes
    it. Two digits (mod 100) instead of one keeps the collision rate ~1% rather
    than ~10%, so it actually catches corruption from the imperfect decoder."""
    acc = 0
    for i, c in enumerate(concepts):
        acc = (acc * 131 + sum((idx + 1) * ord(ch) for idx, ch in enumerate(c)) + i) % 100
    return acc


# ---------------------------------------------------------------------------
# frame model
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    sender: int
    recipient: int
    msg_type: str            # one of MSG_TYPES
    seq: int
    payload: List[str]       # semantic concepts

    def to_concepts(self) -> List[str]:
        _ensure_protocol_morphemes()
        out: List[str] = [PREAMBLE]
        out += [ADDR_PREFIX] + lex.number_to_concepts(self.sender)
        out += [ADDR_PREFIX] + lex.number_to_concepts(self.recipient)
        out += [self.msg_type]
        out += ["NUM"] + lex.number_to_concepts(self.seq)[1:]  # seq as a numeral
        out += list(self.payload)
        out += [CHECKSUM_MARK] + lex.number_to_concepts(_concept_checksum(self.payload))
        return out


@dataclass
class ParseResult:
    frame: Optional[Frame]
    ok: bool
    error: str = ""
    checksum_ok: bool = False


def parse_concepts(concepts: List[str]) -> ParseResult:
    """Parse a decoded concept stream back into a Frame and verify the checksum."""
    c = list(concepts)
    if not c or c[0] != PREAMBLE:
        return ParseResult(None, False, "no preamble")
    i = 1

    def read_addr(i):
        if i >= len(c) or c[i] != ADDR_PREFIX:
            return None, i
        i += 1
        # collect NUM..digits
        j = i
        num = lex.concepts_to_number(c[j:])
        # advance past NUM + its digits
        if num is None:
            return None, i
        k = j + 1
        while k < len(c) and (c[k].startswith("D") and c[k][1:].isdigit()):
            k += 1
        return num, k

    sender, i = read_addr(i)
    recipient, i = read_addr(i)
    if sender is None or recipient is None:
        return ParseResult(None, False, "bad address")
    if i >= len(c) or c[i] not in MSG_TYPES:
        return ParseResult(None, False, "bad msg type")
    msg_type = c[i]; i += 1
    # sequence numeral
    seq = lex.concepts_to_number(c[i:])
    if seq is None:
        return ParseResult(None, False, "bad seq")
    i += 1
    while i < len(c) and (c[i].startswith("D") and c[i][1:].isdigit()):
        i += 1
    # payload up to CHECKSUM_MARK
    if CHECKSUM_MARK not in c[i:]:
        return ParseResult(None, False, "no checksum")
    ck_idx = c.index(CHECKSUM_MARK, i)
    payload = c[i:ck_idx]
    got_ck = lex.concepts_to_number(c[ck_idx + 1:])   # already begins with NUM
    want_ck = _concept_checksum(payload)
    checksum_ok = (got_ck == want_ck)
    frame = Frame(sender, recipient, msg_type, seq, payload)
    return ParseResult(frame, True, "" if checksum_ok else "checksum mismatch", checksum_ok)


# ---------------------------------------------------------------------------
# convenience
# ---------------------------------------------------------------------------
def make_frame(sender, recipient, msg_type, seq, payload) -> List[str]:
    return Frame(sender, recipient, msg_type, seq, payload).to_concepts()
