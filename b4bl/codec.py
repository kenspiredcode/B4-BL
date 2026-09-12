"""
B4-BL codec — meaning <-> phoneme-sequence <-> audio.

Encode path (complete):
    concepts -> phoneme names -> audio
    with word gaps between morphemes and spelling fallback for unknown tokens.

Decode path:
    SYMBOLIC decode (phoneme names -> concepts) is complete here and proves the
    language is unambiguously parseable, including spelling fallback.

    ACOUSTIC decode (microphone audio -> phoneme names) is the Phase 2/4 signal-
    processing task and is NOT implemented yet. `audio_to_phonemes` raises
    NotImplementedError with a pointer, rather than pretending to work. The
    feature model it will target already exists: Phoneme.features. Because synth
    and recognizer share phonology.py, the classifier's job is to recover that
    4-tuple (band, contour, duration, class) per segment.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from . import generators as gen
from . import phonology as ph
from . import lexicon as lex
from .prosody import Prosody, NEUTRAL

WORD_GAP = 0.14      # gap between morphemes (groups phonemes into "words")
PHONE_GAP = 0.04     # gap between phonemes within a morpheme


# ---------------------------------------------------------------------------
# ENCODE
# ---------------------------------------------------------------------------
def concepts_to_phoneme_words(concepts: List[str]) -> List[List[str]]:
    """Each concept -> a list of phoneme names. Unknown concepts are spelled."""
    words = []
    for c in concepts:
        if lex.is_known(c):
            words.append(lex.concept_to_phonemes(c))
        else:
            # spelling fallback: marker + per-char codes
            seq = list(lex.SPELL_MARKER)
            for ch in c.lower():
                if ch in lex.CHAR_TO_PHONES:
                    seq += lex.CHAR_TO_PHONES[ch]
            words.append(seq)
    return words


def _render_phoneme_seq(pnames: List[str], prosody: Prosody) -> List[np.ndarray]:
    clips = []
    for pi, pname in enumerate(pnames):
        clips.append(ph.BY_NAME[pname].render(prosody=prosody))
        if pi != len(pnames) - 1:
            clips.append(_silence(PHONE_GAP))
    return clips


def _render_rep(rep: "lex.Rep", prosody: Prosody) -> List[np.ndarray]:
    """Render a repeated group at its lexical rhythm. The inter-pulse gap comes
    from `rate`; prosody may scale intensity/timing but NEVER the count."""
    unit = list(rep.unit)
    # gap so that pulse_period = 1/rate; clamp so pulses don't overlap.
    period = 1.0 / max(rep.rate, 0.3)
    clips = []
    for k in range(rep.count):
        if rep.alternate:
            pulse = [unit[k % len(unit)]]     # tick/tock: one cycling member per pulse
        else:
            pulse = unit
        clips += _render_phoneme_seq(pulse, prosody)
        if k != rep.count - 1:
            # approximate: subtract a nominal unit duration from the period
            gap = max(0.02, period - 0.16)
            clips.append(_silence(gap))
    return clips


def _render_concept(concept: str, prosody: Prosody) -> List[np.ndarray]:
    if lex.is_known(concept):
        body = lex.morpheme_body(concept)
        if isinstance(body, lex.Rep):
            return _render_rep(body, prosody)
        return _render_phoneme_seq(body, prosody)
    # spelling fallback
    seq = list(lex.SPELL_MARKER)
    for ch in concept.lower():
        if ch in lex.CHAR_TO_PHONES:
            seq += lex.CHAR_TO_PHONES[ch]
    return _render_phoneme_seq(seq, prosody)


def encode(concepts: List[str], prosody: Prosody = NEUTRAL) -> np.ndarray:
    """Top-level: list of concepts -> audio, respecting repetition rhythm."""
    clips = []
    for ci, c in enumerate(concepts):
        clips += _render_concept(c, prosody)
        if ci != len(concepts) - 1:
            clips.append(_silence(WORD_GAP))
    return np.concatenate(clips) if clips else np.zeros(0, dtype=np.float32)


def _silence(dur: float) -> np.ndarray:
    return np.zeros(int(gen.SR * dur), dtype=np.float32)


# ---------------------------------------------------------------------------
# DECODE (symbolic)
# ---------------------------------------------------------------------------
def phoneme_words_to_concepts(words: List[List[str]]) -> List[str]:
    out = []
    for word in words:
        if word[:len(lex.SPELL_MARKER)] == lex.SPELL_MARKER:
            # spelled token
            body = word[len(lex.SPELL_MARKER):]
            chars = []
            for i in range(0, len(body) - 1, 2):
                ch = lex.PHONES_TO_CHAR.get(tuple(body[i:i + 2]))
                if ch:
                    chars.append(ch)
            out.append("".join(chars).upper())
        else:
            concept = lex.phonemes_to_concept(word)
            out.append(concept if concept else "?" + "-".join(word))
    return out


def audio_to_phonemes(audio: np.ndarray) -> List[List[str]]:
    """Acoustic front-end: segment audio and classify each segment into a phoneme
    name, grouping by word gaps. Implemented in b4bl.decoder (Phase-2 v1,
    file/loopback). Not yet perfect per-phoneme; the protocol layer's checksum/FEC
    is what makes it reliable end to end."""
    from . import decoder
    return decoder.audio_to_phoneme_words(audio)


def decode_to_concepts(audio: np.ndarray) -> List[str]:
    """Full acoustic decode: audio -> phoneme words -> concepts."""
    return phoneme_words_to_concepts(audio_to_phonemes(audio))


# ---------------------------------------------------------------------------
# convenience round-trip used by tests
# ---------------------------------------------------------------------------
def roundtrip_symbolic(concepts: List[str]) -> Tuple[List[List[str]], List[str]]:
    words = concepts_to_phoneme_words(concepts)
    back = phoneme_words_to_concepts(words)
    return words, back
