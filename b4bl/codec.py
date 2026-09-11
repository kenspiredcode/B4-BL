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


def phoneme_words_to_audio(words: List[List[str]], prosody: Prosody = NEUTRAL) -> np.ndarray:
    clips = []
    for wi, word in enumerate(words):
        for pi, pname in enumerate(word):
            phoneme = ph.BY_NAME[pname]
            clips.append(phoneme.render(prosody=prosody))
            if pi != len(word) - 1:
                clips.append(_silence(PHONE_GAP))
        if wi != len(words) - 1:
            clips.append(_silence(WORD_GAP))
    return np.concatenate(clips) if clips else np.zeros(0, dtype=np.float32)


def encode(concepts: List[str], prosody: Prosody = NEUTRAL) -> np.ndarray:
    """Top-level: list of concepts -> audio."""
    return phoneme_words_to_audio(concepts_to_phoneme_words(concepts), prosody)


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
    name, grouping by word gaps.

    NOT YET IMPLEMENTED. This is the Phase 2/4 DSP task. It will:
      1. segment on silence (PHONE_GAP vs WORD_GAP thresholds),
      2. per segment estimate (band, contour, duration, class) = Phoneme.features,
      3. nearest-match against phonology.INVENTORY.
    See docs/phase-0-sound.md and the plan's Layer-1 section.
    """
    raise NotImplementedError(
        "Acoustic decode (mic -> phonemes) is the Phase 2/4 signal-processing task; "
        "the symbolic round-trip (phoneme_words_to_concepts) is implemented and tested."
    )


# ---------------------------------------------------------------------------
# convenience round-trip used by tests
# ---------------------------------------------------------------------------
def roundtrip_symbolic(concepts: List[str]) -> Tuple[List[List[str]], List[str]]:
    words = concepts_to_phoneme_words(concepts)
    back = phoneme_words_to_concepts(words)
    return words, back
