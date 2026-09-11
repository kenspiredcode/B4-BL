"""
B4-BL Layer 2 — semantic layer: morphemes + compositional grammar.

Design (from the plan):
  - The audible unit is a MORPHEME (a compact concept), not an English word or letter.
  - Common concepts get short phoneme sequences; unknown tokens fall back to
    character spelling (see codec.py).
  - A small compositional grammar builds sentences from morphemes.

Each morpheme maps to a sequence of phoneme names from phonology.INVENTORY.
These assignments are a FIRST-PASS PROPOSAL. They are deliberately short (1-2
phonemes for the commonest concepts) to keep utterances compact. The exact
mapping is open to revision; what matters structurally is that meaning <->
phoneme-sequence is a clean bijection over the dictionary, with spelling fallback
for everything else.

Grammar (minimal, verb-second-ish, machine-friendly):
  UTTERANCE := [MARKER] SUBJECT? PREDICATE OBJECT? [MODIFIER*]
  e.g. QUERY YOU ENERGY LOW   -> "is your energy low?"
       SELF MOVE LOCATION     -> "I am moving to the location"
The grammar is not enforced strictly here — the codec just concatenates morpheme
phoneme-sequences with word gaps. Grammar mostly governs how humans/decoders
group meaning, and where the protocol layer will later insert structure.
"""

from __future__ import annotations
from typing import Dict, List

# concept -> phoneme-name sequence. Kept short for common concepts.
# (phoneme names are defined in phonology.INVENTORY)
# Every sequence must be UNIQUE (enforced by tests). The commonest concepts get
# the shortest (1-phoneme) codes; the rest use distinct 2-phoneme codes. Single-
# phoneme codes are a scarce resource — only 15 phonemes exist — so they are
# spent on the highest-frequency concepts.
MORPHEMES: Dict[str, List[str]] = {
    # --- 1-phoneme codes: the most frequent concepts ---
    "ACK":      ["Hf"],             # yes/ok/acknowledged
    "DENY":     ["Ld"],             # no
    "SELF":     ["Lf"],
    "YOU":      ["Mf"],
    "OBJECT":   ["Ma"],
    "LOCATION": ["Mi"],
    "STOP":     ["Mfl"],
    "FRONT":    ["Ha"],
    "FAR":      ["Grl"],
    # --- 2-phoneme codes ---
    # markers / speech acts
    "QUERY":    ["Hr", "Mr"],       # rising-rising = "?"
    "WARNING":  ["Hr", "Hr"],       # double high rise = attention
    # predicates / verbs
    "MOVE":     ["Mr", "Hf"],
    "FOLLOW":   ["Mr", "Mr"],
    "QUERY_STATE": ["Hr", "Ma"],
    # states / properties
    "ENERGY":   ["Ma", "Mf"],
    "LOW":      ["Lr", "Ld"],
    "HIGH":     ["Hr", "Hd"],
    "OBSTACLE": ["Rz", "Mfl"],      # rasp = something bad
    "URGENT":   ["Hr", "Hr", "Hr"],
    # directions
    "BACK":     ["Ld", "Lf"],
    "NEAR":     ["Mf", "Mf"],
}

# reverse map for decoding: tuple(phoneme names) -> concept
_BY_SEQ = {tuple(v): k for k, v in MORPHEMES.items()}


def is_known(concept: str) -> bool:
    return concept in MORPHEMES


def concept_to_phonemes(concept: str) -> List[str]:
    return list(MORPHEMES[concept])


def phonemes_to_concept(seq):
    return _BY_SEQ.get(tuple(seq))


# --- character-spelling fallback for out-of-dictionary tokens ----------------
# Map a small alphabet to phoneme pairs so anything can be spelled. This uses a
# distinct 2-phoneme code space so the decoder can tell "spelled" from "morpheme".
_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
# build deterministic 2-phoneme codes from a compact phoneme subset
_SPELL_PHONES = ["Lf", "Lr", "Mf", "Mr", "Mfl", "Hf", "Hr", "Hd"]


def _char_code(i: int) -> List[str]:
    a = _SPELL_PHONES[i // len(_SPELL_PHONES)]
    b = _SPELL_PHONES[i % len(_SPELL_PHONES)]
    return [a, b]


CHAR_TO_PHONES = {ch: _char_code(i) for i, ch in enumerate(_ALPHABET)}
PHONES_TO_CHAR = {tuple(v): k for k, v in CHAR_TO_PHONES.items()}

# a dedicated marker phoneme sequence that means "spelling mode follows"
SPELL_MARKER = ["Grm"]


def example_sentences():
    """Return (english, [concepts]) pairs used in demos/tests."""
    return [
        ("Is your energy low?",      ["QUERY", "YOU", "ENERGY", "LOW"]),
        ("My energy is low.",        ["SELF", "ENERGY", "LOW"]),
        ("Warning: obstacle ahead.", ["WARNING", "OBSTACLE", "FRONT"]),
        ("Acknowledged.",            ["ACK"]),
        ("Follow me.",               ["YOU", "FOLLOW", "SELF"]),
        ("Stop.",                    ["STOP"]),
    ]
