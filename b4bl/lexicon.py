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
from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass(frozen=True)
class Rep:
    """A repeated phoneme group — repetition is a LEXICAL axis of meaning.

    unit  : phoneme names rendered as one pulse.
    count : how many pulses (fixed; part of identity, prosody must not change it).
    rate  : pulses per second (the rhythm — slow pulses read very differently from
            fast ones, e.g. slow gargle = 'calculating', fast rising whistle = 'alarm').

    Reserving repetition COUNTS as lexical (and forbidding prosody from adding or
    removing repeats) is what keeps 'ALARM' distinct from 'an urgently-repeated
    single whistle'. See prosody.py.
    """
    unit: tuple
    count: int
    rate: float = 3.0
    alternate: bool = False   # if True, each pulse uses ONE cycling member of unit
                              # (tick/tock), instead of playing the whole unit together

    def flatten(self) -> List[str]:
        """Phoneme-name view for unambiguity checks / decoding."""
        out = []
        for k in range(self.count):
            if self.alternate:
                out.append(self.unit[k % len(self.unit)])
            else:
                out.extend(self.unit)
        return out


# A morpheme body is either a flat list of phoneme names, or a Rep group.
MorphemeBody = Union[List[str], Rep]

# concept -> phoneme-name sequence. Kept short for common concepts.
# (phoneme names are defined in phonology.INVENTORY)
# Every sequence must be UNIQUE (enforced by tests). The commonest concepts get
# the shortest (1-phoneme) codes; the rest use distinct 2-phoneme codes. Single-
# phoneme codes are a scarce resource — only 15 phonemes exist — so they are
# spent on the highest-frequency concepts.
# --- (1) HAND-DESIGNED morphemes: speech acts get intentional, human-legible
#     shapes; repetition morphemes carry count+rhythm. These are the sounds a
#     human learns by living with the droid. ---
HAND_MORPHEMES: Dict[str, MorphemeBody] = {
    # speech acts — the human-legible layer (see vocabulary-spec.md)
    "QUERY":    ["Hr", "Mr"],        # clear rising "?" — asked a question
    "CONFIRM":  ["Ma", "Ld"],        # settled arch->fall "got it"
    "CLARIFY":  ["Mc", "Mr"],        # scoop + rise — "which?"
    "ACK":      ["Hf"],              # crisp high blip — "heard you"
    "DENY":     ["Ld"],              # firm low fall — "no"
    "WARN":     ["Hr", "Hr"],        # double high rise — "heads up"
    "DONE":     ["Mr", "Ha"],        # rise-resolve — "finished"
    "REPEAT":   ["Hw"],              # stutter (double contour) — "say again"
    "WAIT":     ["MfL"],             # long flat hold — "hold on"
    "ERROR":    ["Rz", "Ld"],        # rasp + fall — "something's wrong"
    # repetition-axis morphemes (identity includes count + rhythm)
    "ALARM":    Rep(unit=("Vr",), count=3, rate=6.0),        # 3 fast very-high rises
    "WORKING":  Rep(unit=("Hum1", "Hum0"), count=6, rate=3.0, alternate=True),
}
# alias kept for older references
HAND_MORPHEMES["CALCULATING"] = HAND_MORPHEMES["WORKING"]

# --- (2) BULK vocabulary: concepts by category. Phoneme sequences are
#     AUTO-ALLOCATED (see _allocate below) so we don't hand-write ~250 codes.
#     Order matters: earlier = more frequent = gets a shorter code. ---
VOCAB_CATEGORIES: Dict[str, List[str]] = {
    "pronoun": [
        "SELF", "YOU", "IT", "THIS", "THAT", "OTHER_BOT", "ALL", "NONE", "WE",
        # household role slots (user-assigned; glossed as roles, not names)
        "PARENT_F", "PARENT_M", "CHILD_GIRL", "CHILD_BOY", "GUEST",
    ],
    "verb": [
        "MOVE", "STOP", "FOLLOW", "COME", "GO", "BRING", "TAKE", "GIVE", "FIND",
        "SEARCH", "SCAN", "CHARGE", "DOCK", "OPEN", "CLOSE", "PICK", "PLACE",
        "TURN", "LOOK", "TELL", "ASK", "HELP", "CARRY", "CLEAN", "WAIT_V",
        "START", "FINISH", "STORE", "STOP_V", "STAY", "STAND", "STANDBY",
        "STANDDOWN", "STANDUP", "REMIND", "PLAY", "STOP_MEDIA", "CALL", "SEND",
        "STORE_V", "STOPPED", "STANDBY_V", "SLEEP", "WAKE", "STANDGUARD",
    ],
    "spatial": [
        "FRONT", "BACK", "LEFT", "RIGHT", "UP", "DOWN", "NEAR", "FAR", "HERE",
        "THERE", "IN", "ON", "UNDER", "TOWARD", "AWAY", "LOCATION", "ROOM",
        "DOORWAY", "CORNER", "CENTER", "EDGE", "ABOVE", "BELOW", "BESIDE",
        "BETWEEN", "AROUND", "THROUGH", "DISTANCE", "METER", "STEP",
    ],
    "state": [
        "ENERGY", "LOW", "HIGH", "OK", "FAULT", "BUSY", "IDLE", "FULL", "EMPTY",
        "HOT", "COLD", "FAST", "SLOW", "ONSTATE", "OFFSTATE", "LOCKED", "UNLOCKED",
        "READY", "NOTREADY", "MOVING", "STOPPEDSTATE", "CHARGED", "UNCHARGED",
        "CONNECTED", "OFFLINE", "SAFE", "UNSAFE", "CLEAR", "BLOCKED",
    ],
    "object": [
        "OBJECT", "TOOL", "DOOR", "PERSON", "CHARGER", "CONTAINER", "OBSTACLE",
        "TARGET", "ITEM", "BOX", "CUP", "BOTTLE", "KEY", "PHONE", "REMOTE",
        "LIGHT", "TABLE", "CHAIR", "FLOOR", "WALL", "STAIRS", "PET", "FOOD",
        "WATER", "PACKAGE", "MAIL", "BAG", "CLOTHES", "TOY", "TRASH",
    ],
    "quantity": [
        "MORE", "LESS", "HALF", "MANY", "FEW", "SOME", "ENOUGH",
    ],
    "time": [
        "NOW", "THEN", "BEFORE", "AFTER", "SOON", "LATER", "EVERY", "ONCE",
        "AGAIN_T", "UNTIL",
    ],
    "logic": [
        "AND", "OR", "NOT", "IF", "BECAUSE", "VERY", "MAYBE", "SAME",
        "DIFFERENT", "AGAIN", "WITH", "WITHOUT",
    ],
    "social": [
        "GREETING", "FAREWELL", "THANKS", "PLEASE", "SORRY", "WELCOME",
    ],
    "digit": [f"D{i}" for i in range(10)] + ["NUM", "AXIS"],  # numerals + markers
}


def _body_seq(body: "MorphemeBody"):
    return tuple(body.flatten()) if isinstance(body, Rep) else tuple(body)


# --- auto-allocation of phoneme sequences to bulk concepts ------------------
# Frequency-ranked concepts get short codes. We allocate from a pool of TONE
# phonemes (texture classes are reserved for special/hand morphemes), using
# length-1 codes first, then length-2, skipping any sequence already taken by a
# hand morpheme so nothing collides. This is deterministic and unambiguous by
# construction (enforced by tests).
def _alloc_pool():
    from . import phonology as _ph
    # tone phonemes only, stable order, excluding:
    #  - very-high band (reserved for alarm/surprise, kept sparse)
    #  - LONG FLAT tones: a sustained flat pitch reads as a cheap-sci-fi UFO hum,
    #    the least astromech sound there is. Excluded so no ordinary morpheme uses it.
    def ok(p):
        if p.cls != _ph.SoundClass.TONE:
            return False
        if p.band == _ph.Band.VHIGH:
            return False
        if p.contour == _ph.Contour.FLAT and p.dur == _ph.Dur.LONG:
            return False
        return True
    return [p.name for p in _ph.INVENTORY if ok(p)]


def _build_morphemes() -> "Dict[str, MorphemeBody]":
    morphemes: Dict[str, MorphemeBody] = dict(HAND_MORPHEMES)
    taken = {_body_seq(v) for v in morphemes.values()}
    pool = _alloc_pool()

    # candidate codes: length-1 then length-2, in pool order (freq-ranked pool)
    def code_stream():
        for a in pool:
            yield [a]
        for a in pool:
            for b in pool:
                yield [a, b]

    codes = code_stream()

    # flatten categories in priority order (pronoun/verb/... already freq-ish)
    for cat, concepts in VOCAB_CATEGORIES.items():
        for concept in concepts:
            if concept in morphemes:
                continue
            # find next unused code
            while True:
                cand = next(codes)
                if tuple(cand) not in taken:
                    break
            morphemes[concept] = cand
            taken.add(tuple(cand))
    return morphemes


MORPHEMES: Dict[str, MorphemeBody] = _build_morphemes()

# reverse map for decoding: tuple(flattened phoneme names) -> concept
_BY_SEQ = {_body_seq(v): k for k, v in MORPHEMES.items()}


def is_known(concept: str) -> bool:
    return concept in MORPHEMES


def morpheme_body(concept: str) -> MorphemeBody:
    return MORPHEMES[concept]


def concept_to_phonemes(concept: str) -> List[str]:
    """Flat phoneme-name view (repetition expanded). Used for unambiguity/decoding."""
    return list(_body_seq(MORPHEMES[concept]))


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


# documented aliases: two concept names that intentionally share one sound.
ALIASES = {"CALCULATING": "WORKING"}


# --- compositional numerals -------------------------------------------------
def number_to_concepts(n: int) -> List[str]:
    """Integer -> concept sequence: NUM marker then digits, e.g. 42 -> NUM D4 D2."""
    return ["NUM"] + [f"D{int(d)}" for d in str(abs(int(n)))]


def concepts_to_number(concepts: List[str]):
    """Parse a NUM..D.. run back to an int, or None if not a number run."""
    if not concepts or concepts[0] != "NUM":
        return None
    digits = []
    for c in concepts[1:]:
        if c.startswith("D") and c[1:].isdigit():
            digits.append(c[1:])
        else:
            break
    return int("".join(digits)) if digits else None


def example_sentences():
    """Return (english, [concepts]) pairs used in demos/tests."""
    return [
        ("Is your energy low?",      ["QUERY", "YOU", "ENERGY", "LOW"]),
        ("My energy is low.",        ["SELF", "ENERGY", "LOW"]),
        ("Warning: obstacle ahead.", ["WARN", "OBSTACLE", "FRONT"]),
        ("Acknowledged.",            ["ACK"]),
        ("Follow me.",               ["YOU", "FOLLOW", "SELF"]),
        ("Bring me the cup.",        ["YOU", "BRING", "SELF", "CUP"]),
        ("Stop.",                    ["STOP"]),
    ]
