"""
B4-BL — interjections: standalone emotive vocalizations.

A THIRD thing, distinct from both morphemes and prosody:
  - a morpheme is a word (carries semantic payload, gets decoded);
  - prosody modulates a word you are already saying;
  - an interjection IS the whole utterance and carries NO semantic payload.

This is the "tell the bot no and it just goes *awww...* with a falling tone"
charm — R2's sad descending whistle, the excited trill. They are not decoded for
meaning, so they don't consume morpheme slots and don't need unique feature
tuples; they're rendered directly by the generators.

Each interjection is a small hand-tuned gesture. They can stand alone or tag a
reply (e.g. CONFIRM followed by PLEASED = "yes! :)").
"""

from __future__ import annotations
import numpy as np
from . import generators as gen

SR = gen.SR


def _seq(*clips, gap=0.03):
    return gen.sequence(list(clips), gap=gap)


def aww() -> np.ndarray:
    """Disappointed — a long, soft, whimpering downward whistle. Deliberately
    contrasted with DENY (short firm low fall): AWW starts HIGH, is much LONGER,
    droops with a wavering vibrato, and has a couple of little 'sad' bumps so it
    reads as a whimper/voice, not a beep."""
    return gen.tonal(gen.Gesture(
        [(0, 1650), (0.2, 1500), (0.45, 1550), (0.7, 1050), (1, 620)],
        1.0, "swell", vib=(6.5, 0.09)))


def yay() -> np.ndarray:
    """Excited — fast rising trill, then a little pop up."""
    a = gen.tonal(gen.Gesture([(0, 700), (1, 1900)], 0.28, "even", vib=(18, 0.10)))
    b = gen.tonal(gen.Gesture([(0, 1900), (1, 2300)], 0.12, "stab"))
    return _seq(a, b, gap=0.02)


def pleased() -> np.ndarray:
    """Pleased/praised — warm short arch."""
    return gen.tonal(gen.Gesture([(0, 900), (0.5, 1500), (1, 1150)], 0.35, "swell", vib=(6, 0.03)))


def annoyed() -> np.ndarray:
    """Annoyed — clipped low double, flat and terse."""
    a = gen.tonal(gen.Gesture([(0, 520), (1, 470)], 0.12, "stab"))
    return _seq(a, a, gap=0.05)


def curious() -> np.ndarray:
    """Curious — lilting up-turn (scoop up), questioning."""
    return gen.tonal(gen.Gesture([(0, 900), (0.35, 780), (1, 1500)], 0.4, "even", vib=(7, 0.04)))


def relieved() -> np.ndarray:
    """Relieved — descending settle with a little sigh (soft tail)."""
    a = gen.tonal(gen.Gesture([(0, 1400), (0.5, 1000), (1, 750)], 0.45, "decay"))
    b = gen.tonal(gen.Gesture([(0, 700), (1, 640)], 0.2, "decay", vib=(4, 0.05)))
    return _seq(a, b, gap=0.02)


def surprised() -> np.ndarray:
    """Surprised — sharp very-high blip up."""
    return gen.tonal(gen.Gesture([(0, 1900), (1, 2800)], 0.14, "stab"))


def playful() -> np.ndarray:
    """Playful — bouncy warble, affectionate idle."""
    return gen.tonal(gen.Gesture([(0, 800), (0.25, 1300), (0.5, 950), (0.75, 1400), (1, 1050)],
                                 0.5, "even", vib=(9, 0.06)))


INTERJECTIONS = {
    "AWW": aww, "YAY": yay, "PLEASED": pleased, "ANNOYED": annoyed,
    "CURIOUS": curious, "RELIEVED": relieved, "SURPRISED": surprised, "PLAYFUL": playful,
}


def render(name: str) -> np.ndarray:
    return INTERJECTIONS[name]()


def montage() -> np.ndarray:
    return gen.sequence([f() for f in INTERJECTIONS.values()], gap=0.3)
