"""
B4-BL Layer 1 — acoustic phonology (the "phonemes").

A phoneme here is a robustly-classifiable acoustic GESTURE, not an exact
waveform. Meaning attaches to invariant features; everything else is free to
vary for prosody/affect.

MEANING vs EXPRESSION split (PROPOSAL — decision #1 in the plan, needs review)
------------------------------------------------------------------------------
Lexical (carries meaning; the classifier keys on these):
  - PITCH BAND       : LOW / MID / HIGH        (categorical region, not exact Hz)
  - CONTOUR TOPOLOGY : FLAT / RISE / FALL / ARCH / DIP   (sign pattern, not shape)
  - DURATION CLASS   : SHORT / LONG
  - SOUND CLASS      : TONE / GARGLE / RASP     (which generator family)

Expressive (free; prosody owns these, classifier ignores them):
  - exact pitch within the band
  - exact contour shape / ornament
  - vibrato rate & depth
  - onset hardness, gap timing, loudness contour

This gives an inventory of  bands(3) x topology(5) x duration(2) x class(3) = 90
theoretical cells, but most are not acoustically distinct or not useful. We pick
a working set of ~16 that are pairwise separable through a cheap mic. That keeps
the alphabet learnable and the decoder reliable.

Each Phoneme knows how to RENDER itself (via generators) and exposes the feature
tuple a decoder would recover. Synth and recognizer share this one definition.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import numpy as np
from . import generators as gen


class Band(Enum):
    LOW = "L"
    MID = "M"
    HIGH = "H"


class Contour(Enum):
    FLAT = "flat"
    RISE = "rise"
    FALL = "fall"
    ARCH = "arch"   # up then down
    DIP = "dip"     # down then up


class Dur(Enum):
    SHORT = "short"
    LONG = "long"


class SoundClass(Enum):
    TONE = "tone"
    GARGLE = "gargle"
    RASP = "rasp"


# Representative center frequencies per band (Hz). The decoder only needs the
# band, but the synth needs an actual number; exact value is expressive.
BAND_CENTER = {Band.LOW: 500, Band.MID: 1100, Band.HIGH: 2000}
BAND_SPAN = 380  # how far the contour swings around the center
DUR_SEC = {Dur.SHORT: 0.16, Dur.LONG: 0.5}


@dataclass(frozen=True)
class Phoneme:
    name: str
    band: Band
    contour: Contour
    dur: Dur
    cls: SoundClass = SoundClass.TONE
    freq_hz: float = 0.0   # optional exact-center override (0 = use band center).
                           # Used for sub-band texture pulses like the hum, which
                           # sit below the LOW tonal band.

    @property
    def features(self) -> Tuple[str, str, str, str]:
        """The invariant tuple a decoder recovers. This IS the phoneme's identity."""
        return (self.band.value, self.contour.value, self.dur.value, self.cls.value)

    @property
    def center(self) -> float:
        return self.freq_hz if self.freq_hz else BAND_CENTER[self.band]

    # -- rendering -----------------------------------------------------------
    def _contour_points(self):
        c = BAND_CENTER[self.band]
        s = BAND_SPAN
        if self.contour == Contour.FLAT:
            return [(0, c), (1, c)]
        if self.contour == Contour.RISE:
            return [(0, c - s), (1, c + s)]
        if self.contour == Contour.FALL:
            return [(0, c + s), (1, c - s)]
        if self.contour == Contour.ARCH:
            return [(0, c - s), (0.5, c + s), (1, c)]
        if self.contour == Contour.DIP:
            return [(0, c + s), (0.5, c - s), (1, c)]
        return [(0, c), (1, c)]

    def render(self, prosody=None) -> np.ndarray:
        """Render this phoneme to audio. `prosody` (optional) may perturb the
        expressive dimensions; see b4bl.prosody."""
        dur = DUR_SEC[self.dur]
        pts = self._contour_points()
        env = "stab" if self.dur == Dur.SHORT else "even"
        vib = (7, 0.03)
        if prosody is not None:
            pts, dur, env, vib = prosody.apply(self, pts, dur, env, vib)

        if self.cls == SoundClass.TONE:
            return gen.tonal(gen.Gesture(pts, dur, env, vib))
        if self.cls == SoundClass.GARGLE:
            # low hum pulses want a gentler, slower flutter than the sharp default
            # gargle, so they read as humming rather than a buzzy roll.
            if self.center < 400:
                return gen.gargle(dur=dur, center=self.center, tremolo_hz=22, depth=0.5)
            return gen.gargle(dur=dur, center=self.center)
        if self.cls == SoundClass.RASP:
            return gen.raspberry(dur=dur, center=max(250, self.center / 3))
        raise ValueError(self.cls)


# ---------------------------------------------------------------------------
# The working inventory (~16 separable primitives).
# Named with short mnemonic codes: <band><contour-initial><s/l>[+class]
# ---------------------------------------------------------------------------
def _p(name, band, contour, dur, cls=SoundClass.TONE, freq_hz=0.0):
    return Phoneme(name, band, contour, dur, cls, freq_hz)


INVENTORY = [
    # LOW band
    _p("Lf",  Band.LOW,  Contour.FLAT, Dur.SHORT),
    _p("Lr",  Band.LOW,  Contour.RISE, Dur.SHORT),
    _p("Ld",  Band.LOW,  Contour.FALL, Dur.LONG),      # groan-like
    # MID band
    _p("Mf",  Band.MID,  Contour.FLAT, Dur.SHORT),
    _p("Mr",  Band.MID,  Contour.RISE, Dur.SHORT),
    _p("Mfl", Band.MID,  Contour.FALL, Dur.SHORT),
    _p("Ma",  Band.MID,  Contour.ARCH, Dur.LONG),
    _p("Mi",  Band.MID,  Contour.DIP,  Dur.LONG),
    # HIGH band
    _p("Hf",  Band.HIGH, Contour.FLAT, Dur.SHORT),
    _p("Hr",  Band.HIGH, Contour.RISE, Dur.SHORT),      # surprised blip
    _p("Hd",  Band.HIGH, Contour.FALL, Dur.SHORT),
    _p("Ha",  Band.HIGH, Contour.ARCH, Dur.LONG),       # whistle
    # texture classes (used sparingly, high separability from tones)
    _p("Grm", Band.MID,  Contour.FLAT, Dur.SHORT, SoundClass.GARGLE),
    _p("Grl", Band.MID,  Contour.FLAT, Dur.LONG,  SoundClass.GARGLE),
    _p("Rz",  Band.LOW,  Contour.FALL, Dur.SHORT, SoundClass.RASP),
]

# Texture sub-units used INSIDE morphemes (e.g. the hum), not part of the
# decodable phoneme alphabet — so they don't need unique feature tuples. They
# are distinguished only by exact pitch, which the hum uses as flavour.
SUBUNITS = [
    # Two LOW hum pulses for "muttering to itself" — both low so it reads as
    # HMM-hmmm, not beep-boop. Exact centers overridden below the tonal bands.
    _p("Hum1", Band.LOW, Contour.FLAT, Dur.SHORT, SoundClass.GARGLE, freq_hz=300),  # higher hmm
    _p("Hum0", Band.LOW, Contour.FLAT, Dur.SHORT, SoundClass.GARGLE, freq_hz=190),  # lower hmmm
]

BY_NAME = {p.name: p for p in INVENTORY + SUBUNITS}


def render_inventory_montage():
    clips = []
    for p in INVENTORY:
        clips.append(p.render())
    return gen.sequence(clips, gap=0.12)


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "audio_samples", "phoneme_inventory.wav")
    gen.write_wav(out, render_inventory_montage())
    print("wrote", out, "-", len(INVENTORY), "phonemes")
