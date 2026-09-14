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
    SUB = "S"        # below the tonal range — hums/growls
    LOW = "L"
    MID = "M"
    HIGH = "H"
    VHIGH = "V"      # very-high — used SPARINGLY (alarm/surprise), stays salient


class Contour(Enum):
    FLAT = "flat"
    RISE = "rise"
    FALL = "fall"
    ARCH = "arch"     # up then down
    DIP = "dip"       # down then up
    SCOOP = "scoop"   # down a bit then up past start (question-like)
    DOUBLE = "double" # two little bumps (bip-bip within one gesture)


class Dur(Enum):
    SHORT = "short"
    LONG = "long"


class SoundClass(Enum):
    TONE = "tone"       # pure sine
    WHISTLE = "whistle" # breathier sine w/ slight air (rendered as tone for now)
    TRILL = "trill"     # fast pitch flutter
    GARGLE = "gargle"   # amplitude flutter texture
    RASP = "rasp"       # noisy buzz


# Representative center frequencies per band (Hz). The decoder only needs the
# band, but the synth needs an actual number; exact value is expressive.
BAND_CENTER = {
    Band.SUB: 260, Band.LOW: 520, Band.MID: 1000, Band.HIGH: 1750, Band.VHIGH: 2700,
}
BAND_SPAN = 300  # how far the contour swings around the center
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
        c = self.center
        # span PROPORTIONAL to center: a rise/fall should be a clear musical
        # interval (~a fifth), so contours are unmistakable both to the decoder
        # and to a human ear, and never blur into 'flat'.
        s = c * 0.33
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
        if self.contour == Contour.SCOOP:
            # dip early then rise clearly past start — reads as questioning. Peak
            # reached before the end so the defining upswing renders (and isn't
            # lost to envelope decay / edge trimming), keeping it distinct from DIP.
            return [(0, c), (0.2, c - s * 0.6), (0.8, c + s), (1, c + s)]
        if self.contour == Contour.DOUBLE:
            # two bumps within one gesture
            return [(0, c - s * 0.5), (0.25, c + s * 0.5), (0.5, c - s * 0.3),
                    (0.75, c + s * 0.5), (1, c)]
        return [(0, c), (1, c)]

    def render(self, prosody=None, register_mult: float = 1.0) -> np.ndarray:
        """Render this phoneme to audio. `prosody` (optional) may perturb the
        expressive dimensions; see b4bl.prosody. `register_mult` shifts the whole
        phoneme's pitch by a constant factor — used to place a WORD in a target
        pitch register (register-cycling word-boundary cue), applied uniformly so
        it doesn't disturb the contour/duration/class the decoder keys on."""
        dur = DUR_SEC[self.dur]
        pts = self._contour_points()
        if register_mult != 1.0:
            pts = [(f, hz * register_mult) for (f, hz) in pts]
        env = "stab" if self.dur == Dur.SHORT else "even"
        vib = (7, 0.03)
        # A long FLAT tone with vibrato reads as a cheap-sci-fi UFO warble. Keep it
        # dead steady so a held pitch reads as a deliberate "hold", not a UFO.
        if self.contour == Contour.FLAT and self.dur == Dur.LONG:
            vib = None
        if prosody is not None:
            pts, dur, env, vib = prosody.apply(self, pts, dur, env, vib)

        if self.cls == SoundClass.TONE:
            return gen.tonal(gen.Gesture(pts, dur, env, vib))
        if self.cls == SoundClass.WHISTLE:
            # breathier whistle: pure tone with a gentle onset swell
            return gen.tonal(gen.Gesture(pts, dur, "swell" if env == "even" else env, vib))
        if self.cls == SoundClass.TRILL:
            # fast pitch flutter on top of the contour
            rate, depth = vib
            return gen.tonal(gen.Gesture(pts, dur, env, (max(rate, 16), max(depth, 0.08))))
        if self.cls == SoundClass.GARGLE:
            # low hum pulses want a gentler, slower flutter than the sharp default
            # gargle, so they read as humming rather than a buzzy roll.
            gc = self.center * register_mult
            if gc < 400:
                return gen.gargle(dur=dur, center=gc, tremolo_hz=22, depth=0.5)
            return gen.gargle(dur=dur, center=gc)
        if self.cls == SoundClass.RASP:
            return gen.raspberry(dur=dur, center=max(250, self.center * register_mult / 3))
        raise ValueError(self.cls)


# ---------------------------------------------------------------------------
# The working inventory (~16 separable primitives).
# Named with short mnemonic codes: <band><contour-initial><s/l>[+class]
# ---------------------------------------------------------------------------
def _p(name, band, contour, dur, cls=SoundClass.TONE, freq_hz=0.0):
    return Phoneme(name, band, contour, dur, cls, freq_hz)


# ---------------------------------------------------------------------------
# The widened inventory (~40 separable primitives), generated from a curated
# subset of the band x contour x dur x class grid. We do NOT take the full
# cross-product (that would include hard-to-classify cells and overuse very-
# high); we pick combinations that are pairwise separable through a cheap mic.
# The final set is confirmed by the Phase-2 separability test.
#
# Names are stable mnemonics: <band-letter><contour-initial><dur><class?>.
# ---------------------------------------------------------------------------
_CONTOUR_LETTER = {
    Contour.FLAT: "f", Contour.RISE: "r", Contour.FALL: "d", Contour.ARCH: "a",
    Contour.DIP: "i", Contour.SCOOP: "c", Contour.DOUBLE: "w",
}
_CLASS_LETTER = {
    SoundClass.TONE: "", SoundClass.WHISTLE: "W", SoundClass.TRILL: "T",
    SoundClass.GARGLE: "G", SoundClass.RASP: "R",
}


def _name(band, contour, dur, cls):
    return f"{band.value}{_CONTOUR_LETTER[contour]}{'L' if dur == Dur.LONG else ''}{_CLASS_LETTER[cls]}"


def _gen_inventory():
    S, L = Dur.SHORT, Dur.LONG
    T = SoundClass.TONE
    combos = []
    # TONE phonemes across bands x a curated contour/dur set.
    # LOW: flat, rise, fall(long), dip
    combos += [(Band.LOW, c, d, T) for (c, d) in
               [(Contour.FLAT, S), (Contour.RISE, S), (Contour.FALL, S),
                (Contour.FALL, L), (Contour.DIP, L), (Contour.SCOOP, S)]]
    # MID: the richest band (most contours) — the workhorse
    combos += [(Band.MID, c, d, T) for (c, d) in
               [(Contour.FLAT, S), (Contour.FLAT, L), (Contour.RISE, S),
                (Contour.FALL, S), (Contour.ARCH, L), (Contour.DIP, L),
                (Contour.SCOOP, S), (Contour.DOUBLE, S), (Contour.RISE, L),
                (Contour.FALL, L)]]
    # HIGH: flat, rise, fall, arch(long), scoop, double
    combos += [(Band.HIGH, c, d, T) for (c, d) in
               [(Contour.FLAT, S), (Contour.RISE, S), (Contour.FALL, S),
                (Contour.ARCH, L), (Contour.SCOOP, S), (Contour.DOUBLE, S)]]
    # SUB: low growly tones — flat, rise, fall
    combos += [(Band.SUB, c, S, T) for c in (Contour.FLAT, Contour.RISE, Contour.FALL)]
    # VHIGH: SPARINGLY — only rise + double (alarm/surprise)
    combos += [(Band.VHIGH, Contour.RISE, S, T), (Band.VHIGH, Contour.DOUBLE, S, T)]
    # NOTE: WHISTLE and TRILL are TONE sub-flavors (breathier / fluttered), not
    # distinct meaning-bearing classes — a decoder can't reliably separate them
    # from plain TONE, and no morpheme uses them. So they are NOT decodable
    # phonemes; the sound-class meaning axis is the coarse TONE / GARGLE / RASP
    # (talking vs texture). They remain available for expressive rendering.
    # GARGLE class textures
    combos += [(Band.MID, Contour.FLAT, S, SoundClass.GARGLE),
               (Band.MID, Contour.FLAT, L, SoundClass.GARGLE),
               (Band.LOW, Contour.FLAT, S, SoundClass.GARGLE)]
    # RASP class (rude/error) — low fall
    combos += [(Band.LOW, Contour.FALL, S, SoundClass.RASP),
               (Band.MID, Contour.FALL, S, SoundClass.RASP)]

    out, seen = [], set()
    for band, contour, dur, cls in combos:
        key = (band, contour, dur, cls)
        if key in seen:
            continue
        seen.add(key)
        out.append(Phoneme(_name(band, contour, dur, cls), band, contour, dur, cls))
    return out


INVENTORY = _gen_inventory()

# --- backward-compatible aliases for the original hand-named phonemes, so the
#     existing lexicon/tests keep working while we migrate names. ---
_ALIAS = {
    "Lf": (Band.LOW, Contour.FLAT, Dur.SHORT, SoundClass.TONE),
    "Lr": (Band.LOW, Contour.RISE, Dur.SHORT, SoundClass.TONE),
    "Ld": (Band.LOW, Contour.FALL, Dur.LONG, SoundClass.TONE),
    "Mf": (Band.MID, Contour.FLAT, Dur.SHORT, SoundClass.TONE),
    "Mr": (Band.MID, Contour.RISE, Dur.SHORT, SoundClass.TONE),
    "Mfl": (Band.MID, Contour.FALL, Dur.SHORT, SoundClass.TONE),
    "Ma": (Band.MID, Contour.ARCH, Dur.LONG, SoundClass.TONE),
    "Mi": (Band.MID, Contour.DIP, Dur.LONG, SoundClass.TONE),
    "Hf": (Band.HIGH, Contour.FLAT, Dur.SHORT, SoundClass.TONE),
    "Hr": (Band.HIGH, Contour.RISE, Dur.SHORT, SoundClass.TONE),
    "Hd": (Band.HIGH, Contour.FALL, Dur.SHORT, SoundClass.TONE),
    "Ha": (Band.HIGH, Contour.ARCH, Dur.LONG, SoundClass.TONE),
    "Grm": (Band.MID, Contour.FLAT, Dur.SHORT, SoundClass.GARGLE),
    "Grl": (Band.MID, Contour.FLAT, Dur.LONG, SoundClass.GARGLE),
    "Rz": (Band.LOW, Contour.FALL, Dur.SHORT, SoundClass.RASP),
}

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

# resolve aliases to the canonical generated phoneme with the same feature tuple
_BY_FEATURES = {(p.band, p.contour, p.dur, p.cls): p for p in INVENTORY}
for _alias_name, _feat in _ALIAS.items():
    if _alias_name not in BY_NAME and _feat in _BY_FEATURES:
        BY_NAME[_alias_name] = _BY_FEATURES[_feat]


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
