"""Render A/B (legacy vs symbol-clock/slotted) audio for ear-verification.

For each demo sentence we render the SAME concepts two ways and write them to
audio_samples/slot_ab/ so Ken can hear whether the slotted symbol-clock + widened
gaps + geminate marker preserve the R2 character.

Run: python3 src/render_slot_ab.py
"""
import os
import numpy as np
from b4bl import codec, generators as gen, lexicon as lex
from b4bl.prosody import NEUTRAL

OUT = os.path.join(os.path.dirname(__file__), "..", "audio_samples", "slot_ab")
os.makedirs(OUT, exist_ok=True)

# sentences chosen to cover: single word, multi-word, adjacent-identical phonemes
# (WARN = Hr,Hr and "...FRONT" ends Hd,Hd -> exercises the geminate marker), and
# a rhythm-defined Rep morpheme (ALARM) which must stay UNslotted.
SENTENCES = [
    ("query_energy_low", ["QUERY", "YOU", "ENERGY", "LOW"]),
    ("self_energy_low",  ["SELF", "ENERGY", "LOW"]),
    ("warn_obstacle_front", ["WARN", "OBSTACLE", "FRONT"]),   # geminate x2
    ("follow_me",        ["YOU", "FOLLOW", "SELF"]),
    ("bring_cup",        ["YOU", "BRING", "SELF", "CUP"]),
    ("stop",             ["STOP"]),
    ("alarm",            ["ALARM"]),                          # rhythm morpheme
]


def _pair(concepts):
    legacy = codec.encode(concepts, prosody=NEUTRAL, slots=False)
    slotted = codec.encode(concepts, prosody=NEUTRAL, slots=True)
    return legacy, slotted


def main():
    for name, concepts in SENTENCES:
        legacy, slotted = _pair(concepts)
        gen.write_wav(os.path.join(OUT, f"{name}__A_legacy.wav"), legacy)
        gen.write_wav(os.path.join(OUT, f"{name}__B_slotted.wav"), slotted)
        # also a back-to-back A|B in one file (short silence between) for easy compare
        both = np.concatenate([legacy, np.zeros(int(gen.SR * 0.6), np.float32), slotted])
        gen.write_wav(os.path.join(OUT, f"{name}__AB.wav"), both)
        print(f"{name:24s} legacy={len(legacy)/gen.SR:.2f}s  slotted={len(slotted)/gen.SR:.2f}s")

    # isolate the geminate marker so Ken can hear exactly what it sounds like:
    # WARN alone (Hr,Hr) with and without slotting.
    warn_leg = codec.encode(["WARN"], slots=False)
    warn_slot = codec.encode(["WARN"], slots=True)
    gen.write_wav(os.path.join(OUT, "geminate_demo__WARN_A_legacy.wav"), warn_leg)
    gen.write_wav(os.path.join(OUT, "geminate_demo__WARN_B_slotted.wav"), warn_slot)
    print("wrote geminate WARN demo (legacy merges the two Hr; slotted separates them)")
    print("\nfiles in", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
