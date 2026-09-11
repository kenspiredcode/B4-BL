#!/usr/bin/env python3
"""
B4-BL language demo — render the example sentences to audio, both neutral and
with prosody, plus the phoneme inventory montage.

Run: python3 src/demo_language.py
Outputs into audio_samples/.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from b4bl import codec, lexicon as lex, phonology as ph, generators as gen
from b4bl import prosody

OUT = os.path.join(os.path.dirname(__file__), "..", "audio_samples")


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1. phoneme inventory
    gen.write_wav(os.path.join(OUT, "phoneme_inventory.wav"),
                  ph.render_inventory_montage())

    # 2. each example sentence, neutral
    clips = []
    for eng, concepts in lex.example_sentences():
        audio = codec.encode(concepts, prosody.NEUTRAL)
        safe = eng.strip("?.").replace(" ", "_").replace(":", "").lower()[:24]
        gen.write_wav(os.path.join(OUT, f"say_{safe}.wav"), audio)
        clips.append(audio)
        clips.append(codec._silence(0.4))
    gen.write_wav(os.path.join(OUT, "sentences_neutral.wav"),
                  __import__("numpy").concatenate(clips))

    # 3. same sentence, four prosodies -> proves affect rides on fixed payload
    concepts = ["QUERY", "YOU", "ENERGY", "LOW"]
    import numpy as np
    variants = []
    for name, pr in [("neutral", prosody.NEUTRAL), ("uncertain", prosody.UNCERTAIN),
                     ("urgent", prosody.URGENT), ("calm", prosody.CALM)]:
        a = codec.encode(concepts, pr)
        gen.write_wav(os.path.join(OUT, f"prosody_{name}.wav"), a)
        variants.append(a)
        variants.append(codec._silence(0.4))
    gen.write_wav(os.path.join(OUT, "prosody_compare.wav"), np.concatenate(variants))

    print("rendered inventory, sentences, and prosody comparison into audio_samples/")


if __name__ == "__main__":
    main()
