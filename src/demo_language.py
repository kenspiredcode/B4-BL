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

    # 4. repetition-axis morphemes
    gen.write_wav(os.path.join(OUT, "alarm.wav"), codec.encode(["ALARM"], prosody.URGENT))
    gen.write_wav(os.path.join(OUT, "calculating.wav"),
                  codec.encode(["CALCULATING"], prosody.NEUTRAL))
    # a little scene: "thinking... then alarm: obstacle ahead!"
    scene = np.concatenate([
        codec.encode(["CALCULATING"], prosody.CALM),
        codec._silence(0.35),
        codec.encode(["ALARM", "OBSTACLE", "FRONT"], prosody.URGENT),
    ])
    gen.write_wav(os.path.join(OUT, "scene_think_then_alarm.wav"), scene)

    # 5. interjections (emotive, standalone)
    from b4bl import interjections as itj
    gen.write_wav(os.path.join(OUT, "interjections.wav"), itj.montage())
    # "tell the bot no -> it goes awww"
    no_aww = np.concatenate([codec.encode(["DENY"], prosody.NEUTRAL),
                             codec._silence(0.2), itj.render("AWW")])
    gen.write_wav(os.path.join(OUT, "scene_no_aww.wav"), no_aww)
    # "good job -> yay!"
    praise_yay = np.concatenate([codec.encode(["DONE"], prosody.CALM),
                                 codec._silence(0.15), itj.render("YAY")])
    gen.write_wav(os.path.join(OUT, "scene_done_yay.wav"), praise_yay)

    # 6. the human-legible speech acts, in a row
    acts = ["QUERY", "CONFIRM", "CLARIFY", "ACK", "DENY", "WARN", "DONE",
            "REPEAT", "WAIT", "ERROR", "ALARM", "WORKING"]
    clips2 = []
    for a in acts:
        clips2.append(codec.encode([a], prosody.NEUTRAL))
        clips2.append(codec._silence(0.35))
    gen.write_wav(os.path.join(OUT, "speech_acts.wav"), np.concatenate(clips2))

    # 7. a number: "go to room 3" (MOVE ROOM NUM D3)
    num = codec.encode(["GO", "ROOM"] + lex.number_to_concepts(3), prosody.NEUTRAL)
    gen.write_wav(os.path.join(OUT, "say_go_room_3.wav"), num)

    print("rendered inventory, sentences, prosody, repetition, interjections, "
          "speech-acts, numerals into audio_samples/")


if __name__ == "__main__":
    main()
