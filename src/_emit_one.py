#!/usr/bin/env python3
"""
Single-emission worker — plays+records ONE message and writes the clipped segment
to a wav, or exits nonzero on failure. Run as a subprocess by collect_dataset so a
hung audio call (AirPlay drop wedging CoreAudio) can be killed by the parent and
the OS reclaims the audio state — which an in-process thread timeout cannot do.

Usage:
  python3 src/_emit_one.py <out_wav> <latency> <gain> <concept1> [concept2 ...]
  (special: if the first concept is '__SELFTEST__', run the self-test instead)
Exit: 0 = wrote a good segment, 2 = no sync / too quiet, 3 = self-test fail.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.io import wavfile
from b4bl import capture, codec, prosody, generators as gen

MIN_RMS = 0.006


def main():
    out_wav = sys.argv[1]
    latency = float(sys.argv[2])
    gain = float(sys.argv[3])
    concepts = sys.argv[4:]
    pname = os.environ.get("B4BL_PROSODY", "neutral")
    input_dev = os.environ.get("B4BL_INPUT_DEVICE") or None
    output_dev = os.environ.get("B4BL_OUTPUT_DEVICE") or None
    capture.set_channel_profile(latency=latency, gain=gain,
                                input_device=input_dev, output_device=output_dev)

    if concepts[:1] == ["__SELFTEST__"]:
        sys.exit(0 if capture.self_test() else 3)

    prmap = {"neutral": prosody.NEUTRAL, "uncertain": prosody.UNCERTAIN,
             "urgent": prosody.URGENT, "calm": prosody.CALM}
    slots = os.environ.get("B4BL_SLOTS") == "1"   # symbol-clock encoding
    audio = codec.encode(concepts, prmap.get(pname, prosody.NEUTRAL), slots=slots)
    rec = capture.play_and_record(audio)          # may block if device wedges;
    seg = capture.find_message(rec)               # parent kills us on timeout.
    if seg is None or float(np.sqrt(np.mean(seg ** 2))) < MIN_RMS:
        sys.exit(2)
    wavfile.write(out_wav, gen.SR, (np.clip(seg, -1, 1) * 32767).astype(np.int16))
    sys.exit(0)


if __name__ == "__main__":
    main()
