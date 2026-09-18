#!/usr/bin/env python3
"""Render the optional compact-packet spoken replies; never plays audio."""
from pathlib import Path
import sys

from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from b4bl import clocked, compact_clocked
from b4bl.prosody import NEUTRAL, UNCERTAIN


def main():
    output = Path(__file__).resolve().parent.parent/'audio_samples'
    output.mkdir(exist_ok=True)
    samples = {
        'packet_reply_confirm.wav': (compact_clocked.CONFIRM_REPLY, NEUTRAL),
        'packet_reply_repeat.wav': (compact_clocked.REPEAT_REPLY, UNCERTAIN),
    }
    for filename, (concepts, expression) in samples.items():
        path = output/filename
        wavfile.write(path, clocked.SR, clocked.encode(concepts, expression))
        print(path)


if __name__ == '__main__':
    main()
