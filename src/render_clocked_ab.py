"""Save a short listening comparison; never play audio or open a microphone.
Run from the repository: python3 src/render_clocked_ab.py
A = current widened-gap slots; B = experimental synchronized word framing.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b4bl import codec, clocked, generators as gen, prosody

OUT = Path(__file__).resolve().parents[1] / 'audio_samples/scratch/clocked_timing'
CASES = [
    ('01_battery_neutral', ['SELF', 'ENERGY', 'LOW'], prosody.NEUTRAL),
    ('02_question_uncertain', ['QUERY', 'YOU', 'ENERGY', 'LOW'], prosody.UNCERTAIN),
    ('03_warning_urgent', ['WARN', 'OBSTACLE', 'FRONT'], prosody.URGENT),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, words, affect in CASES:
        old = codec.encode(words, affect, slots=True)
        new = clocked.encode(words, affect)
        # One shared gain preserves the relative levels of each A/B pair.
        combined = np.concatenate([old, np.zeros(gen.SR, np.float32), new])
        path = OUT / (name + '__AB.wav')
        gen.write_wav(str(path), combined)
        sr, saved = wavfile.read(path)
        assert sr == gen.SR and saved.ndim == 1 and len(saved) == len(combined)
        assert np.isfinite(saved).all() and np.max(np.abs(saved.astype(float))) < 32768
        records.append(dict(file=str(path), concepts=words, first='widened-gap slots (current)',
                            second='clocked prototype', prototype_starts_seconds=len(old)/gen.SR+1,
                            duration_seconds=len(combined)/gen.SR))
    voices = []
    for affect in (prosody.NEUTRAL, prosody.UNCERTAIN, prosody.URGENT, prosody.CALM):
        voices.extend([clocked.encode(['SELF', 'ENERGY', 'LOW'], affect), np.zeros(gen.SR, np.float32)])
    gen.write_wav(str(OUT/'04_prototype_prosodies.wav'), np.concatenate(voices[:-1]))
    gen.write_wav(str(OUT/'05_word_marker.wav'), np.concatenate([np.zeros(gen.SR//4), clocked.MARKER, np.zeros(gen.SR//4)]))
    (OUT/'listening_index.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
