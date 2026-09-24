"""Stage everything the GitHub Pages demo needs into docs/.

GitHub Pages serves docs/ as the site root, so the demo cannot reach
../audio_samples/. This copies the handful of samples the page actually uses
into docs/audio/ and regenerates docs/data/language.json.

Run after changing the lexicon, phonology, grammar, or the demo's sample list.
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, "docs")
SRC_AUDIO = os.path.join(ROOT, "audio_samples")
DST_AUDIO = os.path.join(DOCS, "audio")

# Only what the page references. Keep this list short: it ships with the site.
SAMPLES = [
    "prosody_neutral.wav",
    "prosody_uncertain.wav",
    "prosody_urgent.wav",
    "prosody_calm.wav",
    "phoneme_inventory.wav",
    "sentences_neutral.wav",
    "clocked_packet_full.wav",
    "clocked_message.wav",
]


def main():
    for script in ("tools_export_language.py", "tools_render_packet_figure.py"):
        subprocess.check_call([sys.executable, os.path.join(ROOT, script)])

    # The browser synth must stay equivalent to codec.encode. Skipped silently
    # when node is unavailable; run tools_verify_port.py directly to see why.
    from shutil import which
    if which("node"):
        subprocess.check_call([sys.executable,
                               os.path.join(ROOT, "tools_verify_port.py")])
    else:
        print("note: node not found; skipping synth port verification")

    os.makedirs(DST_AUDIO, exist_ok=True)
    missing = []
    total = 0
    for name in SAMPLES:
        src = os.path.join(SRC_AUDIO, name)
        if not os.path.exists(src):
            missing.append(name)
            continue
        shutil.copy2(src, os.path.join(DST_AUDIO, name))
        total += os.path.getsize(src)

    if missing:
        raise SystemExit(
            "missing audio samples: " + ", ".join(missing) +
            "\nrun: python3 src/demo_language.py"
        )

    print(f"staged {len(SAMPLES)} samples into docs/audio/ "
          f"({total / 1e6:.1f} MB)")

    data = json.load(open(os.path.join(DOCS, "data", "language.json")))
    print(f"language.json: {len(data['phonemes'])} phonemes, "
          f"{len(data['morphemes'])} morphemes, "
          f"{len(data['templates'])} templates")


if __name__ == "__main__":
    main()
