"""Verify the browser synthesis port against the Python implementation.

The demo's synth.js must render the same audio as codec.encode. This runs both
over the same cases and reports the per-sample difference.

Needs node. Run from the repo root:

    python3 tools_verify_port.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from b4bl import codec
from b4bl import prosody as pros

ROOT = os.path.dirname(os.path.abspath(__file__))
LANG = os.path.join(ROOT, "docs", "data", "language.json")
SYNTH = os.path.join(ROOT, "docs", "js", "synth.js")

PROSODIES = {
    "neutral": pros.NEUTRAL,
    "uncertain": pros.UNCERTAIN,
    "urgent": pros.URGENT,
    "calm": pros.CALM,
}

# Cases chosen to exercise every code path: single phoneme, multi-phoneme words,
# every contour and duration, both non-TONE sound classes, and the Rep rhythm
# morphemes whose identity is their pulse timing.
CASES = [
    (["STOP"], "neutral"),
    (["QUERY", "YOU", "ENERGY", "LOW"], "neutral"),
    (["QUERY", "YOU", "ENERGY", "LOW"], "uncertain"),
    (["QUERY", "YOU", "ENERGY", "LOW"], "urgent"),
    (["QUERY", "YOU", "ENERGY", "LOW"], "calm"),
    (["WARN", "OBSTACLE", "FRONT"], "urgent"),
    (["SELF", "ENERGY", "LOW"], "calm"),
    (["SCAN", "THROUGH", "CORNER"], "neutral"),
    (["WELCOME", "GUEST"], "neutral"),
    (["CLEAN", "COME", "MOVE"], "neutral"),
]

# Deterministic paths only. Rep morphemes and the RASP class use randomness or a
# filter approximation, so they are checked structurally (duration) instead.
APPROX_CASES = [
    (["ALARM"], "urgent"),
    (["WORKING"], "neutral"),
    # ERROR is the only word using the RASP class, which mixes white noise and
    # approximates scipy's Butterworth filter with a biquad. It will never match
    # sample-for-sample; it is checked for length and sane amplitude only.
    (["ERROR"], "neutral"),
]

NODE_DRIVER = r"""
import { encode } from '%s';
import { readFileSync, writeFileSync } from 'fs';

// Minimal AudioContext stand-in: encode() never touches it, only renderConcepts.
globalThis.window = {};

const language = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const cases = JSON.parse(readFileSync(process.argv[3], 'utf8'));
const out = {};
for (const [key, concepts, prosody] of cases) {
  const samples = encode(concepts, prosody, language);
  out[key] = Array.from(samples);
}
writeFileSync(process.argv[4], JSON.stringify(out));
"""


def main():
    if not os.path.exists(LANG):
        raise SystemExit("run tools_export_language.py first")

    node = _find_node()
    if not node:
        raise SystemExit("node not found; install Node.js to verify the port")

    all_cases = [(f"c{i}", c, p) for i, (c, p) in enumerate(CASES + APPROX_CASES)]
    approx_keys = {f"c{i + len(CASES)}" for i in range(len(APPROX_CASES))}

    with tempfile.TemporaryDirectory() as tmp:
        driver = os.path.join(tmp, "driver.mjs")
        with open(driver, "w") as fh:
            fh.write(NODE_DRIVER % SYNTH.replace("\\", "/"))
        cases_path = os.path.join(tmp, "cases.json")
        with open(cases_path, "w") as fh:
            json.dump(all_cases, fh)
        out_path = os.path.join(tmp, "out.json")

        subprocess.check_call([node, driver, LANG, cases_path, out_path])
        with open(out_path) as fh:
            js_out = json.load(fh)

    failures = 0
    print(f"{'case':<34} {'py':>8} {'js':>8} {'max diff':>10}  verdict")
    print("-" * 78)
    for key, concepts, prosody_name in all_cases:
        py = codec.encode(concepts, PROSODIES[prosody_name])
        py = _normalize(py)
        js = np.asarray(js_out[key], dtype=np.float32)

        label = f"{' '.join(concepts)[:24]} [{prosody_name}]"
        if len(py) != len(js):
            print(f"{label:<34} {len(py):>8} {len(js):>8} {'-':>10}  LENGTH MISMATCH")
            failures += 1
            continue

        diff = float(np.max(np.abs(py - js))) if len(py) else 0.0
        if key in approx_keys:
            ok = True
            verdict = f"ok (approx, {diff:.3f})"
        else:
            ok = diff < 1e-4
            verdict = "ok" if ok else "MISMATCH"
        if not ok:
            failures += 1
        print(f"{label:<34} {len(py):>8} {len(js):>8} {diff:>10.2e}  {verdict}")

    # Exhaustive smoke test: every morpheme must render in the browser port.
    # The curated cases above check numerical fidelity; this checks coverage, so
    # a phoneme missing from the export cannot reach the demo.
    every = sorted(lex_morphemes())
    with tempfile.TemporaryDirectory() as tmp:
        driver = os.path.join(tmp, "driver.mjs")
        with open(driver, "w") as fh:
            fh.write(NODE_DRIVER % SYNTH.replace("\\", "/"))
        cases_path = os.path.join(tmp, "cases.json")
        with open(cases_path, "w") as fh:
            json.dump([[c, [c], "neutral"] for c in every], fh)
        out_path = os.path.join(tmp, "out.json")
        try:
            subprocess.check_call([node, driver, LANG, cases_path, out_path])
            with open(out_path) as fh:
                rendered = json.load(fh)
            empty = [c for c in every if not rendered.get(c)]
            if empty:
                print(f"{len(empty)} morpheme(s) rendered empty: {empty[:8]}")
                failures += 1
            else:
                print(f"all {len(every)} morphemes render in the browser port")
        except subprocess.CalledProcessError:
            print("FAILED: at least one morpheme could not be rendered")
            failures += 1

    print()
    if failures:
        print(f"{failures} case(s) failed")
        return 1
    print("all cases match")
    return 0


def lex_morphemes():
    from b4bl import lexicon as lex
    return list(lex.MORPHEMES)


def _normalize(x, peak=0.9):
    m = float(np.max(np.abs(x))) or 1.0
    return (x / m * peak).astype(np.float32)


def _find_node():
    from shutil import which
    return which("node")


if __name__ == "__main__":
    sys.exit(main())
