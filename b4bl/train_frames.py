"""
Train the FRAME classifier for the CTC-style frame_decoder.

Ground-truth frame labels are free from synthesis: render each phoneme (with
prosody + augmentation), then every frame fully inside it is labeled with that
phoneme; frames of silence/padding are labeled SILENCE. We also render short
2-3 phoneme sequences with realistic inter-phoneme gaps so the model sees frames
at phoneme BOUNDARIES and in silence between them — the situations the segmenter
couldn't handle.

Run:  python3 -m b4bl.train_frames
Saves: b4bl/models/frame_clf.joblib
"""

from __future__ import annotations
import os
import numpy as np

from . import phonology as ph
from . import prosody, generators as gen
from . import features as F
from .frame_decoder import SILENCE, FRAME_MS, HOP_MS
from .train_classifier import _augment, _random_prosody

SR = gen.SR
_rng = np.random.default_rng(4242)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "frame_clf.joblib")
SAMPLES_PER_PHONEME = 120


def _label_frames(audio, spans, out_X, out_y):
    """spans: list of (start_sample, end_sample, label). Emit a feature vector +
    label for each frame; a frame's label is the span it falls in, else SILENCE."""
    w = int(FRAME_MS / 1000 * SR)
    hop = int(HOP_MS / 1000 * SR)
    for i in range(0, max(1, len(audio) - w), hop):
        c = i + w // 2   # frame center
        label = SILENCE
        for s, e, lab in spans:
            if s <= c < e:
                label = lab
                break
        out_X.append(F.extract(audio[i:i + w]))
        out_y.append(label)


def build_frame_dataset(samples_per=SAMPLES_PER_PHONEME):
    X, y = [], []
    sil = np.zeros(int(0.12 * SR), dtype=np.float32)
    for p in ph.INVENTORY:
        for _ in range(samples_per):
            pr = _random_prosody()
            sig = p.render(pr)
            # surround with silence so boundary + silence frames are labeled too
            audio = np.concatenate([sil, sig, sil])
            audio = _augment(audio)
            s = len(sil); e = s + len(sig)
            _label_frames(audio, [(s, e, p.name)], X, y)
    # a batch of pure-silence frames (noise floor) so SILENCE is well represented
    for _ in range(samples_per * 4):
        audio = _augment(np.zeros(int(0.2 * SR), dtype=np.float32) + 1e-4)
        _label_frames(audio, [], X, y)
    return list(X), list(y)


def build_frame_dataset_from_recordings(manifest_path):
    """Frame-label real recordings. We only know the phoneme SEQUENCE per file,
    not per-frame timing — so we segment each recording and, when the segment
    count matches the expected phoneme count (clean alignment), label each
    segment's frames with its phoneme and the gaps between as SILENCE. Recordings
    that don't align are skipped (same policy as the segment retrain)."""
    import json
    from scipy.io import wavfile
    from . import decoder as _dec, lexicon as lex, codec
    rec_dir = os.path.dirname(manifest_path)
    X, y = [], []
    used = skipped = 0
    for line in open(manifest_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        exp = [p for w in codec.concepts_to_phoneme_words(r["concepts"]) for p in w]
        sr, d = wavfile.read(os.path.join(rec_dir, r["file"]))
        audio = d.astype(np.float32) / 32768.0
        segs = [(s, e) for (s, e, _g) in _dec._segments(audio)]
        if len(segs) != len(exp):
            skipped += 1
            continue
        spans = [(s, e, pname) for (s, e), pname in zip(segs, exp)]
        _label_frames(audio, spans, X, y)   # gaps between spans auto-labeled SILENCE
        used += 1
    print(f"  real: {used} recordings aligned, {skipped} skipped; {len(X)} frames")
    return X, y


def train(samples_per=SAMPLES_PER_PHONEME, save=True, recordings=None, mix_synth=True):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib
    X, y = [], []
    if recordings:
        print(f"building frame dataset from real recordings {recordings}...")
        Xr, yr = build_frame_dataset_from_recordings(recordings)
        X += Xr; y += yr
        if mix_synth:
            print(f"mixing synthetic frames ({len(ph.INVENTORY)} x {samples_per})...")
            Xs, ys = build_frame_dataset(samples_per)
            X += Xs; y += ys
    else:
        print(f"building frame dataset ({len(ph.INVENTORY)} phonemes x {samples_per})...")
        Xs, ys = build_frame_dataset(samples_per)
        X += Xs; y += ys
    import numpy as _np
    X = _np.array(X); y = _np.array(y)
    print(f"  {X.shape[0]} frames, {X.shape[1]} features")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    # cap trees + leaf size so the saved model stays small (unbounded trees on
    # ~190k frames produced a ~1GB model); accuracy is barely affected.
    clf = RandomForestClassifier(n_estimators=60, max_depth=16, min_samples_leaf=4,
                                 n_jobs=-1, random_state=1)
    clf.fit(Xtr, ytr)
    print(f"  frame holdout accuracy: {clf.score(Xte, yte):.3f}")
    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({"model": clf, "feature_len": X.shape[1]}, MODEL_PATH)
        print(f"saved -> {MODEL_PATH}")
    return clf


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default=None,
                    help="manifest.jsonl of real recordings to frame-label + train on")
    ap.add_argument("--no-mix-synth", action="store_true")
    ap.add_argument("--samples", type=int, default=SAMPLES_PER_PHONEME)
    args = ap.parse_args()
    train(samples_per=args.samples, recordings=args.recordings,
          mix_synth=not args.no_mix_synth)
