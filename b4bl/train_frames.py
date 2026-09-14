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


def build_frame_dataset_from_recordings(manifest_path, force_align_model=None):
    """Frame-label real recordings.

    First try SEGMENT-alignment (segment count == expected phoneme count -> label
    each segment's frames). For recordings that fail that, if a `force_align_model`
    is given, FORCE-ALIGN them: use the model's per-frame predictions + the known
    phoneme sequence to recover per-frame labels via constrained DP (see
    b4bl.align). This unlocks the ~55% of recordings that silence-segmentation
    rejects — the harder ones we're failing on."""
    import json
    from scipy.io import wavfile
    from . import decoder as _dec, lexicon as lex, codec
    from . import align as _align
    rec_dir = os.path.dirname(manifest_path)
    X, y = [], []
    seg_used = forced = skipped = 0
    w = int(90 / 1000 * SR)  # must match frame_decoder FRAME_MS for label spans
    for line in open(manifest_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        exp = [p for w2 in codec.concepts_to_phoneme_words(r["concepts"]) for p in w2]
        sr, d = wavfile.read(os.path.join(rec_dir, r["file"]))
        audio = d.astype(np.float32) / 32768.0
        segs = [(s, e) for (s, e, _g) in _dec._segments(audio)]
        if len(segs) == len(exp):
            spans = [(s, e, pname) for (s, e), pname in zip(segs, exp)]
            _label_frames(audio, spans, X, y)
            seg_used += 1
        elif force_align_model is not None:
            res = _align.aligned_frame_labels(audio, exp, force_align_model)
            if res is None:
                skipped += 1
                continue
            starts, labels = res
            # emit a feature+label per aligned frame directly (labels already
            # per-frame, aligned to the frame_decoder framing)
            a = audio / (np.max(np.abs(audio)) or 1.0)
            for st, lab in zip(starts, labels):
                X.append(F.extract(a[st:st + w])); y.append(lab)
            forced += 1
        else:
            skipped += 1
    print(f"  real: {seg_used} seg-aligned + {forced} force-aligned, "
          f"{skipped} skipped; {len(X)} frames")
    return X, y


def train(samples_per=SAMPLES_PER_PHONEME, save=True, recordings=None, mix_synth=True,
          force_align=False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib
    X, y = [], []
    if recordings:
        fa_model = None
        if force_align:
            from .frame_decoder import load as _load
            m = _load()
            fa_model = m["model"] if m else None
            print("force-align: using existing frame model as aligner"
                  if fa_model else "force-align requested but no model found; skipping")
        print(f"building frame dataset from real recordings {recordings}...")
        Xr, yr = build_frame_dataset_from_recordings(recordings, force_align_model=fa_model)
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
    ap.add_argument("--force-align", action="store_true",
                    help="force-align recordings that fail segment-alignment, using "
                         "the existing frame model (bootstrap: unlocks the hard ~55%)")
    ap.add_argument("--samples", type=int, default=SAMPLES_PER_PHONEME)
    args = ap.parse_args()
    train(samples_per=args.samples, recordings=args.recordings,
          mix_synth=not args.no_mix_synth, force_align=args.force_align)
