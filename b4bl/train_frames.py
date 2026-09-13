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
    return np.array(X), np.array(y)


def train(samples_per=SAMPLES_PER_PHONEME, save=True):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib
    print(f"building frame dataset ({len(ph.INVENTORY)} phonemes x {samples_per})...")
    X, y = build_frame_dataset(samples_per)
    print(f"  {X.shape[0]} frames, {X.shape[1]} features")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, max_depth=18, n_jobs=-1, random_state=1)
    clf.fit(Xtr, ytr)
    print(f"  frame holdout accuracy: {clf.score(Xte, yte):.3f}")
    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({"model": clf, "feature_len": X.shape[1]}, MODEL_PATH)
        print(f"saved -> {MODEL_PATH}")
    return clf


if __name__ == "__main__":
    train()
