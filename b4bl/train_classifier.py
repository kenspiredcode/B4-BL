"""
B4-BL — train the phoneme classifier on synth-generated, augmented data.

Why this is feasible on a laptop CPU:
  - the synth IS the label source (render a phoneme -> we know its true features),
    so we generate unlimited labeled data for free;
  - ~32 classes, short feature vectors, classical ML (random forests) -> trains in
    well under a minute on CPU, no GPU.

We train FOUR small classifiers, one per meaning-bearing feature dimension
(band / contour / duration / sound-class), which is more robust and more
inspectable than one 32-way classifier.

Augmentation makes the model robust and is where the real-world V2 will plug in:
  - random prosody (confidence/urgency) — what the runtime actually varies
  - additive noise, gain changes, pitch jitter, light reverb — stand-ins for a
    real room until we have actual mic recordings to train on.

Run:  python3 -m b4bl.train_classifier
Saves: b4bl/models/phoneme_clf.joblib
"""

from __future__ import annotations
import os
import numpy as np

from . import phonology as ph
from . import prosody
from . import features as F
from . import generators as gen

SR = gen.SR
_rng = np.random.default_rng(20260912)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "phoneme_clf.joblib")
SAMPLES_PER_PHONEME = 400


# ---------------------------------------------------------------------------
# augmentation — stand-ins for real-world variation until we have mic data
# ---------------------------------------------------------------------------
def _augment(sig: np.ndarray) -> np.ndarray:
    x = sig.astype(float).copy()
    # gain
    x *= _rng.uniform(0.6, 1.0)
    # additive noise (SNR ~ 15-40 dB)
    snr = _rng.uniform(15, 40)
    p_sig = np.mean(x ** 2) + 1e-9
    p_noise = p_sig / (10 ** (snr / 10))
    x += _rng.normal(0, np.sqrt(p_noise), len(x))
    # light reverb: a couple of decaying echoes
    if _rng.random() < 0.5 and len(x) > 2000:
        y = x.copy()
        for delay_ms, g in [(_rng.uniform(8, 25), _rng.uniform(0.1, 0.3)),
                            (_rng.uniform(30, 60), _rng.uniform(0.05, 0.15))]:
            d = int(delay_ms / 1000 * SR)
            if d < len(x):
                y[d:] += g * x[:-d]
        x = y
    return x.astype(np.float32)


def _random_prosody():
    conf = float(np.clip(_rng.normal(0.75, 0.22), 0.15, 1.0))
    urg = float(np.clip(_rng.normal(0.35, 0.25), 0.0, 1.0))
    return prosody.Prosody(confidence=conf, urgency=urg)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
def _segmented_render(p, pr):
    """Render a phoneme the way the runtime will actually hear it: as a single
    utterance passed through the SAME segmentation the decoder uses, so the
    training features match inference features (avoids the clean-render vs
    segmented-render mismatch). Falls back to the raw render if segmentation
    finds nothing."""
    from . import decoder as _dec
    sig = p.render(pr)
    # small leading/trailing silence so the segmenter sees a real onset/offset
    pad = np.zeros(int(0.05 * SR), dtype=np.float32)
    utt = np.concatenate([pad, sig, pad])
    segs = _dec._segments(utt)
    if segs:
        s, e, _g = max(segs, key=lambda z: z[1] - z[0])
        return utt[s:e]
    return sig


def build_dataset(samples_per=SAMPLES_PER_PHONEME):
    """Mix CLEAN and AUGMENTED samples so the model is good on both quiet
    close-mic audio and noisy/reverberant audio. ~40% clean, 60% augmented."""
    X, y_band, y_contour, y_dur, y_cls = [], [], [], [], []
    for p in ph.INVENTORY:
        for i in range(samples_per):
            seg = _segmented_render(p, _random_prosody())
            if i % 5 >= 2:            # 60% augmented, 40% clean
                seg = _augment(seg)
            X.append(F.extract(seg))
            y_band.append(p.band.value)
            y_contour.append(p.contour.value)
            y_dur.append(p.dur.value)
            y_cls.append(p.cls.value)
    return (np.array(X), np.array(y_band), np.array(y_contour),
            np.array(y_dur), np.array(y_cls))


def train(samples_per=SAMPLES_PER_PHONEME, save=True):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib

    print(f"generating dataset ({len(ph.INVENTORY)} phonemes x {samples_per})...")
    X, yb, yc, yd, ycl = build_dataset(samples_per)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features each")

    models = {}
    reports = {}
    for name, y in [("band", yb), ("contour", yc), ("dur", yd), ("cls", ycl)]:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
        # capped depth + fewer trees keeps the saved model small (committable)
        # with negligible accuracy loss on this well-separated problem.
        clf = RandomForestClassifier(n_estimators=80, max_depth=16, n_jobs=-1,
                                     random_state=1)
        clf.fit(Xtr, ytr)
        acc = clf.score(Xte, yte)
        models[name] = clf
        reports[name] = acc
        print(f"  {name:8} holdout accuracy: {acc:.3f}")

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({"models": models, "feature_len": X.shape[1]}, MODEL_PATH)
        print(f"saved -> {MODEL_PATH}")
    return models, reports


if __name__ == "__main__":
    train()
