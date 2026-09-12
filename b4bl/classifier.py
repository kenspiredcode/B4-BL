"""
B4-BL — trained-classifier phoneme recognition.

Loads the random-forest models trained by train_classifier.py and classifies an
audio segment into a phoneme by predicting the four meaning-bearing feature
dimensions (band / contour / duration / class) and matching to the inventory.

Falls back to the threshold decoder if no trained model is present, so the system
always works (just less accurately) without the model file.
"""

from __future__ import annotations
import os
from typing import Optional, Tuple
import numpy as np

from . import phonology as ph
from . import features as F
from . import decoder as _dec

_MODEL = None
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "phoneme_clf.joblib")


def load(path: str = _MODEL_PATH):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not os.path.exists(path):
        return None
    import joblib
    _MODEL = joblib.load(path)
    return _MODEL


def available() -> bool:
    return load() is not None


def _nearest_by_features(band, contour, dur, cls) -> str:
    # exact, then progressively relaxed (same order of preference as the
    # threshold decoder's fallback)
    want = (band, contour, dur, cls)
    for p in ph.INVENTORY:
        if (p.band.value, p.contour.value, p.dur.value, p.cls.value) == want:
            return p.name
    for p in ph.INVENTORY:
        if (p.band.value, p.contour.value, p.cls.value) == (band, contour, cls):
            return p.name
    for p in ph.INVENTORY:
        if (p.band.value, p.cls.value) == (band, cls):
            return p.name
    for p in ph.INVENTORY:
        if p.band.value == band:
            return p.name
    return ph.INVENTORY[0].name


def classify_segment(seg: np.ndarray) -> Tuple[str, tuple]:
    """Trained-model classification. Returns (phoneme name, feature tuple)."""
    m = load()
    if m is None:
        return _dec.classify_segment(seg)   # fallback
    feats = F.extract(seg).reshape(1, -1)
    models = m["models"]
    band = models["band"].predict(feats)[0]
    contour = models["contour"].predict(feats)[0]
    dur = models["dur"].predict(feats)[0]
    cls = models["cls"].predict(feats)[0]
    name = _nearest_by_features(band, contour, dur, cls)
    return name, (band, contour, dur, cls)


def _topk(model, feats, k):
    """Return [(label, prob), ...] top-k for one sklearn model."""
    proba = model.predict_proba(feats)[0]
    classes = model.classes_
    order = np.argsort(proba)[::-1][:k]
    return [(classes[i], float(proba[i])) for i in order]


def phoneme_candidates(seg: np.ndarray, k: int = 2):
    """Return a ranked list of (phoneme_name, score) candidates for a segment.

    Combines top-k per feature dimension into candidate feature tuples, scored by
    the product of the per-dimension probabilities. This N-best output is what
    lets the lexicon layer recover a misheard phoneme by choosing the candidate
    that forms a valid word."""
    m = load()
    if m is None:
        name, _ = _dec.classify_segment(seg)
        return [(name, 1.0)]
    feats = F.extract(seg).reshape(1, -1)
    models = m["models"]
    bands = _topk(models["band"], feats, k)
    contours = _topk(models["contour"], feats, k)
    durs = _topk(models["dur"], feats, 1)      # duration is ~perfect, no need for k
    classes = _topk(models["cls"], feats, 1)   # class is ~perfect too
    cands = {}
    for b, pb in bands:
        for c, pc in contours:
            for d, pd in durs:
                for cl, pcl in classes:
                    name = _nearest_by_features(b, c, d, cl)
                    score = pb * pc * pd * pcl
                    # keep the best score per resulting phoneme name
                    if name not in cands or score > cands[name]:
                        cands[name] = score
    return sorted(cands.items(), key=lambda kv: kv[1], reverse=True)
