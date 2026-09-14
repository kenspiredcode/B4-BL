"""
B4-BL frame-based decoder (CTC-style) — no explicit segmentation.

The gap-threshold segmenter fails on real audio: within-morpheme phoneme gaps
(~30ms) and noise-induced gaps are the same size, so no threshold separates them
(over-segmenting one way, under the other). Modern ASR avoids the problem entirely
by never deciding "where does a phoneme start": it classifies short overlapping
FRAMES across the whole utterance, then collapses runs of the same label into one
phoneme (Connectionist Temporal Classification, simplified here to greedy collapse
with a blank/silence class).

Pipeline:
  1. slide a short window over the utterance (FRAME_MS, HOP_MS)
  2. classify each frame -> phoneme label or SILENCE (blank)
  3. collapse consecutive equal labels; drop SILENCE; enforce a min run length
  4. group into words by long SILENCE runs

This module needs a model trained on FRAMES (see train_classifier frame mode),
not on pre-segmented phonemes. Falls back gracefully if that model is absent.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from . import generators as gen
from . import phonology as ph
from . import features as F

SR = gen.SR
FRAME_MS = 90        # window length for per-frame classification
HOP_MS = 20          # step between frames
SILENCE = "_"        # blank label
MIN_RUN = 2          # a phoneme run must persist >= this many frames to count
WORD_SIL_FRAMES = 5  # a SILENCE run >= this many frames = word boundary

_FRAME_MODEL = None
import os
_FRAME_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "frame_clf.joblib")


def load(path: str = _FRAME_MODEL_PATH):
    global _FRAME_MODEL
    if _FRAME_MODEL is not None:
        return _FRAME_MODEL
    if not os.path.exists(path):
        return None
    import joblib
    _FRAME_MODEL = joblib.load(path)
    return _FRAME_MODEL


def available() -> bool:
    return load() is not None


def frame_features(win: np.ndarray) -> np.ndarray:
    """Per-frame feature vector. Reuses the segment feature extractor on the short
    window (pitch shape is degenerate on a tiny window, but pitch center + spectral
    shape + flatness carry the frame's phoneme identity)."""
    return F.extract(win)


def _frames(audio: np.ndarray):
    w = int(FRAME_MS / 1000 * SR)
    hop = int(HOP_MS / 1000 * SR)
    for i in range(0, max(1, len(audio) - w), hop):
        yield i, audio[i:i + w]


def frame_probabilities(audio: np.ndarray):
    """Return (probs [T x C], classes) — per-frame class probabilities from the RF
    frame model. Feeds the CTC beam decoder (which needs the full distribution, not
    just the argmax the greedy collapse used)."""
    m = load()
    if m is None:
        return None, None
    model = m["model"]
    a = audio.astype(np.float32)
    if np.max(np.abs(a)) > 0:
        a = a / np.max(np.abs(a))
    feats = np.array([frame_features(w) for _, w in _frames(a)], dtype=np.float32)
    if len(feats) == 0:
        return None, None
    return model.predict_proba(feats), list(model.classes_)


def decode_frames(audio: np.ndarray) -> List[List[str]]:
    """Audio -> list of words, each a list of phoneme names. CTC-style greedy
    collapse. Requires the frame model; returns [] if unavailable."""
    m = load()
    if m is None:
        return []
    model = m["model"]
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    labels = []
    feats = []
    idxs = []
    for i, win in _frames(audio):
        feats.append(frame_features(win))
        idxs.append(i)
    if not feats:
        return []
    preds = model.predict(np.array(feats))

    # Split the frame stream into VOICED REGIONS separated by silence runs. Each
    # voiced region between short silences is ONE phoneme; the transition frames
    # at its edges get mislabeled, but the TRUE phoneme dominates the region (its
    # runs are long, the phantoms are 1-2 frames). So we emit the single label with
    # the most total frames in each region. A silence run >= WORD_SIL_FRAMES ends
    # a word. This is robust to boundary/transition-frame noise that defeated the
    # per-run-threshold collapse.
    from collections import Counter

    words: List[List[str]] = []
    current: List[str] = []          # phonemes of the current word
    region: Counter = Counter()      # frame counts per label in the current region
    sil_run = 0

    def flush_region():
        nonlocal region
        if region:
            lab, n = region.most_common(1)[0]
            if n >= MIN_RUN:
                current.append(lab)
        region = Counter()

    for p in preds:
        if p == SILENCE:
            sil_run += 1
            # a short silence ends the current phoneme region but stays in-word;
            # a long silence ends the word.
            if region:
                flush_region()
            if sil_run >= WORD_SIL_FRAMES and current:
                words.append(current)
                current = []
        else:
            sil_run = 0
            region[p] += 1
    flush_region()
    if current:
        words.append(current)
    return words
