"""
B4-BL — decode using the neural-net frame classifier (Option 2).

Same CTC-style pipeline as frame_decoder, but per-frame predictions come from the
MLP-with-temporal-context (train_nn.py) instead of the random forest. Reuses the
region-dominant collapse + word splitting.
"""

from __future__ import annotations
from typing import List
import os
import numpy as np

from . import generators as gen
from . import features as F
from . import frame_decoder as fd
from .train_nn import CONTEXT, MODEL_PATH, _stack_context

SR = gen.SR
_NET = None
_LABELS = None
_FEAT_DIM = None


def load():
    global _NET, _LABELS, _FEAT_DIM
    if _NET is not None:
        return _NET
    if not os.path.exists(MODEL_PATH):
        return None
    import torch, torch.nn as nn
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    labels = ckpt["labels"]
    net = nn.Sequential(
        nn.Linear(ckpt["in_dim"], 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, len(labels)),
    )
    net.load_state_dict(ckpt["state"]); net.eval()
    _NET = net; _LABELS = labels; _FEAT_DIM = ckpt["feat_dim"]
    return _NET


def available() -> bool:
    return load() is not None


def _predict_frames(audio):
    import torch
    net = load()
    a = audio.astype(np.float32)
    if np.max(np.abs(a)) > 0:
        a = a / np.max(np.abs(a))
    feats = np.array([F.extract(w) for _, w in fd._frames(a)], dtype=np.float32)
    if len(feats) == 0:
        return []
    seq = np.zeros(len(feats))          # single utterance
    Xc = _stack_context(feats, seq)
    with torch.no_grad():
        out = net(torch.tensor(Xc))
        idx = out.argmax(1).numpy()
    return [_LABELS[i] for i in idx]


def decode_frames(audio: np.ndarray) -> List[List[str]]:
    """NN version of frame_decoder.decode_frames (region-dominant collapse)."""
    net = load()
    if net is None:
        return fd.decode_frames(audio)   # fall back to RF frame decoder
    preds = _predict_frames(audio)
    if not preds:
        return []
    from collections import Counter
    words: List[List[str]] = []
    current: List[str] = []
    region: Counter = Counter()
    sil_run = 0

    def flush():
        nonlocal region
        if region:
            lab, n = region.most_common(1)[0]
            if n >= fd.MIN_RUN:
                current.append(lab)
        region = Counter()

    for p in preds:
        if p == fd.SILENCE:
            sil_run += 1
            if region:
                flush()
            if sil_run >= fd.WORD_SIL_FRAMES and current:
                words.append(current); current = []
        else:
            sil_run = 0
            region[p] += 1
    flush()
    if current:
        words.append(current)
    return words
