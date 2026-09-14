"""
B4-BL — force-alignment of real recordings to their known phoneme sequence.

~55% of real recordings fail silence-based segmentation (wrong segment count), so
they never contribute frame-training data. But we KNOW each recording's phoneme
SEQUENCE (its label) — we just don't know the per-frame timing. Force-alignment
recovers the timing: given the frame classifier's per-frame probabilities and the
known ordered phoneme sequence, find the monotonic frame->phoneme assignment that
best agrees with the classifier. This is a constrained DP (like the alignment step
in ASR / DTW), and it unlocks the hard recordings as labeled frame data.

Output: for each frame, the phoneme it belongs to (or SILENCE), consistent with
the known sequence order. Frames whose best fit is a big disagreement can be left
as-is (we trust the constraint).
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from . import generators as gen
from . import features as F
from . import frame_decoder as fd

SR = gen.SR


def _frame_logprobs(audio, model):
    """Per-frame log-probability over the model's classes; returns (logP, classes,
    frame_starts)."""
    a = audio.astype(np.float32)
    if np.max(np.abs(a)) > 0:
        a = a / np.max(np.abs(a))
    feats, starts = [], []
    for i, win in fd._frames(a):
        feats.append(fd.frame_features(win)); starts.append(i)
    if not feats:
        return None, None, None
    proba = model.predict_proba(np.array(feats))
    logP = np.log(proba + 1e-9)
    return logP, list(model.classes_), starts


def force_align(audio, phoneme_seq, model) -> Optional[List[str]]:
    """Return a per-frame label list aligned to phoneme_seq (with SILENCE allowed
    between/around phonemes), or None if it can't be aligned.

    DP over frames x sequence-position. State = which phoneme of the sequence we
    are 'in' (or in a silence gap before it). Monotonic: we only advance. Each
    frame scores the log-prob of the current state's label.
    """
    logP, classes, starts = _frame_logprobs(audio, model)
    if logP is None:
        return None
    cls_idx = {c: i for i, c in enumerate(classes)}
    sil = cls_idx.get(fd.SILENCE)
    T = len(logP)
    # Build the state sequence: SIL, p0, SIL, p1, SIL, ..., pN, SIL
    # (silence optional between phonemes). Each phoneme state must be entered.
    states = []          # (label, is_phoneme)
    states.append((fd.SILENCE, False))
    for p in phoneme_seq:
        states.append((p, True))
        states.append((fd.SILENCE, False))
    S = len(states)
    NEG = -1e18
    # dp[t][s] = best score aligning first t frames ending in state s
    dp = np.full((T, S), NEG)
    bk = np.zeros((T, S), dtype=int)

    def score(t, label):
        i = cls_idx.get(label)
        return logP[t][i] if i is not None else NEG

    # init: frame 0 in state 0 (leading silence) or state 1 (first phoneme)
    dp[0][0] = score(0, states[0][0])
    if S > 1:
        dp[0][1] = score(0, states[1][0]); bk[0][1] = 0
    for t in range(1, T):
        for s in range(S):
            # can stay in s, or arrive from s-1 (monotonic advance by 1)
            best_prev, best_s = dp[t - 1][s], s
            if s > 0 and dp[t - 1][s - 1] > best_prev:
                best_prev, best_s = dp[t - 1][s - 1], s - 1
            # also allow skipping an optional silence state (advance by 2) so a
            # phoneme can follow another with no silence between
            if s > 1 and states[s][1] and dp[t - 1][s - 2] > best_prev:
                best_prev, best_s = dp[t - 1][s - 2], s - 2
            if best_prev <= NEG:
                continue
            dp[t][s] = best_prev + score(t, states[s][0])
            bk[t][s] = best_s
    # must end having passed through every phoneme -> end in last state or the
    # final phoneme state
    end_candidates = [S - 1, S - 2] if S >= 2 else [S - 1]
    best_end = max(end_candidates, key=lambda s: dp[T - 1][s])
    if dp[T - 1][best_end] <= NEG:
        return None
    # backtrack
    labels = [None] * T
    s = best_end
    for t in range(T - 1, -1, -1):
        labels[t] = states[s][0]
        s = bk[t][s]
    # verify every phoneme state was visited (alignment covered the sequence)
    visited_phon = set(l for l in labels if l != fd.SILENCE)
    if visited_phon != set(phoneme_seq):
        # some phoneme never got a frame -> alignment failed
        return None
    return labels, starts


def aligned_frame_labels(audio, phoneme_seq, model):
    """Convenience: return (frame_start_samples, labels) for training, or None."""
    res = force_align(audio, phoneme_seq, model)
    if res is None:
        return None
    labels, starts = res
    return starts, labels
