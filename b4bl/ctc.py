"""
B4-BL — CTC prefix-beam decoding (A1).

The greedy region-vote collapse threw away the per-frame probability distribution
and committed to one argmax path. CTC prefix-beam search keeps the full
distribution and searches over collapsed label sequences, summing the probability
of ALL frame-alignments that yield each sequence. This is the standard, correct
frame->sequence decoder and recovers information the greedy version discarded.

We keep it framework-agnostic: input is a [T x C] probability matrix and the class
list (one class is the blank/silence). Output is the best collapsed phoneme
sequence (blanks removed, repeats merged per CTC rules), which the caller then
splits into words and maps to concepts.

A2 (lexicon fusion) hooks in via an optional `word_scorer` that biases the beam
toward valid morphemes — implemented in the lexical variant.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Callable
import numpy as np
from collections import defaultdict

NEG_INF = -1e30


def _logsumexp(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def ctc_beam_decode(probs: np.ndarray, classes: List[str], blank: str,
                    beam_width: int = 24,
                    prune: float = 1e-3) -> List[str]:
    """CTC prefix-beam search. Returns the best collapsed label sequence (list of
    class names, blank removed, CTC repeat-merge applied).

    probs: [T, C] per-frame probabilities. classes[j] is column j's label.
    """
    T, C = probs.shape
    blank_idx = classes.index(blank)
    logp = np.log(np.clip(probs, 1e-12, 1.0))

    # beam entries keyed by prefix tuple; track prob of prefix ending in blank
    # (p_b) and ending in a non-blank (p_nb), in log space.
    # start: empty prefix, prob 1 ending in blank.
    beams = {(): (0.0, NEG_INF)}   # prefix -> (log p_b, log p_nb)

    for t in range(T):
        # only consider classes with non-negligible prob this frame (speed)
        top = np.where(probs[t] > prune)[0]
        if blank_idx not in top:
            top = np.append(top, blank_idx)
        next_beams = defaultdict(lambda: (NEG_INF, NEG_INF))

        for prefix, (pb, pnb) in beams.items():
            prefix_prob = _logsumexp(pb, pnb)
            for c in top:
                lp = logp[t, c]
                if c == blank_idx:
                    # stay same prefix, extend blank
                    nb, nnb = next_beams[prefix]
                    nb = _logsumexp(nb, prefix_prob + lp)
                    next_beams[prefix] = (nb, nnb)
                else:
                    lab = classes[c]
                    # case 1: repeat of last label -> merges (adds to p_nb of same prefix)
                    if prefix and prefix[-1] == lab:
                        # repeat right after non-blank: stays same prefix
                        nb, nnb = next_beams[prefix]
                        nnb = _logsumexp(nnb, pnb + lp)
                        next_beams[prefix] = (nb, nnb)
                        # repeat after blank -> new occurrence (extends prefix)
                        newp = prefix + (lab,)
                        nb2, nnb2 = next_beams[newp]
                        nnb2 = _logsumexp(nnb2, pb + lp)
                        next_beams[newp] = (nb2, nnb2)
                    else:
                        newp = prefix + (lab,)
                        nb2, nnb2 = next_beams[newp]
                        nnb2 = _logsumexp(nnb2, prefix_prob + lp)
                        next_beams[newp] = (nb2, nnb2)

        # prune to beam_width by total prob
        scored = sorted(next_beams.items(),
                        key=lambda kv: _logsumexp(kv[1][0], kv[1][1]), reverse=True)
        beams = dict(scored[:beam_width])

    best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
    return list(best)
