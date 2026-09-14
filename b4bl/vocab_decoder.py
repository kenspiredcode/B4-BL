"""
B4-BL — vocabulary-constrained decoder.

The key advantage we hadn't spent: we KNOW every possible word. Instead of
classify-each-segment-then-assemble (which fails at segmentation), we SEARCH the
finite set of valid morphemes and score each against the audio, letting the closed
vocabulary do the work DURING decoding.

Scoring primitive: given per-frame class probabilities [T x C] and a candidate
phoneme sequence, compute the CTC forced-alignment score — the total probability
that the frames produced exactly that sequence (blank/silence allowed between and
around, repeats merged per CTC). This is a forward-DP over the CTC lattice for the
specific target sequence. High score = the audio matches this word.

Single word: argmax over all morphemes' scores. Multi-word: split on word gaps,
score each word region independently against the vocabulary (each region is a
closed 1-word problem). This never considers invalid phoneme strings.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from . import generators as gen
from . import lexicon as lex
from . import frame_decoder as fd

SR = gen.SR
NEG_INF = -1e30


def _logsumexp2(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = a if a > b else b
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def ctc_score(logp: np.ndarray, cls_idx: dict, blank_idx: int, seq: List[str]) -> float:
    """CTC forced-alignment log-probability that frames `logp` [T x C] produced the
    label sequence `seq`. Standard CTC forward algorithm over the extended sequence
    (blank between every label and at the ends)."""
    T = logp.shape[0]
    # extended sequence: blank, s0, blank, s1, ..., blank
    ext = [blank_idx]
    for s in seq:
        ext.append(cls_idx.get(s, -1))
        ext.append(blank_idx)
    S = len(ext)
    if any(e < 0 for e in ext):
        return NEG_INF
    a = np.full((S,), NEG_INF)     # forward vars for current t
    a[0] = logp[0, ext[0]]
    if S > 1:
        a[1] = logp[0, ext[1]]
    for t in range(1, T):
        na = np.full((S,), NEG_INF)
        for s in range(S):
            stay = a[s]
            come = a[s - 1] if s >= 1 else NEG_INF
            v = _logsumexp2(stay, come)
            # allow skip over a blank when current label differs from label 2 back
            if s >= 2 and ext[s] != blank_idx and ext[s] != ext[s - 2]:
                v = _logsumexp2(v, a[s - 2])
            if v == NEG_INF:
                continue
            na[s] = v + logp[t, ext[s]]
        a = na
    # valid endings: last label or last blank
    return _logsumexp2(a[S - 1], a[S - 2] if S >= 2 else NEG_INF)


class _Scorer:
    def __init__(self):
        self.vocab = None      # list of (concept, phoneme-seq)
        self.cls_idx = None
        self.blank_idx = None

    def ensure_vocab(self):
        if self.vocab is None:
            self.vocab = [(c, list(lex.concept_to_phonemes(c)))
                          for c in lex.MORPHEMES if c not in getattr(lex, "ALIASES", {})]

    def best_concept(self, logp, classes) -> Tuple[Optional[str], float]:
        """Score every morpheme against the frame log-probs; return the best."""
        self.ensure_vocab()
        cls_idx = {c: i for i, c in enumerate(classes)}
        blank_idx = classes.index(fd.SILENCE)
        best, best_s = None, NEG_INF
        for concept, seq in self.vocab:
            s = ctc_score(logp, cls_idx, blank_idx, seq)
            # length-normalize so short words aren't unfairly favored
            s = s / max(1, len(seq))
            if s > best_s:
                best, best_s = concept, s
        return best, best_s


_SCORER = _Scorer()


def _word_regions(audio):
    """Split audio into word regions on long silence (reuse boundaries.detect_words
    logic via the decoder segmenter)."""
    from . import boundaries as bd
    return bd.detect_words(audio)


def decode(audio: np.ndarray) -> List[str]:
    """Vocabulary-constrained decode. Split into word regions, score each region's
    frames against the whole morpheme vocabulary, emit the best-matching concept."""
    probs, classes = fd.frame_probabilities(audio)
    if probs is None:
        return []
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    # map audio frame index -> which word region it belongs to
    words = _word_regions(audio)
    if not words:
        return []
    hop = int(fd.HOP_MS / 1000 * SR)
    win = int(fd.FRAME_MS / 1000 * SR)
    out = []
    for (ws, we) in words:
        # frames whose center lies in this word region
        idx = [i for i in range(logp.shape[0])
               if ws <= i * hop + win // 2 < we]
        if not idx:
            continue
        sub = logp[idx]
        concept, _score = _SCORER.best_concept(sub, classes)
        if concept:
            out.append(concept)
    return out
