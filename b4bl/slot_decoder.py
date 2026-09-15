"""B4-BL — slot-grid (symbol-clock) decoder.

The counterpart to codec.encode(slots=True). Because the slotted encoder puts a
mandatory, room-survivable gap after every phoneme, boundaries are no longer
something we must *infer* from a continuous stream — every phoneme is a cleanly
silence-delimited voiced run. So decoding collapses to the thing the classifier is
already good at (93-99%): classify each isolated segment.

Pipeline:
  1. voiced-run segmentation (decoder._segments) — now reliable, gaps are wide
  2. drop geminate-marker runs (the reserved high pip) — record them as "the next
     phoneme repeats the previous", so identical adjacent phonemes are recovered
  3. group runs into words on the wider word gap
  4. classify each phoneme run with the trained segment classifier
  5. lexicon: phoneme words -> concepts (with N-best repair, as today)

No frame model, no boundary detector, no CTC. Those existed only to fight the
segmentation problem this encoding removes.
"""

from __future__ import annotations
from typing import List
import numpy as np

from . import generators as gen
from . import decoder as _dec
from . import classifier as _clf
from . import lexicon as lex
from . import codec

SR = gen.SR

# The geminate marker is a broadband CLICK: short and spectrally FLAT (energy
# spread across the spectrum) — unlike any phoneme, which is tonal (energy at one
# pitch). We tell them apart by spectral flatness, on a different axis than pitch,
# so the click can never be confused for a phoneme in any band.
_CLICK_MAX_DUR = codec.GEMINATE_DUR * 3.0     # a click is short
_CLICK_MIN_FLATNESS = 0.35                    # a click is broadband (flat); tones are ~0


def _is_geminate(seg: np.ndarray) -> bool:
    """True if this run is the reserved geminate click: short AND spectrally flat."""
    if len(seg) / SR > _CLICK_MAX_DUR:
        return False
    if len(seg) < 32:
        return False
    return _dec._spectral_flatness(seg) >= _CLICK_MIN_FLATNESS


_MIN_PHONEME_DUR = 0.05    # runs shorter than this that AREN'T clicks are noise blips


def _raw_runs(a: np.ndarray):
    """Voiced runs WITHOUT the min-duration drop, so the short geminate click
    survives. Returns (start, end, gap_before_seconds) triples."""
    times, rms = _dec._rms_envelope(a)
    voiced = rms > _dec.SILENCE_RMS
    raw = _dec._raw_segments(voiced, times)
    out, prev_end = [], 0
    for s, e in raw:
        out.append((s, e, (s - prev_end) / SR))
        prev_end = e
    return out


def _segment_words(a: np.ndarray):
    """Shared segmentation: returns a list of words, each a list of phoneme audio
    segments, plus a parallel 'repeat' flag per phoneme (True if a click marked it
    as an identical repeat of the previous phoneme in its word)."""
    runs = _raw_runs(a)
    if not runs:
        return []
    word_gap_thr = 0.5 * (codec.SLOT_PHONE_GAP + codec.SLOT_WORD_GAP)
    words = []
    current = []          # list of (segment_audio, repeat_flag)
    repeat_next = False
    for (s, e, gap_before) in runs:
        seg = a[s:e]
        dur = (e - s) / SR
        if _is_geminate(seg):
            repeat_next = True
            continue
        if dur < _MIN_PHONEME_DUR:
            continue
        if current and gap_before >= word_gap_thr and not repeat_next:
            words.append(current)
            current = []
        current.append((seg, repeat_next))
        repeat_next = False
    if current:
        words.append(current)
    return words


def audio_to_phoneme_words(audio: np.ndarray) -> List[List[str]]:
    """Slotted audio -> list of words, each a list of phoneme names (top-1)."""
    a = audio.astype(np.float32)
    if np.max(np.abs(a)) > 0:
        a = a / np.max(np.abs(a))
    out = []
    for word in _segment_words(a):
        names = []
        for seg, repeat in word:
            if repeat and names:
                names.append(names[-1])          # click: identical to previous
            else:
                names.append(_clf.classify_segment(seg)[0])
        out.append(names)
    return out


def audio_to_candidate_words(audio: np.ndarray, k: int = 2):
    """Slotted audio -> list of words, each a list of per-slot N-best candidates
    [(name, prob), ...]. This is what the lexicon-constrained decoder consumes to
    repair a misheard phoneme by snapping to the nearest valid morpheme."""
    a = audio.astype(np.float32)
    if np.max(np.abs(a)) > 0:
        a = a / np.max(np.abs(a))
    out = []
    for word in _segment_words(a):
        cand_word = []
        for seg, repeat in word:
            if repeat and cand_word:
                cand_word.append(list(cand_word[-1]))   # click: copy prev candidates
            else:
                cand_word.append(_clf.phoneme_candidates(seg, k=k))
        out.append(cand_word)
    return out


def audio_to_flat_candidates(audio: np.ndarray, k: int = 3):
    """All phoneme segments across the WHOLE utterance as one flat list of N-best
    candidates, IGNORING word grouping. Word boundaries are left for the vocabulary
    DP to decide (see decode_vocab), because on a real channel the within-word and
    between-word gap distributions overlap and no gap threshold separates them."""
    a = audio.astype(np.float32)
    if np.max(np.abs(a)) > 0:
        a = a / np.max(np.abs(a))
    flat = []
    prev = None
    for word in _segment_words(a):
        for seg, repeat in word:
            if repeat and prev is not None:
                flat.append(list(prev))               # click: copy prev candidates
            else:
                prev = _clf.phoneme_candidates(seg, k=k)
                flat.append(prev)
    return flat


_MORPH_INDEX = None


def _morph_index():
    global _MORPH_INDEX
    if _MORPH_INDEX is None:
        idx = []
        for c in lex.MORPHEMES:
            if c in lex.ALIASES:
                continue                      # skip alias duplicates (same sound)
            seq = tuple(lex.concept_to_phonemes(c))
            if not seq:
                continue                      # skip anything with no phonemes
            idx.append((c, seq))
        _MORPH_INDEX = idx
    return _MORPH_INDEX


def _score_seq(cands, seq) -> float:
    """Log-score that the phoneme candidate slots `cands` (each [(name,prob),...])
    spell exactly the morpheme phoneme-sequence `seq` (same length). Product of per-
    position probabilities, with a floor for a phoneme not among the candidates."""
    import math
    s = 0.0
    for slot, want in zip(cands, seq):
        p = dict(slot).get(want, 0.02)
        s += math.log(max(p, 1e-6))
    return s


def decode_vocab(audio: np.ndarray, k: int = 3, word_penalty: float = 1.0) -> List[str]:
    """Vocabulary-driven decode: classify every phoneme (clean, thanks to slotting),
    then DP over the flat phoneme sequence to find the split into VALID morphemes
    that maximizes total score. The closed vocabulary decides word boundaries — no
    gap threshold — so it is robust to the real channel's overlapping gap sizes.

    word_penalty discourages over-splitting (each extra word must earn its score)."""
    flat = audio_to_flat_candidates(audio, k=k)
    N = len(flat)
    if N == 0:
        return []
    index = _morph_index()
    # group morphemes by phoneme length for a quick per-span lookup
    by_len = {}
    for c, seq in index:
        by_len.setdefault(len(seq), []).append((c, seq))
    maxlen = max(by_len) if by_len else 1

    NEG = -1e30
    dp = [NEG] * (N + 1)
    dp[0] = 0.0
    back = [(-1, None)] * (N + 1)
    for j in range(1, N + 1):
        for L in range(1, min(maxlen, j) + 1):
            i = j - L
            if dp[i] <= NEG:
                continue
            cands = flat[i:j]
            best_c, best_s = None, NEG
            for c, seq in by_len.get(L, ()):
                s = _score_seq(cands, seq)
                if s > best_s:
                    best_c, best_s = c, s
            if best_c is None:
                continue
            total = dp[i] + best_s - word_penalty
            if total > dp[j]:
                dp[j] = total
                back[j] = (i, best_c)
    # backtrack
    words, j = [], N
    while j > 0 and back[j][0] >= 0:
        i, c = back[j]
        words.append(c)
        j = i
    return list(reversed(words))


def decode(audio: np.ndarray, lexical: bool = True, vocab: bool = False) -> List[str]:
    """Slotted audio -> concept list.

    vocab=True: vocabulary-driven word segmentation (robust on a real channel where
    gap sizes overlap). lexical=True (default, vocab=False): trust word gaps, then
    lexicon-repair each word. Plain top-1 if neither."""
    if vocab and _clf.available():
        return decode_vocab(audio)
    if lexical and _clf.available():
        return codec.candidate_words_to_concepts(audio_to_candidate_words(audio))
    return codec.phoneme_words_to_concepts(audio_to_phoneme_words(audio))
