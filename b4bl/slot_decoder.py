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


def decode(audio: np.ndarray, lexical: bool = True) -> List[str]:
    """Slotted audio -> concept list. With lexical=True (default) uses the lexicon-
    constrained repair (recovers a misheard phoneme when the correction forms a
    valid morpheme); else plain top-1 lookup."""
    if lexical and _clf.available():
        return codec.candidate_words_to_concepts(audio_to_candidate_words(audio))
    return codec.phoneme_words_to_concepts(audio_to_phoneme_words(audio))
