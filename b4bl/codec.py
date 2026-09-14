"""
B4-BL codec — meaning <-> phoneme-sequence <-> audio.

Encode path (complete):
    concepts -> phoneme names -> audio
    with word gaps between morphemes and spelling fallback for unknown tokens.

Decode path:
    SYMBOLIC decode (phoneme names -> concepts) is complete here and proves the
    language is unambiguously parseable, including spelling fallback.

    ACOUSTIC decode (microphone audio -> phoneme names) is the Phase 2/4 signal-
    processing task and is NOT implemented yet. `audio_to_phonemes` raises
    NotImplementedError with a pointer, rather than pretending to work. The
    feature model it will target already exists: Phoneme.features. Because synth
    and recognizer share phonology.py, the classifier's job is to recover that
    4-tuple (band, contour, duration, class) per segment.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from . import generators as gen
from . import phonology as ph
from . import lexicon as lex
from .prosody import Prosody, NEUTRAL

WORD_GAP = 0.14      # gap between morphemes (groups phonemes into "words")
PHONE_GAP = 0.04     # gap between phonemes within a morpheme


# ---------------------------------------------------------------------------
# ENCODE
# ---------------------------------------------------------------------------
def concepts_to_phoneme_words(concepts: List[str]) -> List[List[str]]:
    """Each concept -> a list of phoneme names. Unknown concepts are spelled."""
    words = []
    for c in concepts:
        if lex.is_known(c):
            words.append(lex.concept_to_phonemes(c))
        else:
            # spelling fallback: marker + per-char codes
            seq = list(lex.SPELL_MARKER)
            for ch in c.lower():
                if ch in lex.CHAR_TO_PHONES:
                    seq += lex.CHAR_TO_PHONES[ch]
            words.append(seq)
    return words


def _render_phoneme_seq(pnames: List[str], prosody: Prosody) -> List[np.ndarray]:
    clips = []
    for pi, pname in enumerate(pnames):
        clips.append(ph.BY_NAME[pname].render(prosody=prosody))
        if pi != len(pnames) - 1:
            clips.append(_silence(PHONE_GAP))
    return clips


def _render_rep(rep: "lex.Rep", prosody: Prosody) -> List[np.ndarray]:
    """Render a repeated group at its lexical rhythm. The inter-pulse gap comes
    from `rate`; prosody may scale intensity/timing but NEVER the count."""
    unit = list(rep.unit)
    # gap so that pulse_period = 1/rate; clamp so pulses don't overlap.
    period = 1.0 / max(rep.rate, 0.3)
    clips = []
    for k in range(rep.count):
        if rep.alternate:
            pulse = [unit[k % len(unit)]]     # tick/tock: one cycling member per pulse
        else:
            pulse = unit
        clips += _render_phoneme_seq(pulse, prosody)
        if k != rep.count - 1:
            # approximate: subtract a nominal unit duration from the period
            gap = max(0.02, period - 0.16)
            clips.append(_silence(gap))
    return clips


def _render_concept(concept: str, prosody: Prosody) -> List[np.ndarray]:
    if lex.is_known(concept):
        body = lex.morpheme_body(concept)
        if isinstance(body, lex.Rep):
            return _render_rep(body, prosody)
        return _render_phoneme_seq(body, prosody)
    # spelling fallback
    seq = list(lex.SPELL_MARKER)
    for ch in concept.lower():
        if ch in lex.CHAR_TO_PHONES:
            seq += lex.CHAR_TO_PHONES[ch]
    return _render_phoneme_seq(seq, prosody)


def encode(concepts: List[str], prosody: Prosody = NEUTRAL) -> np.ndarray:
    """Top-level: list of concepts -> audio, respecting repetition rhythm."""
    clips = []
    for ci, c in enumerate(concepts):
        clips += _render_concept(c, prosody)
        if ci != len(concepts) - 1:
            clips.append(_silence(WORD_GAP))
    return np.concatenate(clips) if clips else np.zeros(0, dtype=np.float32)


def _silence(dur: float) -> np.ndarray:
    return np.zeros(int(gen.SR * dur), dtype=np.float32)


# ---------------------------------------------------------------------------
# DECODE (symbolic)
# ---------------------------------------------------------------------------
def phoneme_words_to_concepts(words: List[List[str]]) -> List[str]:
    out = []
    for word in words:
        if word[:len(lex.SPELL_MARKER)] == lex.SPELL_MARKER:
            # spelled token
            body = word[len(lex.SPELL_MARKER):]
            chars = []
            for i in range(0, len(body) - 1, 2):
                ch = lex.PHONES_TO_CHAR.get(tuple(body[i:i + 2]))
                if ch:
                    chars.append(ch)
            out.append("".join(chars).upper())
        else:
            concept = lex.phonemes_to_concept(word)
            out.append(concept if concept else "?" + "-".join(word))
    return out


def audio_to_phonemes(audio: np.ndarray) -> List[List[str]]:
    """Acoustic front-end: segment audio and classify each segment into a phoneme
    name, grouping by word gaps. Implemented in b4bl.decoder (Phase-2 v1,
    file/loopback). Not yet perfect per-phoneme; the protocol layer's checksum/FEC
    is what makes it reliable end to end."""
    from . import decoder
    return decoder.audio_to_phoneme_words(audio)


def decode_to_concepts(audio: np.ndarray) -> List[str]:
    """Full acoustic decode. Uses lexicon-constrained decoding when the trained
    classifier is available (recovers misheard phonemes by snapping to the nearest
    valid morpheme — works with the error-correcting codebook), else the plain
    top-1 lookup."""
    try:
        from . import classifier as _clf
        if _clf.available():
            return decode_to_concepts_lexical(audio)
    except Exception:
        pass
    return phoneme_words_to_concepts(audio_to_phonemes(audio))


def decode_to_concepts_plain(audio: np.ndarray) -> List[str]:
    """Plain top-1 decode without lexicon correction (kept for comparison/tests)."""
    return phoneme_words_to_concepts(audio_to_phonemes(audio))


# ---------------------------------------------------------------------------
# LEXICON-CONSTRAINED DECODE
# Instead of classifying each phoneme to a single best then looking up (where one
# misheard phoneme dooms the word), score each gap-separated word against the set
# of REAL morphemes using the classifier's ranked candidates. A phoneme that was
# misheard is recovered when the correcting choice forms a valid word — the same
# principle a spell-checker uses. Most random phoneme sequences aren't words, so
# this collapses many errors onto the intended morpheme.
# ---------------------------------------------------------------------------
def _lexicon_index():
    """concept -> flat phoneme sequence, for all morphemes (repetition expanded)."""
    idx = {}
    for concept in lex.MORPHEMES:
        idx[concept] = tuple(lex.concept_to_phonemes(concept))
    return idx


_LEX_INDEX = None


def _score_word(cand_word, seq) -> float:
    """Score how well a morpheme phoneme-sequence `seq` explains a candidate word
    (list of per-position [(name, prob), ...]). Length mismatch is penalized; each
    matched position adds its candidate probability (or a small floor if the
    needed phoneme isn't among the candidates)."""
    if len(seq) != len(cand_word):
        # allow it but penalize — segmentation may split/merge one phoneme
        length_pen = 0.35 ** abs(len(seq) - len(cand_word))
    else:
        length_pen = 1.0
    score = 1.0
    for i, phon in enumerate(seq):
        if i < len(cand_word):
            probs = dict(cand_word[i])
            score *= probs.get(phon, 0.03)   # floor for "not in candidates"
        else:
            score *= 0.03
    return score * length_pen


def candidate_words_to_concepts(cand_words) -> List[str]:
    """Lexicon-constrained decode: each candidate-word -> best real concept."""
    global _LEX_INDEX
    if _LEX_INDEX is None:
        _LEX_INDEX = _lexicon_index()
    out = []
    for cand_word in cand_words:
        # top-1 phoneme sequence, for spelling-marker detection + fallback
        top1 = [c[0][0] for c in cand_word]
        if top1[:len(lex.SPELL_MARKER)] == lex.SPELL_MARKER:
            out.append(phoneme_words_to_concepts([top1])[0])
            continue
        best, best_s = None, -1.0
        for concept, seq in _LEX_INDEX.items():
            s = _score_word(cand_word, seq)
            if s > best_s:
                best, best_s = concept, s
        out.append(best if best is not None else "?" + "-".join(top1))
    return out


def decode_to_concepts_lexical(audio: np.ndarray, k: int = 2) -> List[str]:
    """Full acoustic decode with lexicon correction (the accurate path)."""
    from . import decoder
    cand_words = decoder.audio_to_candidate_words(audio, k=k)
    return candidate_words_to_concepts(cand_words)


def _phone_word_distance(got, seq) -> float:
    """Edit-distance-like cost between two phoneme-name sequences (Levenshtein
    with unit costs). Lower = closer."""
    n, m = len(got), len(seq)
    if n == 0:
        return float(m)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (0 if got[i - 1] == seq[j - 1] else 1))
            prev = cur
    return float(dp[m])


def phoneme_words_to_concepts_lexical(words) -> List[str]:
    """Lexicon-constrained decode for a plain phoneme-word list (e.g. from the
    frame decoder, which emits one phoneme per position, not candidates). Snaps
    each word to the nearest real morpheme by phoneme edit-distance — recovering
    misheard phonemes when the correction forms a valid word (the same win the
    segment path gets from lexicon decoding, now for frames)."""
    global _LEX_INDEX
    if _LEX_INDEX is None:
        _LEX_INDEX = _lexicon_index()
    out = []
    for w in words:
        w = [str(p) for p in w]
        if not w:
            continue
        if w[:len(lex.SPELL_MARKER)] == lex.SPELL_MARKER:
            out.append(phoneme_words_to_concepts([w])[0])
            continue
        best, best_d = None, 1e9
        for concept, seq in _LEX_INDEX.items():
            d = _phone_word_distance(w, list(seq))
            if d < best_d:
                best, best_d = concept, d
        # only accept if reasonably close (avoid snapping garbage to a random word)
        if best is not None and best_d <= max(1, len(w)):
            out.append(best)
        else:
            out.append("?" + "-".join(w))
    return out


def decode_to_concepts_frames(audio: np.ndarray, lexical: bool = True) -> List[str]:
    """Full acoustic decode via the CTC-style frame decoder, with optional
    lexicon correction. This is the real-audio path."""
    from . import frame_decoder
    words = frame_decoder.decode_frames(audio)
    if lexical:
        return phoneme_words_to_concepts_lexical(words)
    return [lex.phonemes_to_concept(w) for w in words]


def decode_to_concepts_ctc(audio: np.ndarray, lexical: bool = True,
                           beam_width: int = 24) -> List[str]:
    """A1: CTC prefix-beam decode over the RF frame probabilities. CTC removes the
    blank/silence, so word boundaries are lost here — this returns the whole
    utterance as ONE phoneme sequence mapped to concept(s). Best evaluated on
    single-morpheme utterances first; word-splitting comes with A2."""
    from . import frame_decoder, ctc
    probs, classes = frame_decoder.frame_probabilities(audio)
    if probs is None:
        return []
    seq = ctc.ctc_beam_decode(probs, classes, blank=frame_decoder.SILENCE,
                              beam_width=beam_width)
    if not seq:
        return []
    if lexical:
        return phoneme_words_to_concepts_lexical([seq])
    c = lex.phonemes_to_concept(seq)
    return [c] if c else ["?" + "-".join(seq)]


def decode_to_concepts_nn(audio: np.ndarray, lexical: bool = True) -> List[str]:
    """Full acoustic decode via the neural-net frame classifier (Option 2), with
    optional lexicon correction."""
    from . import nn_decoder
    words = nn_decoder.decode_frames(audio)
    if lexical:
        return phoneme_words_to_concepts_lexical(words)
    return [lex.phonemes_to_concept(w) for w in words]


# ---------------------------------------------------------------------------
# convenience round-trip used by tests
# ---------------------------------------------------------------------------
def roundtrip_symbolic(concepts: List[str]) -> Tuple[List[List[str]], List[str]]:
    words = concepts_to_phoneme_words(concepts)
    back = phoneme_words_to_concepts(words)
    return words, back
