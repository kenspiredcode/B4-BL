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

# --- symbol-clock (slotted) rendering -------------------------------------
# The decoder's whole bottleneck is finding phoneme boundaries. In slotted mode
# every phoneme lands on a fixed time grid so boundaries are known a priori, and
# the mandatory inter-phoneme gap is widened enough to survive a real room's
# reverb/decay. See docs/symbol-clock-reset.md.
# Per-CLASS slot widths. A single universal slot sized for the longest phoneme
# padded every SHORT phoneme with dead air and sounded sluggish. Instead each
# duration class gets its own slot width, each = the phoneme body + the mandatory
# gap. The grid is still known to the decoder: SHORT vs LONG is itself a decodable
# meaning axis, so once a slot's phoneme is classified its width is known too.
# Widths = body duration (phonology.DUR_SEC) + SLOT_PHONE_GAP.
SLOT_PHONE_GAP = 0.10  # mandatory gap after every phoneme in slotted mode
SLOT_WORD_GAP = 0.20   # gap between morphemes in slotted mode (> phone gap)


def _slot_width(dur) -> float:
    """Slot width for a phoneme of the given Dur: its body length plus the gap."""
    return ph.DUR_SEC[dur] + SLOT_PHONE_GAP


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
        clips.append(ph.BY_NAME[pname].render(prosody=prosody, register_mult=_REG_MULT[0]))
        if pi != len(pnames) - 1:
            clips.append(_silence(PHONE_GAP))
    return clips


# --- geminate marker: a short reserved pip inserted ONLY between two adjacent,
#     identical phonemes, so the decoder does not merge them into one long tone.
#     It is not a lexical phoneme; no morpheme uses it. Kept short and pitched so
#     it reads as a little R2 "tick" rather than a mechanical click. -------------
# The geminate marker is a broadband CLICK, not a tone. A click can't be confused
# for any phoneme because phonemes are tonal (energy at one pitch) while the click
# is spectrally FLAT (energy everywhere) — a different axis (sound class), not a
# pitch to dodge. It reads as a short mechanical "tk", on-aesthetic for R2.
GEMINATE_DUR = 0.02          # very short — a tick, not a beep
GEMINATE_HZ = 0              # unused (kept for any external reference); click is broadband


def _render_geminate() -> np.ndarray:
    """A short broadband click: a burst of noise with a fast decay. Spectrally flat,
    so the decoder tells it from phonemes by flatness, never by pitch."""
    n = int(gen.SR * GEMINATE_DUR)
    noise = gen._rng.uniform(-1, 1, n).astype(np.float32)
    env = np.exp(-np.linspace(0, 6, n)).astype(np.float32)   # sharp attack, fast decay
    return (noise * env * 0.9).astype(np.float32)


def _fit_slot(clip: np.ndarray, width_sec: float) -> np.ndarray:
    """Place a rendered phoneme at the start of a width_sec-wide window, padding the
    remainder with the mandatory inter-phoneme gap (silence). The phoneme body may
    be shorter or longer than its nominal duration (prosody flexes it); we clamp to
    the slot window and always leave at least SLOT_PHONE_GAP of trailing silence so
    the boundary is recoverable."""
    window = int(gen.SR * width_sec)
    gap = int(gen.SR * SLOT_PHONE_GAP)
    body_room = max(1, window - gap)
    out = np.zeros(window, dtype=np.float32)
    if len(clip) <= body_room:
        out[:len(clip)] = clip
    else:
        # body longer than the slot allows (e.g. prosody stretched it): fade the
        # last few ms instead of hard-cutting, so we never chop the waveform mid-
        # cycle (which clicks / sounds clipped).
        body = clip[:body_room].copy()
        fade = min(len(body), int(0.02 * gen.SR))
        body[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        out[:len(body)] = body
    return out


def _render_phoneme_seq_slotted(pnames: List[str], prosody: Prosody) -> List[np.ndarray]:
    """Slotted render: each phoneme fills a fixed 1- or 2-slot window (SHORT/LONG),
    with a mandatory trailing gap. A geminate marker is inserted between adjacent
    identical phonemes so they don't merge."""
    clips = []
    for pi, pname in enumerate(pnames):
        p = ph.BY_NAME[pname]
        body = p.render(prosody=prosody, register_mult=_REG_MULT[0])
        clips.append(_fit_slot(body, _slot_width(p.dur)))
        # geminate marker before the next phoneme if it is identical to this one.
        # The slot already carries a trailing gap; add a leading gap before the
        # next phoneme too, so the pip is flanked by silence on both sides and
        # reads as its own separate "tick" rather than fusing with either tone.
        if pi != len(pnames) - 1 and pnames[pi + 1] == pname:
            clips.append(_silence(SLOT_PHONE_GAP))
            clips.append(_render_geminate())
            clips.append(_silence(SLOT_PHONE_GAP))
    return clips


# register-cycling word-boundary cue: consecutive words are TRANSPOSED so their
# center lands on an ABSOLUTE target register (low/mid/high Hz), cycling. A
# boundary is then a register jump regardless of the words' natural bands. (A plain
# multiplier failed: a low-band word x1.85 is still lower than a high-band word x1,
# so registers overlapped — we must transpose to absolute targets.)
REGISTER_TARGETS = [450.0, 850.0, 1600.0]   # low / mid / high absolute centers (Hz)
_REG_MULT = [1.0]


def _word_natural_center(concept: str) -> float:
    """Geometric-mean center pitch of a word's phonemes at natural register."""
    import numpy as _np
    names = (lex.concept_to_phonemes(concept) if lex.is_known(concept)
             else [p for ch in concept.lower() for p in lex.CHAR_TO_PHONES.get(ch, [])])
    centers = [ph.BY_NAME[n].center for n in names if n in ph.BY_NAME]
    return float(_np.exp(_np.mean(_np.log(centers)))) if centers else 850.0


def _render_rep(rep: "lex.Rep", prosody: Prosody) -> List[np.ndarray]:
    """Render a repeated group at its lexical rhythm. The inter-pulse gap comes
    from `rate`; prosody may scale intensity/timing but NEVER the count.

    Rep morphemes (ALARM, WORKING) ARE the rhythmic-repetition axis of meaning, so
    they keep their natural pulse timing even in slotted mode — the rhythm IS the
    word. Slotting them would destroy the very cue that identifies them."""
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


def _render_concept(concept: str, prosody: Prosody, slots: bool = False) -> List[np.ndarray]:
    seq_fn = _render_phoneme_seq_slotted if slots else _render_phoneme_seq
    if lex.is_known(concept):
        body = lex.morpheme_body(concept)
        if isinstance(body, lex.Rep):
            return _render_rep(body, prosody)   # rhythm-defined; never slotted
        return seq_fn(body, prosody)
    # spelling fallback
    seq = list(lex.SPELL_MARKER)
    for ch in concept.lower():
        if ch in lex.CHAR_TO_PHONES:
            seq += lex.CHAR_TO_PHONES[ch]
    return seq_fn(seq, prosody)


def encode(concepts: List[str], prosody: Prosody = NEUTRAL,
           register_cycle: bool = False, slots: bool = False) -> np.ndarray:
    """Top-level: list of concepts -> audio, respecting repetition rhythm.

    slots (symbol-clock mode): each phoneme lands on a fixed time grid (SHORT = 1
    slot, LONG = 2 slots) with a widened mandatory inter-phoneme gap, and identical
    adjacent phonemes are separated by a geminate marker. This makes phoneme
    boundaries known a priori instead of detected — the segmentation fix. See
    docs/symbol-clock-reset.md. Word gap is widened to SLOT_WORD_GAP.

    register_cycle (legacy experiment): each successive word is transposed to a
    cycling absolute register so a word boundary is a pitch jump. Superseded by
    slotted mode; kept for comparison. Mutually exclusive with slots in practice."""
    clips = []
    word_gap = SLOT_WORD_GAP if slots else WORD_GAP
    for ci, c in enumerate(concepts):
        if register_cycle:
            target = REGISTER_TARGETS[ci % len(REGISTER_TARGETS)]
            _REG_MULT[0] = target / _word_natural_center(c)   # transpose to target
        else:
            _REG_MULT[0] = 1.0
        clips += _render_concept(c, prosody, slots=slots)
        if ci != len(concepts) - 1:
            clips.append(_silence(word_gap))
    _REG_MULT[0] = 1.0
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
