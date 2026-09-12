"""Symbolic round-trip and inventory sanity tests. Run: python3 -m pytest -q"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from b4bl import lexicon as lex
from b4bl import phonology as ph
from b4bl import codec


def test_inventory_features_unique():
    """Every phoneme must have a distinct feature tuple, or the decoder can't
    tell them apart."""
    feats = [p.features for p in ph.INVENTORY]
    assert len(feats) == len(set(feats)), "duplicate phoneme feature tuples"


def test_morpheme_sequences_unambiguous():
    """No two DISTINCT concepts may share a phoneme-sequence (documented aliases
    excepted — they intentionally name the same sound)."""
    concepts = [c for c in lex.MORPHEMES if c not in lex.ALIASES]
    seqs = [tuple(lex.concept_to_phonemes(c)) for c in concepts]
    assert len(seqs) == len(set(seqs)), "two distinct morphemes share a sequence"


def test_all_morpheme_phonemes_exist():
    for concept in lex.MORPHEMES:
        for pname in lex.concept_to_phonemes(concept):
            assert pname in ph.BY_NAME, f"{concept} uses unknown phoneme {pname}"


def test_repetition_morphemes_roundtrip():
    for concept in ["ALARM", "CALCULATING"]:
        _w, back = codec.roundtrip_symbolic([concept])
        assert back == [concept], f"{concept} -> {back}"


def test_known_concepts_roundtrip():
    for _eng, concepts in lex.example_sentences():
        _words, back = codec.roundtrip_symbolic(concepts)
        assert back == concepts, f"{concepts} -> {back}"


def test_spelling_fallback_roundtrip():
    concepts = ["QUERY", "R2D2", "OK"]  # R2D2 is out-of-dictionary -> spelled
    _words, back = codec.roundtrip_symbolic(concepts)
    assert back[0] == "QUERY"
    assert back[1] == "R2D2"
    assert back[2] == "OK"


def test_encode_produces_audio():
    audio = codec.encode(["WARN", "OBSTACLE", "FRONT"])
    assert audio.ndim == 1 and len(audio) > 1000


def test_vocabulary_size():
    """A primitive-but-real language: expect a couple hundred morphemes."""
    assert len(lex.MORPHEMES) >= 180


def test_numerals_roundtrip():
    for n in [0, 7, 42, 100, 2599]:
        assert lex.concepts_to_number(lex.number_to_concepts(n)) == n


def test_number_concepts_are_known():
    for c in lex.number_to_concepts(42):
        assert lex.is_known(c), f"{c} not a known morpheme"


def test_interjections_render():
    from b4bl import interjections as itj
    for name in itj.INTERJECTIONS:
        a = itj.render(name)
        assert a.ndim == 1 and len(a) > 100
