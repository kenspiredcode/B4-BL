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
    """No morpheme's flattened phoneme-sequence may be identical to another's."""
    seqs = [tuple(lex.concept_to_phonemes(c)) for c in lex.MORPHEMES]
    assert len(seqs) == len(set(seqs)), "two morphemes share a phoneme sequence"


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
    audio = codec.encode(["WARNING", "OBSTACLE", "FRONT"])
    assert audio.ndim == 1 and len(audio) > 1000
