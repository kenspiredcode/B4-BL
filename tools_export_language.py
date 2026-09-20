"""Export the language tables to JSON for the browser demo.

The demo must never hand-maintain a copy of the lexicon or phoneme inventory;
it consumes this file so it cannot drift from the Python implementation.
"""
import json
import os

from b4bl import lexicon as lex
from b4bl import phonology as ph
from b4bl import grammar
from b4bl import codec
from b4bl import generators as gen
from b4bl import prosody

OUT = os.path.join(os.path.dirname(__file__), "docs", "data", "language.json")


def main():
    # Export every phoneme reachable by name, not just INVENTORY: aliases
    # (Ha, Ma, Rz), subunits (the hum pulses), the spelling phoneme Mfl and the
    # spell marker Grm all appear in morphemes or the spelling fallback, and a
    # curated subset silently breaks those words in the browser.
    phonemes = {
        name: {
            "band": p.band.name,
            "contour": p.contour.name,
            "dur": p.dur.name,
            "cls": p.cls.name,
            "center_hz": p.freq_hz or ph.BAND_CENTER[p.band],
            "seconds": ph.DUR_SEC[p.dur],
        }
        for name, p in ph.BY_NAME.items()
    }
    # Mark the 32 lexical primitives. The rest are aliases, hum subunits and
    # spelling-only phonemes: real sounds the synth must render, but not part of
    # the inventory the page presents.
    inventory_names = {p.name for p in ph.INVENTORY}
    for name, entry in phonemes.items():
        entry["inventory"] = name in inventory_names

    # Most morphemes are a flat phoneme list. ALARM/WORKING/CALCULATING are Rep
    # morphemes whose identity IS the rhythm, so they carry their pulse structure
    # instead; the browser synth must reproduce the timing, not just the sounds.
    morphemes = {}
    for concept in sorted(lex.MORPHEMES):
        body = lex.MORPHEMES[concept]
        if isinstance(body, lex.Rep):
            morphemes[concept] = {
                "rep": {
                    "unit": list(body.unit),
                    "count": body.count,
                    "rate": body.rate,
                    "alternate": bool(body.alternate),
                },
            }
        else:
            morphemes[concept] = lex.concept_to_phonemes(concept)

    categories = {
        cat: [w for w in words if w in lex.MORPHEMES]
        for cat, words in lex.VOCAB_CATEGORIES.items()
    }
    categories["act"] = [m for m in lex.HAND_MORPHEMES if m in lex.MORPHEMES]

    templates = [
        {
            "name": name,
            "slots": [{"category": cat, "optional": opt} for cat, opt in tmpl],
        }
        for name, tmpl in _named_templates()
    ]

    data = {
        "band_center_hz": {b.name: hz for b, hz in ph.BAND_CENTER.items()},
        "band_span_hz": ph.BAND_SPAN,
        "contour_swing": ph.CONTOUR_SWING,
        "dur_sec": {d.name: s for d, s in ph.DUR_SEC.items()},
        "phonemes": phonemes,
        "morphemes": morphemes,
        "categories": categories,
        "spell_marker": list(lex.SPELL_MARKER),
        "char_to_phones": {c: list(v) for c, v in lex.CHAR_TO_PHONES.items()},
        "templates": templates,
        "timing": {
            "word_gap": codec.WORD_GAP,
            "phone_gap": codec.PHONE_GAP,
            "geminate_dur": codec.GEMINATE_DUR,
            "sample_rate": gen.SR,
        },
        "prosody": {
            name: {"confidence": pr.confidence, "urgency": pr.urgency}
            for name, pr in (("neutral", prosody.NEUTRAL),
                             ("uncertain", prosody.UNCERTAIN),
                             ("urgent", prosody.URGENT),
                             ("calm", prosody.CALM))
        },
        "examples": [
            {"english": en, "concepts": cs} for en, cs in lex.example_sentences()
        ],
    }

    needed = set()
    for concept, body in lex.MORPHEMES.items():
        needed.update(body.unit if isinstance(body, lex.Rep)
                      else lex.concept_to_phonemes(concept))
    needed.update(lex.SPELL_MARKER)
    for code in lex.CHAR_TO_PHONES.values():
        needed.update(code)
    missing = sorted(needed - set(phonemes))
    if missing:
        raise SystemExit(f"phonemes used but not exported: {missing}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    print(f"wrote {OUT}")
    print(f"  {len(phonemes)} phonemes, {len(morphemes)} morphemes, "
          f"{len(templates)} templates")


def _named_templates():
    """Pair grammar.TEMPLATES with human-facing names, in declaration order."""
    names = [
        "single word",
        "command",
        "query state",
        "move",
        "subject predicate",
        "pronoun state",
        "warning",
    ]
    if len(names) != len(grammar.TEMPLATES):
        raise SystemExit(
            f"template names ({len(names)}) out of sync with "
            f"grammar.TEMPLATES ({len(grammar.TEMPLATES)})"
        )
    return list(zip(names, grammar.TEMPLATES))


if __name__ == "__main__":
    main()
