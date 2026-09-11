# B4-BL ("Babble")

An intelligible acoustic language for robots, styled after Star Wars astromech
droidspeak. B4-BL is both a constructed language and a low-bandwidth,
human-perceivable, spatially-local side channel for embodied machines.

Two acoustic layers travel together:

- a **symbolic layer** — sparse, robust, exactly machine-decodable;
- an **expressive layer** — prosody/affect (confidence, urgency) that a human can
  read but a machine may ignore.

The goal: a human who lives around the droids gradually learns roughly *what is
going on*, while a droid recovers the *exact packet*.

## Status

Early. Phase 0 (sound design) is done — the sound palette is chosen and
implemented. Layers 1–3 exist as a working symbolic implementation with audio
output. The acoustic decoder (microphone → symbols) is not built yet.

| Layer | What | State |
|-------|------|-------|
| Sound palette | pure-tonal + rough generators | ✅ implemented, ear-tested |
| L1 Phonology | ~16 acoustic primitives w/ shared synth+recognizer feature model | ✅ inventory + render; classifier pending |
| L2 Lexicon | morpheme vocabulary + spelling fallback | ✅ encode + symbolic decode |
| L3 Prosody | affect as transforms on expressive dimensions | ✅ implemented |
| Codec | meaning ↔ phonemes ↔ audio | ✅ encode + symbolic round-trip; ⛔ acoustic decode |
| Protocol | preamble/identity/seq/FEC/ACK | ⛔ not started |

## Layout

```
b4bl/
  generators.py   the locked Phase 0 sound palette (tonal + rough families)
  phonology.py    Layer 1: acoustic primitive inventory (the "phonemes")
  lexicon.py      Layer 2: morpheme vocabulary + grammar + spelling fallback
  prosody.py      Layer 3: affect as transforms on expressive dimensions
  codec.py        meaning <-> phoneme-sequence <-> audio
src/
  astromech_synth.py   Phase 0 A/B/C style comparison (historical)
  whistle_synth.py     Phase 0 whistle-direction study
  palette_synth.py     the palette listening-test renderer
  demo_language.py     renders sentences + prosody + inventory to audio
tests/            symbolic round-trip + inventory sanity tests
audio_samples/    rendered demos (WAV/MP3)
```

## Try it

```bash
pip install numpy scipy          # only hard deps
python3 src/demo_language.py     # renders sentences, prosody, inventory to audio_samples/
python3 -m pytest -q             # round-trip + unambiguity tests
```

Then listen to `audio_samples/`:
- `phoneme_inventory.*` — the ~16 acoustic primitives in a row
- `sentences_neutral.*` — example sentences ("is your energy low?", "warning: obstacle ahead", …)
- `prosody_compare.*` — the SAME sentence rendered neutral / uncertain / urgent / calm

## Design notes

Phase 0 listening tests settled the sound direction:

- **Pure tone reads as R2.** Added harmonics / ring-modulation read as
  "kazoo / SNES sound effects." Character lives in the **pitch gesture**, not the timbre.
- Sample-and-hold random burble read as low-budget sci-fi → dropped.
- Grit is per-family: wrong on whistles, right on raspberries.

The full plan (three layers, phased roadmap, open decisions) lives in the
project's design docs. Key open decision: exactly which acoustic dimensions are
lexical vs. expressive — the current split is documented in `b4bl/phonology.py`
as a reviewed proposal.

## License

MIT. All audio is newly synthesized; no film recordings are used or redistributed.
