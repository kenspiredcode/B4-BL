# B4-BL demo site

Static site served by GitHub Pages from this `docs/` folder. No build tooling, no
framework, no server — plain ES modules and WebAudio.

## Publishing

GitHub repo → Settings → Pages → Source: `main` branch, `/docs` folder.

## Regenerating

```bash
python3 tools_build_demo.py     # from the repo root
```

That regenerates `data/language.json`, re-renders the annotated packet figure, and
stages the audio samples into `audio/`. Run it after any change to `lexicon.py`,
`phonology.py`, `grammar.py`, or `compact_clocked.py`.

## Structure

```
index.html          page skeleton, section by section
css/style.css       all styling
js/main.js          entry point; loads language.json, inits modules
js/hear.js          four-prosody players (static audio, no synthesis)
js/builder.js       grammar-driven sentence builder
js/synth.js         WebAudio port of generators.py + codec.encode
js/spectrogram.js   canvas FFT spectrogram for the builder
js/inventory.js     phoneme grid + vocabulary tables
data/language.json  GENERATED — do not hand-edit
audio/              GENERATED — staged from ../audio_samples/
img/                GENERATED — annotated packet spectrogram
```

## Ground rules

**`data/language.json` is generated.** The demo must never hand-maintain a copy of
the lexicon or phoneme inventory. If the demo and the Python implementation disagree
about what a word sounds like, the demo is wrong and the trained decoder is the reason
it matters.

**`js/synth.js` is a port, not a reimplementation.** It must produce audio equivalent
to `codec.encode` for the same concepts and prosody. Changes to how B4-BL *sounds* go
into the Python implementation first, because the decoder was trained on it.

`tools_verify_port.py` proves this numerically: it renders the same cases in both
implementations and compares sample by sample. Current agreement is 5.96e-08 (float32
rounding). `ERROR` is the one exception — it uses the RASP class, which mixes white
noise and approximates scipy's Butterworth filter, so it matches perceptually rather
than exactly. The check also renders all 215 morphemes, so a phoneme missing from the
export cannot silently reach the page. It runs as part of `tools_build_demo.py`.

**No decoder here.** Decoding needs a ~140 MB trained classifier. The demo is
transmit-only by design; the receiver lives in the repository.

## Build status

- [x] Page skeleton and styling
- [x] Generated language data
- [x] Audio staging
- [x] `hear.js` — prosody players
- [x] `synth.js` — WebAudio synthesis port (verified sample-accurate)
- [x] `builder.js` — sentence builder
- [x] `inventory.js` — phoneme grid + vocab tables
- [x] Annotated packet image (light + dark)
- [x] Live spectrogram in the builder
- [ ] Publish to GitHub Pages and link from the README
