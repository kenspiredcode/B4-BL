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

## Status (2026-09-16)

The sound palette, vocabulary, synthesis, and acoustic decoder exist. Reliable
multi-word room communication is **not solved**. Older 95% figures were small
loopback experiments and must not be read as real-channel delivery accuracy.

A reproducible offline replay of 2,267 existing slotted recordings gives:

| Exact multi-word messages | Before alias fix | After alias fix |
|---|---:|---:|
| Close/clean, 291 messages | 71.5% | 77.0% |
| Distant/noisy, 287 messages | 40.4% | 44.9% |

These are **retrospective** scores with a model trained on these collections.
A separate model trained without the noisy round or a deterministic 20% of concept
sequences scores 81.8% on 66 unseen close-channel multi-word sequences and 29.6%
on 287 noisy-channel multi-word messages. Historical channel tags stand in for
missing session IDs; this is held out from model training, not a prospectively
unseen environment. Reported word error rate includes insertions and may exceed 100%.

`codec.encode(slots=True)` is the historical widened-gap format; its receiver still
segments phonemes by energy. Oversized gestures now render completely at the slot
body duration instead of losing their tails. New captures of this corrected format
must use a distinct channel/version, and the historical model may need retraining.

`b4bl.clocked` is a **separate experimental format**: tonal word markers, a real
300 ms clock (SHORT=1 tick, LONG=2), complete contours, and optional repetition.
It has no microphone validation or listening approval. On 132 fresh synthetic
cases crossing lengths 2–5 with all four prosodies, accepted exact decoding is 99.2%
clean, 84.8% with moderate echo and 15 dB nominal noise, 9.1% under the harsher 5 dB
condition, and 86.4% at 0.1% sample-clock skew. Best hypotheses reach 91.7% in the
moderate condition, but some are rejected by the uncalibrated confidence gate.
Five of 132 moderate-condition messages are incorrectly accepted; acoustic
acceptance must not be treated as CRC-verified delivery.
Triple repetition did not improve the moderate condition and increases mean
utterance duration from 4.41 to 12.58 seconds; it remains optional, off by default.

`b4bl.verified_clocked` adds experimental CRC32 protection over headers and payload.
It is deliberately verbose (a three-word payload example takes 30.28 seconds).
It is not an acoustic ACK/retry implementation. Fixed tempo, rhythm-defined
ALARM/WORKING, spelling support, compact protection, and realistic noisy-room
performance remain open. Rhythm-defined words/spelling are explicitly unsupported
by the new transport, not silently changed.

No audio was played or recorded during the offline work. Full findings and the
next capture plan live in the canonical vault document
`~/CollabHarnessVault/projects/B4-BL/docs/offline-validation-2026-09-16.md`.

## Silent evaluation

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
# Needs the existing recordings and historical model; never plays audio.
python3 -m b4bl.benchmark --output experiments/my-offline-run
python3 -m b4bl.reference_audit --cache experiments/my-offline-run
# Trains a separate synthetic clock-window model and tests without device I/O.
python3 -m b4bl.clocked_benchmark --output experiments/my-clocked-run
# Fresh evaluation of an already frozen model:
python3 -m b4bl.clocked_benchmark --model experiments/my-clocked-run/clocked_classifier.joblib --output experiments/my-clocked-holdout --seed 180027 --messages 120
# Prepares the future capture corpus only. --dry-run guarantees no playback.
python3 src/collect_dataset.py --clocked --dry-run --channel clocked_office_v1 --limit 500
```

Output directories must be new, so runs cannot silently overwrite earlier results.
JSON summaries are retained in `experiments/`; recording-level predictions,
feature caches, and model snapshots stay local and are ignored by Git. The
historical default classifier is not overwritten by these benchmark commands.
Random forests now validate with one shared split grouping entire recordings,
not separate phoneme-row splits for each output dimension. That internal split
still does not replace a held-out capture session/channel benchmark.

The clocked receiver estimates one affine sample-rate scale from all detected
marker intervals (within a fixed ±4% search range), then decodes windows on that
received clock. This handles cumulative drift such as a 0.1% sample-rate mismatch;
the estimate and residual are returned on `DecodeResult`. It remains conservative
about missing or extra markers.

Future `--clocked` capture runs record session IDs, format/source fingerprints,
raw audio, transmitter timing labels, device settings, and failed-attempt logs.
Actual capture must be started separately for an overnight burst. Keep room/session
test sets out of training before any decoder tuning. Source-derived alignment of
old recordings is diagnostic only; it is not verified oracle segmentation.

The prepared overnight command is intentionally explicit and loudness-controlled;
run it only when ready, with a fresh channel tag:

```bash
python3 src/collect_dataset.py --clocked --channel clocked_office_v1 \
  --limit 500 --input-device AUKEY --output-device "<speaker>" \
  --gain 2 --max-hang-streak 8 --hang-pause 3
```

The collector self-tests first, refuses to mix formats in a channel, logs every
attempt, saves raw capture plus transmitter timing, and can resume after hangs.
Use `--dry-run` to inspect the corpus without touching audio devices.

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
pip install -r requirements.txt
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
