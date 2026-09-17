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

## Status (2026-09-17)

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
Its timing samples have listening approval, and real recordings now cover MacBook
and AUKEY microphones, Bluetooth playback, and multiple rooms. On 132 fresh synthetic
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

The initial offline work used no new captures; subsequent collection was explicitly
authorized and includes preserved raw audio even for trim-rejected attempts.
Initial findings live in the canonical vault document
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
  --gain 1 --max-hang-streak 8 --hang-pause 3
```

The collector self-tests first, refuses to mix formats in a channel, logs every
attempt, saves raw capture plus transmitter timing, and can resume after hangs.
Use `--dry-run` to inspect the corpus without touching audio devices.
Explicit device selectors now reject unknown/ambiguous names; numeric device IDs
are supported. The self-test locates a sustained 1 kHz tone instead of assuming
the loudest captured sound is the test signal.

### Real clocked recordings

`b4bl.clocked_receiver` is an experimental frontend that detects the marker's
spectral sweep rather than waveform phase and filters low-frequency microphone
noise before phoneme classification. It does not change transmitted sounds and
does not replace the historical/default decoder.

The frozen attempt-level split retains capture failures, groups identical concept
sequences across prosodies/channels, and excludes all room 3 audio from training:

```bash
python3 -m b4bl.real_clocked_experiment \
  --split experiments/real-clocked-20260917/split.json \
  --output experiments/my-real-clocked-run
```

This silent experiment saves a separate model and compares the original receiver,
the new frontend alone, and real-data training on held-out recordings. Load its
model with `joblib.load` and use `b4bl.clocked_receiver.decode(audio, model)`;
`read_audio(path)` handles both integer and floating-point WAVs. Earlier room 3
diagnostic pilots are reported separately from the full collection. The room is
not a pristine unseen development environment because those pilots informed
diagnosis. Acoustic acceptance remains distinct from CRC-verified delivery.

The first frozen comparison (`experiments/real-clocked-20260917/run1/summary.json`)
trained on 1,191 aligned recordings from three configurations. Exact multi-word
recovery on composition holdouts was 68/68 MacBook→AUKEY, 68/68 Bluetooth→AUKEY,
and 67/68 Bluetooth→MacBook in room 2. The full room 3 run, excluded from training,
scored 232/328 (70.7%), including a missing capture as unsuccessful. Its 327
available multi-word raw recordings all had the expected marker count, but 23
messages were wrongly accepted and 72 rejected. This is strong same-environment
recognition, not 90% generalization to new rooms. No waveform redesign was needed.

Continued development found that the pitch tracker was treating low-periodicity
reverb/noise after a tone release as additional pitch. A versioned feature profile
that ignores frames below 0.5 periodicity raises the training-excluded room 3
multi-word result to 282/328 accepted-exact (86.0%), with 11 wrong acceptances,
34 rejections, and one missing capture. Its best hypotheses are correct on 300/328
(91.5%). The correct whole message is within the top two candidates per word in
320/328 (97.6%) and within the top three in 325/328 (99.1%); those are diagnostic
oracle bounds, not delivered accuracy.

`verified_clocked.decode` now supports bounded candidate-list search. An alternate
path is accepted only when the complete packet is canonical and its CRC validates.
This mechanism is tested symbolically; the project still needs real captures of
protected packets before claiming the oracle bounds as verified delivery.

When room 3 non-holdout recordings are included in training, its reserved
multi-word compositions score 67/68 (98.5% including one missing capture; 67/67
available), with no wrong acceptances. Other represented channels remain at
98.5–100%. This is the practical calibrated-environment path above 90%; the 86.0%
leave-one-room-out result remains the honest cross-room development figure.

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
